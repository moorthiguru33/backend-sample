"""
main.py – PSD Pipeline Orchestrator (Excel-based tracking).

╔══════════════════════════════════════════════════════════════════════╗
║  HOW IT WORKS — EVERY RUN                                           ║
║                                                                      ║
║  PHASE 1 — COLLECT                                                   ║
║  • Load psd_tracker.xlsx from GitHub                                 ║
║  • Count pending rows                                                ║
║  • If not enough pending (or nothing at all): SCRAPE forpsd.com     ║
║    → Add new URLs with status="pending"                              ║
║    → Push updated Excel to GitHub                                    ║
║                                                                      ║
║  PHASE 2 — PROCESS                                                   ║
║  • Loop through pending rows (up to ITEM_LIMIT per run)             ║
║    a. Resolve category from detail page (if not already set)         ║
║    b. Resolve GDrive download URL                                    ║
║    c. Download archive (ZIP/RAR/7Z)                                  ║
║    d. Extract → rename files to tamilpsd-XXXX                       ║
║    e. Upload renamed PSD/TIF to GDrive/<category>/ subfolder        ║
║    f. Mark row "completed" in Excel → push to GitHub immediately     ║
║                                                                      ║
║  PHASE 3 — RESCRAPE (when all pending done)                         ║
║  • Re-scrape forpsd.com for new links                                ║
║  • Add only truly new URLs (not already in Excel)                   ║
║  • If new links found → push Excel → pipeline continues next run    ║
║  • If nothing new → touch ALL_DONE → stop automation               ║
╚══════════════════════════════════════════════════════════════════════╝

Error rows:
  • Rows with status="error" are automatically retried every run.
  • After MAX_RETRIES consecutive errors on a URL it stays "error" (skipped).

File counter:
  • Derived from max tamilpsd-N in the Excel "completed" rows.
  • Also cross-checked against Drive scan and Excel log.
  • Guaranteed to never collide even after state loss.
"""

import logging
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FORPSD_COOKIE,
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN,
    GDRIVE_PSD_FOLDER,
    RUN_MINUTES, PAGE_LIMIT, ITEM_LIMIT, WORK_DIR, DONE_FILE, LOG_FILE,
)
from psd_tracker import PSDTracker
from scraper     import ForPSDScraper
from downloader  import download_from_drive
from processor   import process_archive
from uploader    import DriveUploader

import os
GH_PAT       = os.environ.get("GH_PAT", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
CONTENT_REPO = os.environ.get("CONTENT_REPO", "moorthiguru33/Gurumoorthi")

# ── Constants ──────────────────────────────────────────────────────────────────
POSTS_PER_PAGE = 17   # forpsd.com items per listing page
TRACKER_PATH   = str(Path(WORK_DIR) / "psd_tracker.xlsx")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ── Secret validation ──────────────────────────────────────────────────────────

def check_secrets() -> bool:
    missing = []
    if not FORPSD_COOKIE:       missing.append("FORPSD_COOKIE")
    if not GOOGLE_CLIENT_ID:    missing.append("GOOGLE_CLIENT_ID")
    if not GOOGLE_CLIENT_SECRET:missing.append("GOOGLE_CLIENT_SECRET")
    if not GOOGLE_REFRESH_TOKEN:missing.append("GOOGLE_REFRESH_TOKEN")
    if not GDRIVE_PSD_FOLDER:   missing.append("GDRIVE_PSD_FOLDER")
    if not (GH_PAT or GITHUB_TOKEN):
        missing.append("GH_PAT or GITHUB_TOKEN")
    if missing:
        log.error(f"Missing GitHub Secrets: {', '.join(missing)}")
        return False
    return True


# ── Phase 1: Scrape + add to tracker ──────────────────────────────────────────

def collect_new_items(
    tracker: PSDTracker,
    scraper: ForPSDScraper,
    need_count: int,
) -> int:
    """
    Scrape forpsd.com and add new items to the tracker.
    `need_count`: how many NEW items we want (0 = scrape all pages).
    Returns number of new items actually added.
    """
    existing_urls = tracker.all_urls()
    completed     = sum(1 for r in tracker._rows if r.get("status") == "completed")
    start_page    = (completed // POSTS_PER_PAGE) + 1

    log.info(
        f"🔍 Scraping forpsd.com — "
        f"start_page={start_page} | "
        f"need={need_count if need_count > 0 else 'unlimited'} new items | "
        f"known URLs={len(existing_urls)}"
    )

    scraped = scraper.get_all_items(
        page_limit    = PAGE_LIMIT,
        stop_at_known_urls = existing_urls if existing_urls else None,
        start_page    = start_page,
        max_new_items = need_count,
    )

    if not scraped:
        log.info("  No new items found on forpsd.com")
        return 0

    added = tracker.add_items(scraped, push=False)   # bulk push at the end
    log.info(f"  ✅ Scraped {len(scraped)} items → {added} new added to tracker")
    return added


# ── Archive original-name extraction ──────────────────────────────────────────

import zipfile
import rarfile
import py7zr

def _list_archive_originals(archive_path: Path) -> list[str]:
    """Return PSD/TIF/TIFF filenames inside the archive (in extraction order)."""
    TARGET_EXT = {".psd", ".tif", ".tiff"}
    names: list[str] = []
    try:
        suffix = archive_path.suffix.lower()
        if suffix == ".zip":
            with zipfile.ZipFile(archive_path) as zf:
                for info in zf.infolist():
                    p = Path(info.filename)
                    if p.suffix.lower() in TARGET_EXT and not info.is_dir():
                        names.append(p.name)
        elif suffix == ".rar":
            with rarfile.RarFile(archive_path) as rf:
                for info in rf.infolist():
                    p = Path(info.filename)
                    if p.suffix.lower() in TARGET_EXT and not info.is_dir():
                        names.append(p.name)
        elif suffix == ".7z":
            with py7zr.SevenZipFile(archive_path, mode="r") as sz:
                for entry in sz.list():
                    p = Path(entry.filename)
                    if p.suffix.lower() in TARGET_EXT and not entry.is_directory:
                        names.append(p.name)
    except Exception as exc:
        log.warning(f"  Could not list archive contents for {archive_path.name}: {exc}")
    return names


# ── Phase 2: Process one pending item ─────────────────────────────────────────

def process_one_item(
    item:       dict,
    tracker:    PSDTracker,
    scraper:    ForPSDScraper,
    uploader:   DriveUploader,
    file_counter: int,
    work_root:  Path,
) -> tuple[bool, int]:
    """
    Process a single pending item end-to-end.

    Returns (success: bool, next_file_counter: int).
    On success  → marks completed + pushes Excel.
    On failure  → marks error   + pushes Excel.
    """
    url        = item["url"]
    detail_url = item.get("detail_url", "")
    card_title = item.get("card_title", "")
    category   = item.get("category", "")

    item_dir = work_root / f"item_{int(time.time() * 1000)}"
    item_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Step a: Resolve category ───────────────────────────────────────
        if not category:
            log.info(f"  📂 Resolving category …")
            category = scraper.get_category(detail_url, hint_title=card_title)
            tracker.update_category(url, category)
        log.info(f"  📂 Category: [{category}]")

        # ── Step b: Resolve GDrive URL ─────────────────────────────────────
        drive_url = scraper.resolve_drive_url(url)
        if not drive_url:
            raise RuntimeError("Could not resolve Google Drive download URL")
        log.info(f"  🔗 Drive URL: {drive_url}")

        # ── Step c: Download archive ───────────────────────────────────────
        dl_dir  = item_dir / "download"
        archive = download_from_drive(
            drive_url, dl_dir,
            client_id     = GOOGLE_CLIENT_ID,
            client_secret = GOOGLE_CLIENT_SECRET,
            refresh_token = GOOGLE_REFRESH_TOKEN,
        )
        if not archive:
            raise RuntimeError("Archive download failed")

        # ── Step d: Read original filenames before extraction ──────────────
        original_names = _list_archive_originals(archive)
        log.info(f"  📦 Archive: {archive.name} → {len(original_names)} PSD/TIF file(s)")

        # Skip if ALL originals already done (all uploaded in a previous run)
        if original_names and all(
            tracker.is_original_done(n) for n in original_names
        ):
            log.info(
                f"  ⏭  All {len(original_names)} file(s) already uploaded "
                f"(found in completed rows) — marking done"
            )
            tracker.mark_completed(
                url, original_names, [], [],   # no new renamed files this time
                push=True,
            )
            return True, file_counter

        # ── Step e: Extract + rename ───────────────────────────────────────
        result = process_archive(
            archive_path = archive,
            work_dir     = item_dir / "process",
            start_index  = file_counter,
        )
        if not result:
            raise RuntimeError(f"process_archive failed for {archive.name}")

        renamed_files, next_index = result

        # ── Step f: Build original→renamed mapping ─────────────────────────
        orig_map: dict[Path, str] = {}
        for idx, renamed_path in enumerate(renamed_files):
            orig_name = original_names[idx] if idx < len(original_names) else renamed_path.name
            orig_map[renamed_path] = orig_name

        # ── Step g: Upload each file to GDrive/<category>/ ─────────────────
        uploaded_originals: list[str] = []
        uploaded_renamed:   list[str] = []
        uploaded_links:     list[str] = []

        for renamed_path, orig_name in orig_map.items():
            if not renamed_path.exists():
                log.warning(f"  ⚠  File missing after rename: {renamed_path}")
                continue

            log.info(f"  ☁️  Uploading: {renamed_path.name}  →  [{category}/]")
            upload_result = uploader.upload_to_category(
                file_path        = renamed_path,
                parent_folder_id = GDRIVE_PSD_FOLDER,
                category         = category,
                # Pass tracker as excel_tracker for Drive-level dedup
                excel_tracker    = _TrackerAdapter(tracker),
                original_name    = orig_name,
            )

            if upload_result.get("skipped"):
                reason = upload_result.get("reason", "")
                log.info(f"  ⏭  Skipped ({reason}): {renamed_path.name}")
                # Still count as processed so we don't retry forever
                uploaded_originals.append(orig_name)
                uploaded_renamed.append(renamed_path.name)
                uploaded_links.append("")
            else:
                link = upload_result.get("webViewLink", "")
                uploaded_originals.append(orig_name)
                uploaded_renamed.append(renamed_path.name)
                uploaded_links.append(link)

        if not uploaded_originals:
            raise RuntimeError("No files were uploaded or skipped — nothing to record")

        # ── Step h: Mark completed + push Excel ────────────────────────────
        tracker.mark_completed(
            url,
            uploaded_originals,
            uploaded_renamed,
            uploaded_links,
            push=True,
        )

        log.info(
            f"  ✅ Done: {len(uploaded_renamed)} file(s) uploaded | "
            f"Next counter: tamilpsd-{next_index:04d}"
        )
        return True, next_index

    except Exception as exc:
        log.error(f"  ❌ Error: {exc}", exc_info=True)
        tracker.mark_error(url, str(exc)[:500], push=True)
        return False, file_counter

    finally:
        shutil.rmtree(item_dir, ignore_errors=True)


# ── Adapter: make PSDTracker look like ExcelTracker for DriveUploader ─────────

class _TrackerAdapter:
    """
    Thin adapter so DriveUploader.upload_to_category() can use PSDTracker
    for its Excel-based duplicate checks (it expects .is_original_done and
    .is_renamed_used and .add_entry methods).
    """
    def __init__(self, tracker: PSDTracker):
        self._tracker = tracker

    def is_original_done(self, original_filename: str) -> bool:
        return self._tracker.is_original_done(original_filename)

    def is_renamed_used(self, renamed_filename: str) -> bool:
        return self._tracker.is_renamed_used(renamed_filename)

    def add_entry(self, original_filename: str, renamed_filename: str) -> None:
        # The tracker updates its in-memory sets when mark_completed is called.
        # This no-op satisfies the uploader interface during upload.
        pass

    def stats(self) -> str:
        return self._tracker.stats()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not check_secrets():
        sys.exit(2)

    start_time = datetime.now(timezone.utc)
    deadline   = start_time + timedelta(minutes=RUN_MINUTES)

    log.info("=" * 70)
    log.info(f"🚀 PSD Pipeline started at {start_time.strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"   Deadline: {deadline.strftime('%H:%M UTC')} ({RUN_MINUTES} min)")
    log.info(f"   Item limit this run: {ITEM_LIMIT if ITEM_LIMIT > 0 else 'unlimited'}")
    log.info("=" * 70)

    work_root = Path(WORK_DIR)
    work_root.mkdir(parents=True, exist_ok=True)

    gh_token = GH_PAT or GITHUB_TOKEN
    scraper  = ForPSDScraper(FORPSD_COOKIE)
    uploader = DriveUploader(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN)

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 1 — Load tracker from GitHub
    # ══════════════════════════════════════════════════════════════════════
    log.info("\n📥 STEP 1 — Loading psd_tracker.xlsx from GitHub …")
    tracker = PSDTracker(
        local_path    = TRACKER_PATH,
        github_token  = gh_token,
        github_repo   = CONTENT_REPO,
    )
    tracker.load_from_github()

    # Retry error rows from previous run
    retried = tracker.reset_errors_to_pending()
    if retried:
        log.info(f"  🔄 {retried} error rows reset → pending (will retry)")

    log.info(f"  📊 {tracker.stats()}")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2 — Pre-scan Drive for existing files (counter + skip guard)
    # ══════════════════════════════════════════════════════════════════════
    log.info("\n🔍 STEP 2 — Pre-scanning Google Drive for existing files …")
    uploader.preload_existing_names(GDRIVE_PSD_FOLDER)

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 3 — Collect: scrape if not enough pending items
    # ══════════════════════════════════════════════════════════════════════
    log.info("\n📋 STEP 3 — Checking if we have enough pending items …")

    pending = tracker.get_pending()
    need_scrape = (
        (ITEM_LIMIT == 0 and len(pending) == 0) or
        (ITEM_LIMIT >  0 and len(pending) < ITEM_LIMIT)
    )

    if need_scrape:
        deficit = (ITEM_LIMIT - len(pending)) if ITEM_LIMIT > 0 else 0
        log.info(
            f"  Only {len(pending)} pending | need {ITEM_LIMIT or 'unlimited'} | "
            f"scraping {deficit or 'all'} more …"
        )
        added = collect_new_items(tracker, scraper, need_count=deficit)

        if added > 0:
            # Push updated Excel with new pending rows to GitHub
            tracker.push_to_github(
                f"tracker: +{added} new pending items scraped [{start_time.strftime('%Y-%m-%d %H:%M UTC')}]"
            )
        else:
            log.info("  No new items found on site. Checking remaining pending …")

        pending = tracker.get_pending()
    else:
        log.info(f"  ✅ Enough pending ({len(pending)}) — no scrape needed this run")

    log.info(f"  📊 {tracker.stats()}")

    if not pending:
        log.info("\n✅ All items processed and no new items found — pipeline complete!")
        log.info("   Touching ALL_DONE to stop automation trigger.")
        Path(DONE_FILE).touch()
        return

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 4 — Process pending items
    # ══════════════════════════════════════════════════════════════════════
    log.info(f"\n🤖 STEP 4 — Processing {len(pending)} pending item(s) …")

    # Sync file_counter from all sources
    file_counter = max(
        tracker.max_counter + 1,
        uploader.max_counter + 1,
        1,
    )
    log.info(
        f"  File counter starts: tamilpsd-{file_counter:04d} "
        f"(tracker max={tracker.max_counter:04d}, "
        f"Drive max={uploader.max_counter:04d})"
    )

    processed_this_run = 0
    errors_this_run    = 0
    limit              = ITEM_LIMIT if ITEM_LIMIT > 0 else len(pending)

    for item in pending:

        # ── Time limit ─────────────────────────────────────────────────────
        if datetime.now(timezone.utc) >= deadline:
            log.info(
                f"\n⏰ Time limit reached — "
                f"processed {processed_this_run} this run. "
                f"Remaining: {len(tracker.get_pending())} pending."
            )
            break

        # ── Item limit ─────────────────────────────────────────────────────
        if processed_this_run >= limit:
            log.info(
                f"\n🔢 Item limit ({limit}) reached — "
                f"processed {processed_this_run} this run. "
                f"Remaining: {len(tracker.get_pending())} pending."
            )
            break

        log.info(
            f"\n{'─' * 60}\n"
            f"  [{processed_this_run + 1}/{limit}]  {item['url'][:70]}\n"
            f"  Category (current): [{item.get('category') or 'TBD'}]\n"
            f"{'─' * 60}"
        )

        success, file_counter = process_one_item(
            item         = item,
            tracker      = tracker,
            scraper      = scraper,
            uploader     = uploader,
            file_counter = file_counter,
            work_root    = work_root,
        )

        if success:
            processed_this_run += 1
        else:
            errors_this_run += 1

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 5 — Final status
    # ══════════════════════════════════════════════════════════════════════
    remaining = len(tracker.get_pending())
    log.info("\n" + "=" * 70)
    log.info(f"✅ Run finished at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"   Processed this run : {processed_this_run}")
    log.info(f"   Errors this run    : {errors_this_run}")
    log.info(f"   Pending remaining  : {remaining}")
    log.info(f"   📊 {tracker.stats()}")
    log.info("=" * 70)

    # ── Phase 3: If nothing pending, re-scrape for new items ──────────────
    if remaining == 0:
        log.info("\n🔁 PHASE 3 — All pending done! Re-scraping for new items …")
        added = collect_new_items(tracker, scraper, need_count=0)

        if added > 0:
            tracker.push_to_github(
                f"tracker: +{added} new items found on re-scrape [{_now_iso()}]"
            )
            log.info(f"  🆕 {added} new items found — pipeline will continue next run")
        else:
            log.info("  ✅ No new items on site — everything is done!")
            log.info("     Touching ALL_DONE to stop automation.")
            Path(DONE_FILE).touch()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


if __name__ == "__main__":
    main()
