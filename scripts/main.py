"""
main.py – Orchestrator (JobTracker Excel-primary, category-aware, continuous counter).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SOURCE OF TRUTH: jobs.xlsx  (job_tracker.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Flow per run
────────────
  1. Load jobs.xlsx (JobTracker) + rename_log.xlsx (ExcelTracker).
  2. Determine pending items:
       A. No items in jobs.xlsx → FULL scrape (all pages), batch-save to xlsx.
       B. Pending items exist → use them directly (no scrape needed).
       C. All Completed → re-scrape from page 1 (stop early at known URLs).
  3. Loop over pending:
       ① Skip if jobs.xlsx Status=Completed         (fastest, 0 API calls)
       ② Skip if ALL archive files in rename_log    (0 API calls)
       ③ Resolve category → download → extract → upload
       ④ mark_completed in jobs.xlsx (targeted cell update — fast)

Skip logic (fastest-first)
───────────────────────────
  ① jobs.xlsx Status=Completed      → skip (0 API calls)
  ② rename_log.xlsx original-name   → skip (0 API calls)
  ③ Drive preloaded name cache       → skip (0 API calls)
  ④ Live Drive query per file        → skip (handles interrupted runs)

File counter continuity
───────────────────────
  counter = max(state.json, Drive scan max, rename_log.xlsx max) + 1
  state.json is now minimal: only stores file_counter + last_run.

FIX: Fresh download URL on every run
──────────────────────────────────────
  forpsd.com encodes a JWT token in the /download/ URL that can expire
  between scrape time and processing time. When resolve_drive_url() fails,
  the code now re-scrapes the product detail page to obtain a fresh
  /download/ URL before giving up and marking the item as Error.
"""

import logging
import shutil
import sys
import time
import zipfile
import rarfile
import py7zr
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FORPSD_COOKIE,
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN,
    GDRIVE_PSD_FOLDER,
    RUN_MINUTES, PAGE_LIMIT, ITEM_LIMIT, WORK_DIR, STATE_FILE, DONE_FILE, LOG_FILE,
)
from state_manager import StateManager
from scraper import ForPSDScraper
from downloader import download_from_drive
from processor import process_archive
from uploader import DriveUploader
from excel_tracker import ExcelTracker
from job_tracker import JobTracker

POSTS_PER_PAGE: int = 17

_REPO_ROOT      = Path(STATE_FILE).parent
EXCEL_LOG_FILE  = str(_REPO_ROOT / "rename_log.xlsx")
JOBS_EXCEL_FILE = str(_REPO_ROOT / "jobs.xlsx")

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ── Secrets check ──────────────────────────────────────────────────────────

def check_secrets() -> bool:
    missing = []
    if not FORPSD_COOKIE:         missing.append("FORPSD_COOKIE")
    if not GOOGLE_CLIENT_ID:      missing.append("GOOGLE_CLIENT_ID")
    if not GOOGLE_CLIENT_SECRET:  missing.append("GOOGLE_CLIENT_SECRET")
    if not GOOGLE_REFRESH_TOKEN:  missing.append("GOOGLE_REFRESH_TOKEN")
    if not GDRIVE_PSD_FOLDER:     missing.append("GDRIVE_PSD_FOLDER")
    if missing:
        log.error(f"Missing GitHub Secrets: {', '.join(missing)}")
        return False
    return True


# ── Scraping ───────────────────────────────────────────────────────────────

def _full_scrape(scraper: ForPSDScraper, job_tracker: JobTracker) -> int:
    log.info("🌐 Full scrape: page 1 → end (all pages, no stop condition)…")
    all_items = scraper.get_all_items(
        page_limit=PAGE_LIMIT,
        stop_at_known_urls=None,
        start_page=1,
        max_new_items=0,
    )
    if not all_items:
        log.warning("Full scrape returned 0 items — check FORPSD_COOKIE and site.")
        return 0

    log.info(f"Full scrape complete: {len(all_items)} items found. Saving to jobs.xlsx…")
    added = job_tracker.add_pending_items(all_items)
    log.info(f"📋 jobs.xlsx: +{added} new Pending (total: {job_tracker.total()})")
    return added


def _rescrape_for_new(scraper: ForPSDScraper, job_tracker: JobTracker,
                      item_limit: int) -> int:
    known_urls = job_tracker.all_known_urls()
    log.info(
        f"🌐 Re-scrape from page 1 (looking for items not in {len(known_urls)} known URLs)…"
    )
    new_items = scraper.get_all_items(
        page_limit=PAGE_LIMIT,
        stop_at_known_urls=known_urls,
        start_page=1,
        max_new_items=item_limit if item_limit > 0 else 0,
    )
    if not new_items:
        return 0
    added = job_tracker.add_pending_items(new_items)
    log.info(f"Re-scrape done: found {len(new_items)}, +{added} new Pending.")
    return added


def _topup_scrape(scraper: ForPSDScraper, job_tracker: JobTracker,
                  item_limit: int) -> int:
    known_urls = job_tracker.all_known_urls()
    done_count = job_tracker.completed_count()
    start_page = max(1, (done_count // POSTS_PER_PAGE))
    needed     = item_limit - len(job_tracker.get_pending())

    log.info(
        f"🌐 Top-up scrape: start_page={start_page}, need {needed} more items…"
    )
    new_items = scraper.get_all_items(
        page_limit=PAGE_LIMIT,
        stop_at_known_urls=known_urls,
        start_page=start_page,
        max_new_items=needed,
    )
    if not new_items:
        return 0
    added = job_tracker.add_pending_items(new_items)
    log.info(f"Top-up done: +{added} new Pending.")
    return added


# ── Archive helpers ────────────────────────────────────────────────────────

def _list_archive_originals(archive_path: Path) -> list[str]:
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
        log.warning(f"  Could not list archive: {archive_path.name}: {exc}")
    return names


# ── FIX: Resolve Drive URL with automatic fresh-URL fallback ──────────────

def _resolve_drive_url_with_retry(
    scraper: ForPSDScraper,
    download_url: str,
    detail_url: str,
) -> str | None:
    """
    Resolve the forpsd /download/ URL to a Google Drive URL.

    Step 1: Try the stored download_url directly.
    Step 2: If that fails and we have a detail_url, re-scrape the product
            page to get a fresh /download/ URL (the JWT token may have
            expired since it was scraped), then try resolving again.
    Returns the Google Drive URL, or None if both attempts fail.
    """
    # ── Attempt 1: stored URL ─────────────────────────────────────────────
    drive_url = scraper.resolve_drive_url(download_url)
    if drive_url:
        return drive_url

    # ── Attempt 2: fresh URL from detail page ─────────────────────────────
    if not detail_url:
        log.warning("  resolve_drive_url failed and no detail_url available — cannot retry")
        return None

    log.warning("  Stored download URL failed — fetching fresh URL from detail page…")
    fresh_dl_url = scraper.get_fresh_download_url(detail_url)
    if not fresh_dl_url:
        log.warning("  Could not obtain fresh download URL from detail page")
        return None

    if fresh_dl_url == download_url:
        # Same URL — site hasn't changed, permanent failure
        log.warning("  Fresh URL is identical to stored URL — permanent failure")
        return None

    log.info(f"  Retrying with fresh URL: {fresh_dl_url}")
    drive_url = scraper.resolve_drive_url(fresh_dl_url)
    if drive_url:
        log.info(f"  ✅ Fresh URL resolved successfully")
    else:
        log.warning("  Fresh URL also failed to resolve Drive URL")
    return drive_url


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    if not check_secrets():
        sys.exit(2)

    start_time = datetime.utcnow()
    deadline   = start_time + timedelta(minutes=RUN_MINUTES)

    log.info("=" * 70)
    log.info(f"Run started:    {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log.info(f"Deadline:       {deadline.strftime('%H:%M:%S UTC')}  ({RUN_MINUTES} min)")
    log.info(f"Item limit:     {ITEM_LIMIT if ITEM_LIMIT > 0 else 'unlimited'}")
    log.info("=" * 70)

    work_root = Path(WORK_DIR)
    work_root.mkdir(parents=True, exist_ok=True)

    scraper  = ForPSDScraper(FORPSD_COOKIE)
    uploader = DriveUploader(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN)

    # ── Load trackers ──────────────────────────────────────────────────────
    job_tracker    = JobTracker(JOBS_EXCEL_FILE)
    rename_tracker = ExcelTracker(EXCEL_LOG_FILE)
    state          = StateManager(STATE_FILE)

    log.info(f"📋 Job tracker:    {job_tracker.stats()}")
    log.info(f"📊 Rename tracker: {rename_tracker.stats()}")

    uploader.preload_existing_names(GDRIVE_PSD_FOLDER)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 1: Ensure pending items exist in jobs.xlsx
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pending = job_tracker.get_pending()

    if job_tracker.total() == 0:
        # ── Scenario A: Brand new — full scrape ───────────────────────────
        log.info("🆕 First run — full scrape to populate jobs.xlsx…")
        found = _full_scrape(scraper, job_tracker)
        if found == 0:
            log.error("No items found. Check FORPSD_COOKIE and site availability.")
            sys.exit(1)
        pending = job_tracker.get_pending()

    elif not pending:
        # ── Scenario B: All Completed → check for new items on site ──────
        log.info(
            f"✅ All {job_tracker.completed_count()} items Completed. "
            "Re-scraping from page 1 for new uploads…"
        )
        found = _rescrape_for_new(scraper, job_tracker, ITEM_LIMIT)
        if found == 0:
            log.info("No new items found on site — all done! 🎉")
            Path(DONE_FILE).touch()
            return
        pending = job_tracker.get_pending()

    elif ITEM_LIMIT > 0 and len(pending) < ITEM_LIMIT:
        # ── Scenario C: Need more pending items to hit ITEM_LIMIT ─────────
        log.info(
            f"Only {len(pending)} pending but ITEM_LIMIT={ITEM_LIMIT}. "
            "Fetching more items…"
        )
        _topup_scrape(scraper, job_tracker, ITEM_LIMIT)
        pending = job_tracker.get_pending()

    log.info(f"📋 Ready to process: {job_tracker.stats()}")

    if not pending:
        log.info("Nothing pending — exiting.")
        Path(DONE_FILE).touch()
        return

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # File counter — synced from ALL sources so collisions are impossible
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    file_counter: int = max(
        state.get("file_counter", 1),
        uploader.max_counter + 1,
        rename_tracker.max_counter + 1,
    )
    state.set("file_counter", file_counter)
    state.save()
    log.info(
        f"Counter starts: tamilpsd-{file_counter:04d}  "
        f"(Drive max={uploader.max_counter:04d}, "
        f"Rename log max={rename_tracker.max_counter:04d})"
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 2: Process pending items
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    processed_this_run = 0
    errors_this_run    = 0

    for item in pending:
        download_url = item.get("download_url", "")
        detail_url   = item.get("detail_url",   "")
        card_title   = item.get("card_title",   "")
        category     = item.get("category",     "")

        # ── Time limit ────────────────────────────────────────────────────
        if datetime.utcnow() >= deadline:
            log.info(
                f"⏰ Time limit reached. "
                f"Processed={processed_this_run}, "
                f"Remaining={len(job_tracker.get_pending())}"
            )
            break

        # ── Item limit ────────────────────────────────────────────────────
        if ITEM_LIMIT > 0 and processed_this_run >= ITEM_LIMIT:
            log.info(f"🔢 Item limit ({ITEM_LIMIT}) reached.")
            break

        # ── SKIP ① — Already Completed in jobs.xlsx ──────────────────────
        if job_tracker.is_completed(download_url):
            log.info(f"  ⏭  Already Completed (jobs.xlsx) → skip: {download_url[:60]}")
            processed_this_run += 1
            continue

        item_dir = work_root / f"item_{int(time.time() * 1000)}"
        item_dir.mkdir(parents=True, exist_ok=True)

        try:
            log.info(f"\n{'─'*65}")
            log.info(f"  [{processed_this_run+1}] {download_url}")

            # ── Resolve category ──────────────────────────────────────────
            if not category:
                if card_title or detail_url:
                    category = scraper.get_category(detail_url, hint_title=card_title)
                else:
                    category = "uncategorized"
                    log.warning("  No detail URL or card_title — using 'uncategorized'")
                job_tracker.update_category(download_url, category)

            log.info(f"  Category: [{category}]")

            # ── FIX: Resolve Google Drive URL with fresh-URL fallback ─────
            # Previously: one attempt only → mark Error if failed.
            # Now: if the stored /download/ URL fails (JWT expired), we
            # re-scrape the detail page to get a fresh URL and retry once.
            drive_url = _resolve_drive_url_with_retry(
                scraper, download_url, detail_url
            )
            if not drive_url:
                log.warning("  Could not resolve Drive URL (both attempts failed) — marking Error")
                job_tracker.mark_error(download_url)
                errors_this_run += 1
                continue

            log.info(f"  Drive URL: {drive_url}")

            # ── Download archive ──────────────────────────────────────────
            dl_dir  = item_dir / "download"
            archive = download_from_drive(
                drive_url, dl_dir,
                client_id=GOOGLE_CLIENT_ID,
                client_secret=GOOGLE_CLIENT_SECRET,
                refresh_token=GOOGLE_REFRESH_TOKEN,
            )
            if not archive:
                log.error("  Download failed — marking Error")
                job_tracker.mark_error(download_url)
                errors_this_run += 1
                continue

            # ── List original filenames ───────────────────────────────────
            original_names = _list_archive_originals(archive)
            log.info(f"  Archive: {len(original_names)} PSD/TIF file(s) inside")

            # ── SKIP ② — ALL files already in rename_log.xlsx ────────────
            if original_names and all(
                rename_tracker.is_original_done(n) for n in original_names
            ):
                log.info(
                    f"  ⏭  All {len(original_names)} file(s) already in rename_log.xlsx "
                    f"— marking Completed, no upload needed."
                )
                job_tracker.mark_completed(download_url)
                processed_this_run += 1
                continue

            # ── Extract + rename ──────────────────────────────────────────
            result = process_archive(
                archive_path=archive,
                work_dir=item_dir / "process",
                start_index=file_counter,
            )
            if not result:
                log.error(f"  process_archive failed for {archive.name}")
                job_tracker.mark_error(download_url)
                errors_this_run += 1
                continue

            renamed_files, next_index = result

            orig_map: dict[Path, str] = {}
            for idx, renamed_path in enumerate(renamed_files):
                orig_name = original_names[idx] if idx < len(original_names) else renamed_path.name
                orig_map[renamed_path] = orig_name

            # ── Upload to Drive category subfolder ────────────────────────
            uploaded_count = 0
            for renamed_path, orig_name in orig_map.items():
                if renamed_path.exists():
                    log.info(f"  ↑ Uploading: {renamed_path.name} → [{category}/]")
                    result_info = uploader.upload_to_category(
                        file_path=renamed_path,
                        parent_folder_id=GDRIVE_PSD_FOLDER,
                        category=category,
                        excel_tracker=rename_tracker,
                        original_name=orig_name,
                    )
                    if not result_info.get("skipped"):
                        uploaded_count += 1
                else:
                    log.warning(f"  File missing after rename: {renamed_path}")

            # ── Persist counter + mark Completed ──────────────────────────
            file_counter = next_index
            state.set("file_counter", file_counter)
            state.save()

            job_tracker.mark_completed(download_url)
            processed_this_run += 1

            log.info(
                f"  ✅ Done ({processed_this_run} this run) "
                f"[{category}] uploaded={uploaded_count} "
                f"next=tamilpsd-{file_counter:04d}"
            )
            log.info(f"  📋 {job_tracker.stats()}")

        except Exception as exc:
            log.error(f"  ❌ Unhandled error: {exc}", exc_info=True)
            try:
                job_tracker.mark_error(download_url)
            except Exception:
                pass
            errors_this_run += 1

        finally:
            shutil.rmtree(item_dir, ignore_errors=True)

    # ── Final summary ──────────────────────────────────────────────────────
    state.set("last_run", datetime.utcnow().isoformat())
    state.save()

    remaining = len(job_tracker.get_pending())
    log.info(f"\n{'='*70}")
    log.info(
        f"Run finished: Processed={processed_this_run} | "
        f"Errors={errors_this_run} | Remaining={remaining} | "
        f"Next=tamilpsd-{file_counter:04d}"
    )
    log.info(f"📋 Job tracker final:    {job_tracker.stats()}")
    log.info(f"📊 Rename tracker final: {rename_tracker.stats()}")

    if remaining == 0:
        log.info("🎉 All items Completed! Touching ALL_DONE.")
        Path(DONE_FILE).touch()
    else:
        log.info("📌 Items still pending — next run continues from here.")


if __name__ == "__main__":
    main()
