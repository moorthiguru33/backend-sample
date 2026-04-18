"""
main.py – Orchestrator (category-aware upload, continuous file counter).

Flow per run:
  1. Scrape listing pages to collect {download_url, detail_url} items.
     - First run (state is empty): full scrape of all pages.
     - Pending items exist: skip scrape, use existing state.
     - All known items done: INCREMENTAL re-scrape (stops early once it
       hits a page of already-known URLs). Fast — usually 1-2 pages only.

  2. Loop over pending items until ITEM_LIMIT reached or no more pending:
       a. Resolve category from product detail page (or cached in state).
       b. Resolve /download/eyJ… -> Google Drive URL.
       c. Download archive (ZIP/RAR/7Z).
       d. Extract -> rename files to tamilpsd-XXXX (global counter, continuous).
       e. Upload renamed PSD/TIF files into GDRIVE_PSD_FOLDER/<category>/
       f. Mark URL as processed, persist file counter to state.

File counter continuity:
  Counter is global across ALL categories and ALL runs.
  On startup, counter = max(state.json value, Drive scan max, Excel log max) + 1
  so even a lost state.json cannot cause collisions.

ITEM_LIMIT behaviour:
  0   -> process everything pending (default; also triggers auto-continue)
  N>0 -> process exactly N items then stop (no auto-continue triggered)

Skip logic (three layers, fastest-first):
  - Excel log  : skip files whose original OR renamed name is already logged.
                 Zero API calls — purely in-memory after initial load.
  - state.json : skip URLs already fully processed in a previous run.
  - Drive check: skip individual files already on Drive
                 (handles interrupted runs where state was not committed).

Excel log (rename_log.xlsx):
  Path: <WORK_DIR>/rename_log.xlsx
  Col A "Original File Name" – filename inside the source archive.
  Col B "Renamed File Name"  – assigned tamilpsd-XXXX name.
  Written immediately after each successful upload — crash-safe.
  Loaded at startup to rebuild in-memory skip sets.
"""

import logging
import shutil
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

# Optional archive-format libraries — only needed for RAR / 7Z archives.
# Install with: pip install rarfile py7zr
# If missing, those formats are still handled by process_archive itself;
# the original-name pre-listing step is simply skipped for that format.
try:
    import rarfile as _rarfile
    _HAS_RARFILE = True
except ImportError:
    _HAS_RARFILE = False

try:
    import py7zr as _py7zr
    _HAS_PY7ZR = True
except ImportError:
    _HAS_PY7ZR = False

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

# How many posts forpsd.com shows per listing page.
POSTS_PER_PAGE: int = 17

# Excel tracker file lives next to state.json so it persists across runs.
EXCEL_LOG_FILE: str = str(Path(STATE_FILE).parent / "rename_log.xlsx")

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


def check_secrets() -> bool:
    missing = []
    if not FORPSD_COOKIE:          missing.append("FORPSD_COOKIE")
    if not GOOGLE_CLIENT_ID:       missing.append("GOOGLE_CLIENT_ID")
    if not GOOGLE_CLIENT_SECRET:   missing.append("GOOGLE_CLIENT_SECRET")
    if not GOOGLE_REFRESH_TOKEN:   missing.append("GOOGLE_REFRESH_TOKEN")
    if not GDRIVE_PSD_FOLDER:      missing.append("GDRIVE_PSD_FOLDER")
    if missing:
        log.error(f"Missing GitHub Secrets: {', '.join(missing)}")
        return False
    return True


def _smart_scrape(scraper: ForPSDScraper, state: StateManager, item_limit: int) -> int:
    """
    Targeted scrape: fetch ONLY the listing pages that contain the next
    `item_limit` unprocessed items.
    """
    done_count = len(state.get("processed", []))
    start_page = (done_count // POSTS_PER_PAGE) + 1

    known_urls: set[str] = {
        item.get("download_url", "")
        for item in state.get("all_items", [])
    }

    log.info(
        f"Smart scrape: done={done_count} → start_page={start_page} | "
        f"need={item_limit if item_limit > 0 else 'unlimited'} new items"
    )

    new_items = scraper.get_all_items(
        page_limit=PAGE_LIMIT,
        stop_at_known_urls=known_urls if known_urls else None,
        start_page=start_page,
        max_new_items=item_limit,
    )

    if not new_items:
        return 0

    all_items: list = state.get("all_items", [])
    all_items.extend(new_items)
    state.set("all_items", all_items)
    state.set("all_urls", [i["download_url"] for i in all_items])
    state.save()
    log.info(
        f"Smart scrape done: +{len(new_items)} new items "
        f"(total known: {len(all_items)})"
    )
    return len(new_items)


# ── Archive original-name extraction ──────────────────────────────────────

def _list_archive_originals(archive_path: Path) -> list[str]:
    """
    Return all PSD/TIF/TIFF filenames (original names) inside the archive,
    in the order they would be extracted by process_archive.

    Used to build the original_name → renamed_name mapping for the Excel log.
    Supports ZIP (stdlib), RAR (requires rarfile), and 7Z (requires py7zr).
    Returns [] on any error or if the required library is not installed.
    """
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
            if not _HAS_RARFILE:
                log.warning("  rarfile not installed — skipping original-name listing for .rar")
                return []
            with _rarfile.RarFile(archive_path) as rf:
                for info in rf.infolist():
                    p = Path(info.filename)
                    if p.suffix.lower() in TARGET_EXT and not info.is_dir():
                        names.append(p.name)

        elif suffix == ".7z":
            if not _HAS_PY7ZR:
                log.warning("  py7zr not installed — skipping original-name listing for .7z")
                return []
            with _py7zr.SevenZipFile(archive_path, mode="r") as sz:
                for entry in sz.list():
                    p = Path(entry.filename)
                    if p.suffix.lower() in TARGET_EXT and not entry.is_directory:
                        names.append(p.name)

    except Exception as exc:
        log.warning(f"  Could not list archive originals for {archive_path.name}: {exc}")

    return names


def main() -> None:
    if not check_secrets():
        sys.exit(2)

    start_time = datetime.utcnow()
    deadline   = start_time + timedelta(minutes=RUN_MINUTES)

    log.info("=" * 70)
    log.info(f"Run started at {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log.info(f"Will work until {deadline.strftime('%H:%M:%S UTC')}  ({RUN_MINUTES} min)")
    log.info(f"Item limit this run: {ITEM_LIMIT if ITEM_LIMIT > 0 else 'unlimited'}")
    log.info("=" * 70)

    work_root = Path(WORK_DIR)
    work_root.mkdir(parents=True, exist_ok=True)

    state    = StateManager(STATE_FILE)
    scraper  = ForPSDScraper(FORPSD_COOKIE)
    uploader = DriveUploader(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN)

    # ── Load Excel tracker ─────────────────────────────────────────────────
    # Reads rename_log.xlsx once; builds in-memory sets for O(1) skip checks.
    tracker = ExcelTracker(EXCEL_LOG_FILE)
    log.info(f"📊 Excel tracker: {tracker.stats()}")

    # ── Preload all existing Drive filenames (cross-subfolder duplicate guard)
    uploader.preload_existing_names(GDRIVE_PSD_FOLDER)

    # ── Phase 1: Ensure we have pending items ─────────────────────────────
    pending = state.pending_items()

    if not pending:
        if not state.get("all_items"):
            log.info("First run — smart scrape for initial batch ...")
        else:
            log.info(
                f"All {len(state.get('all_items', []))} known items are done. "
                "Smart scrape to check for more on site ..."
            )

        found_new = _smart_scrape(scraper, state, ITEM_LIMIT)
        if found_new == 0:
            log.info("No new items found on site. Everything is done!")
            Path(DONE_FILE).touch()
            return

        pending = state.pending_items()
    log.info(f"State: {state.summary()}")

    if not pending:
        log.info("Nothing pending — exiting.")
        Path(DONE_FILE).touch()
        return

    # ── Global file counter — synced from ALL sources ──────────────────────
    # Use the maximum of:
    #   • state.json value           (normal run-to-run continuity)
    #   • Drive scan max + 1         (handles lost state.json)
    #   • Excel log max + 1          (NEW: most reliable source of truth)
    # This guarantees the counter is always ABOVE any existing file on Drive
    # regardless of which state sources have been lost or reset.
    file_counter: int = max(
        state.get("file_counter", 1),
        uploader.max_counter + 1,
        tracker.max_counter + 1,
    )
    state.set("file_counter", file_counter)
    state.save()
    log.info(
        f"File counter starts at: {file_counter}  "
        f"(Drive max: {uploader.max_counter:04d}, "
        f"Excel max: {tracker.max_counter:04d}, "
        f"state: {state.get('file_counter', 1)}) "
        f"-> next tamilpsd-{file_counter:04d}"
    )

    # ── Phase 2: Process pending items ────────────────────────────────────
    processed_this_run = 0
    errors_this_run    = 0

    for item in pending:

        download_url = item.get("download_url", "")
        detail_url   = item.get("detail_url", "")

        # ── Time limit ────────────────────────────────────────────────────
        if datetime.utcnow() >= deadline:
            log.info(
                f"Time limit reached. "
                f"Processed {processed_this_run} this run. "
                f"Remaining: {len(state.pending_items())}"
            )
            break

        # ── Item limit ────────────────────────────────────────────────────
        if ITEM_LIMIT > 0 and processed_this_run >= ITEM_LIMIT:
            log.info(
                f"Item limit ({ITEM_LIMIT}) reached. "
                f"Processed {processed_this_run} this run. "
                f"Remaining: {len(state.pending_items())}"
            )
            break

        item_dir = work_root / f"item_{int(time.time() * 1000)}"
        item_dir.mkdir(parents=True, exist_ok=True)

        try:
            log.info(f"--- Processing: {download_url}")

            # ── 2a: Resolve category ──────────────────────────────────────
            category = item.get("category", "")
            if not category:
                card_title = item.get("card_title", "")
                if detail_url or card_title:
                    if card_title:
                        log.info(f"  Getting category (card_title hint): {card_title!r}")
                    else:
                        log.info(f"  Fetching category from: {detail_url}")
                    category = scraper.get_category(detail_url, hint_title=card_title)
                    for stored_item in state.get("all_items", []):
                        if stored_item.get("download_url") == download_url:
                            stored_item["category"] = category
                            break
                    state.save()
                else:
                    category = "uncategorized"
                    log.warning("  No detail URL or card title — using 'uncategorized'")

            log.info(f"  Category: [{category}]")

            # ── 2b: Resolve Google Drive URL ──────────────────────────────
            drive_url = scraper.resolve_drive_url(download_url)
            if not drive_url:
                log.warning("  No Drive URL – skipping (will retry next run)")
                errors_this_run += 1
                continue

            log.info(f"  Drive URL: {drive_url}")

            # ── 2c: Download archive ──────────────────────────────────────
            dl_dir  = item_dir / "download"
            archive = download_from_drive(
                drive_url, dl_dir,
                client_id=GOOGLE_CLIENT_ID,
                client_secret=GOOGLE_CLIENT_SECRET,
                refresh_token=GOOGLE_REFRESH_TOKEN,
            )
            if not archive:
                log.error("  Download failed – skipping")
                errors_this_run += 1
                continue

            # ── 2d: Extract original filenames from archive ────────────────
            # We read the archive manifest BEFORE process_archive so we can
            # map original[i] → renamed[i] for the Excel log.
            original_names = _list_archive_originals(archive)
            log.info(f"  Archive contains {len(original_names)} PSD/TIF file(s)")

            # ── 2e: Excel pre-check — skip entire item if ALL files are done
            if original_names and all(
                tracker.is_original_done(n) for n in original_names
            ):
                log.info(
                    f"  ⏭  All {len(original_names)} file(s) from this archive "
                    f"already in Excel log — marking item done, skipping download"
                )
                state.mark_processed(download_url)
                processed_this_run += 1
                continue

            # ── 2f: Extract and rename with global counter ─────────────────
            result = process_archive(
                archive_path=archive,
                work_dir=item_dir / "process",
                start_index=file_counter,
            )
            if not result:
                log.error(f"  Processing failed for {archive.name} – skipping")
                errors_this_run += 1
                continue

            renamed_files, next_index = result

            # ── 2g: Build original→renamed mapping ────────────────────────
            # process_archive renames in the same order files appear in the archive.
            # Zip the two lists; if counts differ, fall back to renamed name as key.
            orig_map: dict[Path, str] = {}
            for idx, renamed_path in enumerate(renamed_files):
                orig_name = original_names[idx] if idx < len(original_names) else renamed_path.name
                orig_map[renamed_path] = orig_name

            # ── 2h: Upload to Drive category subfolder ─────────────────────
            uploaded_count = 0
            for renamed_path, orig_name in orig_map.items():
                if renamed_path.exists():
                    log.info(f"  Uploading: {renamed_path.name}  -> [{category}/]")
                    upload_result = uploader.upload_to_category(
                        file_path=renamed_path,
                        parent_folder_id=GDRIVE_PSD_FOLDER,
                        category=category,
                        excel_tracker=tracker,      # ← Excel integration
                        original_name=orig_name,    # ← original filename logged
                    )
                    if not upload_result.get("skipped"):
                        uploaded_count += 1
                else:
                    log.warning(f"  File missing after rename: {renamed_path}")

            # ── 2i: Persist counter and mark item done ─────────────────────
            file_counter = next_index
            state.set("file_counter", file_counter)
            state.mark_processed(download_url)
            processed_this_run += 1
            log.info(
                f"  Done ({processed_this_run} this run). "
                f"Category=[{category}] | "
                f"Uploaded {uploaded_count} file(s). "
                f"Next file counter: tamilpsd-{file_counter:04d}. "
                f"Excel: {tracker.stats()}. "
                f"State: {state.summary()}"
            )

        except Exception as exc:
            log.error(f"  Unhandled error: {exc}", exc_info=True)
            errors_this_run += 1

        finally:
            shutil.rmtree(item_dir, ignore_errors=True)

    # ── Final ──────────────────────────────────────────────────────────────
    state.set("last_run", datetime.utcnow().isoformat())
    state.save()

    remaining = len(state.pending_items())
    log.info(
        f"Run finished. Processed={processed_this_run}, "
        f"Errors={errors_this_run}, Remaining={remaining}, "
        f"Next file counter=tamilpsd-{file_counter:04d}"
    )
    log.info(f"📊 Excel tracker final: {tracker.stats()}")

    if remaining == 0:
        log.info("All items processed! Touching ALL_DONE to stop automation.")
        Path(DONE_FILE).touch()
    else:
        log.info("Items still pending – next run will continue from here.")


if __name__ == "__main__":
    main()
