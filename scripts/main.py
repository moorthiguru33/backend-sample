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
  If 500 items are done and counter is at tamilpsd-0738, the next run
  continues at tamilpsd-0739 — persisted in state.json.

ITEM_LIMIT behaviour:
  0   -> process everything pending (default; also triggers auto-continue)
  N>0 -> process exactly N items then stop (no auto-continue triggered)

Skip logic (two layers):
  - state.json  : skips URLs already fully processed in a previous run.
  - Drive check : uploader skips individual files already on Drive
                  (handles interrupted runs where state was not committed).
"""

import logging
import shutil
import sys
import time
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

# How many posts forpsd.com shows per listing page.
# Used to calculate which page the next pending items are on.
POSTS_PER_PAGE: int = 17

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

    Logic
    -----
    Each forpsd.com listing page has POSTS_PER_PAGE (17) items.
    If N items are already done, the next unprocessed item is at
    position N (0-indexed), which lives on page:

        start_page = (done_count // POSTS_PER_PAGE) + 1

    Examples
    --------
        done=0   → start_page=1   (first 17 items are on page 1)
        done=17  → start_page=2   (items 18-34 are on page 2)
        done=30  → start_page=2   (item 31 is on page 2, 32-34 too, 35+ on page 3)
        done=34  → start_page=3   (items 35-51 are on page 3)

    For ITEM_LIMIT=2 with 34 already done, we only scrape page 3
    and stop the moment we have 2 new items — never touching pages 4-400+.

    For ITEM_LIMIT=0 (unlimited), we start at start_page and scrape
    to the end — still much smarter than always starting at page 1.

    Returns the number of new items added to state.
    """
    done_count = len(state.get("processed", []))
    start_page = (done_count // POSTS_PER_PAGE) + 1

    # URLs already known — skip them so we don't re-add to state
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
        max_new_items=item_limit,          # 0 = no cap (scrape to end)
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

    # ── Phase 1: Ensure we have pending items ─────────────────────────────
    pending = state.pending_items()

    if not pending:
        # Either truly first run (state empty) or all known items are done.
        # Either way: smart-scrape to fetch exactly the pages we need.
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

    # ── Global file counter — persisted across ALL runs & categories ───────
    file_counter: int = state.get("file_counter", 1)
    log.info(f"File counter starts at: {file_counter}  (next -> tamilpsd-{file_counter:04d})")

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

        # ── Item limit — checked BEFORE starting each item ─────────────
        # If ITEM_LIMIT=1 and processed_this_run=1, we stop here immediately.
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
                    # Cache so we never re-fetch on retry
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
            archive = download_from_drive(drive_url, dl_dir)
            if not archive:
                log.error("  Download failed – skipping")
                errors_this_run += 1
                continue

            # ── 2d: Extract and rename with global counter ─────────────────
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

            # ── 2e: Upload to Drive category subfolder ─────────────────────
            uploaded_count = 0
            for orig in renamed_files:
                if orig.exists():
                    log.info(f"  Uploading: {orig.name}  -> [{category}/]")
                    upload_result = uploader.upload_to_category(
                        file_path=orig,
                        parent_folder_id=GDRIVE_PSD_FOLDER,
                        category=category,
                    )
                    if not upload_result.get("skipped"):
                        uploaded_count += 1
                else:
                    log.warning(f"  File missing after rename: {orig}")

            # ── 2f: Persist counter and mark item done ─────────────────────
            file_counter = next_index
            state.set("file_counter", file_counter)
            state.mark_processed(download_url)
            processed_this_run += 1
            log.info(
                f"  Done ({processed_this_run} this run). "
                f"Category=[{category}] | "
                f"Uploaded {uploaded_count} file(s). "
                f"Next file counter: tamilpsd-{file_counter:04d}. "
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

    if remaining == 0:
        log.info("All items processed! Touching ALL_DONE to stop automation.")
        Path(DONE_FILE).touch()
    else:
        log.info("Items still pending – next run will continue from here.")


if __name__ == "__main__":
    main()
