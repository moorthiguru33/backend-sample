"""
main.py – Orchestrator (category-aware upload, continuous file counter).

Flow per run:
  1. Scrape all listing pages → collect {download_url, detail_url} items
     (first run only — stored in state.json).
  2. Loop over pending items until the 5.5 h deadline OR ITEM_LIMIT reached:
       a. Resolve category from product detail page (or cached in state).
       b. Resolve /download/eyJ… → Google Drive URL.
       c. Download archive (ZIP/RAR/7Z).
       d. Extract → rename files to tamilpsd-XXXX (global counter, continuous).
       e. Upload renamed PSD/TIF files into:
            GDRIVE_PSD_FOLDER / <category> /   ← e.g. psd/wedding/
       f. Mark URL as processed, save file counter to state.

File counter continuity:
  Counter is global across ALL categories.
  If category "wedding" ends at tamilpsd-2548, the next item (any category)
  starts at tamilpsd-2549. The counter is persisted in state.json.

Skip logic (two layers):
  • state.json  – skips URLs already fully processed in a previous run.
  • Drive check – uploader skips individual files that already exist on Drive
                  (handles interrupted runs where the state wasn't committed).

Folder structure created on Drive:
  GDRIVE_PSD_FOLDER/
    wedding/
      tamilpsd-0001.psd
      tamilpsd-0002.psd
    birthday/
      tamilpsd-0003.psd
    business-cards/
      tamilpsd-0004.psd
      tamilpsd-0005.psd
    uncategorized/
      tamilpsd-0006.psd
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

    # ── Phase 1: collect all items (first run only) ────────────────────────
    if not state.get("all_items"):
        limit_msg = f"first {PAGE_LIMIT} pages" if PAGE_LIMIT > 0 else "all pages"
        log.info(f"First run – scraping {limit_msg} …")

        all_items = scraper.get_all_items(page_limit=PAGE_LIMIT)
        if not all_items:
            log.error("Could not scrape any items. Check FORPSD_COOKIE secret.")
            sys.exit(2)

        state.set("all_items", all_items)
        # Also keep legacy all_urls for state summary compatibility
        state.set("all_urls", [item["download_url"] for item in all_items])
        state.set("processed", [])
        state.save()
        log.info(f"Collected {len(all_items)} items → saved to state.json")

    pending = state.pending_items()
    log.info(f"State: {state.summary()}")

    if not pending:
        # All KNOWN items are done — re-scrape to check for newly added items
        limit_msg = f"first {PAGE_LIMIT} pages" if PAGE_LIMIT > 0 else "all pages"
        log.info(f"All known items processed. Re-scraping ({limit_msg}) to check for new items …")

        refreshed_items = scraper.get_all_items(page_limit=PAGE_LIMIT)

        existing_urls = {
            item.get("download_url", "")
            for item in state.get("all_items", [])
        }
        truly_new = [
            item for item in refreshed_items
            if item.get("download_url") and item["download_url"] not in existing_urls
        ]

        if truly_new:
            all_items = state.get("all_items", [])
            all_items.extend(truly_new)
            state.set("all_items", all_items)
            state.set("all_urls", [i["download_url"] for i in all_items])
            state.save()
            log.info(f"Found {len(truly_new)} new item(s) → added to state.json")
            pending = state.pending_items()
        else:
            log.info("🎉 No new items found on site. All done!")
            Path(DONE_FILE).touch()
            return

    # ── Global file counter — persisted across ALL runs and ALL categories ──
    # This ensures continuous numbering: tamilpsd-0001 … tamilpsd-9999+
    # regardless of category changes between items.
    file_counter: int = state.get("file_counter", 1)
    log.info(f"File counter starts at: {file_counter}  (next → tamilpsd-{file_counter:04d})")

    # ── Phase 2: process pending items ────────────────────────────────────
    processed_this_run = 0
    errors_this_run    = 0

    for item in pending:

        download_url = item.get("download_url", "")
        detail_url   = item.get("detail_url", "")

        # ── Time limit ────────────────────────────────────────────────────
        if datetime.utcnow() >= deadline:
            log.info(
                f"⏰ Time limit reached. "
                f"Processed {processed_this_run} this run. "
                f"Remaining: {len(state.pending_items())}"
            )
            break

        # ── Item limit ────────────────────────────────────────────────────
        if ITEM_LIMIT > 0 and processed_this_run >= ITEM_LIMIT:
            log.info(
                f"✋ Item limit ({ITEM_LIMIT}) reached. "
                f"Processed {processed_this_run} this run. "
                f"Remaining: {len(state.pending_items())}"
            )
            break

        item_dir = work_root / f"item_{int(time.time() * 1000)}"
        item_dir.mkdir(parents=True, exist_ok=True)

        try:
            log.info(f"─── Processing: {download_url}")

            # ── 2a: Get category from post title / product detail page ──────
            # Use cached category if already stored in the item dict
            category = item.get("category", "")
            if not category:
                card_title = item.get("card_title", "")
                if detail_url or card_title:
                    if card_title:
                        log.info(f"  Getting category (card_title hint): {card_title!r}")
                    else:
                        log.info(f"  Fetching category from: {detail_url}")
                    category = scraper.get_category(detail_url, hint_title=card_title)
                    # Cache in state so we don't re-fetch on retry
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

            # ── 2d: Process (extract → rename with global counter) ────────
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

            # ── 2e: Upload renamed PSD/TIF → category subfolder ───────────
            # Files go into:  GDRIVE_PSD_FOLDER / <category> /
            # Folder is auto-created on Drive if it does not exist.
            uploaded_count = 0
            for orig in renamed_files:
                if orig.exists():
                    log.info(f"  Uploading: {orig.name}  → [{category}/]")
                    upload_result = uploader.upload_to_category(
                        file_path=orig,
                        parent_folder_id=GDRIVE_PSD_FOLDER,
                        category=category,
                    )
                    if not upload_result.get("skipped"):
                        uploaded_count += 1
                else:
                    log.warning(f"  PSD file missing after rename: {orig}")

            # ── 2f: Update global counter and mark done ───────────────────
            file_counter = next_index
            state.set("file_counter", file_counter)
            state.mark_processed(download_url)
            processed_this_run += 1
            log.info(
                f"  ✅ Done ({processed_this_run} this run). "
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
        log.info("🎉 All items processed! Touching ALL_DONE to stop automation.")
        Path(DONE_FILE).touch()
    else:
        log.info("Items still pending – next run will be triggered automatically.")


if __name__ == "__main__":
    main()
