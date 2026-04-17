"""
main.py – Orchestrator (updated workflow).

Flow per run:
  1. Scrape all listing pages → collect download URLs (first run only).
  2. Loop over pending URLs until the 5.5 h deadline:
       a. Resolve /download/eyJ… → Google Drive URL.
       b. Download archive (ZIP/RAR).
       c. Extract → create WebP with watermark (no layer deletion).
       d. Upload original PSD/TIF files → GDRIVE_PSD_FOLDER   (psd/)
       e. Upload WebP previews          → GDRIVE_WEBP_FOLDER  (preview/)
       f. Mark URL as processed and save state.
  3. If all done, touch ALL_DONE (stops self-triggering).
  4. Otherwise exit normally – workflow sleeps 10 min then re-triggers.
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
    GDRIVE_PSD_FOLDER, GDRIVE_WEBP_FOLDER,
    RUN_MINUTES, PAGE_LIMIT, WORK_DIR, STATE_FILE, DONE_FILE, LOG_FILE,
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
    if not GDRIVE_WEBP_FOLDER:     missing.append("GDRIVE_WEBP_FOLDER")
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
    log.info("=" * 70)

    work_root = Path(WORK_DIR)
    work_root.mkdir(parents=True, exist_ok=True)

    state    = StateManager(STATE_FILE)
    scraper  = ForPSDScraper(FORPSD_COOKIE)
    uploader = DriveUploader(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN)

    # ── Phase 1: collect all URLs (first run only) ─────────────────────────
    if not state.get("all_urls"):
        limit_msg = f"first {PAGE_LIMIT} pages" if PAGE_LIMIT > 0 else "all pages"
        log.info(f"First run – scraping {limit_msg} …")
        all_urls = scraper.get_all_download_urls(page_limit=PAGE_LIMIT)
        if not all_urls:
            log.error("Could not scrape any URLs. Check FORPSD_COOKIE secret.")
            sys.exit(2)
        state.set("all_urls", all_urls)
        state.set("processed", [])
        state.save()
        log.info(f"Collected {len(all_urls)} download URLs → saved to state.json")

    pending = state.pending_urls()
    log.info(f"State: {state.summary()}")

    if not pending:
        log.info("🎉 All items already processed!")
        Path(DONE_FILE).touch()
        return

    # ── Phase 2: process pending items ────────────────────────────────────
    processed_this_run = 0
    errors_this_run    = 0

    for download_url in pending:

        if datetime.utcnow() >= deadline:
            log.info(
                f"⏰ Time limit reached. "
                f"Processed {processed_this_run} this run. "
                f"Remaining: {len(state.pending_urls())}"
            )
            break

        item_dir = work_root / f"item_{int(time.time() * 1000)}"
        item_dir.mkdir(parents=True, exist_ok=True)

        try:
            log.info(f"─── Processing: {download_url}")

            # ── 2a: Resolve Google Drive URL ──────────────────────────────
            drive_url = scraper.resolve_drive_url(download_url)
            if not drive_url:
                log.warning("  No Drive URL – skipping (will retry next run)")
                errors_this_run += 1
                continue

            log.info(f"  Drive URL: {drive_url}")

            # ── 2b: Download archive ──────────────────────────────────────
            dl_dir  = item_dir / "download"
            archive = download_from_drive(drive_url, dl_dir)
            if not archive:
                log.error("  Download failed – skipping")
                errors_this_run += 1
                continue

            # ── 2c: Process (extract → watermark WebP, no layer delete) ───
            result = process_archive(
                archive_path=archive,
                work_dir=item_dir / "process",
            )
            if not result:
                log.error(f"  Processing failed for {archive.name} – skipping")
                errors_this_run += 1
                continue

            original_files, webp_list = result

            # ── 2d: Upload original PSD/TIF → psd/ folder ────────────────
            for orig in original_files:
                if orig.exists():
                    log.info(f"  Uploading original: {orig.name}")
                    uploader.upload(orig, GDRIVE_PSD_FOLDER)
                else:
                    log.warning(f"  Original file missing: {orig}")

            # ── 2e: Upload WebP previews → preview/ folder ────────────────
            for webp in webp_list:
                if webp.exists():
                    log.info(f"  Uploading WebP: {webp.name}")
                    uploader.upload(webp, GDRIVE_WEBP_FOLDER)

            # ── 2f: Mark done ──────────────────────────────────────────────
            state.mark_processed(download_url)
            processed_this_run += 1
            log.info(
                f"  ✅ Done ({processed_this_run} this run). "
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

    remaining = len(state.pending_urls())
    log.info(
        f"Run finished. Processed={processed_this_run}, "
        f"Errors={errors_this_run}, Remaining={remaining}"
    )

    if remaining == 0:
        log.info("🎉 All items processed! Touching ALL_DONE to stop automation.")
        Path(DONE_FILE).touch()
    else:
        log.info("Items still pending – next run will be triggered automatically.")


if __name__ == "__main__":
    main()
