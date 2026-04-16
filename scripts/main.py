"""
main.py – Orchestrator.

Flow per run:
  1. If state has no URLs yet → scrape all listing pages to collect them.
  2. Loop over pending URLs until the 5.5 h deadline:
       a. Resolve /download/eyJ… → Google Drive URL
       b. Download archive (ZIP/RAR)
       c. Extract → delete top layer → create WebP with watermark → re-zip
       d. Upload ZIP to GDRIVE_PSD_FOLDER
       e. Upload WebP(s) to GDRIVE_WEBP_FOLDER
       f. Mark URL as processed and save state
  3. If all done, touch ALL_DONE (workflow stops self-triggering).
  4. Otherwise exit normally – the workflow sleeps 10 min then re-triggers.
"""

import logging
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── bring scripts/ into sys.path ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FORPSD_COOKIE, GDRIVE_SA_JSON, GDRIVE_PSD_FOLDER, GDRIVE_WEBP_FOLDER,
    RUN_MINUTES, WORK_DIR, STATE_FILE, DONE_FILE, LOG_FILE,
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


# ═══════════════════════════════════════════════════════════════════════════
#  Validation helpers
# ═══════════════════════════════════════════════════════════════════════════

def check_secrets() -> bool:
    missing = []
    if not FORPSD_COOKIE:    missing.append("FORPSD_COOKIE")
    if not GDRIVE_SA_JSON:   missing.append("GDRIVE_SA_JSON")
    if not GDRIVE_PSD_FOLDER:  missing.append("GDRIVE_PSD_FOLDER")
    if not GDRIVE_WEBP_FOLDER: missing.append("GDRIVE_WEBP_FOLDER")
    if missing:
        log.error(f"Missing GitHub Secrets: {', '.join(missing)}")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

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

    # ── Components ─────────────────────────────────────────────────────────
    state    = StateManager(STATE_FILE)
    scraper  = ForPSDScraper(FORPSD_COOKIE)
    uploader = DriveUploader(GDRIVE_SA_JSON)

    # ── Phase 1: collect all URLs (first run only) ─────────────────────────
    if not state.get("all_urls"):
        log.info("First run detected – scraping all listing pages …")
        all_urls = scraper.get_all_download_urls()
        if not all_urls:
            log.error("Could not scrape any URLs. Check FORPSD_COOKIE secret.")
            sys.exit(2)
        state.set("all_urls", all_urls)
        state.set("processed", [])
        state.save()
        log.info(f"Collected {len(all_urls)} download URLs and saved to state.json")

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

        # Time check
        if datetime.utcnow() >= deadline:
            log.info(
                f"⏰ Time limit reached. "
                f"Processed {processed_this_run} items this run. "
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
                log.warning(f"  No Drive URL – skipping (will retry next run)")
                errors_this_run += 1
                continue

            log.info(f"  Drive URL: {drive_url}")

            # ── 2b: Download archive ──────────────────────────────────────
            dl_dir   = item_dir / "download"
            archive  = download_from_drive(drive_url, dl_dir)
            if not archive:
                log.error(f"  Download failed – skipping")
                errors_this_run += 1
                continue

            # ── 2c: Process (extract → layer delete → watermark → zip) ────
            result = process_archive(
                archive_path=archive,
                work_dir=item_dir / "process",
            )
            if not result:
                log.error(f"  Processing failed for {archive.name} – skipping")
                errors_this_run += 1
                continue

            new_zip, webp_list = result

            # ── 2d: Upload ZIP ─────────────────────────────────────────────
            if new_zip and new_zip.exists():
                uploader.upload(new_zip, GDRIVE_PSD_FOLDER)
            else:
                log.warning("  ZIP not found after processing")

            # ── 2e: Upload WebP(s) ─────────────────────────────────────────
            for webp in webp_list:
                if webp.exists():
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
            # Always clean up temp files to avoid disk-full
            shutil.rmtree(item_dir, ignore_errors=True)

    # ── Final state save ───────────────────────────────────────────────────
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
