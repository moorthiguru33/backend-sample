"""
cleanup_duplicates.py
══════════════════════════════════════════════════════════════════════
PURPOSE
  Remove all EXTRA tamilpsd files from Google Drive and the local
  preview_image/ folder that are NOT recorded in rename_log.xlsx.

  rename_log.xlsx is the SINGLE SOURCE OF TRUTH.
  Every tamilpsd file that is NOT in that log is a duplicate / stale
  upload and should be deleted.

HOW TO RUN
  python scripts/cleanup_duplicates.py [--dry-run] [--drive] [--preview]

  --dry-run   Show what WOULD be deleted without actually deleting
  --drive     Clean Google Drive (requires GOOGLE_* env vars)
  --preview   Clean local preview_image/ folder
  (default: --dry-run --drive --preview)

WHAT IT DOES
  1. Reads rename_log.xlsx  → builds set of VALID tamilpsd names
  2. Scans Google Drive     → deletes files NOT in valid set
  3. Scans preview_image/   → deletes files NOT in valid set
  4. Prints a full report
══════════════════════════════════════════════════════════════════════
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
RENAME_LOG  = str(_REPO_ROOT / "rename_log.xlsx")
PREVIEW_DIR = _REPO_ROOT / "preview_image"


# ── Read rename_log ────────────────────────────────────────────────────────

def load_valid_names(rename_log_path: str) -> set:
    """
    Return a set of lowercase filenames (e.g. 'tamilpsd-0001.psd')
    that exist in rename_log.xlsx — these are VALID / wanted files.
    """
    wb = openpyxl.load_workbook(rename_log_path, read_only=True)
    ws = wb.active
    valid = set()
    for row in ws.iter_rows(min_row=2, values_only=True):
        renamed = row[1]
        if renamed:
            valid.add(str(renamed).strip().lower())
    wb.close()
    log.info(f"✅ rename_log.xlsx: {len(valid)} valid tamilpsd file names loaded")
    return valid


# ── Drive cleanup ──────────────────────────────────────────────────────────

def cleanup_drive(valid_names: set, dry_run: bool) -> None:
    """Scan all subfolders in GDRIVE_PSD_FOLDER and delete extras."""
    from config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN, GDRIVE_PSD_FOLDER

    if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN, GDRIVE_PSD_FOLDER]):
        log.error("Missing GOOGLE_* env vars — cannot clean Drive.")
        return

    import google.auth.transport.requests
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    SCOPES    = ["https://www.googleapis.com/auth/drive"]
    TOKEN_URI = "https://oauth2.googleapis.com/token"

    creds = Credentials(
        token=None,
        refresh_token=GOOGLE_REFRESH_TOKEN,
        token_uri=TOKEN_URI,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=SCOPES,
    )
    creds.refresh(google.auth.transport.requests.Request())
    svc = build("drive", "v3", credentials=creds, cache_discovery=False)

    def list_files(folder_id):
        items, page_token = [], None
        while True:
            params = dict(
                q=(
                    f"'{folder_id}' in parents and trashed=false "
                    "and mimeType!='application/vnd.google-apps.folder'"
                ),
                fields="nextPageToken,files(id,name)",
                pageSize="1000",
            )
            if page_token:
                params["pageToken"] = page_token
            r = svc.files().list(**params).execute()
            items.extend(r.get("files", []))
            page_token = r.get("nextPageToken")
            if not page_token:
                break
        return items

    def list_subfolders(folder_id):
        items, page_token = [], None
        while True:
            params = dict(
                q=(
                    f"'{folder_id}' in parents and trashed=false "
                    "and mimeType='application/vnd.google-apps.folder'"
                ),
                fields="nextPageToken,files(id,name)",
                pageSize="1000",
            )
            if page_token:
                params["pageToken"] = page_token
            r = svc.files().list(**params).execute()
            items.extend(r.get("files", []))
            page_token = r.get("nextPageToken")
            if not page_token:
                break
        return items

    log.info("🔍 Scanning Google Drive for ALL tamilpsd files…")

    total_scanned  = 0
    total_deleted  = 0
    total_kept     = 0
    to_delete      = []

    # Root-level files
    for f in list_files(GDRIVE_PSD_FOLDER):
        total_scanned += 1
        if f["name"].lower() not in valid_names:
            to_delete.append((f["id"], f["name"], "root"))
        else:
            total_kept += 1

    # Subfolders
    for sf in list_subfolders(GDRIVE_PSD_FOLDER):
        log.info(f"  📁 Scanning subfolder: {sf['name']}")
        for f in list_files(sf["id"]):
            total_scanned += 1
            if f["name"].lower() not in valid_names:
                to_delete.append((f["id"], f["name"], sf["name"]))
            else:
                total_kept += 1

    log.info(f"\n{'='*60}")
    log.info(f"Drive scan complete:")
    log.info(f"  Total files scanned : {total_scanned}")
    log.info(f"  Valid (keep)        : {total_kept}")
    log.info(f"  Extra (delete)      : {len(to_delete)}")
    log.info(f"{'='*60}")

    if dry_run:
        log.info("DRY RUN — showing first 50 files that WOULD be deleted:")
        for fid, name, folder in to_delete[:50]:
            log.info(f"  [{folder}] {name}")
        if len(to_delete) > 50:
            log.info(f"  ... and {len(to_delete)-50} more")
        return

    import time as _time
    log.info(f"🗑️  Deleting {len(to_delete)} extra files from Drive…")
    log.info(f"   (This may take several minutes for large batches)")
    for i, (fid, name, folder) in enumerate(to_delete, 1):
        try:
            svc.files().delete(fileId=fid).execute()
            total_deleted += 1
            if i % 50 == 0:
                log.info(f"  ✅ Deleted {i}/{len(to_delete)} ({total_deleted} success)…")
                _time.sleep(0.5)   # avoid Drive API rate limit
        except Exception as exc:
            log.warning(f"  Could not delete {name}: {exc}")

    log.info(f"✅ Drive cleanup done: {total_deleted} files deleted, {total_kept} files kept")


# ── Preview folder cleanup ─────────────────────────────────────────────────

def cleanup_preview(valid_names: set, dry_run: bool) -> None:
    """Remove preview_image/*.webp files NOT in valid_names."""
    if not PREVIEW_DIR.exists():
        log.info("preview_image/ folder not found — skipping")
        return

    webp_files = list(PREVIEW_DIR.glob("*.webp"))
    log.info(f"\n🔍 preview_image/: {len(webp_files)} .webp files found")

    to_delete = []
    to_keep   = []

    for f in webp_files:
        # Preview is named e.g. tamilpsd-0001.webp
        # valid_names contains tamilpsd-0001.psd / tamilpsd-0001.png etc.
        # Match by the numeric part: tamilpsd-XXXX
        stem = f.stem.lower()   # e.g. "tamilpsd-0001"
        # Check if ANY valid name starts with this stem
        if any(v.startswith(stem + ".") for v in valid_names):
            to_keep.append(f)
        else:
            to_delete.append(f)

    log.info(f"  Valid (keep)  : {len(to_keep)}")
    log.info(f"  Extra (delete): {len(to_delete)}")

    if dry_run:
        log.info("DRY RUN — showing first 50 preview files that WOULD be deleted:")
        for f in to_delete[:50]:
            log.info(f"  {f.name}")
        if len(to_delete) > 50:
            log.info(f"  ... and {len(to_delete)-50} more")
        return

    log.info(f"🗑️  Deleting {len(to_delete)} extra preview files…")
    deleted = 0
    for f in to_delete:
        try:
            f.unlink()
            deleted += 1
        except Exception as exc:
            log.warning(f"  Could not delete {f.name}: {exc}")

    log.info(f"✅ Preview cleanup done: {deleted} deleted, {len(to_keep)} kept")
    log.info("ℹ️  Remember to git commit & push after cleanup!")


# ── Report latest uploads ──────────────────────────────────────────────────

def show_latest_uploads(rename_log_path: str, n: int = 20) -> None:
    """Print the last N entries from rename_log (latest uploads)."""
    wb  = openpyxl.load_workbook(rename_log_path, read_only=True)
    ws  = wb.active
    rows = [(r[0], r[1]) for r in ws.iter_rows(min_row=2, values_only=True) if r[0] or r[1]]
    wb.close()

    log.info(f"\n{'='*60}")
    log.info(f"Latest {n} uploads in rename_log.xlsx:")
    log.info(f"{'Original (forpsd)':<35} → {'TamilPSD name'}")
    log.info(f"{'-'*60}")
    for orig, renamed in rows[-n:]:
        log.info(f"  {str(orig):<33} → {renamed}")
    log.info(f"{'='*60}")
    log.info(f"Total logged entries: {len(rows)}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cleanup duplicate tamilpsd files")
    parser.add_argument("--dry-run",  action="store_true", default=True,
                        help="Show what would be deleted (default: True)")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                        help="Actually delete files")
    parser.add_argument("--drive",   action="store_true", default=True,
                        help="Clean Google Drive (default: True)")
    parser.add_argument("--preview", action="store_true", default=True,
                        help="Clean preview_image/ folder (default: True)")
    parser.add_argument("--show-latest", type=int, default=20,
                        help="Show latest N uploads from rename_log (default: 20)")
    args = parser.parse_args()

    if args.dry_run:
        log.info("🔵 DRY RUN MODE — nothing will be deleted")
    else:
        log.info("🔴 LIVE MODE — files WILL be permanently deleted!")
        log.info("   (Confirmation via --no-dry-run flag — no interactive prompt in CI)")

    # Load valid names
    valid_names = load_valid_names(RENAME_LOG)

    # Show latest uploads
    show_latest_uploads(RENAME_LOG, args.show_latest)

    # Clean Drive
    if args.drive:
        cleanup_drive(valid_names, args.dry_run)

    # Clean preview_image/
    if args.preview:
        cleanup_preview(valid_names, args.dry_run)

    log.info("\n✅ Cleanup script finished.")
    if args.dry_run:
        log.info("Run with --no-dry-run to actually delete the files.")


if __name__ == "__main__":
    main()
