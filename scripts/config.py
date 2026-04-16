"""
config.py – All settings loaded from GitHub Actions environment / secrets.
"""
import os

# ── forpsd.com ─────────────────────────────────────────────────────────────
# Paste the full Cookie header value from browser DevTools → Network tab.
# Example: "laravel_session=xxxx; XSRF-TOKEN=yyyy; ..."
FORPSD_COOKIE: str = os.environ.get("FORPSD_COOKIE", "")

BASE_URL: str = "https://forpsd.com"

# ── Google Drive ───────────────────────────────────────────────────────────
# Full JSON content of a Service Account key file (one line, no newlines).
GDRIVE_SA_JSON: str = os.environ.get("GDRIVE_SA_JSON", "")

# Folder IDs (from the URL when you open the folder in Drive):
#   drive.google.com/drive/folders/<FOLDER_ID>
GDRIVE_PSD_FOLDER:  str = os.environ.get("GDRIVE_PSD_FOLDER", "")   # for ZIPs
GDRIVE_WEBP_FOLDER: str = os.environ.get("GDRIVE_WEBP_FOLDER", "")  # for WebP previews

# ── Timing ─────────────────────────────────────────────────────────────────
RUN_MINUTES: int = int(os.environ.get("RUN_MINUTES", "330"))  # 5.5 h default

# ── Watermark ──────────────────────────────────────────────────────────────
WATERMARK_TEXT:    str   = "www.tamilpsd.in"
WATERMARK_OPACITY: int   = 40   # 0–255  (40 ≈ 16% opacity – visible but subtle)

# ── Paths ──────────────────────────────────────────────────────────────────
WORK_DIR:   str = "/tmp/psd_work"
STATE_FILE: str = "state.json"
DONE_FILE:  str = "ALL_DONE"
LOG_FILE:   str = "automation.log"
