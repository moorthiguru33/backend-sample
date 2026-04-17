"""
config.py – All settings loaded from GitHub Actions environment / secrets.
"""
import os

# ── forpsd.com ─────────────────────────────────────────────────────────────
FORPSD_COOKIE: str = os.environ.get("FORPSD_COOKIE", "")
BASE_URL: str = "https://forpsd.com"

# ── Google Drive – OAuth2 credentials (3 GitHub secrets) ──────────────────
# Old method (Service Account JSON) is replaced by these 3 lightweight vars.
GOOGLE_CLIENT_ID:      str = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET:  str = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REFRESH_TOKEN:  str = os.environ.get("GOOGLE_REFRESH_TOKEN", "")

# Folder IDs (from Drive URL → /drive/folders/<FOLDER_ID>)
GDRIVE_PSD_FOLDER:  str = os.environ.get("GDRIVE_PSD_FOLDER", "")    # psd/    ← original files
GDRIVE_WEBP_FOLDER: str = os.environ.get("GDRIVE_WEBP_FOLDER", "")   # preview/ ← WebP previews

# ── Scrape limit ───────────────────────────────────────────────────────────
PAGE_LIMIT: int = int(os.environ.get("PAGE_LIMIT", "0"))

# ── Timing ─────────────────────────────────────────────────────────────────
RUN_MINUTES: int = int(os.environ.get("RUN_MINUTES", "330"))   # 5.5 h

# ── Watermark ──────────────────────────────────────────────────────────────
WATERMARK_TEXT:    str = "www.tamilpsd.in"
WATERMARK_OPACITY: int = 40   # 0–255  (40 ≈ 16% — visible but not obtrusive)

# PNG watermark file path (tree-style logo).
# • Set WATERMARK_PNG env var to an absolute path of your tree logo PNG.
# • Leave empty to use the diagonal text tile watermark instead.
WATERMARK_PNG: str = os.environ.get("WATERMARK_PNG", "")

# ── WebP size limit ────────────────────────────────────────────────────────
WEBP_MAX_KB: int = int(os.environ.get("WEBP_MAX_KB", "100"))

# ── Paths ──────────────────────────────────────────────────────────────────
WORK_DIR:   str = "/tmp/psd_work"
STATE_FILE: str = "state.json"
DONE_FILE:  str = "ALL_DONE"
LOG_FILE:   str = "automation.log"
