"""
config.py – All settings loaded from GitHub Actions environment / secrets.
"""
import os

# ── forpsd.com ─────────────────────────────────────────────────────────────
FORPSD_COOKIE: str = os.environ.get("FORPSD_COOKIE", "")
BASE_URL: str = "https://forpsd.com"

# ── Google Drive – OAuth2 credentials (3 GitHub secrets) ──────────────────
GOOGLE_CLIENT_ID:      str = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET:  str = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REFRESH_TOKEN:  str = os.environ.get("GOOGLE_REFRESH_TOKEN", "")

# Folder ID (from Drive URL → /drive/folders/<FOLDER_ID>)
# Only PSD folder needed — no WebP/preview folder anymore.
GDRIVE_PSD_FOLDER: str = os.environ.get("GDRIVE_PSD_FOLDER", "")   # psd/ ← original files

# ── Scrape / run limits ────────────────────────────────────────────────────
PAGE_LIMIT: int = int(os.environ.get("PAGE_LIMIT", "0"))

# How many items (archives) to download & process per run.
# 0 = no limit (process everything pending).
ITEM_LIMIT: int = int(os.environ.get("ITEM_LIMIT", "0"))

# ── Timing ─────────────────────────────────────────────────────────────────
RUN_MINUTES: int = int(os.environ.get("RUN_MINUTES", "330"))   # 5.5 h

# ── Paths ──────────────────────────────────────────────────────────────────
WORK_DIR:   str = "/tmp/psd_work"
STATE_FILE: str = "state.json"
DONE_FILE:  str = "ALL_DONE"
LOG_FILE:   str = "automation.log"
