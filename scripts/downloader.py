"""
downloader.py – Download a file from Google Drive using gdown.

Handles:
  • Public shared files (standard gdown)
  • Large files (virus-scan warning bypass built into gdown)
  • Extracts the file_id from any Drive URL format
"""

import re
import logging
from pathlib import Path

import gdown

log = logging.getLogger(__name__)

# Regex patterns to extract file_id from various Drive URL formats
_ID_PATTERNS = [
    r"/file/d/([a-zA-Z0-9_-]{20,})",   # /file/d/<id>/view
    r"[?&]id=([a-zA-Z0-9_-]{20,})",    # ?id=<id>
    r"/d/([a-zA-Z0-9_-]{20,})",        # /d/<id>
]


def extract_file_id(drive_url: str) -> str | None:
    for pattern in _ID_PATTERNS:
        m = re.search(pattern, drive_url)
        if m:
            return m.group(1)
    log.error(f"Cannot extract file_id from: {drive_url}")
    return None


def download_from_drive(drive_url: str, dest_dir: Path) -> Path | None:
    """
    Download a file from Google Drive to dest_dir.
    Returns the Path of the downloaded file, or None on failure.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    file_id = extract_file_id(drive_url)
    if not file_id:
        return None

    try:
        # gdown with fuzzy=True handles all URL formats and the large-file
        # "can't scan for viruses" warning page automatically.
        output = gdown.download(
            id=file_id,
            output=str(dest_dir) + "/",
            quiet=False,
            fuzzy=True,
        )
        if output is None:
            log.error(f"gdown returned None for file_id={file_id}")
            return None

        downloaded = Path(output)
        log.info(f"Downloaded: {downloaded.name} ({downloaded.stat().st_size / 1024 / 1024:.1f} MB)")
        return downloaded

    except Exception as exc:
        log.error(f"Download failed for {drive_url}: {exc}")
        return None
