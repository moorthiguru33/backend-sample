"""
downloader.py – Download a file from Google Drive.

Strategy:
  1. For 'usp=drivesdk' URLs (private/SDK-shared): skip gdown entirely
     and use the authenticated Google Drive API directly.
  2. For public 'usp=drive_link' URLs: try gdown first (fast, no auth
     needed), then fall back to the authenticated API if gdown fails
     (e.g. due to 'too many accesses' throttling).

ROOT CAUSE FIX:
  Files shared via the Google Drive mobile app / SDK use `usp=drivesdk`
  in their URL. These are NOT publicly accessible — gdown cannot retrieve
  the download link and throws:
    "Cannot retrieve the public link of the file. You may need to change
     the permission to 'Anyone with the link', or have had many accesses."

  The fix is to detect these private links and download them using the
  OAuth2 credentials (GOOGLE_CLIENT_ID / CLIENT_SECRET / REFRESH_TOKEN)
  that are already present in the project.
"""

import re
import time
import logging
from pathlib import Path

import requests
import gdown

log = logging.getLogger(__name__)

# Regex patterns to extract file_id from various Drive URL formats
_ID_PATTERNS = [
    r"/file/d/([a-zA-Z0-9_-]{20,})",   # /file/d/<id>/view
    r"[?&]id=([a-zA-Z0-9_-]{20,})",    # ?id=<id>
    r"/d/([a-zA-Z0-9_-]{20,})",        # /d/<id>
]

# In-memory token cache — avoids re-fetching on every file
_token_cache: dict = {"access_token": None, "expires_at": 0.0}


def extract_file_id(drive_url: str) -> "str | None":
    for pattern in _ID_PATTERNS:
        m = re.search(pattern, drive_url)
        if m:
            return m.group(1)
    log.error(f"Cannot extract file_id from: {drive_url}")
    return None


# ── OAuth2 helpers ─────────────────────────────────────────────────────────

def _get_oauth_token(client_id: str, client_secret: str, refresh_token: str) -> "str | None":
    """Return a valid OAuth2 access token, refreshing from Google if expired."""
    now = time.time()
    if _token_cache["access_token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["access_token"]

    try:
        resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id":     client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type":    "refresh_token",
            },
            timeout=30,
        )
        data = resp.json()
        if "access_token" not in data:
            log.error(f"OAuth2 token refresh failed: {data}")
            return None
        _token_cache["access_token"] = data["access_token"]
        _token_cache["expires_at"]   = now + data.get("expires_in", 3600)
        log.info("  🔑 OAuth2 token refreshed for authenticated download")
        return _token_cache["access_token"]
    except Exception as exc:
        log.error(f"OAuth2 token refresh error: {exc}")
        return None


def _authenticated_download(
    file_id: str,
    dest_dir: Path,
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> "Path | None":
    """Download a Google Drive file via the Drive v3 API using OAuth2."""
    token = _get_oauth_token(client_id, client_secret, refresh_token)
    if not token:
        log.error("  Cannot download: failed to obtain OAuth2 token")
        return None

    headers = {"Authorization": f"Bearer {token}"}

    # Step 1: get file metadata so we know the real filename
    try:
        meta_resp = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            params={"fields": "name,size"},
            headers=headers,
            timeout=30,
        )
        meta_resp.raise_for_status()
        meta     = meta_resp.json()
        filename = meta.get("name", f"{file_id}.zip")
        size_raw = meta.get("size")
        size_str = f"{int(size_raw) / 1024 / 1024:.1f} MB" if size_raw else "unknown size"
        log.info(f"  Authenticated download: {filename} ({size_str})")
    except Exception as exc:
        log.warning(f"  Could not fetch file metadata: {exc} — using fallback filename")
        filename = f"{file_id}.zip"

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    # Step 2: stream the file content
    try:
        with requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            params={"alt": "media"},
            headers=headers,
            timeout=600,
            stream=True,
        ) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):  # 8 MB chunks
                    fh.write(chunk)

        size_mb = dest_path.stat().st_size / 1024 / 1024
        log.info(f"  ✅ Authenticated download OK: {filename} ({size_mb:.1f} MB)")
        return dest_path

    except Exception as exc:
        log.error(f"  Authenticated download failed for file_id={file_id}: {exc}")
        if dest_path.exists():
            dest_path.unlink()
        return None


# ── Public API ─────────────────────────────────────────────────────────────

def download_from_drive(
    drive_url: str,
    dest_dir: Path,
    client_id:     str = "",
    client_secret: str = "",
    refresh_token: str = "",
) -> "Path | None":
    """
    Download a file from Google Drive to dest_dir.
    Returns the Path of the downloaded file, or None on failure.

    Parameters
    ----------
    drive_url     : Google Drive share URL (any format)
    dest_dir      : directory to save the file into
    client_id     : Google OAuth2 client ID   (from GOOGLE_CLIENT_ID secret)
    client_secret : Google OAuth2 client secret
    refresh_token : Google OAuth2 refresh token

    Download strategy
    -----------------
    • usp=drivesdk  → private file (shared via mobile/SDK app)
                      gdown CANNOT access these — go straight to OAuth API.
    • usp=drive_link → public file
                      try gdown first (no credentials needed, fast),
                      fall back to OAuth API if gdown fails.
    • No credentials → gdown only; fail gracefully if gdown fails.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    file_id = extract_file_id(drive_url)
    if not file_id:
        return None

    has_credentials = bool(client_id and client_secret and refresh_token)

    # ── Private file: usp=drivesdk ────────────────────────────────────────
    # These are NEVER accessible via gdown — skip it and go straight to auth.
    if "usp=drivesdk" in drive_url:
        log.info("  Private Drive link (usp=drivesdk) — using authenticated download")
        if not has_credentials:
            log.error(
                "  No OAuth credentials available to download private file. "
                "Set GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET / GOOGLE_REFRESH_TOKEN."
            )
            return None
        return _authenticated_download(file_id, dest_dir, client_id, client_secret, refresh_token)

    # ── Public file: try gdown first ──────────────────────────────────────
    try:
        output = gdown.download(
            id=file_id,
            output=str(dest_dir) + "/",
            quiet=False,
            fuzzy=True,
        )
        if output is not None:
            downloaded = Path(output)
            log.info(
                f"Downloaded: {downloaded.name} "
                f"({downloaded.stat().st_size / 1024 / 1024:.1f} MB)"
            )
            return downloaded
        log.warning(f"gdown returned None for file_id={file_id}")
    except Exception as exc:
        log.warning(f"gdown failed for {drive_url}: {exc}")

    # ── Fallback: authenticated download ──────────────────────────────────
    if has_credentials:
        log.info("  Falling back to authenticated Drive API download")
        return _authenticated_download(file_id, dest_dir, client_id, client_secret, refresh_token)

    log.error(
        f"Download failed for {drive_url}: gdown failed and no OAuth credentials provided."
    )
    return None
