"""
uploader.py – Upload files to Google Drive using OAuth2 (Refresh Token).

No service-account JSON blob needed. Uses 3 simple GitHub secrets:
  GOOGLE_CLIENT_ID
  GOOGLE_CLIENT_SECRET
  GOOGLE_REFRESH_TOKEN

One-time setup:
  1. Google Cloud Console → Enable Drive API
  2. Create OAuth 2.0 Desktop credentials → get client_id + client_secret
  3. Run get_refresh_token.py locally → get refresh_token
  4. Add all three to GitHub → Settings → Secrets and variables → Actions
  5. Share target Drive folders with the Google account you authenticated with
"""

import logging
import mimetypes
from pathlib import Path

import google.auth.transport.requests
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_URI = "https://oauth2.googleapis.com/token"


class DriveUploader:
    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
        """
        Build Drive service using OAuth2 refresh token.
        The access token is obtained automatically and refreshed as needed.
        """
        creds = Credentials(
            token=None,                  # No access token yet — will be fetched
            refresh_token=refresh_token,
            token_uri=TOKEN_URI,
            client_id=client_id,
            client_secret=client_secret,
            scopes=SCOPES,
        )
        # Force immediate token refresh so any auth error fails fast at startup
        request = google.auth.transport.requests.Request()
        creds.refresh(request)

        self._svc = build("drive", "v3", credentials=creds, cache_discovery=False)
        log.info("✅ Google Drive authenticated via OAuth2 refresh token")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _mime(self, path: Path) -> str:
        mt, _ = mimetypes.guess_type(str(path))
        return mt or "application/octet-stream"

    def _file_exists(self, name: str, folder_id: str) -> bool:
        """Return True if a file with this name already exists in the folder."""
        q = (
            f"name='{name}' and '{folder_id}' in parents "
            f"and trashed=false"
        )
        result = self._svc.files().list(q=q, fields="files(id)").execute()
        return len(result.get("files", [])) > 0

    # ── Public API ─────────────────────────────────────────────────────────

    def upload(self, file_path: "str | Path", folder_id: str) -> dict:
        """
        Upload file_path to the specified Drive folder.
        Skips if a file with the same name already exists.
        Returns the file metadata dict.
        """
        path = Path(file_path)
        name = path.name

        if self._file_exists(name, folder_id):
            log.info(f"Skipping upload (already exists): {name}")
            return {"name": name, "skipped": True}

        metadata = {"name": name, "parents": [folder_id]}
        media = MediaFileUpload(str(path), mimetype=self._mime(path), resumable=True)

        try:
            request = self._svc.files().create(
                body=metadata, media_body=media, fields="id,name,webViewLink"
            )

            # Resumable upload with progress logging
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    pct = int(status.progress() * 100)
                    log.info(f"  Uploading {name}: {pct}%")

            log.info(f"✅ Uploaded: {response.get('name')} → {response.get('webViewLink')}")
            return response

        except HttpError as exc:
            log.error(f"Drive upload error for {name}: {exc}")
            raise
