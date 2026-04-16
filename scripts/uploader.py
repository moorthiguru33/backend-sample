"""
uploader.py – Upload files to Google Drive using a Service Account.

Setup (one-time):
  1. Google Cloud Console → Create a project → Enable Drive API.
  2. IAM → Service Accounts → Create → Download JSON key.
  3. Open both target folders in Drive → Share → Add the service-account
     email (ends in @…iam.gserviceaccount.com) as Editor.
  4. Store the JSON key contents (single-line) in GitHub secret GDRIVE_SA_JSON.
  5. Store the folder IDs in GDRIVE_PSD_FOLDER and GDRIVE_WEBP_FOLDER.
"""

import json
import logging
import mimetypes
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive"]


class DriveUploader:
    def __init__(self, sa_json_string: str):
        info = json.loads(sa_json_string)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=SCOPES
        )
        self._svc = build("drive", "v3", credentials=creds, cache_discovery=False)

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
    def upload(self, file_path: str | Path, folder_id: str) -> dict:
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

            # Resumable upload with progress
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
