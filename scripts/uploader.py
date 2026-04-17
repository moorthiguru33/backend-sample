"""
uploader.py – Upload files to Google Drive using OAuth2 (Refresh Token).

No service-account JSON blob needed. Uses 3 simple GitHub secrets:
  GOOGLE_CLIENT_ID
  GOOGLE_CLIENT_SECRET
  GOOGLE_REFRESH_TOKEN

Category subfolder support:
  • upload_to_category(file, parent_folder_id, category) automatically:
      1. Looks for an existing subfolder named <category> inside parent.
      2. Creates it if it does not exist.
      3. Uploads the file into that subfolder.
  • Folder IDs are cached in memory to avoid repeated API calls.

Skip logic:
  Before every upload, the file name is checked against the target folder.
  If a file with the same name already exists (trashed=false), the upload
  is skipped entirely — no bandwidth wasted, no duplicates created.
  This makes every run safe to re-run after an interruption.

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

SCOPES    = ["https://www.googleapis.com/auth/drive"]
TOKEN_URI = "https://oauth2.googleapis.com/token"


class DriveUploader:
    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
        """
        Build Drive service using OAuth2 refresh token.
        The access token is obtained automatically and refreshed as needed.
        """
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri=TOKEN_URI,
            client_id=client_id,
            client_secret=client_secret,
            scopes=SCOPES,
        )
        request = google.auth.transport.requests.Request()
        creds.refresh(request)

        self._svc = build("drive", "v3", credentials=creds, cache_discovery=False)

        # Cache: { (parent_folder_id, category_name) → subfolder_id }
        self._folder_cache: dict[tuple[str, str], str] = {}

        log.info("✅ Google Drive authenticated via OAuth2 refresh token")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _mime(self, path: Path) -> str:
        mt, _ = mimetypes.guess_type(str(path))
        return mt or "application/octet-stream"

    def _file_exists_in_folder(self, name: str, folder_id: str) -> bool:
        """Return True if a file with this exact name already exists in the folder."""
        q = (
            f"name='{name}' and '{folder_id}' in parents "
            f"and trashed=false and mimeType!='application/vnd.google-apps.folder'"
        )
        result = self._svc.files().list(q=q, fields="files(id,name)").execute()
        return len(result.get("files", [])) > 0

    def _find_or_create_subfolder(self, parent_folder_id: str, folder_name: str) -> str:
        """
        Return the Drive folder ID for <folder_name> inside <parent_folder_id>.
        Creates the subfolder if it does not already exist.
        Caches results to avoid repeated API calls within the same run.
        """
        cache_key = (parent_folder_id, folder_name)
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        # Search for existing subfolder
        q = (
            f"name='{folder_name}' and '{parent_folder_id}' in parents "
            f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
        result = self._svc.files().list(q=q, fields="files(id,name)").execute()
        files = result.get("files", [])

        if files:
            folder_id = files[0]["id"]
            log.info(f"📁 Found existing subfolder '{folder_name}': {folder_id}")
        else:
            # Create the subfolder
            metadata = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_folder_id],
            }
            folder = self._svc.files().create(
                body=metadata, fields="id"
            ).execute()
            folder_id = folder["id"]
            log.info(f"📁 Created new subfolder '{folder_name}': {folder_id}")

        self._folder_cache[cache_key] = folder_id
        return folder_id

    # ── Public API ─────────────────────────────────────────────────────────

    def upload_to_category(
        self,
        file_path: "str | Path",
        parent_folder_id: str,
        category: str,
    ) -> dict:
        """
        Upload file_path into a category subfolder inside parent_folder_id.

        Flow:
          1. Find or create subfolder named <category> in <parent_folder_id>.
          2. Skip if file already exists in that subfolder.
          3. Upload file to the subfolder.

        Returns:
          {"name": ..., "skipped": True}               → already on Drive
          {"id": ..., "name": ..., "webViewLink": ...,
           "category": ..., "folder_id": ...}          → freshly uploaded
        """
        path     = Path(file_path)
        name     = path.name
        category = category.strip() or "uncategorized"

        # Step 1: Resolve category subfolder
        subfolder_id = self._find_or_create_subfolder(parent_folder_id, category)

        # Step 2: Skip if already on Drive
        if self._file_exists_in_folder(name, subfolder_id):
            log.info(f"⏭  Already on Drive [{category}/], skipping: {name}")
            return {"name": name, "skipped": True, "category": category}

        # Step 3: Upload
        metadata = {"name": name, "parents": [subfolder_id]}
        media    = MediaFileUpload(str(path), mimetype=self._mime(path), resumable=True)

        try:
            request = self._svc.files().create(
                body=metadata, media_body=media, fields="id,name,webViewLink"
            )

            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    pct = int(status.progress() * 100)
                    log.info(f"  Uploading {name} → [{category}/]: {pct}%")

            log.info(
                f"✅ Uploaded: {response.get('name')} "
                f"→ [{category}/] {response.get('webViewLink')}"
            )
            return {**response, "category": category, "folder_id": subfolder_id}

        except HttpError as exc:
            log.error(f"Drive upload error for {name}: {exc}")
            raise

    def upload(self, file_path: "str | Path", folder_id: str) -> dict:
        """
        Legacy upload method — uploads directly into folder_id (no subfolder).
        Kept for backward compatibility.
        """
        path = Path(file_path)
        name = path.name

        if self._file_exists_in_folder(name, folder_id):
            log.info(f"⏭  Already on Drive, skipping: {name}")
            return {"name": name, "skipped": True}

        metadata = {"name": name, "parents": [folder_id]}
        media    = MediaFileUpload(str(path), mimetype=self._mime(path), resumable=True)

        try:
            request = self._svc.files().create(
                body=metadata, media_body=media, fields="id,name,webViewLink"
            )
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
