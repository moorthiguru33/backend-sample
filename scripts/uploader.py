"""
uploader.py – Upload files to Google Drive using OAuth2 (Refresh Token).

KEY FIX — Cross-subfolder duplicate prevention + auto-move to duplicates/
=========================================================================
The original code only checked for duplicate names within the SPECIFIC
target subfolder. This caused the same tamilpsd-XXXX name to appear in
multiple subfolders when:
  • Category detection gave different results between runs.
  • state.json was reset and the counter restarted from 1.
  • An interrupted run left a file in one category; the next run put it
    in another.

Fix: On init, `preload_existing_names(parent_folder_id)` scans EVERY
subfolder once and builds a global in-memory set `_known_names` of all
tamilpsd-XXXX filenames that already exist anywhere on Drive.

NEW: If the same filename is found in MULTIPLE subfolders, the extra
copies are automatically moved to a `duplicates/` subfolder. This keeps
Drive clean without losing any files.

Every subsequent upload checks this global set first — if the name
exists ANYWHERE (regardless of subfolder), the upload is skipped.
When a file is successfully uploaded, its name is added to the set so
the same check works for all uploads within the same run.

Excel Tracker Integration
=========================
`upload_to_category` now accepts two optional parameters:
  • excel_tracker  – an ExcelTracker instance (from excel_tracker.py)
  • original_name  – the filename as it existed inside the source archive

Duplicate-check order (upload_to_category):
  1. EXCEL check (fastest): is this original_name already in the log?
     → skip immediately, zero API calls.
  2. EXCEL check: is this renamed name already in the log?
     → skip immediately, zero API calls.
  3. GLOBAL in-memory set: is this renamed name known from Drive scan?
     → skip, no API call.
  4. LOCAL Drive query: belt-and-suspenders API check inside subfolder.
  5. Upload, then log to Excel and in-memory set.

Counter sync
============
`max_counter` (the highest tamilpsd-XXXX number found on Drive) is
exposed so main.py can sync the state.json file_counter at startup,
guaranteeing the counter is always ABOVE any existing file — even after
a state.json loss or reset.
"""

import logging
import mimetypes
import re
from pathlib import Path

import google.auth.transport.requests
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

log = logging.getLogger(__name__)

SCOPES    = ["https://www.googleapis.com/auth/drive"]
TOKEN_URI = "https://oauth2.googleapis.com/token"

# Regex to extract the numeric part from a tamilpsd-XXXX filename
_TAMILPSD_RE = re.compile(r"^tamilpsd-(\d+)\.", re.IGNORECASE)

# Reserved folder names to skip during scans
_RESERVED_FOLDERS = {"duplicates", "final"}


class DriveUploader:
    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
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

        # Cache: { (parent_folder_id, category_name) -> subfolder_id }
        self._folder_cache: dict = {}

        # Global duplicate registry — populated by preload_existing_names().
        # Keys are lowercase filenames (e.g. "tamilpsd-0001.psd").
        self._known_names: set = set()

        # Highest tamilpsd-XXXX number found across all subfolders on Drive.
        # Exposed so main.py can sync file_counter at startup.
        self.max_counter: int = 0

        log.info("✅ Google Drive authenticated via OAuth2 refresh token")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _mime(self, path: Path) -> str:
        mt, _ = mimetypes.guess_type(str(path))
        return mt or "application/octet-stream"

    def _list_all_files_in_folder(self, folder_id: str) -> list:
        """Return all non-folder files directly inside folder_id."""
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
            result = self._svc.files().list(**params).execute()
            items.extend(result.get("files", []))
            page_token = result.get("nextPageToken")
            if not page_token:
                break
        return items

    def _list_subfolders(self, parent_folder_id: str) -> list:
        """Return all subfolder items directly inside parent_folder_id."""
        items, page_token = [], None
        while True:
            params = dict(
                q=(
                    f"'{parent_folder_id}' in parents and trashed=false "
                    "and mimeType='application/vnd.google-apps.folder'"
                ),
                fields="nextPageToken,files(id,name)",
                pageSize="1000",
            )
            if page_token:
                params["pageToken"] = page_token
            result = self._svc.files().list(**params).execute()
            items.extend(result.get("files", []))
            page_token = result.get("nextPageToken")
            if not page_token:
                break
        return items

    # ── Duplicate management ────────────────────────────────────────────────

    def _find_or_create_duplicates_folder(self, parent_folder_id: str) -> str:
        """Find or create a 'duplicates' subfolder inside parent_folder_id."""
        cache_key = (parent_folder_id, "duplicates")
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        q = (
            "name='duplicates' and '{}' in parents "
            "and mimeType='application/vnd.google-apps.folder' and trashed=false"
        ).format(parent_folder_id)
        result = self._svc.files().list(q=q, fields="files(id,name)").execute()
        files = result.get("files", [])

        if files:
            folder_id = files[0]["id"]
            log.info(f"📁 Found existing 'duplicates' folder: {folder_id}")
        else:
            metadata = {
                "name":     "duplicates",
                "mimeType": "application/vnd.google-apps.folder",
                "parents":  [parent_folder_id],
            }
            folder = self._svc.files().create(body=metadata, fields="id").execute()
            folder_id = folder["id"]
            log.info(f"📁 Created new 'duplicates' folder: {folder_id}")

        self._folder_cache[cache_key] = folder_id
        return folder_id

    def _move_to_duplicates(self, file_id: str, file_name: str,
                            parent_folder_id: str) -> bool:
        """Move a duplicate file to the duplicates/ subfolder."""
        try:
            dup_folder_id = self._find_or_create_duplicates_folder(parent_folder_id)

            # Get current parents
            file_info = self._svc.files().get(
                fileId=file_id, fields="parents"
            ).execute()
            current_parents = ",".join(file_info.get("parents", []))

            # Move
            self._svc.files().update(
                fileId=file_id,
                addParents=dup_folder_id,
                removeParents=current_parents,
                fields="id,parents",
            ).execute()
            log.info(f"  📦 Moved duplicate '{file_name}' → duplicates/")
            return True
        except Exception as exc:
            log.warning(f"  ⚠️  Could not move duplicate '{file_name}': {exc}")
            return False

    # ── Startup scan ───────────────────────────────────────────────────────

    def preload_existing_names(self, parent_folder_id: str) -> None:
        """
        Scan every subfolder inside parent_folder_id ONCE at startup and
        populate:
          - self._known_names  : set of all tamilpsd filenames on Drive
          - self.max_counter   : highest tamilpsd number found anywhere

        Also pre-populates the folder cache so _find_or_create_subfolder
        never makes redundant API calls for already-known subfolders.

        NEW: Detects cross-subfolder duplicates (same filename in multiple
        subfolders) and auto-moves extras to a 'duplicates/' subfolder.

        Call this ONCE after creating DriveUploader, before the main loop.
        """
        log.info("🔍 Scanning Drive for existing tamilpsd files (all subfolders)…")

        total_files = 0
        max_num     = 0
        duplicates_moved = 0

        # Track: filename_lower → [(file_id, subfolder_name)]
        # Used to detect same file in multiple subfolders
        file_locations: dict[str, list[tuple[str, str]]] = {}

        # Root-level files
        for f in self._list_all_files_in_folder(parent_folder_id):
            name_lower = f["name"].lower()
            self._known_names.add(name_lower)
            m = _TAMILPSD_RE.match(f["name"])
            if m:
                max_num = max(max_num, int(m.group(1)))
            total_files += 1
            file_locations.setdefault(name_lower, []).append((f["id"], "root"))

        # Every subfolder
        subfolders = self._list_subfolders(parent_folder_id)
        for sf in subfolders:
            sf_id   = sf["id"]
            sf_name = sf["name"]

            # Skip reserved folders
            if sf_name.lower() in _RESERVED_FOLDERS:
                log.info(f"  📁 Skipping '{sf_name}' (reserved folder)")
                continue

            # Pre-populate folder cache
            self._folder_cache[(parent_folder_id, sf_name)] = sf_id
            log.info(f"  📁 Scanning '{sf_name}' …")

            for f in self._list_all_files_in_folder(sf_id):
                name_lower = f["name"].lower()
                self._known_names.add(name_lower)
                m = _TAMILPSD_RE.match(f["name"])
                if m:
                    max_num = max(max_num, int(m.group(1)))
                total_files += 1
                file_locations.setdefault(name_lower, []).append((f["id"], sf_name))

        self.max_counter = max_num

        # ── Auto-move cross-subfolder duplicates ──────────────────────────
        cross_dupes = {
            name: locs for name, locs in file_locations.items()
            if len(locs) > 1
        }
        if cross_dupes:
            log.info(
                f"\n⚠️  Found {len(cross_dupes)} filenames appearing in "
                f"multiple subfolders — moving extras to duplicates/"
            )
            for name, locs in cross_dupes.items():
                # Keep the FIRST occurrence, move the rest
                keep_id, keep_folder = locs[0]
                log.info(f"  🔸 '{name}' — keeping in [{keep_folder}], "
                         f"moving {len(locs)-1} extra(s)")
                for dup_id, dup_folder in locs[1:]:
                    if self._move_to_duplicates(dup_id, name, parent_folder_id):
                        duplicates_moved += 1

        log.info(
            f"✅ Drive scan complete: {total_files} files across "
            f"{len(subfolders)} subfolders | "
            f"Highest tamilpsd number on Drive: {max_num:04d}"
        )
        if duplicates_moved > 0:
            log.info(
                f"📦 Moved {duplicates_moved} duplicate files to duplicates/ folder"
            )

    # ── Internal folder management ─────────────────────────────────────────

    def _find_or_create_subfolder(self, parent_folder_id: str, folder_name: str) -> str:
        cache_key = (parent_folder_id, folder_name)
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        q = (
            f"name='{folder_name}' and '{parent_folder_id}' in parents "
            f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
        result = self._svc.files().list(q=q, fields="files(id,name)").execute()
        files  = result.get("files", [])

        if files:
            folder_id = files[0]["id"]
            log.info(f"📁 Found existing subfolder '{folder_name}': {folder_id}")
        else:
            metadata = {
                "name":     folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents":  [parent_folder_id],
            }
            folder    = self._svc.files().create(body=metadata, fields="id").execute()
            folder_id = folder["id"]
            log.info(f"📁 Created new subfolder '{folder_name}': {folder_id}")

        self._folder_cache[cache_key] = folder_id
        return folder_id

    # ── Public API ─────────────────────────────────────────────────────────

    def name_exists_anywhere(self, filename: str) -> bool:
        """
        Return True if filename already exists in ANY subfolder on Drive.
        Uses the in-memory registry — no extra API call needed.
        """
        return filename.lower() in self._known_names

    def upload_to_category(
        self,
        file_path,
        parent_folder_id: str,
        category: str,
        *,
        excel_tracker=None,
        original_name: str = "",
    ) -> dict:
        """
        Upload file_path into a category subfolder inside parent_folder_id.

        Parameters
        ----------
        file_path         : Path to the renamed local file (tamilpsd-XXXX.ext)
        parent_folder_id  : Google Drive parent folder ID
        category          : subfolder name inside parent_folder_id
        excel_tracker     : ExcelTracker instance (optional but strongly recommended)
        original_name     : original filename from source archive
                            (stored in Excel's "Original File Name" column)

        Duplicate-check order
        ---------------------
          1. Excel — original name already logged? → skip (zero API calls)
          2. Excel — renamed name already logged?  → skip (zero API calls)
          3. Global in-memory Drive set            → skip (zero API calls)
          4. Local Drive query (belt-and-suspenders)
          5. Upload → log to Excel + in-memory set
        """
        path     = Path(file_path)
        name     = path.name
        category = category.strip() or "uncategorized"

        # ── 1. Excel: original name check ─────────────────────────────────
        if excel_tracker and original_name:
            if excel_tracker.is_original_done(original_name):
                log.info(
                    f"⏭  Original '{original_name}' already in Excel log — "
                    f"skipping upload of '{name}'"
                )
                return {"name": name, "skipped": True, "reason": "excel_original", "category": category}

        # ── 2. Excel: renamed name check ──────────────────────────────────
        if excel_tracker:
            if excel_tracker.is_renamed_used(name):
                log.info(
                    f"⏭  Renamed '{name}' already in Excel log — "
                    f"skipping (not uploading to [{category}/])"
                )
                return {"name": name, "skipped": True, "reason": "excel_renamed", "category": category}

        # ── 3. Global cross-subfolder Drive check (in-memory) ─────────────
        if self.name_exists_anywhere(name):
            log.info(
                f"⏭  '{name}' already exists somewhere on Drive — "
                f"skipping (not uploading to [{category}/])"
            )
            # Back-fill Excel so future runs skip via Excel (faster, no Drive scan needed)
            if excel_tracker and not excel_tracker.is_renamed_used(name):
                excel_tracker.add_entry(original_name or name, name)
            return {"name": name, "skipped": True, "reason": "drive_global", "category": category}

        # ── 4. Resolve category subfolder ─────────────────────────────────
        subfolder_id = self._find_or_create_subfolder(parent_folder_id, category)

        # ── 5. Local duplicate check (belt-and-suspenders) ────────────────
        q = (
            f"name='{name}' and '{subfolder_id}' in parents "
            f"and trashed=false and mimeType!='application/vnd.google-apps.folder'"
        )
        result = self._svc.files().list(q=q, fields="files(id,name)").execute()
        if result.get("files"):
            log.info(f"⏭  Already on Drive [{category}/], skipping: {name}")
            self._known_names.add(name.lower())
            # Back-fill Excel
            if excel_tracker and not excel_tracker.is_renamed_used(name):
                excel_tracker.add_entry(original_name or name, name)
            return {"name": name, "skipped": True, "reason": "drive_local", "category": category}

        # ── 6. Upload ──────────────────────────────────────────────────────
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

            # Register in global in-memory set immediately
            self._known_names.add(name.lower())

            # Log to Excel immediately — before printing success, so a crash
            # here still leaves the entry in Excel for the next run.
            if excel_tracker:
                excel_tracker.add_entry(original_name or name, name)

            log.info(
                f"✅ Uploaded: {response.get('name')} "
                f"→ [{category}/] {response.get('webViewLink')}"
            )
            return {**response, "category": category, "folder_id": subfolder_id}

        except HttpError as exc:
            log.error(f"Drive upload error for {name}: {exc}")
            raise

    def upload(self, file_path, folder_id: str, *, excel_tracker=None, original_name: str = "") -> dict:
        """Legacy upload — uploads directly into folder_id (no subfolder)."""
        path = Path(file_path)
        name = path.name

        if excel_tracker and original_name and excel_tracker.is_original_done(original_name):
            log.info(f"⏭  Original '{original_name}' already in Excel log — skipping '{name}'")
            return {"name": name, "skipped": True, "reason": "excel_original"}

        if excel_tracker and excel_tracker.is_renamed_used(name):
            log.info(f"⏭  Renamed '{name}' already in Excel log — skipping")
            return {"name": name, "skipped": True, "reason": "excel_renamed"}

        if self.name_exists_anywhere(name):
            log.info(f"⏭  Already exists on Drive (any subfolder), skipping: {name}")
            return {"name": name, "skipped": True}

        q = (
            f"name='{name}' and '{folder_id}' in parents "
            f"and trashed=false and mimeType!='application/vnd.google-apps.folder'"
        )
        result = self._svc.files().list(q=q, fields="files(id,name)").execute()
        if result.get("files"):
            log.info(f"⏭  Already on Drive, skipping: {name}")
            self._known_names.add(name.lower())
            if excel_tracker and not excel_tracker.is_renamed_used(name):
                excel_tracker.add_entry(original_name or name, name)
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

            self._known_names.add(name.lower())

            if excel_tracker:
                excel_tracker.add_entry(original_name or name, name)

            log.info(f"✅ Uploaded: {response.get('name')} → {response.get('webViewLink')}")
            return response

        except HttpError as exc:
            log.error(f"Drive upload error for {name}: {exc}")
            raise
