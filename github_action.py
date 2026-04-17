#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║   PSD ALL-IN-ONE TOOL — GITHUB ACTIONS EDITION                      ║
║  ─────────────────────────────────────────────────────────────────  ║
║  1.  Auth to Google Drive via refresh token (no browser)            ║
║  2.  Scan GDRIVE_PSD_FOLDER → subfolders are categories             ║
║  3.  Fetch designs.xlsx from GitHub → smart-skip existing IDs       ║
║  4.  Download each new image from GDrive                            ║
║  5.  Upload preview JPG to GDrive/jpg/ subfolder                    ║
║  6.  ModelScope Qwen-VL (SCOPE token) → vision + full SEO           ║
║  7.  Add row to designs.xlsx                                        ║
║  8.  Auto-push every 20 rows + final push at end                    ║
╚══════════════════════════════════════════════════════════════════════╝

Secrets required in GitHub → Settings → Secrets and variables → Actions:
  GOOGLE_CLIENT_ID       OAuth2 client ID
  GOOGLE_CLIENT_SECRET   OAuth2 client secret
  GOOGLE_REFRESH_TOKEN   OAuth2 refresh token (run get_refresh_token.py once locally)
  GDRIVE_PSD_FOLDER      Google Drive root folder ID (contains category subfolders)
  SCOPE                  ModelScope API token (modelscope.cn)
  GITHUB_TOKEN           auto-provided by GitHub Actions (no need to add manually)
"""

import os, io, re, sys, time, base64, json, random, traceback
import requests as req_lib
import pandas as pd
from PIL import Image as PILImage
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION FROM ENVIRONMENT VARIABLES
# ──────────────────────────────────────────────────────────────────────────────

GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REFRESH_TOKEN = os.environ.get("GOOGLE_REFRESH_TOKEN", "")
GDRIVE_PSD_FOLDER    = os.environ.get("GDRIVE_PSD_FOLDER", "")
GITHUB_TOKEN         = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO          = os.environ.get("GITHUB_REPO", "")
SCOPE_API_KEY        = os.environ.get("SCOPE", "")

# Optional: limit how many files to process in one run (0 = unlimited)
MAX_FILES = int(os.environ.get("MAX_FILES", "0"))

# ── ModelScope AI (OpenAI-compatible endpoint) ────────────────────────────────
# Qwen2.5-VL-72B-Instruct: best vision + text model on ModelScope
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
VISION_MODEL        = "Qwen/Qwen2.5-VL-72B-Instruct"
MAX_TOKENS          = 1200
AI_TEMPERATURE      = 0.75
CALL_DELAY          = 8          # seconds between AI calls (adjust for rate limits)
MAX_AI_RETRIES      = 3          # retries on transient API errors
AI_RETRY_DELAY      = 20         # seconds between retries

scope_client = OpenAI(
    api_key=SCOPE_API_KEY,
    base_url=MODELSCOPE_BASE_URL,
)

# ── GitHub ─────────────────────────────────────────────────────────────────────
GITHUB_FILE_PATH = "designs.xlsx"
AUTO_PUSH_EVERY  = 20      # push to GitHub every N new rows

# ── Image settings ─────────────────────────────────────────────────────────────
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".psd", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
ARCHIVE_EXTS = {".zip", ".rar", ".7z"}
PREVIEW_MAX_PX       = 1280
PREVIEW_JPEG_QUALITY = 88
AI_IMAGE_PX          = 1280   # resize before sending to ModelScope

# ── Excel headers ──────────────────────────────────────────────────────────────
XLSX_HEADERS = [
    "ID", "Download URL", "Title", "Category", "Tags", "Description",
    "Dimensions", "DPI", "File Size", "Color Mode", "Software", "Fonts Used", "Preview URL"
]
COL_WIDTHS = [28, 70, 65, 20, 80, 110, 22, 10, 12, 14, 24, 32, 70]

# Excel style constants
HEADER_FILL = PatternFill("solid", fgColor="0D1117")
HEADER_FONT = Font(color="00D4FF", bold=True, name="Courier New", size=11)
ROW_FILL_A  = PatternFill("solid", fgColor="FFFFFF")
ROW_FILL_B  = PatternFill("solid", fgColor="EBF3FB")
URL_FONT    = Font(color="0563C1", underline="single")
BORDER      = Border(
    bottom=Side(style="thin", color="D0D0D0"),
    right=Side(style="thin",  color="D0D0D0"),
)


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# GOOGLE DRIVE AUTH  (refresh token — no browser needed in CI)
# ──────────────────────────────────────────────────────────────────────────────

def build_gdrive_service():
    """
    Build an authenticated Google Drive service using a stored refresh token.
    No browser popup — works perfectly in GitHub Actions.
    """
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = Credentials(
        token=None,
        refresh_token=GOOGLE_REFRESH_TOKEN,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    creds.refresh(Request())   # exchange refresh token → access token
    svc = build("drive", "v3", credentials=creds, cache_discovery=False)
    log("✅ Google Drive authenticated via refresh token")
    return svc


# ──────────────────────────────────────────────────────────────────────────────
# GOOGLE DRIVE — FILE LISTING
# ──────────────────────────────────────────────────────────────────────────────

def list_gdrive_folder_recursive(service, folder_id: str, subfolder: str = "") -> list:
    """
    Recursively list all image/archive files in GDrive folder.
    subfolder = immediate child-folder name of root  →  used as category.

    Folder structure expected:
        GDRIVE_PSD_FOLDER/
            Wedding/       ← subfolder (category)
                file.jpg
                file.zip
            DMK/           ← subfolder (category)
                banner.jpg
            ...

    Returns list of dicts: {id, name, subfolder, mimeType, size}
    """
    results    = []
    page_token = None

    while True:
        params = {
            "q":      f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken, files(id,name,mimeType,size)",
            "pageSize": 1000,
        }
        if page_token:
            params["pageToken"] = page_token

        resp  = service.files().list(**params).execute()
        items = resp.get("files", [])

        for item in items:
            mime = item.get("mimeType", "")
            name = item.get("name", "")

            if mime == "application/vnd.google-apps.folder":
                # category = immediate child folder name of root
                child_sf = name if not subfolder else subfolder
                results.extend(
                    list_gdrive_folder_recursive(service, item["id"], child_sf)
                )
            else:
                ext = os.path.splitext(name)[1].lower()
                if ext in IMAGE_EXTS or ext in ARCHIVE_EXTS:
                    results.append({
                        "id":        item["id"],
                        "name":      name,
                        "subfolder": subfolder,
                        "mimeType":  mime,
                        "size":      int(item.get("size", 0) or 0),
                    })

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return results


def get_gdrive_download_url(file_id: str) -> str:
    """Direct Google Drive download URL (public-readable)."""
    return (
        f"https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )


def download_gdrive_file_bytes(service, file_id: str) -> bytes:
    """Stream-download a GDrive file into memory."""
    from googleapiclient.http import MediaIoBaseDownload
    buf     = io.BytesIO()
    request = service.files().get_media(fileId=file_id)
    dl      = MediaIoBaseDownload(buf, request, chunksize=4 * 1024 * 1024)
    done    = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# GOOGLE DRIVE — PREVIEW JPG UPLOAD
# ──────────────────────────────────────────────────────────────────────────────

_jpg_subfolder_cache: dict = {}   # root_folder_id → jpg_subfolder_id


def _get_or_create_jpg_subfolder(service, root_folder_id: str) -> str:
    """Get or create the 'jpg' preview subfolder inside root GDrive folder."""
    if root_folder_id in _jpg_subfolder_cache:
        return _jpg_subfolder_cache[root_folder_id]

    q = (
        f"name='jpg' and mimeType='application/vnd.google-apps.folder' "
        f"and '{root_folder_id}' in parents and trashed=false"
    )
    r     = service.files().list(q=q, fields="files(id)", pageSize=1).execute()
    files = r.get("files", [])

    if files:
        fid = files[0]["id"]
        log(f"    📁 GDrive 'jpg' folder found: {fid}")
    else:
        meta = {
            "name":     "jpg",
            "mimeType": "application/vnd.google-apps.folder",
            "parents":  [root_folder_id],
        }
        fid = service.files().create(body=meta, fields="id").execute()["id"]
        log(f"    📁 GDrive 'jpg' folder created: {fid}")

    _jpg_subfolder_cache[root_folder_id] = fid
    return fid


def file_bytes_to_jpg(file_bytes: bytes, filename: str) -> bytes:
    """Convert any image bytes (JPG/PNG/PSD/TIFF/WebP…) → JPEG bytes."""
    try:
        ext = os.path.splitext(filename)[1].lower()

        # PSD → psd-tools composite
        if ext == ".psd":
            try:
                from psd_tools import PSDImage
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".psd", delete=False) as tf:
                    tf.write(file_bytes)
                    tf_path = tf.name
                try:
                    psd = PSDImage.open(tf_path)
                    img = psd.composite()
                    if img is None:
                        return None
                    img = img.convert("RGB")
                finally:
                    try:
                        os.unlink(tf_path)
                    except Exception:
                        pass
            except Exception as e:
                log(f"    ⚠ psd-tools failed: {e}")
                return None
        else:
            img = PILImage.open(io.BytesIO(file_bytes))
            # Multi-frame (GIF/TIFF) → take first frame
            try:
                img.seek(0)
            except Exception:
                pass
            # RGBA → RGB (white background)
            if img.mode == "RGBA":
                bg = PILImage.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

        # Resize if oversized
        if max(img.size) > PREVIEW_MAX_PX:
            img.thumbnail((PREVIEW_MAX_PX, PREVIEW_MAX_PX), PILImage.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=PREVIEW_JPEG_QUALITY,
                 subsampling=0, optimize=False)
        return buf.getvalue()

    except Exception as e:
        log(f"    ⚠ file_bytes_to_jpg error: {e}")
        return None


def upload_preview_jpg(service, jpg_bytes: bytes, name: str,
                       root_folder_id: str) -> str:
    """
    Upload JPEG preview to GDrive/jpg/ subfolder.
    Skip if already exists. Returns thumbnail URL or ''.
    """
    try:
        from googleapiclient.http import MediaIoBaseUpload

        jpg_folder_id = _get_or_create_jpg_subfolder(service, root_folder_id)
        if not jpg_folder_id:
            return ""

        jpg_filename = strip_ext(name) + ".jpg"
        safe_name    = jpg_filename.replace("'", "\\'")
        q = (
            f"name='{safe_name}' and '{jpg_folder_id}' in parents "
            f"and trashed=false"
        )
        existing = (
            service.files()
            .list(q=q, fields="files(id)", pageSize=1)
            .execute()
            .get("files", [])
        )
        if existing:
            fid = existing[0]["id"]
            log(f"    ⏭  Preview already exists: {jpg_filename}")
            return f"https://drive.google.com/thumbnail?id={fid}&sz=w1280"

        # Upload
        media = MediaIoBaseUpload(
            io.BytesIO(jpg_bytes), mimetype="image/jpeg", resumable=False
        )
        meta = {"name": jpg_filename, "parents": [jpg_folder_id]}

        for attempt in range(3):
            try:
                f   = service.files().create(body=meta, media_body=media, fields="id").execute()
                fid = f.get("id", "")
                # Make publicly readable (needed for Qwen-VL URL-based vision)
                service.permissions().create(
                    fileId=fid, body={"type": "anyone", "role": "reader"}
                ).execute()
                url = f"https://drive.google.com/thumbnail?id={fid}&sz=w1280"
                log(f"    ✅ Preview uploaded: {jpg_filename} → {fid}")
                return url
            except Exception as e:
                log(f"    ⚠ Preview upload attempt {attempt+1}/3 failed: {e}")
                time.sleep(4)
        return ""

    except Exception as e:
        log(f"    ❌ upload_preview_jpg error: {e}")
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# MODELSCOPE AI — VISION + SEO  (Qwen2.5-VL-72B-Instruct, single combined call)
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_VISION_SEO_PROMPT = """Tamil PSD marketplace SEO expert. Output DIRECTLY with ##TITLE## — no preamble.

CLASSIFY (silent): A=Person/cutout B=Text-only C=Single object | else=standard PSD
A: Title "[Name] PNG Cutout Download" | no printed name→use FOLDER name from 👤 OVERRIDE
B: Transliterate Tamil→English. Title "[Text] Title PNG Download"
C: Title "[Object] PNG Download"

RULES:
• Title: 8-12 words, end with "PSD Template Download" (PNG Download for types A/B/C)
• Tags: 15-20 comma-separated SEO keywords, Tamil + English mix
• Dimensions: detect from image or write "1800x1200 pixels"
• Description: 150-200 words. Sections: What It Is | Design Details | Customization | Who Uses It | File Info
• DPI: 300 | Software: Adobe Photoshop | Fonts: list visible fonts or "Multiple Fonts"
• ⛔ NEVER say "download for free" or include prices

OUTPUT FORMAT (exact — no extra text before ##TITLE##):
##TITLE##
[title here]
##TAGS##
[tags here]
##DIMENSIONS##
[WxH pixels]
##DESCRIPTION##
[description here]
"""


def call_modelscope_vision(jpg_bytes: bytes, name: str,
                            folder_hint: str = "") -> str:
    """
    Send JPEG image to ModelScope Qwen2.5-VL-72B-Instruct.
    Returns raw response text with ##SECTION## markers.
    Retries up to MAX_AI_RETRIES on transient errors.
    """
    # Build the prompt (folder context prepended if available)
    prompt = (folder_hint.strip() + "\n\n" + _DEFAULT_VISION_SEO_PROMPT
              if folder_hint else _DEFAULT_VISION_SEO_PROMPT)

    # Resize image for API submission (faster, cheaper)
    try:
        img = PILImage.open(io.BytesIO(jpg_bytes))
        if max(img.size) > AI_IMAGE_PX:
            img.thumbnail((AI_IMAGE_PX, AI_IMAGE_PX), PILImage.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        api_b64  = base64.b64encode(buf.getvalue()).decode()
        data_url = f"data:image/jpeg;base64,{api_b64}"
    except Exception:
        data_url = "data:image/jpeg;base64," + base64.b64encode(jpg_bytes).decode()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text",      "text": prompt},
            ],
        }
    ]

    for attempt in range(MAX_AI_RETRIES):
        try:
            # Small random jitter so calls don't look identical to rate-limiters
            time.sleep(random.uniform(0.5, 2.0))
            resp = scope_client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=AI_TEMPERATURE,
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            err = str(e).lower()
            log(f"    ⚠ ModelScope attempt {attempt+1}/{MAX_AI_RETRIES} error: {e}")
            if "rate" in err or "429" in err or "quota" in err:
                wait = AI_RETRY_DELAY * (attempt + 1)
                log(f"    ⏳ Rate-limit hit — waiting {wait}s before retry...")
                time.sleep(wait)
            elif attempt < MAX_AI_RETRIES - 1:
                time.sleep(AI_RETRY_DELAY)
            else:
                raise

    raise RuntimeError(f"ModelScope failed after {MAX_AI_RETRIES} attempts for: {name}")


# ──────────────────────────────────────────────────────────────────────────────
# PARSE AI RESPONSE
# ──────────────────────────────────────────────────────────────────────────────

def parse_response(raw: str) -> tuple:
    """Parse ##SECTION## delimited AI output → (title, tags, dims, desc)."""
    def between(text, start_tag, end_tag=None):
        s = text.find(start_tag)
        if s == -1:
            return ""
        s += len(start_tag)
        if end_tag:
            e = text.find(end_tag, s)
            return text[s:e].strip() if e != -1 else text[s:].strip()
        return text[s:].strip()

    title = between(raw, "##TITLE##",      "##TAGS##")
    tags  = between(raw, "##TAGS##",       "##DIMENSIONS##")
    dims  = between(raw, "##DIMENSIONS##", "##DESCRIPTION##")
    desc  = between(raw, "##DESCRIPTION##")

    # Fallback: if ##TITLE## missing, try first ##...## block
    if not title:
        m = re.match(r'^##([^#]+)##', raw.strip())
        if m:
            title = m.group(1).strip()

    for ch in ["[", "]"]:
        title = title.replace(ch, "").strip()
        tags  = tags.replace(ch, "").strip()
        dims  = dims.replace(ch, "").strip()

    # Normalise dimensions
    dims_m = re.search(r'[\d]+\s*[xX×]\s*[\d]+\s*pixels?', dims)
    if dims_m:
        dims = (dims_m.group(0)
                .replace(" ", "")
                .replace("×", "x")
                .replace("X", "x"))
        if not dims.endswith("pixels"):
            dims += " pixels"

    return title, tags, dims, desc


def truncate_description(desc: str, max_lines: int = 100) -> str:
    if not desc:
        return desc
    lines = desc.splitlines()
    return "\n".join(lines[:max_lines]) if len(lines) > max_lines else desc


# ──────────────────────────────────────────────────────────────────────────────
# INVITATION WORD SANITIZER
# ──────────────────────────────────────────────────────────────────────────────

def sanitize_invitation_words(title: str, tags: str, desc: str,
                               subfolder: str) -> tuple:
    """Strip 'invitation' words from title/tags/desc for non-invitation folders."""
    sf = (subfolder or "").lower().strip().replace("-", " ").replace("_", " ")
    inv_folder_words = ["invitation", "invitations", "invite", "invites",
                        "nimantrana", "amandhippu", "amandhipu"]
    if any(w in sf for w in inv_folder_words):
        return title, tags, desc   # legitimate invitation folder — keep as is

    inv_replacements = [
        (r'\bwedding invitation card\b',        'Wedding Banner'),
        (r'\bwedding invitation template\b',    'Wedding Banner Template'),
        (r'\bwedding invitation\b',             'Wedding Banner'),
        (r'\bbirthday invitation card\b',       'Birthday Banner'),
        (r'\bbirthday invitation template\b',   'Birthday Banner Template'),
        (r'\bbirthday invitation\b',            'Birthday Banner'),
        (r'\bhousewarming invitation card\b',   'House Warming Banner'),
        (r'\bhousewarming invitation\b',        'House Warming Banner'),
        (r'\bhouse warming invitation\b',       'House Warming Banner'),
        (r'\bpuberty invitation card\b',        'Puberty Ceremony Banner'),
        (r'\bpuberty invitation\b',             'Puberty Ceremony Banner'),
        (r'\bnaming ceremony invitation\b',     'Naming Ceremony Banner'),
        (r'\bbaby shower invitation\b',         'Baby Shower Banner'),
        (r'\binvitation card psd\b',            'Banner PSD'),
        (r'\binvitation card template\b',       'Banner Template'),
        (r'\binvitation card\b',                'Banner'),
        (r'\binvitation template\b',            'Banner Template'),
        (r'\binvitation psd\b',                 'Banner PSD'),
        (r'\binvitation design\b',              'Banner Design'),
        (r'\binvitations\b',                    'Banners'),
        (r'\binvitation\b',                     'Banner'),
        (r'\binvited\b',                        ''),
        (r'\binviting\b',                       ''),
        (r'\binvite\b',                         'Banner'),
        (r'\binvites\b',                        'Banners'),
    ]
    for pattern, replacement in inv_replacements:
        title = re.sub(pattern, replacement, title, flags=re.IGNORECASE).strip()
        tags  = re.sub(pattern, replacement, tags,  flags=re.IGNORECASE).strip()
        desc  = re.sub(pattern, replacement, desc,  flags=re.IGNORECASE).strip()

    # Deduplicate tags
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    seen, clean_tags = set(), []
    for tag in tag_list:
        if tag.lower() not in seen:
            seen.add(tag.lower())
            clean_tags.append(tag)
    tags = ", ".join(clean_tags)

    return title, tags, desc


# ──────────────────────────────────────────────────────────────────────────────
# FOLDER CONTEXT → AI HINTS
# ──────────────────────────────────────────────────────────────────────────────

KNOWN_PARTIES = {
    "dmk":      ("DMK", "DMK (Dravida Munnetra Kazhagam)", "rising sun symbol, red+black"),
    "admk":     ("ADMK", "ADMK (All India Anna Dravida Munnetra Kazhagam)", "two leaves symbol"),
    "aiadmk":   ("ADMK", "ADMK (All India Anna Dravida Munnetra Kazhagam)", "two leaves symbol"),
    "tvk":      ("TVK", "TVK (Tamilaga Vettri Kazhagam)", "blue color, Vijay's party"),
    "bjp":      ("BJP", "BJP (Bharatiya Janata Party)", "lotus symbol, saffron"),
    "congress": ("Congress", "INC (Indian National Congress)", "open hand symbol"),
    "vck":      ("VCK", "VCK (Viduthalai Chiruthaigal Katchi)", "fist symbol"),
    "pmk":      ("PMK", "PMK (Pattali Makkal Katchi)", "mango symbol"),
    "ntk":      ("NTK", "NTK (Nam Tamilar Katchi)", "tiger symbol"),
    "mmk":      ("MMK", "MMK (Manithaneya Makkal Katchi)", "yellow+red flag"),
}

FOLDER_TYPE_KEYWORDS = {
    "WEDDING":      ["wedding", "marriage", "kalyanam", "thirumanam", "kadhani", "nikah",
                     "bride", "groom", "muhurtham", "engagement"],
    "BIRTHDAY":     ["birthday", "bday", "pirantha", "birth day"],
    "FESTIVAL":     ["festival", "pongal", "diwali", "deepavali", "onam", "christmas",
                     "eid", "ramadan", "navratri", "ganesha", "murugan"],
    "DEATH":        ["death", "condolence", "memorial", "rip", "anjali", "ninaivil",
                     "varsha", "obituary"],
    "INVITATION":   ["invitation", "invite", "amandhippu", "amandhipu", "nimantrana"],
    "PUBERTY":      ["puberty", "manjal", "half saree", "langa voni"],
    "BABY":         ["naming ceremony", "namakaranam", "cradle", "thottil", "baby shower"],
    "HOUSE_WARM":   ["house warm", "grihapravesh", "housewarming", "new house"],
    "BUSINESS":     ["shop", "business", "store", "inauguration", "company", "restaurant",
                     "hotel", "clinic", "hospital", "pharmacy"],
    "VISITING_CARD":["visiting card", "business card", "name card", "visit card"],
    "PAMPHLET":     ["pamphlet", "brochure", "leaflet", "flyer", "trifold", "prospectus"],
    "PNG":          ["png", "cutout", "cut out", "transparent"],
    "POLITICAL":    ["political", "election", "vote", "mla", "campaign", "party banner"],
}

FOLDER_HINTS = {
    "WEDDING": lambda sf: f"""💍 FOLDER CONTEXT OVERRIDE — WEDDING FOLDER: "{sf}"
ALL images are wedding / kadhani BANNERS. Title MUST include Wedding/Kadhani.
⛔ ABSOLUTE RULE: NEVER use the word "invitation" in title/tags/description.
✅ CORRECT: "Tamil Hindu Wedding Flex Banner PSD Template Download"
❌ WRONG:   "Tamil Wedding Invitation PSD" — FORBIDDEN
Tags MUST include: wedding banner psd, thirumanam banner psd, wedding flex banner""",

    "BIRTHDAY": lambda sf: f"""🎂 FOLDER CONTEXT OVERRIDE — BIRTHDAY FOLDER: "{sf}"
ALL images are birthday BANNERS. Title MUST include "Birthday Banner".
⛔ ABSOLUTE RULE: NEVER use the word "invitation".
✅ "Tamil Kids Birthday Flex Banner PSD Template Download"
Tags MUST include: birthday banner psd, birthday flex banner, pirantha naal banner""",

    "FESTIVAL": lambda sf: f"""🎉 FOLDER CONTEXT OVERRIDE — FESTIVAL FOLDER: "{sf}"
ALL images are festival banners. Identify the specific festival from visual elements.
Include festival name in title and tags.""",

    "DEATH": lambda sf: f"""🕊️ FOLDER CONTEXT OVERRIDE — DEATH/MEMORIAL FOLDER: "{sf}"
ALL images are death announcement, condolence, or memorial banners.
Title examples: "Tamil Kanneer Anjali Death Banner PSD" | "Death Anniversary Memorial Banner PSD"
Tags MUST include: condolence banner psd, death announcement, ninaivu anjali, kanneer anjali
NEVER call this a festival, wedding, or birthday banner.""",

    "INVITATION": lambda sf: f"""💌 FOLDER CONTEXT OVERRIDE — INVITATION FOLDER: "{sf}"
ALL images are invitation cards/designs. Identify the occasion.
Title format: "[Occasion] Invitation Card PSD Template Download"
Tags MUST include: invitation card psd, invitation template, tamil invitation, editable invitation
NEVER call this a banner, flex, or poster.""",

    "PUBERTY": lambda sf: f"""🌸 FOLDER CONTEXT OVERRIDE — PUBERTY CEREMONY FOLDER: "{sf}"
ALL images are puberty / manjal neerattu / half-saree ceremony banners.
Title: "[Style] Puberty Ceremony Banner PSD Template Download"
Tags: puberty ceremony banner, manjal neerattu psd, half saree banner, tamil puberty""",

    "BABY": lambda sf: f"""👶 FOLDER CONTEXT OVERRIDE — BABY CEREMONY FOLDER: "{sf}"
ALL images are naming ceremony / cradle ceremony / baby shower designs.
Identify specific ceremony from visual. End title with "Ceremony Banner PSD Download".""",

    "HOUSE_WARM": lambda sf: f"""🏠 FOLDER CONTEXT OVERRIDE — HOUSE WARMING FOLDER: "{sf}"
ALL images are house warming banners. Title must include "House Warming Banner".
Tags: house warming banner psd, grihapravesh banner, new house banner, kudil thirappu""",

    "BUSINESS": lambda sf: f"""🏪 FOLDER CONTEXT OVERRIDE — BUSINESS/SHOP FOLDER: "{sf}"
ALL images are shop/business banners or commercial designs.
Identify the specific business type from the image.
Title: "[Business Type] Shop Banner PSD Template Download" """,

    "VISITING_CARD": lambda sf: f"""💼 FOLDER CONTEXT OVERRIDE — VISITING CARD FOLDER: "{sf}"
ALL images are visiting cards / business cards.
Title format: "[Profession] Visiting Card PSD Template Download"
Tags: visiting card psd, business card template, tamil visiting card""",

    "PAMPHLET": lambda sf: f"""📄 FOLDER CONTEXT OVERRIDE — PAMPHLET/BROCHURE FOLDER: "{sf}"
ALL images are pamphlets, brochures, or leaflets.
Title format: "[Business/Topic] Pamphlet PSD Template Download"
Tags: pamphlet psd, brochure template, leaflet design, tamil pamphlet""",

    "PNG": lambda sf: f"""🖼️ FOLDER CONTEXT OVERRIDE — PNG/CUTOUT FOLDER: "{sf}"
ALL images are transparent PNG cutouts or elements.
Title: "[Subject] PNG Cutout Free Download" or "[Subject] Elements PNG Download"
Tags: png cutout, transparent background, hd png, free png download, tamil png""",

    "POLITICAL": lambda sf: f"""🏛️ FOLDER CONTEXT OVERRIDE — POLITICAL FOLDER: "{sf}"
ALL images are political banners. Identify party and leader from visuals.
Include party name and leader name in title and tags.""",
}


def get_folder_context(subfolder: str) -> dict:
    """
    Return AI hint text and category hint based on subfolder name.
    Mirrors the original GUI script's get_folder_context() logic.
    """
    if not subfolder:
        return {"ai_hint": "", "category_hint": ""}

    sf_lower = subfolder.lower().strip().replace("-", " ").replace("_", " ")

    # ── 1. Known political parties ───────────────────────────────────────────
    for key, (short, full_name, desc) in KNOWN_PARTIES.items():
        if key in sf_lower:
            hint = (
                f"🏛️ FOLDER CONTEXT OVERRIDE — POLITICAL PARTY FOLDER: \"{subfolder}\"\n"
                f"This is a {full_name} political banner folder.\n"
                f"Party visual: {desc}\n"
                f"ALL images are {full_name} political designs.\n"
                f"Title MUST include the party name '{short}'.\n"
                f"Tags MUST include: {key} banner psd, political banner, {key} flex banner"
            )
            return {"ai_hint": hint, "category_hint": short}

    # ── 2. Keyword-based folder type ─────────────────────────────────────────
    for ftype, keywords in FOLDER_TYPE_KEYWORDS.items():
        if any(kw in sf_lower for kw in keywords):
            hint_fn = FOLDER_HINTS.get(ftype)
            hint    = hint_fn(subfolder) if hint_fn else (
                f'📁 FOLDER CONTEXT: "{subfolder}" — use this as context for title/tags.'
            )
            return {"ai_hint": hint, "category_hint": ftype}

    # ── 3. Unknown Title-Case folder → treat as person name ─────────────────
    if subfolder and subfolder[0].isupper() and len(sf_lower.split()) <= 4:
        hint = (
            f"👤 FOLDER CONTEXT OVERRIDE — PERSON FOLDER: \"{subfolder}\"\n"
            f"The folder is named \"{subfolder}\" — a dedicated folder for a specific person "
            f"(politician, actor, leader, public figure).\n"
            f"MANDATORY RULES:\n"
            f"  • Title MUST include the person's name: \"{subfolder}\"\n"
            f"  • If no printed name is visible in image → STILL use \"{subfolder}\" as the name\n"
            f"  • NEVER write \"Unknown Person\"\n"
            f"  • Title: \"{subfolder} PNG Cutout Download\" or \"{subfolder} HD PNG Download\"\n"
            f"  • Tags MUST include: {subfolder.lower()} png, {subfolder.lower()} cutout, "
            f"transparent background, hd png"
        )
        return {"ai_hint": hint, "category_hint": "Person PNG"}

    return {
        "ai_hint": f'📁 FOLDER CONTEXT: "{subfolder}" — use this as context for title/tags.',
        "category_hint": "",
    }


# ──────────────────────────────────────────────────────────────────────────────
# EXCEL / XLSX
# ──────────────────────────────────────────────────────────────────────────────

def df_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame → styled XLSX bytes (in memory)."""
    wb = Workbook()
    ws = wb.active
    ws.title = "PSD Data"

    # Header row
    for ci, (hdr, w) in enumerate(zip(XLSX_HEADERS, COL_WIDTHS), 1):
        cell = ws.cell(row=1, column=ci, value=hdr)
        cell.fill      = HEADER_FILL
        cell.font      = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.column_dimensions[get_column_letter(ci)].width = w
    ws.row_dimensions[1].height = 24

    # Data rows
    for ri, (_, row_data) in enumerate(df.iterrows(), 2):
        fill = ROW_FILL_B if ri % 2 == 0 else ROW_FILL_A
        for ci, col_name in enumerate(XLSX_HEADERS, 1):
            val  = row_data.get(col_name, "")
            val  = "" if pd.isna(val) else str(val)
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.fill      = fill
            cell.border    = BORDER
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if ci in (2, 13) and val and val.startswith("http"):
                cell.font = URL_FONT

    ws.freeze_panes       = "A2"
    ws.auto_filter.ref    = f"A1:{get_column_letter(len(XLSX_HEADERS))}1"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# GITHUB XLSX INTEGRATION
# ──────────────────────────────────────────────────────────────────────────────

def fetch_github_xlsx() -> tuple:
    """Download designs.xlsx from GitHub. Returns (DataFrame, sha)."""
    gh_headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    api_url = (
        f"https://api.github.com/repos/{GITHUB_REPO}"
        f"/contents/{GITHUB_FILE_PATH}"
    )
    log(f"📥 Fetching {GITHUB_FILE_PATH} from GitHub: {GITHUB_REPO}")

    try:
        resp = req_lib.get(api_url, headers=gh_headers, timeout=30)
        if resp.status_code == 200:
            data      = resp.json()
            sha       = data.get("sha")
            raw_bytes = base64.b64decode(
                data["content"].replace("\n", "")
            )
            df = pd.read_excel(
                io.BytesIO(raw_bytes), engine="openpyxl", dtype=str
            )
            df = df.fillna("").astype(str).replace("nan", "")
            for col in XLSX_HEADERS:
                if col not in df.columns:
                    df[col] = ""
            df = df[XLSX_HEADERS]
            log(f"✅ Downloaded: {len(df)} rows  SHA={str(sha)[:10]}...")
            return df, sha

        elif resp.status_code == 404:
            log("ℹ️  designs.xlsx not found in repo — will create fresh")
            return pd.DataFrame(columns=XLSX_HEADERS), None
        else:
            log(f"⚠ GitHub API HTTP {resp.status_code}: {resp.text[:200]}")
            return pd.DataFrame(columns=XLSX_HEADERS), None

    except Exception as e:
        log(f"⚠ fetch_github_xlsx error: {e}")
        return pd.DataFrame(columns=XLSX_HEADERS), None


def push_to_github(xlsx_bytes: bytes, sha, commit_msg: str) -> tuple:
    """Push designs.xlsx to GitHub. Returns (success: bool, new_sha: str)."""
    url = (
        f"https://api.github.com/repos/{GITHUB_REPO}"
        f"/contents/{GITHUB_FILE_PATH}"
    )
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept":        "application/vnd.github.v3+json",
        "Content-Type":  "application/json",
    }
    payload = {
        "message": commit_msg,
        "content": base64.b64encode(xlsx_bytes).decode("utf-8"),
    }
    if sha:
        payload["sha"] = sha

    try:
        resp = req_lib.put(
            url, headers=headers,
            data=json.dumps(payload), timeout=60
        )
        if resp.status_code in (200, 201):
            new_sha = resp.json().get("content", {}).get("sha", sha)
            log(f"✅ GitHub push SUCCESS! SHA: {str(new_sha)[:10]}...")
            return True, new_sha
        else:
            log(f"❌ GitHub push failed: HTTP {resp.status_code}")
            log(f"   Body: {resp.text[:300]}")
            return False, sha

    except Exception as e:
        log(f"❌ push_to_github error: {e}")
        return False, sha


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def strip_ext(filename: str) -> str:
    return os.path.splitext(filename)[0]


def clean_key(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"[\s_]+copy[\s_]*\d*$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"[()]", " ", s)
    s = s.lower().strip()
    s = re.sub(r"[\s_-]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1_048_576:
        return f"{size_bytes/1024:.1f} KB"
    else:
        return f"{size_bytes/1_048_576:.2f} MB"


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("PSD All-In-One Tool — GitHub Actions Edition")
    log(f"Model : {VISION_MODEL}")
    log(f"Repo  : {GITHUB_REPO}")
    log("=" * 70)

    # ── Validate required env vars ────────────────────────────────────────────
    required = {
        "GOOGLE_CLIENT_ID":     GOOGLE_CLIENT_ID,
        "GOOGLE_CLIENT_SECRET": GOOGLE_CLIENT_SECRET,
        "GOOGLE_REFRESH_TOKEN": GOOGLE_REFRESH_TOKEN,
        "GDRIVE_PSD_FOLDER":    GDRIVE_PSD_FOLDER,
        "GITHUB_TOKEN":         GITHUB_TOKEN,
        "GITHUB_REPO":          GITHUB_REPO,
        "SCOPE":                SCOPE_API_KEY,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        log(f"❌ Missing required env vars: {', '.join(missing)}")
        sys.exit(1)

    # ── Step 1: Connect to Google Drive ──────────────────────────────────────
    log("\n[Step 1] Connecting to Google Drive...")
    service = build_gdrive_service()

    # ── Step 2: Fetch designs.xlsx from GitHub ────────────────────────────────
    log("\n[Step 2] Fetching designs.xlsx from GitHub...")
    df, current_sha = fetch_github_xlsx()

    # Build a set of normalised existing IDs (name without extension, lower-case)
    existing_ids: set = set()
    if len(df) > 0:
        for raw_id in df["ID"].tolist():
            s = str(raw_id).strip().lower()
            if s and s not in ("nan", "none", ""):
                existing_ids.add(s)
                existing_ids.add(clean_key(s))
    log(f"    Existing rows: {len(df)}  |  Unique IDs cached: {len(existing_ids)}")

    # ── Step 3: List files from Google Drive ─────────────────────────────────
    log(f"\n[Step 3] Listing files from GDrive folder: {GDRIVE_PSD_FOLDER}")
    all_files = list_gdrive_folder_recursive(service, GDRIVE_PSD_FOLDER)
    log(f"    Total GDrive items found: {len(all_files)}")

    # Separate images from archives
    image_files = [
        f for f in all_files
        if os.path.splitext(f["name"])[1].lower() in IMAGE_EXTS
    ]
    archive_map: dict = {}   # name_no_ext (lower) → file_info
    for f in all_files:
        ext = os.path.splitext(f["name"])[1].lower()
        if ext in ARCHIVE_EXTS:
            key = strip_ext(f["name"]).lower().strip()
            archive_map[key]            = f
            archive_map[clean_key(key)] = f

    log(f"    Image files : {len(image_files)}")
    log(f"    Archive files: {len(archive_map) // 2}")

    # Apply MAX_FILES limit (useful for testing or rate-limit budgets)
    if MAX_FILES > 0:
        image_files = image_files[:MAX_FILES]
        log(f"    ⚠ MAX_FILES={MAX_FILES} — processing first {len(image_files)} images only")

    # ── Step 4: Process new files ─────────────────────────────────────────────
    log("\n[Step 4] Processing new files...")
    new_rows:  list = []
    processed: int  = 0
    skipped:   int  = 0
    errors:    int  = 0

    for idx, file_info in enumerate(image_files):
        file_id   = file_info["id"]
        filename  = file_info["name"]
        subfolder = file_info["subfolder"]
        file_size = file_info["size"]

        name_no_ext = strip_ext(filename)
        key_lower   = name_no_ext.lower().strip()
        key_clean   = clean_key(name_no_ext)

        # ── Smart skip: already processed ────────────────────────────────────
        if key_lower in existing_ids or key_clean in existing_ids:
            skipped += 1
            continue

        log(
            f"\n[{idx+1}/{len(image_files)}] {filename}"
            f"  category={subfolder or 'root'}"
            f"  size={human_size(file_size)}"
        )

        try:
            # ── Download image bytes from GDrive ─────────────────────────────
            log(f"    ⬇️  Downloading from GDrive...")
            file_bytes = download_gdrive_file_bytes(service, file_id)
            log(f"    ✅ Downloaded {human_size(len(file_bytes))}")

            # ── Convert to JPEG for preview upload + AI ───────────────────────
            jpg_bytes   = file_bytes_to_jpg(file_bytes, filename)
            preview_url = ""

            if jpg_bytes:
                preview_url = upload_preview_jpg(
                    service, jpg_bytes, name_no_ext, GDRIVE_PSD_FOLDER
                )
            else:
                log(f"    ⚠ Could not convert to JPG (unsupported format or PSD error)")

            # ── Download URL — prefer matching archive, else use image file ───
            archive_info = archive_map.get(key_lower) or archive_map.get(key_clean)
            if archive_info:
                download_url = get_gdrive_download_url(archive_info["id"])
                log(f"    📦 Download URL → matching archive: {archive_info['name']}")
            else:
                download_url = get_gdrive_download_url(file_id)
                log(f"    🖼️  Download URL → image file: {filename}")

            # ── Get folder context (AI hint) ──────────────────────────────────
            folder_ctx  = get_folder_context(subfolder)
            folder_hint = folder_ctx["ai_hint"]

            # ── ModelScope Qwen-VL vision + SEO ──────────────────────────────
            if jpg_bytes:
                log(f"    🤖 ModelScope {VISION_MODEL} → vision + SEO...")
                raw_ai = call_modelscope_vision(jpg_bytes, name_no_ext, folder_hint)
                title, tags, dims, desc = parse_response(raw_ai)

                # Post-process: strip invitation words for non-invitation folders
                title, tags, desc = sanitize_invitation_words(
                    title, tags, desc, subfolder
                )
                desc = truncate_description(desc)

                if not dims or "x" not in dims.lower():
                    dims = "1800x1200 pixels"

                log(f"    ✅ AI done: {title[:65]}")
            else:
                # Fallback: filename-based title (no vision available)
                log(f"    ⚠ No JPG available — using filename fallback")
                clean_name = (
                    name_no_ext.replace("_", " ").replace("-", " ").title()
                )
                title = f"{clean_name} PSD Template Download"
                tags  = f"psd template, tamil psd, editable psd, {subfolder or 'design'}"
                dims  = "1800x1200 pixels"
                desc  = (
                    f"Tamil PSD design template from {subfolder or 'General'} category. "
                    f"Fully editable in Adobe Photoshop. High resolution 300 DPI. "
                    f"Customize colors, text, and images easily. Perfect for print and digital use."
                )

            # Category = subfolder name; root-level files → "Others"
            category = subfolder if subfolder else "Others"

            # Color mode: PSD files are typically CMYK
            color_mode = "CMYK" if filename.lower().endswith(".psd") else "RGB"

            # ── Build row dict ────────────────────────────────────────────────
            row = {
                "ID":           name_no_ext,
                "Download URL": download_url,
                "Title":        title,
                "Category":     category,
                "Tags":         tags,
                "Description":  desc[:3000] if desc else "",
                "Dimensions":   dims,
                "DPI":          "300 DPI",
                "File Size":    human_size(file_size),
                "Color Mode":   color_mode,
                "Software":     "Adobe Photoshop CC",
                "Fonts Used":   "Multiple Fonts",
                "Preview URL":  preview_url,
            }
            new_rows.append(row)

            # Mark as processed so we don't re-add within the same run
            existing_ids.add(key_lower)
            existing_ids.add(key_clean)
            processed += 1

            log(f"    ✅ Row queued [{processed}]: {name_no_ext}")

            # ── Auto-push checkpoint every AUTO_PUSH_EVERY rows ───────────────
            if processed % AUTO_PUSH_EVERY == 0 and new_rows:
                log(f"\n🚀 AUTO-PUSH checkpoint — {processed} rows processed so far...")
                snapshot_df = pd.concat(
                    [df, pd.DataFrame(new_rows)], ignore_index=True
                )
                xlsx_bytes = df_to_xlsx_bytes(snapshot_df)
                ok, current_sha = push_to_github(
                    xlsx_bytes, current_sha,
                    f"Auto-push: {processed} new rows "
                    f"[{time.strftime('%Y-%m-%d %H:%M')} UTC]"
                )
                if ok:
                    # Absorb queued rows into df so next checkpoint delta is clean
                    df       = snapshot_df
                    new_rows = []

            # Delay between AI calls (respect ModelScope rate limits)
            if jpg_bytes:
                time.sleep(CALL_DELAY)

        except KeyboardInterrupt:
            log("\n⛔ Interrupted by user!")
            break

        except Exception as e:
            log(f"    ❌ Error processing {filename}: {e}")
            log(traceback.format_exc())
            errors += 1
            # Don't abort — skip this file and continue
            continue

    # ── Step 5: Final push ────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log(f"[Step 5] Run Summary:")
    log(f"    ✅ Newly processed : {processed}")
    log(f"    ⏭  Skipped (exist) : {skipped}")
    log(f"    ❌ Errors          : {errors}")

    if processed > 0 or new_rows:
        final_df = (
            pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            if new_rows else df
        )
        log(f"\n🚀 Final push → {len(final_df)} total rows in designs.xlsx")
        xlsx_bytes = df_to_xlsx_bytes(final_df)
        ok, _ = push_to_github(
            xlsx_bytes, current_sha,
            f"Update designs.xlsx: +{processed} new rows "
            f"[{time.strftime('%Y-%m-%d %H:%M')} UTC]"
        )
        if not ok:
            log("❌ Final push FAILED!")
            sys.exit(1)
    else:
        log("ℹ️  No new files to process — designs.xlsx unchanged")

    log(f"\n{'='*70}")
    log("✅ Pipeline complete!")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
