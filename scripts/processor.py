"""
processor.py – Updated pipeline (no layer deletion):

  1. Extract ZIP / RAR archive.
  2. Find all PSD and TIF/TIFF files inside (ignore .txt and others).
  3. For each file:
       a. Composite image from original PSD/TIF (NO layer deletion).
       b. Add professional diagonal watermark (low opacity).
       c. Save as WebP — auto-tune quality to stay UNDER 100 KB.
       d. Rename both PSD and WebP to tamilpsd-XXXX (same base name).
  4. Return (original_files, webp_files, next_index).
     - original PSD/TIF → GDRIVE_PSD_FOLDER  (psd/)
     - webp files       → GDRIVE_WEBP_FOLDER (preview/)

Returns: (original_paths: list[Path], webp_paths: list[Path], next_index: int)
"""

import io
import logging
import math
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
from psd_tools import PSDImage

from config import WATERMARK_TEXT, WATERMARK_OPACITY, WATERMARK_PNG, WEBP_MAX_KB

log = logging.getLogger(__name__)

WEBP_MAX_BYTES = WEBP_MAX_KB * 1024   # default 100 KB → 102 400 bytes


# ═══════════════════════════════════════════════════════════════════════════
#  Archive extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """Extract .zip, .rar, .7z to dest_dir. Returns True on success."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = archive_path.suffix.lower()

    try:
        if ext == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_dir)
            return True

        if ext == ".rar":
            result = subprocess.run(
                ["unrar", "x", "-y", str(archive_path), str(dest_dir) + "/"],
                capture_output=True,
            )
            if result.returncode == 0:
                return True
            import patoollib
            patoollib.extract_archive(str(archive_path), outdir=str(dest_dir))
            return True

        if ext in (".7z", ".7zip"):
            subprocess.run(
                ["7z", "x", str(archive_path), f"-o{dest_dir}", "-y"],
                check=True, capture_output=True,
            )
            return True

        import patoollib
        patoollib.extract_archive(str(archive_path), outdir=str(dest_dir))
        return True

    except Exception as exc:
        log.error(f"Extraction failed for {archive_path}: {exc}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  Composite helpers  (original file — NO layer deletion)
# ═══════════════════════════════════════════════════════════════════════════

def composite_psd(psd_path: Path) -> Optional[Image.Image]:
    """Composite all layers of a PSD into a single PIL image."""
    try:
        psd = PSDImage.open(str(psd_path))
        img = psd.composite()
        if img is not None:
            return img.convert("RGB")
        merged = psd.topil()
        if merged:
            return merged.convert("RGB")
    except Exception as exc:
        log.error(f"composite_psd failed for {psd_path.name}: {exc}")
    return None


def composite_tif(tif_path: Path) -> Optional[Image.Image]:
    """Load first page of TIF/TIFF as preview image."""
    try:
        img = Image.open(str(tif_path))
        return img.convert("RGB")
    except Exception as exc:
        log.error(f"composite_tif failed for {tif_path.name}: {exc}")
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Watermark  — Professional diagonal text (3 spaced rows)
# ═══════════════════════════════════════════════════════════════════════════

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _apply_png_watermark(img: Image.Image, wm_png_path: Path, opacity: int) -> Image.Image:
    """
    Apply a PNG logo watermark — 3 diagonal instances (top-left, center, bottom-right).
    opacity: 0–255.
    """
    img = img.convert("RGBA")
    w, h = img.size

    try:
        wm = Image.open(str(wm_png_path)).convert("RGBA")

        # Scale watermark to ~22% of image width
        wm_w = max(100, w // 5)
        wm_h = int(wm.height * (wm_w / wm.width))
        wm   = wm.resize((wm_w, wm_h), Image.LANCZOS)

        # Apply opacity to alpha channel
        r, g, b, a = wm.split()
        a = a.point(lambda p: int(p * opacity / 255))
        wm.putalpha(a)

        # Rotate the watermark 30 degrees
        wm_rotated = wm.rotate(30, expand=True)

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        # Place 3 instances: top-left area, center, bottom-right area
        positions = [
            (w // 8 - wm_rotated.width // 2,       h // 5 - wm_rotated.height // 2),
            (w // 2 - wm_rotated.width // 2,       h // 2 - wm_rotated.height // 2),
            (w * 7 // 8 - wm_rotated.width // 2,   h * 4 // 5 - wm_rotated.height // 2),
        ]
        for px, py in positions:
            overlay.paste(wm_rotated, (px, py), wm_rotated)

        return Image.alpha_composite(img, overlay).convert("RGB")

    except Exception as exc:
        log.warning(f"PNG watermark failed ({exc}) — using text watermark")
        return _apply_text_watermark(img.convert("RGB"), WATERMARK_TEXT, opacity)


def _apply_text_watermark(img: Image.Image, text: str, opacity: int) -> Image.Image:
    """
    Professional diagonal watermark — 3 evenly-spaced slanted text instances.
    Clean, visible but not overwhelming.
    """
    img  = img.convert("RGBA")
    w, h = img.size

    # Font size relative to image width (decent readability)
    font_size = max(36, w // 18)
    font      = _load_font(font_size)

    # ── Build a rotated text stamp ─────────────────────────────────────────
    # Draw text on a large transparent canvas, rotate 30°, then crop
    ANGLE = 30

    # Measure text
    tmp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = tmp_draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Make stamp canvas (a bit larger than text for clean rotation)
    pad = 20
    stamp_w = tw + pad * 2
    stamp_h = th + pad * 2
    stamp = Image.new("RGBA", (stamp_w, stamp_h), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(stamp)

    # Subtle shadow for depth (professional look)
    draw.text((pad + 2, pad + 2), text, font=font, fill=(0, 0, 0, opacity // 3))
    # Main text
    draw.text((pad, pad), text, font=font, fill=(255, 255, 255, opacity))

    # Rotate the stamp
    rotated = stamp.rotate(ANGLE, expand=True)
    rw, rh  = rotated.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    # 3 positions along the diagonal: top-left zone, center, bottom-right zone
    positions = [
        (w // 7 - rw // 2,       h // 6 - rh // 2),
        (w // 2 - rw // 2,       h // 2 - rh // 2),
        (w * 6 // 7 - rw // 2,   h * 5 // 6 - rh // 2),
    ]

    for px, py in positions:
        overlay.paste(rotated, (px, py), rotated)

    return Image.alpha_composite(img, overlay).convert("RGB")


def add_watermark(img: Image.Image) -> Image.Image:
    """
    Add professional diagonal watermark.
    • If WATERMARK_PNG is set and file exists → PNG logo overlay.
    • Otherwise → diagonal text (3 instances) fallback.
    """
    wm_path = Path(WATERMARK_PNG) if WATERMARK_PNG else None
    if wm_path and wm_path.exists():
        log.info(f"Applying PNG watermark: {wm_path.name}")
        return _apply_png_watermark(img, wm_path, WATERMARK_OPACITY)
    return _apply_text_watermark(img, WATERMARK_TEXT, WATERMARK_OPACITY)


# ═══════════════════════════════════════════════════════════════════════════
#  WebP — auto quality to stay under 100 KB
# ═══════════════════════════════════════════════════════════════════════════

def save_webp_under_limit(img: Image.Image, output_path: Path) -> bool:
    """
    Save WebP at the highest quality that fits within WEBP_MAX_BYTES.
    Steps: 85 → 80 → 75 … → 5.
    Last resort: resize image to 50% and retry.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for quality in range(85, 0, -5):
        buf = io.BytesIO()
        img.save(buf, "WEBP", quality=quality, method=6)
        if buf.tell() <= WEBP_MAX_BYTES:
            output_path.write_bytes(buf.getvalue())
            log.info(
                f"WebP saved: {output_path.name}  "
                f"quality={quality}  size={buf.tell()/1024:.1f} KB"
            )
            return True

    # Last resort — halve the image dimensions
    log.warning(f"Resizing image to fit under {WEBP_MAX_KB} KB: {output_path.name}")
    small = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    for quality in range(85, 0, -5):
        buf = io.BytesIO()
        small.save(buf, "WEBP", quality=quality, method=6)
        if buf.tell() <= WEBP_MAX_BYTES:
            output_path.write_bytes(buf.getvalue())
            log.info(
                f"WebP saved (half-size): {output_path.name}  "
                f"quality={quality}  size={buf.tell()/1024:.1f} KB"
            )
            return True

    log.error(f"Cannot compress {output_path.name} under {WEBP_MAX_KB} KB")
    return False


def create_webp(source_path: Path, output_path: Path) -> bool:
    """Composite source PSD/TIF, apply watermark, save as WebP under limit."""
    ext = source_path.suffix.lower()
    if ext == ".psd":
        img = composite_psd(source_path)
    elif ext in (".tif", ".tiff"):
        img = composite_tif(source_path)
    else:
        log.error(f"Unsupported format: {source_path.suffix}")
        return False

    if img is None:
        log.error(f"Could not composite image from {source_path.name}")
        return False

    return save_webp_under_limit(add_watermark(img), output_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Naming helper — tamilpsd-XXXX
# ═══════════════════════════════════════════════════════════════════════════

def make_tamilpsd_name(index: int) -> str:
    """Return base name like  tamilpsd-0001  for a given 1-based index."""
    return f"tamilpsd-{index:04d}"


# ═══════════════════════════════════════════════════════════════════════════
#  Main pipeline  (no layer deletion, no re-ZIP)
# ═══════════════════════════════════════════════════════════════════════════

def process_archive(
    archive_path: Path,
    work_dir: Path,
    start_index: int = 1,
) -> Optional[tuple[list[Path], list[Path], int]]:
    """
    Full pipeline for one archive.

    Parameters
    ----------
    archive_path : Path
        The .zip/.rar/.7z to process.
    work_dir : Path
        Temporary working directory for this item.
    start_index : int
        Counter used for the first file in this archive.
        Increments for each file found.

    Returns
    -------
    (renamed_psd_files, webp_files, next_index) or None on failure.

    - renamed_psd_files : PSD/TIF files copied & renamed as tamilpsd-XXXX.psd/tif
    - webp_files        : WebP previews named tamilpsd-XXXX.webp
    - next_index        : start_index + number of files successfully processed
                          (caller should persist this in state)

    Callers should:
      upload renamed_psd_files → GDRIVE_PSD_FOLDER   (psd folder)
      upload webp_files        → GDRIVE_WEBP_FOLDER  (preview folder)
    """
    archive_path = Path(archive_path)
    work_dir     = Path(work_dir)

    extract_dir  = work_dir / "extracted"
    webp_dir     = work_dir / "webp"
    renamed_dir  = work_dir / "renamed_psd"
    extract_dir.mkdir(parents=True, exist_ok=True)
    webp_dir.mkdir(parents=True, exist_ok=True)
    renamed_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Extract ───────────────────────────────────────────────────
    log.info(f"Extracting {archive_path.name} …")
    if not extract_archive(archive_path, extract_dir):
        return None

    # ── Step 2: Find PSD / TIF files (skip .txt and others) ───────────────
    source_files: list[Path] = []
    for pattern in ("*.psd", "*.PSD", "*.tif", "*.TIF", "*.tiff", "*.TIFF"):
        source_files.extend(extract_dir.rglob(pattern))

    if not source_files:
        log.warning(f"No PSD/TIF files found in {archive_path.name}")
        return None

    log.info(f"Found {len(source_files)} PSD/TIF file(s). Starting index: {start_index}")

    renamed_files: list[Path] = []
    webp_paths:    list[Path] = []
    current_index = start_index

    for src in source_files:
        base_name = make_tamilpsd_name(current_index)
        new_ext   = src.suffix.lower()   # .psd or .tif/.tiff

        # ── Rename (copy) original PSD/TIF ────────────────────────────────
        renamed_src = renamed_dir / f"{base_name}{new_ext}"
        try:
            shutil.copy2(src, renamed_src)
            renamed_files.append(renamed_src)
            log.info(f"  Renamed: {src.name}  →  {renamed_src.name}")
        except Exception as exc:
            log.warning(f"  Could not rename {src.name}: {exc} — using original")
            renamed_files.append(src)

        # ── Step 3: Create WebP with watermark ────────────────────────────
        webp_path = webp_dir / f"{base_name}.webp"
        if create_webp(src, webp_path):
            webp_paths.append(webp_path)
            log.info(f"  WebP created: {webp_path.name}")
        else:
            log.warning(f"  WebP creation failed for {src.name}")

        current_index += 1

    return renamed_files, webp_paths, current_index
