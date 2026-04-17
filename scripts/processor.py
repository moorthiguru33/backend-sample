"""
processor.py – Updated pipeline (no layer deletion):

  1. Extract ZIP / RAR archive.
  2. Find all PSD and TIF/TIFF files inside (ignore .txt and others).
  3. For each file:
       a. Composite image from original PSD/TIF (NO layer deletion).
       b. Add PNG tree-style watermark (low opacity, protective).
       c. Save as WebP — auto-tune quality to stay UNDER 100 KB.
  4. Return (original_files, webp_files).
     - original PSD/TIF → GDRIVE_PSD_FOLDER  (psd/)
     - webp files       → GDRIVE_WEBP_FOLDER (preview/)

Returns: (original_paths: list[Path], webp_paths: list[Path])
"""

import io
import logging
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
#  Watermark  (PNG tree-style OR text tile fallback)
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
    Tile a PNG watermark (tree logo) diagonally across the image.
    opacity: 0–255.
    """
    img = img.convert("RGBA")
    w, h = img.size

    try:
        wm = Image.open(str(wm_png_path)).convert("RGBA")

        # Scale watermark to ~18% of image width
        wm_w = max(80, w // 6)
        wm_h = int(wm.height * (wm_w / wm.width))
        wm   = wm.resize((wm_w, wm_h), Image.LANCZOS)

        # Apply opacity to alpha channel
        r, g, b, a = wm.split()
        a = a.point(lambda p: int(p * opacity / 255))
        wm.putalpha(a)

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        step_x  = wm_w + 60
        step_y  = wm_h + 60

        for row, y in enumerate(range(-h, h * 2, step_y)):
            x_off = (row % 2) * (step_x // 2)
            for x in range(-w + x_off, w * 2, step_x):
                overlay.paste(wm, (x, y), wm)

        return Image.alpha_composite(img, overlay).convert("RGB")

    except Exception as exc:
        log.warning(f"PNG watermark failed ({exc}) — using text watermark")
        return _apply_text_watermark(img.convert("RGB"), WATERMARK_TEXT, opacity)


def _apply_text_watermark(img: Image.Image, text: str, opacity: int) -> Image.Image:
    """Tile watermark text diagonally — protective but low opacity."""
    img  = img.convert("RGBA")
    w, h = img.size

    font_size = max(28, w // 20)
    font      = _load_font(font_size)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]

    step_x = tw + 70
    step_y = th + 55

    for row, y in enumerate(range(-h, h * 2, step_y)):
        x_off = (row % 2) * (step_x // 2)
        for x in range(-w + x_off, w * 2, step_x):
            draw.text((x, y), text, font=font, fill=(255, 255, 255, opacity))

    return Image.alpha_composite(img, overlay).convert("RGB")


def add_watermark(img: Image.Image) -> Image.Image:
    """
    Add protective watermark.
    • If WATERMARK_PNG is set and the file exists → PNG tree-style overlay.
    • Otherwise → diagonal text tile fallback.
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
#  Main pipeline  (no layer deletion, no re-ZIP)
# ═══════════════════════════════════════════════════════════════════════════

def process_archive(
    archive_path: Path,
    work_dir: Path,
) -> Optional[tuple[list[Path], list[Path]]]:
    """
    Full pipeline for one archive.

    Returns (original_psd_tif_files, webp_files) or None on failure.

    Callers should:
      upload original_psd_tif_files → GDRIVE_PSD_FOLDER   (psd folder)
      upload webp_files             → GDRIVE_WEBP_FOLDER  (preview folder)
    """
    archive_path = Path(archive_path)
    work_dir     = Path(work_dir)

    extract_dir = work_dir / "extracted"
    webp_dir    = work_dir / "webp"
    extract_dir.mkdir(parents=True, exist_ok=True)
    webp_dir.mkdir(parents=True, exist_ok=True)

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

    log.info(f"Found {len(source_files)} PSD/TIF file(s).")

    original_files: list[Path] = []
    webp_paths:     list[Path] = []

    for src in source_files:
        webp_path = webp_dir / (src.stem + ".webp")

        # Original file — no modification
        original_files.append(src)

        # ── Step 3: Create WebP with watermark ────────────────────────────
        if create_webp(src, webp_path):
            webp_paths.append(webp_path)
        else:
            log.warning(f"WebP creation failed for {src.name}")

    return original_files, webp_paths
