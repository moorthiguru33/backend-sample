"""
processor.py – Core processing pipeline:

  1. Extract ZIP or RAR archive.
  2. Find all PSD and TIF/TIFF files inside.
  3. For each file:
       a. Delete the TOP (first) layer → save modified PSD/TIF.
       b. Composite the remaining layers → add watermark → save as WebP.
  4. Re-package the modified PSD/TIF files into a new ZIP.

Returns: (zip_path: Path, webp_paths: list[Path])
"""

import logging
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
from psd_tools import PSDImage

from config import WATERMARK_TEXT, WATERMARK_OPACITY

log = logging.getLogger(__name__)


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
            # Try unrar-free first, then patool fallback
            result = subprocess.run(
                ["unrar", "x", "-y", str(archive_path), str(dest_dir) + "/"],
                capture_output=True,
            )
            if result.returncode == 0:
                return True
            # fallback via patool
            import patoollib
            patoollib.extract_archive(str(archive_path), outdir=str(dest_dir))
            return True

        if ext in (".7z", ".7zip"):
            subprocess.run(
                ["7z", "x", str(archive_path), f"-o{dest_dir}", "-y"],
                check=True,
                capture_output=True,
            )
            return True

        # Generic fallback
        import patoollib
        patoollib.extract_archive(str(archive_path), outdir=str(dest_dir))
        return True

    except Exception as exc:
        log.error(f"Extraction failed for {archive_path}: {exc}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  PSD: delete top layer
# ═══════════════════════════════════════════════════════════════════════════

def delete_top_layer_psd(src: Path, dst: Path) -> bool:
    """
    Remove the topmost layer from a PSD file and save to dst.
    Uses psd-tools internal record access (most reliable approach).
    Returns True if a layer was removed, False otherwise.
    """
    try:
        psd = PSDImage.open(str(src))

        lami = psd._record.layer_and_mask_info
        if lami is None:
            shutil.copy2(src, dst)
            log.warning(f"No layer info in {src.name} – copied as-is")
            return False

        li = lami.layer_info
        if li is None or not li.layer_records:
            shutil.copy2(src, dst)
            log.warning(f"Empty layer records in {src.name} – copied as-is")
            return False

        records = li.layer_records
        channels = li.channel_image_data
        n = len(records)

        if n == 0:
            shutil.copy2(src, dst)
            return False

        # In PSD binary format layers are stored BOTTOM → TOP.
        # Therefore records[-1] = topmost layer in Photoshop's panel.
        records.pop()                        # remove top layer record
        if channels and len(channels) == n:
            channels.pop()                   # remove its pixel data

        psd.save(str(dst))
        log.info(f"PSD top layer removed: {src.name}")
        return True

    except Exception as exc:
        log.error(f"delete_top_layer_psd failed for {src.name}: {exc}")
        shutil.copy2(src, dst)
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  TIF/TIFF: delete top layer (first page/frame)
# ═══════════════════════════════════════════════════════════════════════════

def delete_top_layer_tif(src: Path, dst: Path) -> bool:
    """
    Remove the first page (= top layer) from a multi-page TIFF.
    Falls back to PIL if tifffile fails.
    """
    # ── Method 1: tifffile ────────────────────────────────────────────────
    try:
        import tifffile
        import numpy as np

        with tifffile.TiffFile(str(src)) as tif:
            n_pages = len(tif.pages)
            if n_pages <= 1:
                shutil.copy2(src, dst)
                log.warning(f"TIF {src.name} has only 1 page – copied as-is")
                return False
            arrays = [tif.pages[i].asarray() for i in range(1, n_pages)]

        tifffile.imwrite(
            str(dst),
            arrays,
            photometric=(
                "rgb" if arrays[0].ndim == 3 and arrays[0].shape[-1] >= 3
                else "minisblack"
            ),
        )
        log.info(f"TIF top layer removed: {src.name}")
        return True

    except Exception as exc:
        log.warning(f"tifffile method failed for {src.name}: {exc}. Trying PIL…")

    # ── Method 2: PIL/Pillow fallback ─────────────────────────────────────
    try:
        from PIL import ImageSequence

        img = Image.open(str(src))
        frames = [
            frame.copy()
            for i, frame in enumerate(ImageSequence.Iterator(img))
            if i > 0
        ]

        if not frames:
            shutil.copy2(src, dst)
            return False

        frames[0].save(
            str(dst),
            format="TIFF",
            save_all=True,
            append_images=frames[1:],
        )
        log.info(f"TIF top layer removed (PIL): {src.name}")
        return True

    except Exception as exc2:
        log.error(f"PIL TIF fallback failed for {src.name}: {exc2}")
        shutil.copy2(src, dst)
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  WebP creation with watermark
# ═══════════════════════════════════════════════════════════════════════════

def _load_best_font(size: int) -> ImageFont.FreeTypeFont:
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


def add_watermark(img: Image.Image, text: str, opacity: int) -> Image.Image:
    """
    Tile the watermark text diagonally across the entire image.
    opacity: 0 = invisible, 255 = fully opaque.
    """
    img = img.convert("RGBA")
    w, h = img.size

    font_size = max(24, w // 22)
    font = _load_best_font(font_size)

    # Build a transparent overlay the same size as the image
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    step_x = tw + 80
    step_y = th + 60

    # Tile with a diagonal offset
    for row, y in enumerate(range(-h, h * 2, step_y)):
        x_offset = (row % 2) * (step_x // 2)
        for x in range(-w + x_offset, w * 2, step_x):
            draw.text(
                (x, y),
                text,
                font=font,
                fill=(255, 255, 255, opacity),  # white with low opacity
            )

    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")


def composite_psd(psd_path: Path) -> Optional[Image.Image]:
    """
    Composite all layers of a (already-modified) PSD into a single PIL image.
    """
    try:
        psd = PSDImage.open(str(psd_path))
        img = psd.composite()
        if img is not None:
            return img.convert("RGB")
        # Fallback: use the merged thumbnail inside the PSD
        merged = psd.topil()
        if merged:
            return merged.convert("RGB")
    except Exception as exc:
        log.error(f"composite_psd failed for {psd_path.name}: {exc}")
    return None


def composite_tif(tif_path: Path) -> Optional[Image.Image]:
    """
    Use the first remaining page of the TIF as the preview image.
    """
    try:
        img = Image.open(str(tif_path))
        return img.convert("RGB")
    except Exception as exc:
        log.error(f"composite_tif failed for {tif_path.name}: {exc}")
    return None


def create_webp(source_path: Path, output_path: Path) -> bool:
    """
    Composite source PSD/TIF, apply watermark, save as WebP.
    """
    ext = source_path.suffix.lower()
    if ext == ".psd":
        img = composite_psd(source_path)
    elif ext in (".tif", ".tiff"):
        img = composite_tif(source_path)
    else:
        log.error(f"Unsupported format for WebP creation: {source_path.suffix}")
        return False

    if img is None:
        log.error(f"Could not composite image from {source_path.name}")
        return False

    img_with_wm = add_watermark(img, WATERMARK_TEXT, WATERMARK_OPACITY)
    img_with_wm.save(str(output_path), "WEBP", quality=85)
    log.info(f"WebP created: {output_path.name}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def process_archive(
    archive_path: Path,
    work_dir: Path,
) -> Optional[tuple[Path, list[Path]]]:
    """
    Full pipeline for one archive file.

    Returns (new_zip_path, [webp_path, …]) or None on failure.
    """
    archive_path = Path(archive_path)
    work_dir = Path(work_dir)

    extract_dir  = work_dir / "extracted"
    modified_dir = work_dir / "modified"
    webp_dir     = work_dir / "webp"
    zip_dir      = work_dir / "zip_out"

    for d in (extract_dir, modified_dir, webp_dir, zip_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Extract ───────────────────────────────────────────────────
    log.info(f"Extracting {archive_path.name} …")
    if not extract_archive(archive_path, extract_dir):
        return None

    # ── Step 2: Find PSD / TIF files ──────────────────────────────────────
    source_files = list(extract_dir.rglob("*.psd")) + \
                   list(extract_dir.rglob("*.PSD")) + \
                   list(extract_dir.rglob("*.tif")) + \
                   list(extract_dir.rglob("*.TIF")) + \
                   list(extract_dir.rglob("*.tiff")) + \
                   list(extract_dir.rglob("*.TIFF"))

    if not source_files:
        log.warning(f"No PSD/TIF files found inside {archive_path.name}")
        return None

    log.info(f"Found {len(source_files)} PSD/TIF file(s).")

    modified_files: list[Path] = []
    webp_paths:     list[Path] = []

    for src in source_files:
        stem = src.stem
        ext  = src.suffix.lower()

        mod_path  = modified_dir / src.relative_to(extract_dir)
        mod_path.parent.mkdir(parents=True, exist_ok=True)

        webp_path = webp_dir / (stem + ".webp")

        # ── Step 3a: Delete top layer ──────────────────────────────────────
        if ext == ".psd":
            delete_top_layer_psd(src, mod_path)
        else:
            delete_top_layer_tif(src, mod_path)

        modified_files.append(mod_path)

        # ── Step 3b: Create WebP with watermark ────────────────────────────
        if create_webp(mod_path, webp_path):
            webp_paths.append(webp_path)

    # ── Step 4: Repackage modified files into a new ZIP ───────────────────
    original_stem = archive_path.stem
    new_zip_path  = zip_dir / f"{original_stem}.zip"

    with zipfile.ZipFile(new_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for mod in modified_files:
            # Preserve relative path inside the zip
            arcname = mod.relative_to(modified_dir)
            zf.write(mod, arcname)

    log.info(f"New ZIP created: {new_zip_path.name}  ({new_zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return new_zip_path, webp_paths
