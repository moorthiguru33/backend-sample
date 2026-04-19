"""
processor.py – Simplified pipeline (no WebP, no image processing):

  1. Extract ZIP / RAR / 7Z archive.
  2. Find all PSD and TIF/TIFF files inside (ignore .txt and others).
  3. For each file:
       a. Copy & rename to tamilpsd-XXXX.psd / .tif
  4. Return (renamed_psd_files, next_index).
     - renamed_psd_files → GDRIVE_PSD_FOLDER  (psd/)

Returns: (original_paths: list[Path], next_index: int)
"""

import logging
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

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
            result = subprocess.run(
                ["unrar", "x", "-y", str(archive_path), str(dest_dir) + "/"],
                capture_output=True,
            )
            if result.returncode == 0:
                return True
            # Fallback: use rarfile Python library (already in requirements.txt)
            log.warning("  unrar binary failed — falling back to rarfile library")
            import rarfile as _rarfile
            with _rarfile.RarFile(str(archive_path)) as rf:
                rf.extractall(str(dest_dir))
            return True

        if ext in (".7z", ".7zip"):
            subprocess.run(
                ["7z", "x", str(archive_path), f"-o{dest_dir}", "-y"],
                check=True, capture_output=True,
            )
            return True

        # Generic fallback: use zipfile/rarfile if format is recognized
        log.warning(f"  Unknown archive ext {ext!r} — attempting generic extraction")
        import zipfile as _zf
        if _zf.is_zipfile(str(archive_path)):
            with _zf.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_dir)
            return True
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

    except Exception as exc:
        log.error(f"Extraction failed for {archive_path}: {exc}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  Naming helper — tamilpsd-XXXX
# ═══════════════════════════════════════════════════════════════════════════

def make_tamilpsd_name(index: int) -> str:
    """Return base name like  tamilpsd-0001  for a given 1-based index."""
    return f"tamilpsd-{index:04d}"


# ═══════════════════════════════════════════════════════════════════════════
#  Main pipeline  (extract → rename only, no image processing)
# ═══════════════════════════════════════════════════════════════════════════

def process_archive(
    archive_path: Path,
    work_dir: Path,
    start_index: int = 1,
) -> Optional[tuple[list[Path], int]]:
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
    (renamed_psd_files, next_index) or None on failure.

    - renamed_psd_files : PSD/TIF files copied & renamed as tamilpsd-XXXX.psd/tif
    - next_index        : start_index + number of files successfully processed
                          (caller should persist this in state)

    Callers should:
      upload renamed_psd_files → GDRIVE_PSD_FOLDER   (psd folder)
    """
    archive_path = Path(archive_path)
    work_dir     = Path(work_dir)

    extract_dir = work_dir / "extracted"
    renamed_dir = work_dir / "renamed_psd"
    extract_dir.mkdir(parents=True, exist_ok=True)
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
            log.warning(f"  Could not rename {src.name}: {exc} — using original path")
            renamed_files.append(src)

        current_index += 1

    return renamed_files, current_index
