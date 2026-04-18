"""
job_tracker.py – Excel-based primary job tracker for PSD Automation Pipeline.

┌─────────────────────────────────────────────────────────────────┐
│  THIS IS THE SINGLE SOURCE OF TRUTH FOR WHAT TO PROCESS         │
│                                                                  │
│  First run  → Scrape all URLs → Add to jobs.xlsx as Pending     │
│  Every run  → Read Pending rows → Process → Mark Completed       │
│  All done   → Re-scrape → New items added as Pending → continue │
└─────────────────────────────────────────────────────────────────┘

Excel columns (jobs.xlsx):
  A  Download URL      – the /download/eyJ… URL from forpsd.com
  B  Detail URL        – the product page URL (for category lookup)
  C  Category          – folder category (resolved from detail page)
  D  Status            – Pending / Completed / Error
  E  Date Added        – when this row was first added (UTC)
  F  Date Completed    – when processing succeeded (UTC)

Skip logic (fastest first):
  1. Status = Completed in jobs.xlsx  → skip instantly (zero API calls)
  2. rename_log.xlsx original-name check → skip (ExcelTracker)
  3. Drive name check (preloaded) → skip
"""

import logging
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

log = logging.getLogger(__name__)

# ── Column indices (1-based) ───────────────────────────────────────────────
_C_URL       = 1   # A: Download URL
_C_DETAIL    = 2   # B: Detail URL
_C_CATEGORY  = 3   # C: Category
_C_STATUS    = 4   # D: Status
_C_ADDED     = 5   # E: Date Added
_C_COMPLETED = 6   # F: Date Completed

STATUS_PENDING   = "Pending"
STATUS_COMPLETED = "Completed"
STATUS_ERROR     = "Error"

_HEADERS    = ["Download URL", "Detail URL", "Category",
               "Status", "Date Added", "Date Completed"]
_COL_WIDTHS = [80, 60, 25, 14, 22, 22]

# ── Styles ─────────────────────────────────────────────────────────────────
_HEADER_FONT  = Font(bold=True, color="FFFFFF", name="Calibri", size=11)
_HEADER_FILL  = PatternFill("solid", fgColor="1F3864")   # dark navy
_PENDING_ODD  = PatternFill("solid", fgColor="FFFFF0")   # ivory
_PENDING_EVEN = PatternFill("solid", fgColor="FFF2CC")   # light yellow
_DONE_FILL    = PatternFill("solid", fgColor="E2EFDA")   # light green
_ERROR_FILL   = PatternFill("solid", fgColor="FCE4D6")   # light red
_CENTER       = Alignment(horizontal="center", vertical="center")
_WRAP         = Alignment(wrap_text=True, vertical="top")
_THIN_BORDER  = Border(
    bottom=Side(style="thin", color="D0D0D0"),
    right =Side(style="thin", color="D0D0D0"),
)


class JobTracker:
    """
    Excel-based URL job tracker.

    All lookups are O(1) via in-memory dict { url → info }.
    The Excel file is written after every status change (crash-safe).
    """

    def __init__(self, xlsx_path: str | Path):
        self._path      = Path(xlsx_path)
        # url → {detail, category, status, date_added, date_completed, row}
        self._jobs: dict[str, dict] = {}
        self._next_row: int = 2          # row 1 = header
        self._load()

    # ── Private ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            log.info(f"📋 Job tracker: no jobs.xlsx yet — fresh start.")
            return

        try:
            wb = openpyxl.load_workbook(self._path)
            ws = wb.active
            counts = {STATUS_PENDING: 0, STATUS_COMPLETED: 0, STATUS_ERROR: 0}
            row_num = 1
            for row in ws.iter_rows(min_row=2, values_only=True):
                row_num += 1
                url    = str(row[0]).strip() if row[0] else ""
                if not url or url in ("None", "nan", "Download URL"):
                    continue
                detail = str(row[1]).strip() if row[1] else ""
                cat    = str(row[2]).strip() if row[2] else ""
                status = str(row[3]).strip() if row[3] else STATUS_PENDING
                added  = str(row[4]).strip() if row[4] else ""
                comp   = str(row[5]).strip() if row[5] else ""

                # Normalize any old status values
                if status not in (STATUS_PENDING, STATUS_COMPLETED, STATUS_ERROR):
                    status = STATUS_PENDING

                self._jobs[url] = {
                    "detail": detail, "category": cat,
                    "status": status, "date_added": added,
                    "date_completed": comp, "row": row_num,
                }
                counts[status] = counts.get(status, 0) + 1

            self._next_row = row_num + 1
            log.info(
                f"📋 Job tracker loaded: {len(self._jobs)} total | "
                f"Pending={counts[STATUS_PENDING]} "
                f"Completed={counts[STATUS_COMPLETED]} "
                f"Error={counts[STATUS_ERROR]}"
            )
        except Exception as exc:
            log.error(f"Job tracker load error: {exc} — starting fresh.")
            self._jobs      = {}
            self._next_row  = 2

    def _make_workbook(self) -> openpyxl.Workbook:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Jobs"
        for ci, (hdr, w) in enumerate(zip(_HEADERS, _COL_WIDTHS), 1):
            cell            = ws.cell(1, ci, hdr)
            cell.font       = _HEADER_FONT
            cell.fill       = _HEADER_FILL
            cell.alignment  = _CENTER
            ws.column_dimensions[_col_letter(ci)].width = w
        ws.row_dimensions[1].height = 22
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{_col_letter(len(_HEADERS))}1"
        return wb

    def _full_rewrite(self) -> None:
        """Rewrite the entire Excel from the in-memory dict (used after status change)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            wb = self._make_workbook()
            ws = wb.active

            sorted_items = sorted(self._jobs.items(), key=lambda x: x[1]["row"])
            for ri, (url, info) in enumerate(sorted_items, start=2):
                status = info.get("status", STATUS_PENDING)
                fill   = _row_fill(status, ri)
                values = [
                    url,
                    info.get("detail", ""),
                    info.get("category", ""),
                    status,
                    info.get("date_added", ""),
                    info.get("date_completed", ""),
                ]
                for ci, val in enumerate(values, 1):
                    cell            = ws.cell(ri, ci, val or "")
                    cell.fill       = fill
                    cell.border     = _THIN_BORDER
                    cell.alignment  = _WRAP

            wb.save(self._path)
        except Exception as exc:
            log.error(f"Job tracker full-rewrite error: {exc}")

    def _append_row(self, url: str, info: dict) -> None:
        """Append a single new row to existing file (fast-path for new items)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists():
                try:
                    wb = openpyxl.load_workbook(self._path)
                except Exception:
                    wb = self._make_workbook()
            else:
                wb = self._make_workbook()

            ws  = wb.active
            ri  = self._next_row
            fill = _row_fill(info.get("status", STATUS_PENDING), ri)
            values = [
                url,
                info.get("detail", ""),
                info.get("category", ""),
                info.get("status", STATUS_PENDING),
                info.get("date_added", ""),
                info.get("date_completed", ""),
            ]
            for ci, val in enumerate(values, 1):
                cell            = ws.cell(ri, ci, val or "")
                cell.fill       = fill
                cell.border     = _THIN_BORDER
                cell.alignment  = _WRAP

            wb.save(self._path)
        except Exception as exc:
            log.error(f"Job tracker append-row error: {exc}")

    # ── Public API ─────────────────────────────────────────────────────────

    def has_url(self, url: str) -> bool:
        """True if this URL is already in the tracker (any status)."""
        return url.strip() in self._jobs

    def is_completed(self, url: str) -> bool:
        """True if this URL was successfully processed in a previous run."""
        return self._jobs.get(url.strip(), {}).get("status") == STATUS_COMPLETED

    def total(self) -> int:
        return len(self._jobs)

    def completed_count(self) -> int:
        return sum(1 for v in self._jobs.values() if v.get("status") == STATUS_COMPLETED)

    def get_pending(self) -> list[dict]:
        """
        Return list of {download_url, detail_url, category} for all
        Pending and Error rows (in insertion order).
        """
        result = []
        for url, info in sorted(self._jobs.items(), key=lambda x: x[1]["row"]):
            if info.get("status") in (STATUS_PENDING, STATUS_ERROR):
                result.append({
                    "download_url": url,
                    "detail_url":   info.get("detail", ""),
                    "category":     info.get("category", ""),
                })
        return result

    def all_known_urls(self) -> set[str]:
        """All URLs tracked (any status). Used by scraper to detect new items."""
        return set(self._jobs.keys())

    def add_pending_items(self, items: list[dict]) -> int:
        """
        Add items that are NOT already tracked, with Status=Pending.
        Items with matching download_url already in tracker are SKIPPED.
        Returns count of NEW items added.
        """
        now   = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        added = 0
        for item in items:
            url = item.get("download_url", "").strip()
            if not url or url in self._jobs:
                continue
            info = {
                "detail":         item.get("detail_url", ""),
                "category":       item.get("category", ""),
                "status":         STATUS_PENDING,
                "date_added":     now,
                "date_completed": "",
                "row":            self._next_row,
            }
            self._jobs[url] = info
            self._append_row(url, info)
            self._next_row += 1
            added += 1

        if added:
            log.info(
                f"📋 Job tracker: +{added} new Pending items "
                f"(total: {len(self._jobs)})"
            )
        return added

    def update_category(self, url: str, category: str) -> None:
        """Cache the resolved category (does NOT write to disk — saved on mark_completed)."""
        if url in self._jobs and category:
            self._jobs[url]["category"] = category

    def mark_completed(self, url: str) -> None:
        """Mark URL as Completed and persist the entire file."""
        url = url.strip()
        if url not in self._jobs:
            log.warning(f"Job tracker: mark_completed called for unknown URL: {url[:60]}")
            return
        self._jobs[url]["status"]         = STATUS_COMPLETED
        self._jobs[url]["date_completed"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        self._full_rewrite()
        log.info(f"📋 ✅ Completed: {url[:70]}")

    def mark_error(self, url: str) -> None:
        """Mark URL as Error (will be retried next run) and persist."""
        url = url.strip()
        if url not in self._jobs:
            return
        self._jobs[url]["status"] = STATUS_ERROR
        self._full_rewrite()
        log.warning(f"📋 ❌ Error: {url[:70]}")

    def stats(self) -> str:
        total = len(self._jobs)
        done  = sum(1 for v in self._jobs.values() if v.get("status") == STATUS_COMPLETED)
        err   = sum(1 for v in self._jobs.values() if v.get("status") == STATUS_ERROR)
        pend  = total - done - err
        return f"Total={total} | Pending={pend} | Completed={done} | Error={err}"


# ── Helpers ────────────────────────────────────────────────────────────────

def _col_letter(col: int) -> str:
    """Convert 1-based column index to Excel letter (A, B, …, Z, AA …)."""
    result = ""
    while col:
        col, rem = divmod(col - 1, 26)
        result = chr(65 + rem) + result
    return result


def _row_fill(status: str, row_idx: int) -> PatternFill:
    if status == STATUS_COMPLETED:
        return _DONE_FILL
    if status == STATUS_ERROR:
        return _ERROR_FILL
    return _PENDING_EVEN if row_idx % 2 == 0 else _PENDING_ODD
