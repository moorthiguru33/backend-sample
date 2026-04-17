"""
state_manager.py – Persist progress in state.json so every run continues
                   exactly where the last one stopped.

Stores:
  all_items      : list of {download_url, detail_url} dicts (from scraper)
  processed      : list of download_url strings already done
  file_counter   : global sequential counter (tamilpsd-0001, 0002, ...)
  total_processed: int
  last_run       : ISO timestamp
  status         : "init" | "running" | "done"
"""
import json
import os
from pathlib import Path
from typing import Any


class StateManager:
    def __init__(self, path: str = "state.json"):
        self._path = Path(path)
        self._data: dict = self._load()

    # ── Internal ───────────────────────────────────────────────────────────
    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "all_items":       [],   # list of {download_url, detail_url}
            "all_urls":        [],   # legacy — kept for backward compat
            "processed":       [],   # list of processed download_urls
            "file_counter":    1,    # next tamilpsd-XXXX number
            "total_processed": 0,
            "last_run":        None,
            "status":          "init",
        }

    def save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    # ── Public API ─────────────────────────────────────────────────────────
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def mark_processed(self, url: str) -> None:
        processed: list = self._data.setdefault("processed", [])
        if url not in processed:
            processed.append(url)
        self._data["total_processed"] = len(processed)
        self.save()

    def is_processed(self, url: str) -> bool:
        return url in self._data.get("processed", [])

    def pending_items(self) -> list[dict]:
        """
        Return list of {download_url, detail_url} dicts not yet processed.
        Works with new all_items format.
        """
        all_items = self._data.get("all_items", [])
        processed = set(self._data.get("processed", []))

        # Backward compat: if all_items is empty but all_urls exists
        if not all_items:
            all_urls = self._data.get("all_urls", [])
            return [
                {"download_url": u, "detail_url": ""}
                for u in all_urls
                if u not in processed
            ]

        return [
            item for item in all_items
            if item.get("download_url") not in processed
        ]

    def pending_urls(self) -> list[str]:
        """Legacy method — returns just download_url strings."""
        return [item["download_url"] for item in self.pending_items()]

    def summary(self) -> str:
        all_items = self._data.get("all_items") or self._data.get("all_urls", [])
        total     = len(all_items)
        done      = len(self._data.get("processed", []))
        remaining = total - done
        counter   = self._data.get("file_counter", 1)
        return (
            f"Total={total} | Done={done} | Remaining={remaining} | "
            f"NextFile=tamilpsd-{counter:04d}"
        )
