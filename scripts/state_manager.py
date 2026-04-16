"""
state_manager.py – Persist progress in state.json so every run continues
                   exactly where the last one stopped.
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
            "all_urls":        [],
            "processed":       [],
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

    def pending_urls(self) -> list:
        all_urls  = self._data.get("all_urls", [])
        processed = set(self._data.get("processed", []))
        return [u for u in all_urls if u not in processed]

    def summary(self) -> str:
        total     = len(self._data.get("all_urls", []))
        done      = len(self._data.get("processed", []))
        remaining = total - done
        return f"Total={total} | Done={done} | Remaining={remaining}"
