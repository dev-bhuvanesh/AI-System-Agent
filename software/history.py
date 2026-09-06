"""Bounded append-only software management history."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class SoftwareHistory:
    def __init__(self, path: Path, *, max_records: int = 100) -> None:
        self.path = path.expanduser()
        self.max_records = max(1, max_records)
        self._lock = threading.RLock()

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            records = self.recent(self.max_records - 1)
            records.append(_bounded(record))
            self.path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
                encoding="utf-8",
            )

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        result: list[dict[str, Any]] = []
        for line in lines[-max(1, limit):]:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                result.append(value)
        return result


def _bounded(value: dict[str, Any], limit: int = 160_000) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=False)
        return json.loads(encoded[:limit])
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"value": str(value)[:limit]}


def default_history_path() -> Path:
    from config.config import _data_home

    return _data_home / "system-agent" / "software" / "history.jsonl"

