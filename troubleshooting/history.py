"""Small append-only troubleshooting history store."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class TroubleshootingHistory:
    """Persist bounded, structured records without storing hidden reasoning."""

    def __init__(self, path: Path, *, max_records: int = 100) -> None:
        self.path = path.expanduser()
        self.max_records = max(1, max_records)
        self._lock = threading.RLock()

    def append(self, record: dict[str, Any]) -> None:
        safe_record = _bounded_json(record)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            records = self.recent(self.max_records - 1)
            records.append(safe_record)
            self.path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
                encoding="utf-8",
            )

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records: list[dict[str, Any]] = []
        try:
            lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        for line in lines[-max(1, limit) :]:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records


def _bounded_json(value: Any, *, limit: int = 160_000) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=False)
        if len(encoded) > limit:
            encoded = encoded[:limit]
        decoded = json.loads(encoded)
        return decoded if isinstance(decoded, dict) else {"value": decoded}
    except (TypeError, ValueError, OverflowError):
        return {"value": str(value)[:limit]}
