"""Human-readable formatting for structured Tool Registry results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tools.contracts import ToolResult


_LABELS = {
    "pid": "PID",
    "cpu_model": "CPU model",
    "logical_cpus": "Logical CPUs",
    "memory_total_kib": "Memory total",
    "total_bytes": "Total",
    "used_bytes": "Used",
    "free_bytes": "Free",
    "available_bytes": "Available",
    "usage_percent": "Usage",
    "exit_code": "Exit code",
    "stdout": "Output",
    "stderr": "Error output",
    "timed_out": "Timed out",
    "search_domains": "Search domains",
    "destination_hex": "Destination",
    "mask_hex": "Mask",
}


def format_tool_result(result: ToolResult, *, max_lines: int = 40) -> list[tuple[str, bool]]:
    """Turn one structured result into readable label/value lines.

    The raw JSON shape is intentionally not shown in the chatbot. Error lines
    are returned with a flag so the output widget can style them distinctly.
    """
    if not result.ok:
        lines = [(f"Error: {result.error_message or 'The check failed.'}", True)]
        if result.error_code:
            lines.append((f"Code: {result.error_code}", True))
    else:
        lines = [(line, False) for line in _format_value(result.data)]
    if result.duration_ms:
        lines.append((f"Duration: {_format_duration(result.duration_ms)}", False))
    return lines[:max_lines] or [("No details available.", False)]


def _format_value(value: Any, key: str | None = None, indent: int = 0) -> list[str]:
    prefix = " " * indent
    label = _label(key) if key else None

    if isinstance(value, Mapping):
        lines: list[str] = []
        if label:
            lines.append(f"{prefix}{label}:")
            indent += 2
            prefix = " " * indent
        if not value:
            return lines + [f"{prefix}No details available."]
        for item_key, item_value in value.items():
            if isinstance(item_value, (Mapping, list, tuple)):
                lines.extend(_format_value(item_value, str(item_key), indent))
            else:
                lines.extend(_format_scalar(item_value, str(item_key), indent))
        return lines

    if isinstance(value, (list, tuple)):
        if label:
            lines = [f"{prefix}{label}:"]
            indent += 2
            prefix = " " * indent
        else:
            lines = []
        if not value:
            return lines + [f"{prefix}None found."]
        for item in value:
            if isinstance(item, Mapping):
                parts: list[str] = []
                for item_key, item_value in item.items():
                    if isinstance(item_value, (Mapping, list, tuple)):
                        nested = _format_value(item_value, str(item_key), indent + 2)
                        parts.extend(nested)
                    else:
                        parts.append(_format_scalar_value(item_value, str(item_key)))
                if parts and all("\n" not in part for part in parts):
                    lines.append(f"{prefix}- {', '.join(parts)}")
                else:
                    lines.extend(f"{prefix}- {part}" for part in parts)
            else:
                lines.append(f"{prefix}- {_format_scalar_value(item, key)}")
        return lines

    return _format_scalar(value, key, indent)


def _format_scalar(value: Any, key: str | None, indent: int) -> list[str]:
    prefix = " " * indent
    label = f"{_label(key)}: " if key else ""
    if isinstance(value, str) and "\n" in value:
        lines = [f"{prefix}{label.rstrip()}"]
        lines.extend(f"{prefix}  {line}" for line in value.splitlines())
        return lines
    return [f"{prefix}{label}{_format_scalar_value(value, key)}"]


def _format_scalar_value(value: Any, key: str | None = None) -> str:
    if value is None or value == "":
        return "Unavailable"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if key == "usage_percent":
            return f"{value:.1f}%"
        return f"{value:g}"
    if isinstance(value, (int, float)) and key:
        if key.endswith("_bytes"):
            return _format_bytes(int(value))
        if key.endswith("_kib"):
            return _format_bytes(int(value) * 1024)
        if key == "usage_percent":
            return f"{value:.1f}%"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_scalar_value(item) for item in value) or "None"
    return str(value)


def _label(key: str | None) -> str:
    if not key:
        return "Details"
    return _LABELS.get(key, key.replace("_", " ").capitalize())


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    units = ("KB", "MB", "GB", "TB")
    amount = float(value)
    for unit in units:
        amount /= 1024
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
    return f"{value} B"


def _format_duration(milliseconds: int) -> str:
    if milliseconds < 1000:
        return f"{milliseconds} ms"
    return f"{milliseconds / 1000:.1f} s"
