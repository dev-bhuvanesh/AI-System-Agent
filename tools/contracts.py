"""Contracts shared by the local model, registry, and UI.

These objects deliberately contain data only. A model can request a tool, but
it cannot attach a callable, shell string, file descriptor, or executable
object to a request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Mapping


class PermissionLevel(StrEnum):
    READ_ONLY = "read_only"
    NETWORK = "network"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    TERMINAL = "terminal"


class ToolEventKind(StrEnum):
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolExecutionError(RuntimeError):
    """Structured handler failure preserved by the Tool Registry."""

    def __init__(self, error_code: str, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.data = data


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """A model-produced request for one named registry tool."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    # This is an informational model hint only. The registry never treats it
    # as approval; approval must be supplied by a trusted controller/UI.
    requires_confirmation: bool = True


@dataclass(frozen=True, slots=True)
class ToolApproval:
    """Trusted-UI approval request for one validated system operation."""

    approval_id: str
    request: ToolRequest
    display_name: str
    permission_level: PermissionLevel
    description: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured result returned by a tool, including safe failure details."""

    tool_name: str
    ok: bool
    data: Any = None
    error_code: str = ""
    error_message: str = ""
    duration_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tool": self.tool_name,
            "ok": self.ok,
            "data": self.data,
        }
        if self.error_code:
            result["error_code"] = self.error_code
        if self.error_message:
            result["error_message"] = self.error_message
        result["duration_ms"] = self.duration_ms
        return result


@dataclass(frozen=True, slots=True)
class ToolProgress:
    """Optional bounded progress information emitted by a long-running tool."""

    completed_bytes: int = 0
    total_bytes: int | None = None
    speed_bytes_per_second: float | None = None
    percent: float | None = None


@dataclass(frozen=True, slots=True)
class ToolEvent:
    """One user-safe execution event emitted by the registry."""

    event_id: str
    tool_name: str
    display_name: str
    kind: ToolEventKind
    message: str
    result: ToolResult | None = None
    progress: ToolProgress | None = None
    # Process-log metadata is emitted by the registry rather than model or
    # UI output. These fields are optional for compatibility with test/tool
    # adapters that construct lifecycle events directly.
    step_type: str = ""
    action: str = ""
    input_data: Any = None
    started_at: float | None = None
    ended_at: float | None = None


ToolHandler = Callable[[dict[str, Any], Any], Any]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Metadata and handler for one approved tool."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    permission_level: PermissionLevel
    timeout_seconds: float
    handler: ToolHandler
    display_name: str = ""
    # Trusted troubleshooting may run explicitly marked low-risk network
    # probes without turning on the general network policy.
    safe_diagnostic: bool = False
    # Curated package-management tools may be used by the software controller
    # only through its trusted plan/confirmation path.
    safe_software: bool = False
    # Handlers marked this way receive a trusted in-process progress callback;
    # the callback is never part of model-controlled request arguments.
    reports_progress: bool = False
    # Some network-level tools also write files (software downloads), so they
    # must not inherit the read-only auto-approval path.
    confirmation_required: bool = False
    # A tightly constrained repair handler may run after a trusted UI
    # confirmation without enabling arbitrary terminal access globally.
    safe_troubleshooting: bool = False

    def public_schema(self) -> dict[str, Any]:
        """Return model-facing metadata without exposing the Python handler."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "permission_level": self.permission_level.value,
            "timeout_seconds": self.timeout_seconds,
            "requires_confirmation": self.permission_level
            != PermissionLevel.READ_ONLY,
            "safe_diagnostic": self.safe_diagnostic,
            "safe_software": self.safe_software,
            "reports_progress": self.reports_progress,
            "confirmation_required": self.confirmation_required,
            "safe_troubleshooting": self.safe_troubleshooting,
        }
