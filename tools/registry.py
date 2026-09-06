"""Policy-enforcing registry for all operating-system access."""

from __future__ import annotations

import json
import queue
import re
import shlex
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore, Event
from typing import Any, Iterable, Mapping

from config.config import AgentConfig
from tools.contracts import (
    PermissionLevel,
    ToolDefinition,
    ToolEvent,
    ToolEventKind,
    ToolExecutionError,
    ToolProgress,
    ToolRequest,
    ToolResult,
)
from tools.linux_tools import ToolCancelled, create_tool_definitions
from tools.network_tools import create_network_tool_definitions
from tools.diagnostic_tools import create_diagnostic_tool_definitions


class ToolValidationError(ValueError):
    """The model's typed request did not match an approved tool contract."""


def _request_action(request: ToolRequest | None, fallback: str) -> tuple[str, Any]:
    if request is None:
        return fallback, None
    arguments = dict(request.arguments)
    program = arguments.get("program")
    args = arguments.get("args", [])
    if isinstance(program, str) and isinstance(args, list):
        try:
            return shlex.join([program, *(str(value) for value in args)]), arguments
        except (TypeError, ValueError):
            pass
    encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    return f"{request.name} {encoded}", arguments


def _process_event(
    event_id: str,
    tool_name: str,
    display_name: str,
    kind: ToolEventKind,
    message: str,
    *,
    request: ToolRequest | None = None,
    result: ToolResult | None = None,
    progress: ToolProgress | None = None,
    started_at: float | None = None,
) -> ToolEvent:
    action, input_data = _request_action(request, message)
    ended_at = time.time() if kind in {
        ToolEventKind.BLOCKED,
        ToolEventKind.FAILED,
        ToolEventKind.CANCELLED,
        ToolEventKind.COMPLETED,
    } else None
    return ToolEvent(
        event_id,
        tool_name,
        display_name,
        kind,
        message,
        result=result,
        progress=progress,
        step_type=tool_name,
        action=action,
        input_data=input_data,
        started_at=started_at or time.time(),
        ended_at=ended_at,
    )


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """Trusted execution policy; model fields can never change this policy."""

    allowed_roots: tuple[Path, ...]
    auto_approve_read_only: bool = True
    allow_network: bool = False
    allow_write: bool = False
    allow_destructive: bool = False
    allow_terminal: bool = False
    max_concurrent: int = 1

    @classmethod
    def from_config(cls, config: AgentConfig) -> "ToolPolicy":
        return cls(
            allowed_roots=tuple(config.tool_allowed_roots),
            auto_approve_read_only=config.tool_auto_approve_read_only,
            allow_network=config.tool_allow_network,
            allow_write=config.tool_allow_write,
            allow_destructive=config.tool_allow_destructive,
            allow_terminal=config.tool_allow_terminal,
            max_concurrent=max(1, min(4, config.tool_max_concurrent)),
        )


class ToolRegistry:
    """The sole gateway from model requests to local OS operations.

    ``confirmation`` is intentionally a trusted call-site argument. The
    request's ``requires_confirmation`` field is only a model hint and is
    never used as permission to run a write, destructive, network, or terminal
    operation.
    """

    def __init__(self, definitions: Iterable[ToolDefinition], policy: ToolPolicy) -> None:
        self.policy = policy
        self._definitions = {definition.name: definition for definition in definitions}
        self._execution_slots = BoundedSemaphore(
            max(1, min(4, int(getattr(policy, "max_concurrent", 1))))
        )

    def get(self, name: str) -> ToolDefinition | None:
        return self._definitions.get(name)

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._definitions.values())

    def catalog(self) -> list[dict[str, Any]]:
        return [definition.public_schema() for definition in self._definitions.values()]

    def validate(
        self,
        request: ToolRequest,
        *,
        confirmation: bool = False,
        diagnostic: bool = False,
        software: bool = False,
    ) -> ToolDefinition:
        if not isinstance(request, ToolRequest):
            raise ToolValidationError("request is not a ToolRequest")
        definition = self.get(request.name)
        if definition is None:
            raise ToolValidationError(f"tool is not approved: {request.name}")
        _validate_schema(definition.input_schema, request.arguments)
        if not self._authorized(definition, confirmation, diagnostic, software):
            raise PermissionError(
                f"{definition.display_name or definition.name} requires trusted confirmation "
                f"or an enabled policy for {definition.permission_level.value} access"
            )
        return definition

    def execute_stream(
        self,
        request: ToolRequest,
        cancel_event: Event,
        *,
        confirmation: bool = False,
        diagnostic: bool = False,
        software: bool = False,
    ) -> Iterable[ToolEvent]:
        """Validate, execute with a timeout, and stream safe lifecycle events."""
        event_id = uuid.uuid4().hex
        definition = self.get(request.name if isinstance(request, ToolRequest) else "")
        display_name = definition.display_name if definition else str(getattr(request, "name", "Unknown tool"))
        request_name = str(getattr(request, "name", "unknown"))
        try:
            definition = self.validate(
                request,
                confirmation=confirmation,
                diagnostic=diagnostic,
                software=software,
            )
        except PermissionError as exc:
            result = ToolResult(
                request_name,
                False,
                data=_structured_tool_data(None, ok=False, error_code="permission_denied", error_message=str(exc)),
                error_code="permission_denied",
                error_message=str(exc),
            )
            yield _process_event(
                event_id, request_name, display_name, ToolEventKind.BLOCKED, str(exc),
                request=request, result=result,
            )
            return
        except (ToolValidationError, TypeError, ValueError) as exc:
            result = ToolResult(
                request_name,
                False,
                data=_structured_tool_data(None, ok=False, error_code="invalid_request", error_message=str(exc)),
                error_code="invalid_request",
                error_message=str(exc),
            )
            yield _process_event(
                event_id, request_name, display_name, ToolEventKind.FAILED, f"Rejected: {exc}",
                request=request if isinstance(request, ToolRequest) else None, result=result,
            )
            return

        if cancel_event.is_set():
            yield _process_event(
                event_id, definition.name, definition.display_name, ToolEventKind.CANCELLED,
                "Cancelled before start", request=request,
            )
            return

        if not self._execution_slots.acquire(blocking=False):
            result = ToolResult(
                definition.name,
                False,
                error_code="busy",
                error_message="another system tool is already running",
                data=_structured_tool_data(None, ok=False, error_code="busy", error_message="another system tool is already running"),
            )
            yield _process_event(
                event_id, definition.name, definition.display_name, ToolEventKind.BLOCKED,
                "Tool execution is busy", request=request, result=result,
            )
            return

        started_wall = time.time()
        yield _process_event(
            event_id, definition.name, definition.display_name, ToolEventKind.STARTED,
            f"{definition.display_name} — In progress...", request=request,
            started_at=started_wall,
        )
        started = time.monotonic()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="system-agent-tool")
        progress_queue: queue.Queue[ToolProgress] = queue.Queue()

        def report_progress(
            completed_bytes: int,
            total_bytes: int | None = None,
            speed_bytes_per_second: float | None = None,
        ) -> None:
            completed = max(0, int(completed_bytes))
            total = None if total_bytes is None else max(0, int(total_bytes))
            speed = (
                None
                if speed_bytes_per_second is None
                else max(0.0, float(speed_bytes_per_second))
            )
            percent = None
            if total and total > 0:
                percent = min(100.0, max(0.0, completed * 100.0 / total))
            progress_queue.put(ToolProgress(completed, total, speed, percent))

        try:
            if definition.reports_progress:
                future = executor.submit(
                    definition.handler,
                    dict(request.arguments),
                    cancel_event,
                    report_progress,
                )
            else:
                future = executor.submit(definition.handler, dict(request.arguments), cancel_event)
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            self._execution_slots.release()
            raise
        # Keep the slot occupied until the handler really exits.  A handler
        # that is already running cannot be stopped by Future.cancel(); the
        # handler must observe the cancellation event, and a later request
        # must not overlap it even after this generator returns.
        future.add_done_callback(lambda _future: self._execution_slots.release())
        try:
            deadline = started + definition.timeout_seconds
            try:
                while True:
                    while True:
                        try:
                            progress = progress_queue.get_nowait()
                        except queue.Empty:
                            break
                        yield _process_event(
                            event_id, definition.name, definition.display_name, ToolEventKind.PROGRESS,
                            _progress_message(definition.display_name, progress), request=request,
                            progress=progress, started_at=started_wall,
                        )
                    try:
                        data = future.result(timeout=0.25)
                        break
                    except FutureTimeout:
                        if cancel_event.is_set():
                            future.cancel()
                            yield _process_event(
                                event_id, definition.name, definition.display_name, ToolEventKind.CANCELLED,
                                "Cancelled", request=request, started_at=started_wall,
                            )
                            return
                        if time.monotonic() >= deadline:
                            cancel_event.set()
                            future.cancel()
                            result = ToolResult(
                                tool_name=definition.name,
                                ok=False,
                                data=_structured_tool_data(
                                    {"timed_out": True},
                                    ok=False,
                                    error_code="timeout",
                                    error_message=f"tool exceeded {definition.timeout_seconds:g}s timeout",
                                ),
                                error_code="timeout",
                                error_message=f"tool exceeded {definition.timeout_seconds:g}s timeout",
                                duration_ms=int((time.monotonic() - started) * 1000),
                            )
                            yield _process_event(
                                event_id, definition.name, definition.display_name, ToolEventKind.FAILED,
                                "Timed out", request=request, result=result, started_at=started_wall,
                            )
                            return
                        yield _process_event(
                            event_id, definition.name, definition.display_name, ToolEventKind.PROGRESS,
                            f"{definition.display_name} — Working...", request=request,
                            started_at=started_wall,
                        )
                while True:
                    try:
                        progress = progress_queue.get_nowait()
                    except queue.Empty:
                        break
                    yield _process_event(
                        event_id, definition.name, definition.display_name, ToolEventKind.PROGRESS,
                        _progress_message(definition.display_name, progress), request=request,
                        progress=progress, started_at=started_wall,
                    )
            except ToolCancelled as exc:
                result = ToolResult(
                    tool_name=definition.name,
                    ok=False,
                    data=_structured_tool_data(
                        None,
                        ok=False,
                        error_code="cancelled",
                        error_message=str(exc),
                    ),
                    error_code="cancelled",
                    error_message=str(exc),
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
                yield _process_event(
                    event_id, definition.name, definition.display_name, ToolEventKind.CANCELLED,
                    "Cancelled", request=request, result=result, started_at=started_wall,
                )
                return
            except ToolExecutionError as exc:
                result = ToolResult(
                    tool_name=definition.name,
                    ok=False,
                    data=_structured_tool_data(
                        exc.data,
                        ok=False,
                        error_code=exc.error_code,
                        error_message=str(exc),
                    ),
                    error_code=exc.error_code,
                    error_message=str(exc)[:500],
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
                yield _process_event(
                    event_id, definition.name, definition.display_name, ToolEventKind.FAILED,
                    f"{definition.display_name} — Failed", request=request, result=result,
                    started_at=started_wall,
                )
                return
            except Exception as exc:  # tool faults stay structured and local
                result = ToolResult(
                    tool_name=definition.name,
                    ok=False,
                    data=_structured_tool_data(
                        None,
                        ok=False,
                        error_code=type(exc).__name__,
                        error_message=str(exc),
                    ),
                    error_code=type(exc).__name__,
                    error_message=str(exc)[:500],
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
                yield _process_event(
                    event_id, definition.name, definition.display_name, ToolEventKind.FAILED,
                    f"{definition.display_name} — Failed", request=request, result=result,
                    started_at=started_wall,
                )
                return

            safe_data = _structured_tool_data(data, ok=True)
            result = ToolResult(
                tool_name=definition.name,
                ok=True,
                data=safe_data,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            yield _process_event(
                event_id, definition.name, definition.display_name, ToolEventKind.COMPLETED,
                f"{definition.display_name} — Completed", request=request, result=result,
                started_at=started_wall,
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _authorized(
        self,
        definition: ToolDefinition,
        confirmation: bool,
        diagnostic: bool,
        software: bool,
    ) -> bool:
        level = definition.permission_level
        if not software and definition.safe_troubleshooting and confirmation:
            return True
        if software and definition.safe_software:
            # Read-only discovery/search is allowed under the normal
            # read-only policy. Package downloads and all system changes need
            # an explicit trusted confirmation; the model cannot provide it.
            if definition.confirmation_required:
                return confirmation
            if level in {PermissionLevel.READ_ONLY, PermissionLevel.NETWORK}:
                return self.policy.auto_approve_read_only or confirmation
            return confirmation
        if diagnostic and (
            (level is PermissionLevel.READ_ONLY and not definition.safe_software)
            or (
                definition.safe_diagnostic
                and level in {PermissionLevel.READ_ONLY, PermissionLevel.NETWORK}
            )
        ):
            return True
        if level is PermissionLevel.READ_ONLY:
            return self.policy.auto_approve_read_only or confirmation
        if level is PermissionLevel.NETWORK:
            return self.policy.allow_network and confirmation
        if level is PermissionLevel.WRITE:
            return self.policy.allow_write and confirmation
        if level is PermissionLevel.DESTRUCTIVE:
            return self.policy.allow_destructive and confirmation
        if level is PermissionLevel.TERMINAL:
            return self.policy.allow_terminal and confirmation
        return False


def create_default_registry(config: AgentConfig) -> ToolRegistry:
    policy = ToolPolicy.from_config(config)
    from tools.software_tools import create_software_tool_definitions

    return ToolRegistry(
        (
            *create_tool_definitions(policy.allowed_roots),
            *create_network_tool_definitions(),
            *create_diagnostic_tool_definitions(policy.allowed_roots),
            *create_software_tool_definitions(policy.allowed_roots),
        ),
        policy,
    )


def tool_catalog_for_prompt(registry: ToolRegistry | None = None) -> str:
    """Compact JSON catalog used by the local model; no handlers are exposed."""
    if registry is None:
        # Prompt construction may happen before the application creates its
        # registry. This catalog still contains metadata only.
        defaults = AgentConfig()
        registry = create_default_registry(defaults)
    return json.dumps(registry.catalog(), separators=(",", ":"), ensure_ascii=True)


def _json_safe(value: Any) -> Any:
    """Keep model-facing results JSON-compatible and bounded."""
    try:
        encoded = json.dumps(value, ensure_ascii=False)
        if len(encoded) > 1_500_000:
            return {"truncated": True, "value": encoded[:1_500_000]}
        return json.loads(encoded)
    except (TypeError, ValueError, OverflowError):
        return str(value)[:1_500_000]


def _structured_tool_data(
    value: Any,
    *,
    ok: bool,
    error_code: str = "",
    error_message: str = "",
) -> dict[str, Any]:
    """Wrap every handler result in one predictable observation envelope."""
    safe = _json_safe(value)
    data = dict(safe) if isinstance(safe, dict) else {"value": safe}
    data.setdefault("success", ok)
    data.setdefault("exit_code", 0 if ok else None)
    data.setdefault("stdout", "")
    data.setdefault("stderr", "" if ok else error_message[:2_000])
    data.setdefault("error_type", "" if ok else _classify_error(error_code, error_message, data))
    return data


def _classify_error(error_code: str, error_message: str, data: Any) -> str:
    """Map low-level tool faults to stable troubleshooting categories."""
    haystack = " ".join((error_code, error_message, str(data))).casefold()
    if "permission" in haystack or "access denied" in haystack or "sudo" in haystack:
        return "PERMISSION_ERROR"
    if "package" in haystack and ("not found" in haystack or "locate" in haystack):
        return "PACKAGE_NOT_FOUND"
    if "dependenc" in haystack or "dpkg" in haystack:
        return "DEPENDENCY_ERROR"
    if "dns" in haystack or "resolve" in haystack:
        return "DNS_ERROR"
    if "network" in haystack or "connect" in haystack or "gateway" in haystack:
        return "NETWORK_ERROR"
    if "disk" in haystack or "space" in haystack:
        return "DISK_FULL"
    if "device" in haystack or "adapter" in haystack:
        return "DEVICE_NOT_FOUND"
    if "service" in haystack or "systemd" in haystack:
        return "SERVICE_FAILURE"
    if "driver" in haystack or "gpu" in haystack:
        return "DRIVER_ERROR"
    if "timeout" in haystack:
        return "TIMEOUT_ERROR"
    return error_code.upper() if error_code else "UNKNOWN_ERROR"


def _progress_message(display_name: str, progress: ToolProgress) -> str:
    completed = _format_bytes(progress.completed_bytes)
    if progress.total_bytes:
        total = _format_bytes(progress.total_bytes)
        percent = f"{progress.percent:.0f}%" if progress.percent is not None else ""
        message = f"{display_name} — {percent} ({completed} / {total})"
    else:
        message = f"{display_name} — {completed}"
    if progress.speed_bytes_per_second:
        message += f" · {_format_bytes(progress.speed_bytes_per_second)}/s"
    return message


def _format_bytes(value: int | float) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024 or unit == "GiB":
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} GiB"


def _validate_schema(schema: Mapping[str, Any], value: Any, path: str = "arguments") -> None:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            raise ToolValidationError(f"{path} must be an object")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for name in required:
            if name not in value:
                raise ToolValidationError(f"{path}.{name} is required")
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise ToolValidationError(f"{path} has unknown field(s): {', '.join(sorted(unknown))}")
        for name, item in value.items():
            if name in properties:
                _validate_schema(properties[name], item, f"{path}.{name}")
        return
    if expected == "array":
        if not isinstance(value, list):
            raise ToolValidationError(f"{path} must be an array")
        if len(value) < int(schema.get("minItems", 0)) or len(value) > int(schema.get("maxItems", 10**9)):
            raise ToolValidationError(f"{path} has an invalid item count")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate_schema(item_schema, item, f"{path}[{index}]")
        return
    if expected == "string":
        if not isinstance(value, str):
            raise ToolValidationError(f"{path} must be a string")
        if len(value) < int(schema.get("minLength", 0)) or len(value) > int(schema.get("maxLength", 10**9)):
            raise ToolValidationError(f"{path} has an invalid length")
        if schema.get("pattern") and re.fullmatch(str(schema["pattern"]), value) is None:
            raise ToolValidationError(f"{path} has an invalid format")
        if schema.get("enum") and value not in schema["enum"]:
            raise ToolValidationError(f"{path} is not an allowed value")
        return
    if expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolValidationError(f"{path} must be an integer")
        if value < int(schema.get("minimum", -10**18)) or value > int(schema.get("maximum", 10**18)):
            raise ToolValidationError(f"{path} is outside the allowed range")
        return
    if expected == "boolean" and not isinstance(value, bool):
        raise ToolValidationError(f"{path} must be a boolean")
