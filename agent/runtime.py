"""Orchestrate model planning and registry execution without a bypass path."""

from __future__ import annotations

import json
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from threading import Event, Lock
from typing import Iterator

from llm.provider import ChatMessage, LLMProvider, Plan, ProviderEvent
from tools.contracts import (
    PermissionLevel,
    ToolApproval,
    ToolEvent,
    ToolEventKind,
    ToolRequest,
    ToolResult,
)
from tools.registry import ToolRegistry


logger = logging.getLogger(__name__)


def _task_state_for_status(status: str) -> str:
    return {
        "idle": "IDLE",
        "planning": "PLANNING",
        "planned": "PLANNING",
        "executing": "EXECUTING",
        "waiting_permission": "WAITING_FOR_CONFIRMATION",
        "analyzing": "THINKING",
        "completed": "COMPLETED",
        "failed": "FAILED",
        "cancelled": "CANCELLED",
    }.get(status, "THINKING")


@dataclass(frozen=True, slots=True)
class AgentTaskState:
    """Safe, inspectable state for one serialized agent task."""

    task_id: str = ""
    goal: str = ""
    status: str = "idle"
    plan: Plan | None = None
    observations: tuple[ToolResult, ...] = field(default_factory=tuple)
    executed_tools: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    current_hypothesis: str = ""
    next_action: str = ""
    verification_status: str = "not_started"
    # The lowercase ``status`` field remains for existing callers. These
    # explicit fields describe the whole task rather than the latest tool.
    task_state: str = "IDLE"
    active_step_id: str = ""
    active_process_id: int | None = None
    abort_controller: Event | None = None
    pending_tool_calls: tuple[str, ...] = field(default_factory=tuple)
    verification_required: bool = False
    verification_complete: bool = False


@dataclass(slots=True)
class _PendingApproval:
    approval: ToolApproval
    decision: bool | None = None
    resolved: Event = field(default_factory=Event)


class AssistantRuntime:
    """Turn a user request into a plan, then execute only registry requests."""

    approval_timeout_seconds = 300.0

    def __init__(self, provider: LLMProvider, registry: ToolRegistry) -> None:
        self.provider = provider
        self.registry = registry
        self._active_lock = Lock()
        self._state_lock = Lock()
        self._pending_lock = Lock()
        self._pending: dict[str, _PendingApproval] = {}
        self._state = AgentTaskState()

    @property
    def state(self) -> AgentTaskState:
        with self._state_lock:
            return self._state

    def _set_state(self, **changes: object) -> None:
        status = changes.get("status")
        if isinstance(status, str):
            changes.setdefault("task_state", _task_state_for_status(status))
        with self._state_lock:
            self._state = replace(self._state, **changes)

    def approve_tool(self, approval_id: str, approved: bool) -> bool:
        """Resolve an approval from a trusted UI/controller call site.

        The model cannot call this method. An unknown or expired identifier is
        rejected so an old approval button cannot authorize a later task.
        """
        with self._pending_lock:
            pending = self._pending.get(approval_id)
            if pending is None:
                return False
            pending.decision = bool(approved)
            pending.resolved.set()
            return True

    def stream(
        self,
        request: str,
        cancel_event: Event,
        conversation: tuple[ChatMessage, ...] = (),
    ) -> Iterator[ProviderEvent]:
        """Stream model, tool lifecycle, and final local-model events.

        The model has no registry handler reference and no subprocess API. It
        can only emit data-only ``ToolRequest`` objects inside a validated
        ``Plan``. This controller passes every request through the registry.
        """
        cleaned = request.strip()
        if not cleaned:
            yield ProviderEvent.failure("Please enter a message before sending.")
            return
        if not self._active_lock.acquire(blocking=False):
            yield ProviderEvent.failure("An agent task is already running.")
            return

        task_id = uuid.uuid4().hex
        self._set_state(
            task_id=task_id,
            goal=cleaned,
            status="planning",
            plan=None,
            observations=(),
            executed_tools=(),
            errors=(),
            current_hypothesis="",
            next_action="Understanding the request",
            verification_status="not_started",
            task_state="PLANNING",
            active_step_id="",
            active_process_id=None,
            abort_controller=cancel_event,
            pending_tool_calls=(),
            verification_required=False,
            verification_complete=False,
        )
        plan: Plan | None = None
        emitted_text = False
        results: list[ToolResult] = []
        seen_requests: set[tuple[str, str]] = set()
        try:
            for event in self._plan_stream(cleaned, cancel_event, conversation):
                if cancel_event.is_set():
                    self._set_state(status="cancelled", next_action="", verification_status="cancelled")
                    return
                if event.kind == "error":
                    self._record_error(event.error or "planning failed")
                elif event.kind == "text":
                    emitted_text = emitted_text or bool(event.text.strip())
                elif event.kind == "plan" and event.plan is not None:
                    plan = event.plan
                    self._set_state(
                        status="planned",
                        plan=plan,
                        current_hypothesis=plan.summary,
                        verification_required=bool(plan.tool_requests),
                        next_action=(
                            f"Validate {len(plan.tool_requests)} approved tool request(s)"
                            if plan.tool_requests
                            else "Answer without system access"
                        ),
                    )
                if event.kind != "done":
                    yield event
                    continue

                if plan is not None and plan.tool_requests:
                    self._set_state(status="executing", next_action="Validating and executing approved tools")
                    execution_failed = False
                    for tool_request in plan.tool_requests:
                        if cancel_event.is_set():
                            self._set_state(status="cancelled", next_action="", verification_status="cancelled")
                            return
                        request_key = (
                            tool_request.name,
                            json.dumps(
                                tool_request.arguments,
                                sort_keys=True,
                                ensure_ascii=True,
                                separators=(",", ":"),
                                default=str,
                            ),
                        )
                        if request_key in seen_requests:
                            logger.info("Skipping duplicate model tool request: %s", tool_request.name)
                            continue
                        seen_requests.add(request_key)
                        approved = False
                        if self._requires_approval(tool_request):
                            pending = self._create_approval(task_id, tool_request)
                            self._set_state(
                                status="waiting_permission",
                                next_action=f"Waiting for permission to run {pending.approval.display_name}",
                                verification_status="pending",
                            )
                            yield ProviderEvent.status(
                                f"Waiting for permission: {pending.approval.display_name}"
                            )
                            yield ProviderEvent.tool_approval_ready(pending.approval)
                            deadline = time.monotonic() + self.approval_timeout_seconds
                            try:
                                while not pending.resolved.wait(0.1):
                                    if cancel_event.is_set():
                                        self._set_state(
                                            status="cancelled",
                                            next_action="",
                                            verification_status="cancelled",
                                        )
                                        return
                                    if time.monotonic() >= deadline:
                                        break
                                approved = pending.decision is True
                            finally:
                                with self._pending_lock:
                                    self._pending.pop(pending.approval.approval_id, None)
                            if not pending.resolved.is_set():
                                self._record_error(
                                    f"Permission timed out for {pending.approval.display_name}"
                                )
                            elif not approved:
                                self._record_error(
                                    f"Permission denied for {pending.approval.display_name}"
                                )
                        self._set_state(
                            status="executing",
                            active_step_id=f"tool:{tool_request.name}",
                            pending_tool_calls=(tool_request.name,),
                            next_action=f"Executing {tool_request.name}",
                        )
                        for tool_event in self.registry.execute_stream(
                            tool_request,
                            cancel_event,
                            # This value comes only from approve_tool(), never
                            # from ToolRequest.requires_confirmation.
                            confirmation=approved,
                        ):
                            self._record_tool_event(tool_event, results)
                            yield ProviderEvent.tool_update(tool_event)
                            if tool_event.kind is ToolEventKind.CANCELLED:
                                self._set_state(status="cancelled", next_action="", verification_status="cancelled")
                                return
                            if tool_event.result is not None and not tool_event.result.ok:
                                # Do not continue a model-produced batch after
                                # a failed or blocked operation. Send the
                                # actual structured failure to the provider so
                                # it can explain the result or propose a new,
                                # separately validated next step.
                                execution_failed = True
                                break
                        if execution_failed:
                            self._set_state(
                                status="analyzing",
                                next_action="Analyze the failed tool result before any next action",
                                verification_status="needs_attention",
                            )
                            break
                        self._set_state(
                            active_step_id="",
                            pending_tool_calls=(),
                        )
                    if results and not cancel_event.is_set():
                        self._set_state(
                            status="analyzing",
                            next_action="Analyze validated tool results",
                            verification_status=(
                                "needs_attention"
                                if any(not result.ok for result in results)
                                else "pending"
                            ),
                        )
                        for result_event in self.provider.stream_tool_results(
                            cleaned,
                            plan,
                            tuple(results),
                            cancel_event,
                        ):
                            if cancel_event.is_set():
                                self._set_state(status="cancelled", next_action="", verification_status="cancelled")
                                return
                            if result_event.kind == "error":
                                self._record_error(result_event.error or "result analysis failed")
                            yield result_event
                elif plan is not None and not emitted_text and plan.summary.strip():
                    # JSON-only planners still need a concise user-facing
                    # answer for ordinary conversation without tool access.
                    yield ProviderEvent.text_chunk(plan.summary.strip())
                needs_attention = bool(
                    self.state.errors
                    or any(not result.ok for result in results)
                    or not self._results_verified(results)
                )
                self._set_state(
                    status="failed" if needs_attention else "completed",
                    active_step_id="",
                    pending_tool_calls=(),
                    next_action="",
                    verification_status="needs_attention" if needs_attention else "complete",
                    verification_complete=not needs_attention if self.state.verification_required else True,
                )
                yield event
                return
        finally:
            with self._pending_lock:
                stale = [
                    approval_id
                    for approval_id, pending in self._pending.items()
                    if pending.approval.approval_id.startswith(task_id)
                ]
                for approval_id in stale:
                    pending = self._pending.pop(approval_id)
                    pending.resolved.set()
            if cancel_event.is_set() and self.state.status not in {"cancelled", "completed"}:
                self._set_state(status="cancelled", next_action="", verification_status="cancelled")
            self._active_lock.release()

    def _plan_stream(
        self,
        request: str,
        cancel_event: Event,
        conversation: tuple[ChatMessage, ...],
    ) -> Iterator[ProviderEvent]:
        """Call provider implementations with or without conversation history."""
        try:
            parameters = inspect.signature(self.provider.stream_plan).parameters
            supports_history = "conversation" in parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        except (TypeError, ValueError):
            supports_history = True
        if supports_history:
            return self.provider.stream_plan(request, cancel_event, conversation)
        return self.provider.stream_plan(request, cancel_event)

    def _requires_approval(self, request: ToolRequest) -> bool:
        """Ask the registry before showing a permission prompt.

        Invalid requests are allowed to reach the registry's structured error
        path; only a valid request denied solely for authorization is shown to
        the user as an approval decision.
        """
        try:
            self.registry.validate(request, confirmation=False)
        except PermissionError:
            return True
        except (TypeError, ValueError):
            return False
        return False

    def _create_approval(self, task_id: str, request: ToolRequest) -> _PendingApproval:
        definition = self.registry.get(request.name)
        if definition is None:
            raise RuntimeError("cannot create approval for an unknown tool")
        # Prefixing with the task id prevents an approval from being reused by
        # a later request even if a caller accidentally retains the identifier.
        approval = ToolApproval(
            approval_id=f"{task_id}:{uuid.uuid4().hex}",
            request=request,
            display_name=definition.display_name or definition.name,
            permission_level=definition.permission_level,
            description=definition.description,
        )
        pending = _PendingApproval(approval)
        with self._pending_lock:
            self._pending[approval.approval_id] = pending
        return pending

    def _results_verified(self, results: list[ToolResult]) -> bool:
        """Require explicit handler verification for changing operations."""
        for result in results:
            if not result.ok:
                return False
            definition = self.registry.get(result.tool_name)
            if definition is None or definition.permission_level not in {
                PermissionLevel.WRITE,
                PermissionLevel.DESTRUCTIVE,
            }:
                continue
            if not isinstance(result.data, dict) or result.data.get("verified") is not True:
                return False
        return True

    def _record_error(self, message: str) -> None:
        state = self.state
        self._set_state(errors=(*state.errors, message[:500]), status="failed")

    def _record_tool_event(self, tool_event: ToolEvent, results: list[ToolResult]) -> None:
        state = self.state
        if tool_event.kind is ToolEventKind.STARTED and tool_event.tool_name not in state.executed_tools:
            self._set_state(executed_tools=(*state.executed_tools, tool_event.tool_name))
        if tool_event.result is not None and tool_event.result not in results:
            results.append(tool_event.result)
            state = self.state
            changes: dict[str, object] = {
                "observations": (*state.observations, tool_event.result),
            }
            if not tool_event.result.ok:
                changes["errors"] = (*state.errors, (tool_event.result.error_code or "tool_failed")[:500])
            self._set_state(**changes)
