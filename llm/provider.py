"""Provider and safe structured-plan contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import re
from threading import Event
from typing import Iterator

from troubleshooting.contracts import FixProposal, TroubleshootingStageEvent
from software.contracts import SoftwarePlan, SoftwareState
from tools.contracts import ToolApproval, ToolEvent, ToolRequest, ToolResult


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One role/content pair sent to a local conversational provider."""

    role: str
    content: str


class ChatProvider(ABC):
    """Backend-neutral streaming chat interface for the active UI."""

    @abstractmethod
    def stream_chat(
        self,
        messages: tuple[ChatMessage, ...],
        cancel_event: Event,
    ) -> Iterator["ProviderEvent"]:
        """Stream safe status/text/error events for a conversation."""


_ALLOWED_ACTION_KINDS = frozenset(
    {
        "inspect_system",
        "read_file",
        "check_connection",
        "check_network",
        "install_package",
        "remove_package",
        "run_command",
        "review_request",
    }
)
_TOOL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True, slots=True)
class PlanAction:
    """A typed future action; it is descriptive and never executable here."""

    kind: str
    description: str
    target: str = ""
    requires_confirmation: bool = True


@dataclass(frozen=True, slots=True)
class Plan:
    summary: str
    actions: tuple[PlanAction, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    tool_requests: tuple[ToolRequest, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, value: object, request: str) -> "Plan":
        """Validate model JSON and discard fields outside the future plan API."""
        if not isinstance(value, dict):
            return cls(summary=f"Review request: {request.strip()}")

        summary = str(value.get("summary", "Review the request safely.")).strip()
        actions: list[PlanAction] = []
        raw_actions = value.get("actions", [])
        if isinstance(raw_actions, list):
            for raw_action in raw_actions[:8]:
                if not isinstance(raw_action, dict):
                    continue
                kind = str(raw_action.get("kind", "review_request")).strip()
                if kind not in _ALLOWED_ACTION_KINDS:
                    kind = "review_request"
                description = str(
                    raw_action.get("description", "Review the requested action.")
                ).strip()
                target = str(raw_action.get("target", "")).strip()
                actions.append(
                    PlanAction(
                        kind=kind,
                        description=description[:240],
                        target=target[:160],
                        requires_confirmation=bool(
                            raw_action.get("requires_confirmation", True)
                        ),
                    )
                )
        if not actions:
            actions.append(
                PlanAction(
                    kind="review_request",
                    description="Review the request before any future action.",
                )
            )

        raw_notes = value.get("notes", [])
        notes = tuple(str(note).strip()[:240] for note in raw_notes[:5]) if isinstance(raw_notes, list) else ()
        requests: list[ToolRequest] = []
        raw_requests = value.get("tool_requests", [])
        if isinstance(raw_requests, list):
            for raw_request in raw_requests[:8]:
                # Qwen sometimes compresses a no-argument tool call to its
                # name under constrained JSON generation. Normalize that
                # harmless shorthand; the registry remains authoritative.
                if isinstance(raw_request, str):
                    name = raw_request.strip()[:64]
                    if _TOOL_NAME.fullmatch(name):
                        requests.append(ToolRequest(name=name, arguments={}))
                    continue
                if not isinstance(raw_request, dict):
                    continue
                name = str(raw_request.get("name", "")).strip()[:100]
                arguments = raw_request.get("arguments", {})
                if not name or not _TOOL_NAME.fullmatch(name) or not isinstance(arguments, dict):
                    continue
                # Keep model-controlled data bounded before it reaches the
                # registry. The registry still performs the authoritative
                # schema and permission checks.
                try:
                    encoded_arguments = json.dumps(
                        arguments,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                except (TypeError, ValueError, OverflowError):
                    continue
                if len(encoded_arguments) > 32_000:
                    continue
                requests.append(
                    ToolRequest(
                        name=name,
                        arguments=json.loads(encoded_arguments),
                        requires_confirmation=bool(
                            raw_request.get("requires_confirmation", True)
                        ),
                    )
                )
        return cls(
            summary=summary[:400],
            actions=tuple(actions),
            notes=notes,
            tool_requests=tuple(requests),
        )


@dataclass(frozen=True, slots=True)
class DiagnosticDecision:
    """One bounded next step selected from the read-only diagnostic catalog."""

    done: bool = False
    reason: str = ""
    tool_requests: tuple[ToolRequest, ...] = field(default_factory=tuple)
    available: bool = True


@dataclass(frozen=True, slots=True)
class SoftwareRecoveryDecision:
    """Closed Qwen decision used only to gate a trusted recovery lookup."""

    action: str = "unavailable"
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ProviderEvent:
    """One safe event from a streaming local provider."""

    kind: str
    text: str = ""
    plan: Plan | None = None
    error: str = ""
    tool_event: ToolEvent | None = None
    tool_result: ToolResult | None = None
    stage_event: TroubleshootingStageEvent | None = None
    fix_proposal: FixProposal | None = None
    software_plan: SoftwarePlan | None = None
    software_state: SoftwareState | None = None
    tool_approval: ToolApproval | None = None
    software_recovery: SoftwareRecoveryDecision | None = None
    diagnostic_decision: DiagnosticDecision | None = None

    @classmethod
    def status(cls, text: str) -> "ProviderEvent":
        return cls(kind="status", text=text)

    @classmethod
    def text_chunk(cls, text: str) -> "ProviderEvent":
        return cls(kind="text", text=text)

    @classmethod
    def plan_ready(cls, plan: Plan) -> "ProviderEvent":
        return cls(kind="plan", plan=plan)

    @classmethod
    def done(cls) -> "ProviderEvent":
        return cls(kind="done")

    @classmethod
    def failure(cls, message: str) -> "ProviderEvent":
        return cls(kind="error", error=message)

    @classmethod
    def tool_update(cls, event: ToolEvent) -> "ProviderEvent":
        return cls(kind="tool", tool_event=event, tool_result=event.result)

    @classmethod
    def stage_update(cls, event: TroubleshootingStageEvent) -> "ProviderEvent":
        return cls(kind="stage", stage_event=event)

    @classmethod
    def fix_ready(cls, proposal: FixProposal) -> "ProviderEvent":
        return cls(kind="fix", fix_proposal=proposal)

    @classmethod
    def software_plan_ready(cls, plan: SoftwarePlan) -> "ProviderEvent":
        return cls(kind="software_plan", software_plan=plan)

    @classmethod
    def software_state_ready(cls, state: SoftwareState) -> "ProviderEvent":
        return cls(kind="software_state", software_state=state)

    @classmethod
    def tool_approval_ready(cls, approval: ToolApproval) -> "ProviderEvent":
        return cls(kind="tool_approval", tool_approval=approval)

    @classmethod
    def software_recovery_ready(cls, decision: SoftwareRecoveryDecision) -> "ProviderEvent":
        return cls(kind="software_recovery", software_recovery=decision)

    @classmethod
    def diagnostic_decision_ready(cls, decision: DiagnosticDecision) -> "ProviderEvent":
        return cls(kind="diagnostic_decision", diagnostic_decision=decision)


class LLMProvider(ABC):
    """Backend-independent interface consumed by the desktop UI."""

    @abstractmethod
    def stream_plan(
        self,
        request: str,
        cancel_event: Event,
        conversation: tuple[ChatMessage, ...] = (),
    ) -> Iterator[ProviderEvent]:
        """Stream user-safe statuses/text and finish with a validated plan."""

    def stream_tool_results(
        self,
        request: str,
        plan: Plan,
        results: tuple[ToolResult, ...],
        cancel_event: Event,
    ) -> Iterator[ProviderEvent]:
        """Optionally turn registry results into a streamed local answer.

        The default is intentionally empty so alternate providers can support
        plan-only operation without inheriting an execution mechanism.
        """
        return iter(())

    def stream_software_failure(
        self,
        request: str,
        operation: str,
        attempted_action: str,
        result: ToolResult,
        alternatives: tuple[str, ...],
        cancel_event: Event,
    ) -> Iterator["ProviderEvent"]:
        """Optionally ask the model whether a trusted software retry is useful."""
        del request, operation, attempted_action, result, alternatives, cancel_event
        return iter(())

    def stream_diagnostic_decision(
        self,
        request: str,
        category: str,
        observations: tuple[ToolResult, ...],
        previous_tools: tuple[str, ...],
        cancel_event: Event,
    ) -> Iterator["ProviderEvent"]:
        """Optionally choose the next safe read-only diagnostic tool."""
        del request, category, observations, previous_tools, cancel_event
        return iter(())


_PLAN_MARKER = "PLAN_JSON:"
_ASSISTANT_MARKER = "ASSISTANT:"


def visible_response(raw_output: str, *, final: bool = False) -> str:
    """Expose assistant text while withholding planner protocol and thoughts."""
    if _ASSISTANT_MARKER not in raw_output:
        return ""
    visible = raw_output.split(_ASSISTANT_MARKER, 1)[1]
    marker_position = visible.find(_PLAN_MARKER)
    if marker_position >= 0:
        visible = visible[:marker_position]
    elif not final:
        # Streaming chunks can split PLAN_JSON across several events. Hold
        # back a trailing marker prefix so protocol text never reaches chat.
        for size in range(len(_PLAN_MARKER) - 1, 0, -1):
            if visible.endswith(_PLAN_MARKER[:size]):
                visible = visible[:-size]
                break
    visible = re.sub(r"<think>.*?</think>", "", visible, flags=re.DOTALL | re.IGNORECASE)
    if "<think>" in visible.lower():
        visible = re.split(r"<think>", visible, maxsplit=1, flags=re.IGNORECASE)[0]
    return visible.strip()


def visible_answer(raw_output: str, *, final: bool = False) -> str:
    """Return safe result-review text with or without the optional marker."""
    if _ASSISTANT_MARKER in raw_output:
        return visible_response(raw_output, final=final)
    visible = raw_output
    if _PLAN_MARKER in visible:
        visible = visible.split(_PLAN_MARKER, 1)[0]
    visible = re.sub(r"<think>.*?</think>", "", visible, flags=re.DOTALL | re.IGNORECASE)
    if "<think>" in visible.lower():
        visible = re.split(r"<think>", visible, maxsplit=1, flags=re.IGNORECASE)[0]
    return visible.strip()


def parse_plan_response(raw_output: str, request: str, response_text: str) -> Plan:
    """Parse one provider response into the validated backend-neutral plan."""
    payload = (
        raw_output.split(_PLAN_MARKER, 1)[1].strip()
        if _PLAN_MARKER in raw_output
        else raw_output.strip()
    )
    payload = payload.removeprefix("```").removesuffix("```").strip()
    try:
        value, _end = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError:
        return Plan(summary=response_text.strip()[:400] or "Review the request safely.")
    return Plan.from_dict(value, request) if isinstance(value, dict) else Plan(
        summary=response_text.strip()[:400] or "Review the request safely."
    )


def model_safe_result(result: dict[str, object], *, limit: int = 8_000) -> dict[str, object]:
    """Bound local observations before either provider places them in context."""
    bounded = dict(result)
    if "data" in bounded:
        bounded["data"] = _cap_model_value(bounded["data"], limit)
    return bounded


def _cap_model_value(value: object, limit: int) -> object:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "… [truncated]"
    if isinstance(value, list):
        return [_cap_model_value(item, limit) for item in value[:80]]
    if isinstance(value, dict):
        return {
            str(key): _cap_model_value(item, limit)
            for key, item in list(value.items())[:80]
        }
    return value
