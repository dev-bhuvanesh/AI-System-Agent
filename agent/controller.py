"""Chat controller that owns conversation history and request lifecycle."""

from __future__ import annotations

import logging
from threading import Event, Lock
from typing import Iterator

from agent.classifier import RequestType, classify_request
from agent.runtime import AssistantRuntime
from llm.provider import ChatMessage, ChatProvider, ProviderEvent
from software.manager import SoftwareManager
from troubleshooting.engine import TroubleshootingEngine


logger = logging.getLogger(__name__)


class AIController:
    """Connect chat UI events to one local provider without OS tool access."""

    def __init__(
        self,
        provider: ChatProvider,
        troubleshooter: TroubleshootingEngine | None = None,
        software_manager: SoftwareManager | None = None,
        runtime: AssistantRuntime | None = None,
    ) -> None:
        self.provider = provider
        self.troubleshooter = troubleshooter
        self.software_manager = software_manager
        self.runtime = runtime
        self._history: list[ChatMessage] = []
        self._active_lock = Lock()

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        return tuple(self._history)

    def stream_response(self, request: str, cancel_event: Event) -> Iterator[ProviderEvent]:
        """Append a user message and stream its local Ollama response."""
        cleaned = request.strip()
        if not cleaned:
            yield ProviderEvent.failure("Please enter a message before sending.")
            return

        if not self._active_lock.acquire(blocking=False):
            yield ProviderEvent.failure("A response is already being generated.")
            return

        try:
            user_message = ChatMessage(role="user", content=cleaned)
            self._history.append(user_message)
            classification = classify_request(cleaned)
            logger.info(
                "AIController request: type=%s requires_tools=%s characters=%d history=%d",
                classification.type.value,
                classification.requires_tools,
                len(cleaned),
                len(self._history),
            )
            response_text = ""
            had_error = False
            if classification.type is RequestType.CONVERSATION:
                # Conversational turns bypass planning, the runtime, and all
                # system tools. The model receives only the chat history.
                events = self.provider.stream_chat(tuple(self._history), cancel_event)
            elif self.software_manager is not None and self.software_manager.matches(cleaned):
                events = self.software_manager.stream(
                    cleaned,
                    cancel_event,
                    tuple(self._history),
                )
            elif self.troubleshooter is not None and self.troubleshooter.matches(cleaned):
                events = self.troubleshooter.stream(
                    cleaned,
                    cancel_event,
                    tuple(self._history),
                )
            elif self.runtime is not None:
                events = self.runtime.stream(
                    cleaned,
                    cancel_event,
                    tuple(self._history),
                )
            else:
                events = self.provider.stream_chat(tuple(self._history), cancel_event)
            event_iterator = iter(events)
            try:
                while not cancel_event.is_set():
                    try:
                        event = next(event_iterator)
                    except StopIteration:
                        break
                    if event.kind == "text":
                        response_text += event.text
                    elif event.kind == "error":
                        had_error = True
                    yield event
            finally:
                # Closing the active generator propagates GeneratorExit into
                # nested agent/tool generators. Do not drain it: resuming a
                # provider after cancellation can produce extra tokens and
                # defeats the Stop contract.
                close = getattr(event_iterator, "close", None)
                if callable(close):
                    close()
            if response_text.strip() and not had_error and not cancel_event.is_set():
                self._history.append(ChatMessage(role="assistant", content=response_text))
                logger.info("AIController stored assistant response: characters=%d", len(response_text))
        finally:
            self._active_lock.release()

    def approve_troubleshooting_fix(self, proposal_id: str, approved: bool) -> bool:
        """Resolve a troubleshooting fix decision from a trusted UI action."""
        return bool(
            self.troubleshooter
            and self.troubleshooter.approve_fix(proposal_id, approved)
        )

    def choose_troubleshooting_action(self, proposal_id: str, action: str) -> bool:
        """Resolve the non-privileged action choice from the trusted UI."""
        return bool(
            self.troubleshooter
            and self.troubleshooter.choose_fix(proposal_id, action)
        )

    def approve_software_action(self, plan_id: str, approved: bool) -> bool:
        """Resolve a software plan decision from a trusted UI button."""
        return bool(
            self.software_manager
            and self.software_manager.approve(plan_id, approved)
        )

    def approve_tool(self, approval_id: str, approved: bool) -> bool:
        """Resolve a generic registry approval from the trusted UI."""
        return bool(self.runtime and self.runtime.approve_tool(approval_id, approved))
