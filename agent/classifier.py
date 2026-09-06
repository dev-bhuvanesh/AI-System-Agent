"""Fast, deterministic routing for conversational and system requests.

The classifier intentionally runs locally and synchronously before any task
UI is created.  It does not call the model: classification must not introduce
another visible loading state or a second inference request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from software.catalog import find_spec


class RequestType(StrEnum):
    CONVERSATION = "conversation"
    SYSTEM_TASK = "system_task"


@dataclass(frozen=True, slots=True)
class RequestClassification:
    """The only routing result the chat surface needs."""

    type: RequestType
    requires_tools: bool

    @property
    def is_system_task(self) -> bool:
        return self.type is RequestType.SYSTEM_TASK

    def as_dict(self) -> dict[str, object]:
        return {
            "type": self.type.value,
            "requires_tools": self.requires_tools,
        }


_EXPLANATION = re.compile(
    r"^(?:please\s+)?(?:what(?:'s| is| are| does)\b|why\s+does\b|"
    r"how\s+does\b|explain\b|define\b|meaning\s+of\b|"
    r"tell\s+me\s+about\b|describe\b)"
)
_INSTRUCTIONS = re.compile(
    r"^(?:please\s+)?(?:how\s+(?:do|can|would)\s+i\b|"
    r"show\s+me\s+how\b|steps?\s+to\b|guide\s+me\b|"
    r"instructions?\s+(?:for|to)\b)"
)
_INSTRUCTIONAL_PHRASE = re.compile(
    r"\b(?:how\s+to|tell\s+me\s+how|show\s+me\s+how|steps?\s+to|"
    r"instructions?\s+(?:for|to))\b"
)
_ACTION = re.compile(
    r"\b(?:install|download|update|upgrade|remove|uninstall|reinstall|"
    r"delete|create|rename|copy|move|restart|start|stop|enable|disable|"
    r"repair|fix|configure|troubleshoot|diagnose|inspect|check|list|read|run|"
    r"execute|ping|test|verify|monitor|find|show)\b"
)
_SYSTEM_TARGET = re.compile(
    r"\b(?:my|this|current|the)\s+(?:system|computer|pc|laptop|machine|"
    r"network|wifi|wi-fi|wireless|internet|ethernet|dns|bluetooth|audio|"
    r"sound|display|screen|monitor|gpu|graphics|usb|printer|battery|power|"
    r"cpu|processor|ram|memory|disk|storage|drive|filesystem|file\s+system|"
    r"service|process|firewall|kernel|hardware|logs?)\b"
)
_SYSTEM_NOUN = re.compile(
    r"\b(?:system|computer|pc|laptop|machine|network|wifi|wi-fi|wireless|"
    r"internet|ethernet|dns|bluetooth|audio|sound|display|screen|monitor|"
    r"gpu|graphics|usb|printer|battery|power|cpu|processor|ram|memory|disk|"
    r"storage|drive|filesystem|file\s+system|file|folder|directory|service|"
    r"process|firewall|kernel|hardware|package|software|application|app|"
    r"logs?)\b"
)
_CURRENT_QUERY = re.compile(
    r"\b(?:what\s+(?:is|are)\s+(?:using|running|installed|connected|"
    r"causing)|how\s+much\s+(?:cpu|ram|memory|disk|storage)|"
    r"(?:is|are)\s+(?:my|the)\s+[^?]*(?:status|state|working|running|"
    r"connected|installed|available|normal|usage|load|temperature|"
    r"version|address|configuration|devices|files|processes)\b|"
    r"what\s+(?:is|are)\s+(?:my|current)\s+(?:system|computer|pc|"
    r"laptop|machine|network|wifi|wi-fi|bluetooth|cpu|processor|ram|"
    r"memory|disk|storage|drive|service|process|hardware)\b)"
)
_SYMPTOM = re.compile(
    r"\b(?:not\s+working|doesn['’]?t\s+work|won['’]?t\s+work|"
    r"cannot|can['’]?t|failed|failure|error|problem|issue|broken|"
    r"offline|disconnected|disconnection|slow|freezing|frozen|crash|"
    r"hanging|hangs|missing|unavailable|denied)\b"
)
_SOFTWARE_ACTION = re.compile(
    r"\b(?:install|download|update|upgrade|remove|uninstall|reinstall|"
    r"get|setup|set\s+up|configure|configuration)\b"
)
_SOFTWARE_DIAGNOSTIC_ACTION = re.compile(
    r"^(?:please\s+)?(?:check|troubleshoot|troubleshooting|diagnose|diagnostic|verify)\b"
)
_AGENT_ACTION = re.compile(
    r"\b(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:install|"
    r"download|update|upgrade|remove|uninstall|reinstall|delete|create|"
    r"rename|copy|move|restart|start|stop|enable|disable|repair|fix|configure|"
    r"troubleshoot|diagnose|inspect|check|list|read|run|execute|ping|"
    r"test|verify|monitor|find|show)\b"
)


def _conversation() -> RequestClassification:
    return RequestClassification(RequestType.CONVERSATION, requires_tools=False)


def _system_task() -> RequestClassification:
    return RequestClassification(RequestType.SYSTEM_TASK, requires_tools=True)


def classify_request(request: str) -> RequestClassification:
    """Classify one message before task state or task UI is created.

    Explanation and instructional phrasing is treated as conversation unless
    the same message clearly reports a current symptom or asks the agent to
    inspect the user's machine.  Imperative operations and current-state
    questions are system tasks even when they do not use the word
    ``troubleshoot``.
    """

    text = " ".join(request.casefold().split())
    if not text:
        return _conversation()

    has_system_target = bool(_SYSTEM_TARGET.search(text))
    has_system_noun = bool(_SYSTEM_NOUN.search(text))
    has_current_query = bool(_CURRENT_QUERY.search(text))
    has_symptom = bool(_SYMPTOM.search(text))
    has_action = bool(_ACTION.search(text))

    # A how-to request asks for knowledge, even when it mentions a system
    # operation or a current symptom. It becomes a task only when the user
    # explicitly delegates the action ("for me", "on my system", etc.).
    instructional = bool(_INSTRUCTIONS.match(text) or _INSTRUCTIONAL_PHRASE.search(text))
    delegated = bool(re.search(r"\b(?:for me|on my system|on this (?:computer|machine|laptop))\b", text))
    if instructional and not delegated:
        return _conversation()

    # A request to explain a current failure still needs observation of the
    # actual machine. Pure definitions and how-to questions do not.
    if has_symptom and has_system_noun:
        return _system_task()
    if has_current_query and has_system_noun:
        return _system_task()

    # Named application diagnostics are system tasks even when the request
    # does not contain a generic noun such as "software".  Resolve only the
    # catalog identity here; no package name or command is generated.
    if _SOFTWARE_DIAGNOSTIC_ACTION.match(text) and find_spec(text) is not None:
        return _system_task()

    # "Can you check/install ...?" is an action request. "Can you tell me
    # how to install ...?" is caught by the instructional branch below.
    if _AGENT_ACTION.search(text) and not _INSTRUCTIONS.match(text):
        return _system_task()

    if _EXPLANATION.match(text) or _INSTRUCTIONS.match(text):
        return _conversation()

    # A command-looking imperative, a filesystem operation, or software
    # management request requires the controlled agent path.
    if _SOFTWARE_ACTION.search(text):
        return _system_task()
    if has_action and (has_system_target or has_system_noun):
        return _system_task()
    if re.search(r"(?:`[^`]+`|\$\s+[A-Za-z]|\b(?:sudo|apt|dnf|systemctl|nmcli|"
                 r"ip|ls|cat|mkdir|rm|cp|mv)\b)", text):
        if re.search(r"\b(?:run|execute|use|apply|perform|check|show|read)\b", text):
            return _system_task()

    return _conversation()


__all__ = ["RequestClassification", "RequestType", "classify_request"]
