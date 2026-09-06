"""Data-only contracts for the troubleshooting engine and UI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from tools.contracts import ToolRequest


class TroubleshootingCategory(StrEnum):
    NETWORK = "network"
    ETHERNET = "ethernet"
    WIFI = "wifi"
    DNS = "dns"
    BLUETOOTH = "bluetooth"
    AUDIO = "audio"
    DISPLAY_GPU = "display_gpu"
    USB = "usb"
    PRINTER = "printer"
    STORAGE = "storage"
    PACKAGE = "package"
    APPLICATION = "application"
    PERFORMANCE = "performance"
    BOOT = "boot"
    CRASH = "crash"
    SERVICE = "service"
    SYSTEM_UPDATES = "system_updates"
    POWER = "power"
    PERMISSIONS = "permissions"
    FIREWALL = "firewall"
    VPN = "vpn"
    KERNEL = "kernel"
    HARDWARE = "hardware"
    PHYSICAL = "physical"
    SECURITY = "security"
    GENERAL = "general"


class TroubleshootingOutcome(StrEnum):
    NORMAL = "normal"
    SOFTWARE_PROBLEM = "software_problem"
    HARDWARE_PROBLEM = "hardware_problem"
    PHYSICAL_PROBLEM = "physical_problem"
    UNKNOWN = "unknown"


class TroubleshootingStageStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    WARNING = "warning"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TroubleshootingTaskState(StrEnum):
    """Lifecycle of the complete troubleshooting task, not one tool call."""

    IDLE = "IDLE"
    THINKING = "THINKING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class TroubleshootingSessionState:
    """Inspectable state for the whole task represented by the process log."""

    task_id: str = ""
    task_state: TroubleshootingTaskState = TroubleshootingTaskState.IDLE
    active_step_id: str = ""
    active_process_id: int | None = None
    abort_controller: object | None = None
    pending_tool_calls: tuple[str, ...] = ()
    verification_required: bool = False
    verification_complete: bool = False


# Short compatibility name for callers that only need the state contract.
TroubleshootingState = TroubleshootingSessionState


@dataclass(frozen=True, slots=True)
class TroubleshootingStageEvent:
    stage_id: str
    title: str
    status: TroubleshootingStageStatus
    detail: str
    step_type: str = "troubleshooting"
    action: str = ""
    output: str = ""
    error: str = ""
    started_at: float | None = None
    ended_at: float | None = None


@dataclass(frozen=True, slots=True)
class FixProposal:
    """A modifying operation awaiting a trusted user decision."""

    proposal_id: str
    title: str
    rationale: str
    command_preview: str
    effect: str
    request: ToolRequest
    # ``choice`` displays the action choices; ``confirmation`` displays the
    # final Allow/Cancel gate after the user chooses automatic repair.
    mode: str = "choice"
    # ``automatic`` is a software/configuration repair. Hardware and physical
    # findings expose manual guidance and a check-again action instead.
    action_kind: str = "automatic"
    original_request: str = ""
    manual_instructions: str = ""
    # Optional evidence identifiers remain collapsed in the normal chat view.
    technical_details: tuple[str, ...] = ()
