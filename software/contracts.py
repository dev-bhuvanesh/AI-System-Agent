"""Data contracts for the Linux software agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from tools.contracts import ToolRequest


class SoftwareOperation(StrEnum):
    INSTALL = "install"
    DOWNLOAD = "download"
    UPDATE = "update"
    UPGRADE = "upgrade"
    REMOVE = "remove"
    REINSTALL = "reinstall"
    SEARCH = "search"
    INSTALLED_VERSION = "installed_version"
    AVAILABLE_VERSION = "available_version"
    VERIFY = "verify"


class SoftwareSource(StrEnum):
    PACKAGE_REPOSITORY = "trusted package repository"
    FLATPAK = "Flatpak / Flathub"
    SNAP = "Snap Store"
    OFFICIAL_VENDOR = "official vendor source"


class SoftwareErrorCode(StrEnum):
    """Stable error categories returned by package-management tools."""

    PACKAGE_NOT_FOUND = "PACKAGE_NOT_FOUND"
    NETWORK_ERROR = "NETWORK_ERROR"
    PERMISSION_ERROR = "PERMISSION_ERROR"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    INSTALLATION_ERROR = "INSTALLATION_ERROR"


@dataclass(frozen=True, slots=True)
class SoftwareRequest:
    operation: SoftwareOperation
    software_name: str
    original_request: str
    scope_all: bool = False


@dataclass(frozen=True, slots=True)
class SystemProfile:
    distribution: str
    distribution_id: str
    version: str
    architecture: str
    package_manager: str
    available_managers: tuple[str, ...] = field(default_factory=tuple)
    sources: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class SoftwareState:
    """Current package state used to decide which actions are safe to offer."""

    state_id: str
    software_name: str
    package_name: str
    manager: str
    source: SoftwareSource
    installed: bool
    current_version: str = ""
    available_version: str = ""
    update_available: bool | None = None
    actions: tuple[SoftwareOperation, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class SoftwarePlan:
    plan_id: str
    software_name: str
    operation: SoftwareOperation
    source: SoftwareSource
    package_name: str
    manager: str
    architecture: str
    version: str = ""
    package_size: str = "Not reported by the source"
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    command_preview: str = ""
    details: str = ""
    requires_confirmation: bool = True
    request: ToolRequest = field(default_factory=lambda: ToolRequest("software_install", {}))
    current_version: str = ""
    available_version: str = ""
    risk: str = "Medium"
    what_will_do: str = ""
