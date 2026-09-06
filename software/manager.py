"""Secure software-management orchestration.

This controller resolves curated applications and explicit package names from
trusted Linux repositories, asks the registry to inspect the local package
sources, and waits for a trusted UI decision before any download or system
change. It deliberately does not accept a command or URL from the model.
"""

from __future__ import annotations

import threading
import time
import uuid
import re
import shlex
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterator

from llm.provider import (
    ChatMessage,
    ChatProvider,
    LLMProvider,
    ProviderEvent,
    SoftwareRecoveryDecision,
)
from software.catalog import SoftwareSpec, parse_request
from software.contracts import (
    SoftwareOperation,
    SoftwareErrorCode,
    SoftwarePlan,
    SoftwareRequest,
    SoftwareSource,
    SoftwareState,
    SystemProfile,
)
from software.history import SoftwareHistory, default_history_path
from software.resolver import query_for, repository_candidates
from tools.contracts import ToolEventKind, ToolRequest, ToolResult
from tools.registry import ToolRegistry
from troubleshooting.contracts import (
    TroubleshootingSessionState,
    TroubleshootingStageEvent,
    TroubleshootingStageStatus,
    TroubleshootingTaskState,
)


_SOFTWARE_CHECK_VERBS = ("check", "troubleshoot", "troubleshooting", "diagnose", "diagnostic")


def _software_diagnostic_scope(request: str) -> str | None:
    """Return the diagnostic scope without inventing an application target.

    The package parser intentionally rejects generic nouns such as ``software``.
    These requests still belong to the system-task route, but they need a
    target clarification instead of a broad, unrelated system check.
    """
    normalized = " ".join(request.casefold().split()).strip(" .!?;:")
    if not normalized:
        return None
    if normalized.startswith("please "):
        normalized = normalized[7:].strip()
    if any(
        marker in normalized
        for marker in (
            "overall software",
            "entire software",
            "whole software",
            "all software",
            "software environment",
            "software on my system",
        )
    ) and any(verb in normalized.split() for verb in _SOFTWARE_CHECK_VERBS):
        return "overall"

    if normalized.startswith("troubleshoot and check "):
        remainder = normalized[len("troubleshoot and check "):].strip()
    else:
        prefixes = (
            "check ",
            "troubleshoot ",
            "troubleshooting ",
            "diagnose ",
            "diagnostic ",
        )
        remainder = next((normalized[len(prefix):].strip() for prefix in prefixes if normalized.startswith(prefix)), None)
    if remainder is None:
        return None
    remainder = re.sub(r"^(?:the|my|this)\s+", "", remainder)
    remainder = re.sub(r"\s+(?:on my system|on my computer|in linux)$", "", remainder)
    return "target" if remainder in {"software", "applications", "apps", "programs", "application"} else None


class SoftwareManager:
    """Turn natural-language software requests into safe registry actions."""

    def __init__(
        self,
        provider: ChatProvider | LLMProvider | None,
        registry: ToolRegistry,
        history: SoftwareHistory | None = None,
    ) -> None:
        self._provider = provider
        self.registry = registry
        self.history = history or SoftwareHistory(default_history_path())
        self._pending_lock = threading.RLock()
        self._pending: dict[str, tuple[threading.Event, list[bool | None]]] = {}
        self._state_lock = threading.RLock()
        self._state = TroubleshootingSessionState()

    @staticmethod
    def parse(request: str) -> tuple[SoftwareRequest, SoftwareSpec] | None:
        return parse_request(request)

    @staticmethod
    def matches(request: str) -> bool:
        return parse_request(request) is not None or _software_diagnostic_scope(request) is not None

    def approve(self, plan_id: str, approved: bool) -> bool:
        with self._pending_lock:
            pending = self._pending.get(plan_id)
            if pending is None:
                return False
            decision_event, decision = pending
            decision[0] = bool(approved)
            decision_event.set()
            return True

    @property
    def state(self) -> TroubleshootingSessionState:
        """Return the latest complete software-task lifecycle state."""
        with self._state_lock:
            return self._state

    def _set_task_state(self, **changes: object) -> None:
        with self._state_lock:
            self._state = replace(self._state, **changes)

    def _begin_task(self, cancel_event: threading.Event) -> str:
        task_id = uuid.uuid4().hex
        self._set_task_state(
            task_id=task_id,
            task_state=TroubleshootingTaskState.THINKING,
            active_step_id="",
            active_process_id=None,
            abort_controller=cancel_event,
            pending_tool_calls=(),
            verification_required=False,
            verification_complete=False,
        )
        return task_id

    def stream(
        self,
        request: str,
        cancel_event: threading.Event,
        conversation: tuple[ChatMessage, ...] = (),
    ) -> Iterator[ProviderEvent]:
        del conversation  # The deterministic software boundary needs no hidden model state.
        task_id = self._begin_task(cancel_event)
        diagnostic_scope = _software_diagnostic_scope(request)
        if diagnostic_scope == "target":
            yield from self._stream_missing_diagnostic_target(request, task_id)
            return
        if diagnostic_scope == "overall":
            yield from self._stream_overall_diagnostics(request, cancel_event, task_id)
            return
        parsed = parse_request(request)
        if parsed is None:
            yield ProviderEvent.failure("I could not identify supported software in that request.")
            return
        software_request, spec = parsed
        started = time.monotonic()
        record: dict[str, Any] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "request": request.strip(),
            "software": software_request.software_name,
            "operation": software_request.operation.value,
            "source": None,
            "manager": None,
            "package": None,
            "installed": None,
            "current_version": "",
            "available_version": "",
            "update_available": None,
            "status": "in_progress",
            "task_id": task_id,
            "task_state": TroubleshootingTaskState.THINKING.value,
            "verification_required": False,
            "verification_complete": False,
        }
        try:
            self._set_task_state(task_state=TroubleshootingTaskState.PLANNING)
            record["task_state"] = TroubleshootingTaskState.PLANNING.value
            yield self._stage("understand", "Understanding request", TroubleshootingStageStatus.IN_PROGRESS, "Classifying the software operation")
            yield self._stage("understand", "Understanding request", TroubleshootingStageStatus.COMPLETED, f"Operation: {software_request.operation.value}")
            yield self._stage("identify", "Identifying software", TroubleshootingStageStatus.IN_PROGRESS, "Matching the requested application")
            yield self._stage("identify", "Identifying software", TroubleshootingStageStatus.COMPLETED, spec.display_name)

            yield self._stage("system", "Detecting system", TroubleshootingStageStatus.IN_PROGRESS, "Checking distribution, architecture, and package managers")
            profile_result = yield from self._run_tool(ToolRequest("software_system_profile", {}), cancel_event)
            if not profile_result or not profile_result.ok:
                yield self._stage("system", "Detecting system", TroubleshootingStageStatus.FAILED, "System profile unavailable")
                yield ProviderEvent.text_chunk("I could not detect the Linux distribution or package manager safely.")
                yield ProviderEvent.done()
                return
            profile = _profile_from_result(profile_result)
            record["manager"] = profile.package_manager
            yield self._stage("system", "Detecting system", TroubleshootingStageStatus.COMPLETED, f"{profile.distribution or 'Linux'} · {profile.architecture} · {profile.package_manager or 'no package manager'}")

            if cancel_event.is_set():
                return
            candidates = _candidate_options(spec, profile)
            if software_request.scope_all and profile.package_manager:
                candidates = [
                    (profile.package_manager, "all", SoftwareSource.PACKAGE_REPOSITORY)
                ]
            discovery_result: ToolResult | None = None
            discovery_had_network_error = False
            if not candidates and not software_request.scope_all:
                yield self._stage(
                    "source",
                    "Searching software sources",
                    TroubleshootingStageStatus.IN_PROGRESS,
                    f"Searching trusted repositories for {spec.display_name}",
                )
                discovered, discovery_result, discovery_had_network_error = yield from self._discover_candidate(
                    spec, profile, cancel_event
                )
                if discovered is not None:
                    candidates = [discovered]
            candidate = candidates[0] if candidates else None
            if candidate is None:
                status = TroubleshootingStageStatus.WARNING if discovery_had_network_error else TroubleshootingStageStatus.FAILED
                detail = "Repository search was unavailable" if discovery_had_network_error else "No trusted package matched the requested name"
                yield self._stage("source", "Searching software sources", status, detail)
                message = (
                    f"I could not reach a trusted repository to resolve {spec.display_name}. No package action was prepared."
                    if discovery_had_network_error
                    else f"I could not find {spec.display_name} in the detected trusted repositories or official stores. No package action was prepared."
                )
                yield ProviderEvent.text_chunk(message)
                yield ProviderEvent.done()
                return
            manager, package, source = candidate
            record["source"] = source.value
            record["package"] = package

            if software_request.operation is SoftwareOperation.DOWNLOAD and (
                manager not in {"apt", "dnf"}
                or source not in {SoftwareSource.PACKAGE_REPOSITORY, SoftwareSource.OFFICIAL_VENDOR}
            ):
                yield self._stage(
                    "source",
                    "Searching software sources",
                    TroubleshootingStageStatus.FAILED,
                    "This source does not provide a supported package download",
                )
                yield ProviderEvent.text_chunk(
                    f"I found {spec.display_name}, but this system source cannot safely save a package download."
                )
                yield ProviderEvent.done()
                return

            installed = False
            current_version = ""
            available_version = ""
            update_available: bool | None = None
            if software_request.operation is not SoftwareOperation.SEARCH and not software_request.scope_all:
                yield self._stage(
                    "state",
                    "Checking software status",
                    TroubleshootingStageStatus.IN_PROGRESS,
                    f"Checking whether {spec.display_name} is installed",
                )
                state_result = yield from self._run_tool(
                    ToolRequest("software_query", {"manager": manager, "package": package}),
                    cancel_event,
                )
                if state_result is None or not state_result.ok or not isinstance(state_result.data, dict):
                    yield self._stage(
                        "state",
                        "Checking software status",
                        TroubleshootingStageStatus.FAILED,
                        "Installed state could not be confirmed; no package action was prepared",
                    )
                    yield ProviderEvent.text_chunk(
                        f"I could not safely determine whether {spec.display_name} is installed, so I did not prepare a package command."
                    )
                    yield ProviderEvent.done()
                    return
                state_data = state_result.data
                installed = bool(state_data.get("installed", False))
                current_version = str(state_data.get("version", "")).strip()

                # A package may be installed through Snap or Flatpak even
                # when the primary distro manager is apt/dnf. Probe only the
                # fixed catalog candidates, never arbitrary package names.
                if not installed:
                    for alternate_manager, alternate_package, alternate_source in candidates[1:]:
                        alternate_result = yield from self._run_tool(
                            ToolRequest(
                                "software_query",
                                {"manager": alternate_manager, "package": alternate_package},
                            ),
                            cancel_event,
                        )
                        if (
                            alternate_result is not None
                            and alternate_result.ok
                            and isinstance(alternate_result.data, dict)
                            and alternate_result.data.get("installed", False)
                        ):
                            manager, package, source = alternate_manager, alternate_package, alternate_source
                            state_result = alternate_result
                            state_data = alternate_result.data
                            installed = True
                            current_version = str(state_data.get("version", "")).strip()
                            record["manager"] = manager
                            record["source"] = source.value
                            record["package"] = package
                            break
                record["installed"] = installed
                record["current_version"] = current_version
                yield self._stage(
                    "state",
                    "Checking software status",
                    TroubleshootingStageStatus.COMPLETED,
                    _installed_detail(spec.display_name, installed, current_version),
                )

                if installed and software_request.operation in {
                    SoftwareOperation.DOWNLOAD,
                    SoftwareOperation.INSTALL,
                    SoftwareOperation.UPDATE,
                    SoftwareOperation.UPGRADE,
                }:
                    yield self._stage(
                        "version",
                        "Checking available version",
                        TroubleshootingStageStatus.IN_PROGRESS,
                        "Checking the trusted source for an update",
                    )
                    available_result = yield from self._run_tool(
                        ToolRequest("software_available_version", {"manager": manager, "package": package}),
                        cancel_event,
                    )
                    if available_result is not None and available_result.ok and isinstance(available_result.data, dict):
                        available_version = str(available_result.data.get("version", "")).strip()
                        update_available = _is_newer_version(current_version, available_version)
                        record["available_version"] = available_version
                        record["update_available"] = update_available
                        yield self._stage(
                            "version",
                            "Checking available version",
                            TroubleshootingStageStatus.COMPLETED,
                            _available_detail(available_version, update_available),
                        )
                    else:
                        yield self._stage(
                            "version",
                            "Checking available version",
                            TroubleshootingStageStatus.WARNING,
                            "The trusted source did not return a candidate version",
                        )

                state = _make_state(
                    spec,
                    manager,
                    package,
                    source,
                    installed,
                    current_version,
                    available_version,
                    update_available,
                    software_request,
                )
                record["state_id"] = state.state_id

                if software_request.operation in {SoftwareOperation.DOWNLOAD, SoftwareOperation.INSTALL} and installed:
                    yield ProviderEvent.software_state_ready(state)
                    yield ProviderEvent.text_chunk(_already_installed_text(spec.display_name, current_version, update_available))
                    yield ProviderEvent.done()
                    record["status"] = "completed"
                    return
                if software_request.operation in {SoftwareOperation.UPDATE, SoftwareOperation.UPGRADE}:
                    if not installed:
                        yield ProviderEvent.software_state_ready(state)
                        yield ProviderEvent.text_chunk(
                            f"{spec.display_name} is not installed on this system. No update command was run."
                        )
                        yield ProviderEvent.done()
                        record["status"] = "completed"
                        return
                    if update_available is False:
                        yield ProviderEvent.software_state_ready(state)
                        yield ProviderEvent.text_chunk(
                            f"✅ Everything is normal. No problems were detected.\n\n"
                            f"{spec.display_name} is already up to date. No update is required."
                        )
                        yield ProviderEvent.done()
                        record["status"] = "completed"
                        return
                    if update_available is None:
                        yield self._stage(
                            "version",
                            "Checking available version",
                            TroubleshootingStageStatus.FAILED,
                            "No update command was prepared because the candidate version is unknown",
                        )
                        yield ProviderEvent.text_chunk(
                            f"I could not safely determine whether an update is available for {spec.display_name}, so I did not run an update."
                        )
                        yield ProviderEvent.done()
                        return
                if software_request.operation is SoftwareOperation.REMOVE and not installed:
                    yield ProviderEvent.software_state_ready(state)
                    yield ProviderEvent.text_chunk(f"{spec.display_name} is not installed on this system. No removal command was run.")
                    yield ProviderEvent.done()
                    record["status"] = "completed"
                    return
                if software_request.operation is SoftwareOperation.REINSTALL and not installed:
                    yield ProviderEvent.software_state_ready(state)
                    yield ProviderEvent.text_chunk(f"{spec.display_name} is not installed on this system, so it cannot be reinstalled.")
                    yield ProviderEvent.done()
                    record["status"] = "completed"
                    return

            if cancel_event.is_set():
                return

            needs_search = software_request.operation in {
                SoftwareOperation.INSTALL,
                SoftwareOperation.DOWNLOAD,
                SoftwareOperation.UPDATE,
                SoftwareOperation.UPGRADE,
                SoftwareOperation.REINSTALL,
                SoftwareOperation.SEARCH,
                SoftwareOperation.AVAILABLE_VERSION,
            } and (not software_request.scope_all)
            search_result: ToolResult | None = None
            if needs_search:
                yield self._stage("source", "Searching software sources", TroubleshootingStageStatus.IN_PROGRESS, f"Checking {source.value}")
                if discovery_result is not None:
                    search_result = discovery_result
                    yield self._stage(
                        "source",
                        "Searching software sources",
                        TroubleshootingStageStatus.COMPLETED,
                        f"Found {package} in {source.value}",
                    )
                elif source is SoftwareSource.OFFICIAL_VENDOR:
                    # Vendor selection is a fixed allowlist decision from the
                    # catalog; no network request is made until confirmation.
                    yield self._stage("source", "Searching software sources", TroubleshootingStageStatus.COMPLETED, "Official vendor source allowlisted")
                else:
                    search_result = yield from self._run_tool(
                        ToolRequest("software_search", {"manager": manager, "query": package}),
                        cancel_event,
                    )
                    if search_result is None or not search_result.ok:
                        error_code = search_result.error_code if search_result else ""
                        if error_code == "NETWORK_ERROR":
                            yield self._stage("source", "Searching software sources", TroubleshootingStageStatus.WARNING, "Repository search was unavailable; no package action was prepared")
                            yield ProviderEvent.text_chunk(f"I could not reach the {source.value} for {spec.display_name}. No package action was prepared.")
                            yield ProviderEvent.done()
                            return
                        yield self._stage("source", "Searching software sources", TroubleshootingStageStatus.WARNING, "Source search failed; checking other trusted sources")
                        alternate = yield from self._find_alternate_candidate(
                            candidates,
                            (manager, package, source),
                            spec,
                            cancel_event,
                        )
                        if alternate is not None:
                            manager, package, source, search_result = alternate
                            record["manager"] = manager
                            record["source"] = source.value
                            record["package"] = package
                            yield self._stage("source", "Searching software sources", TroubleshootingStageStatus.COMPLETED, f"Found {package} in {source.value}")
                        else:
                            yield self._stage("source", "Searching software sources", TroubleshootingStageStatus.FAILED, "Package was not found in the trusted sources")
                            yield ProviderEvent.text_chunk(f"I could not find {spec.display_name} in the trusted repositories or approved installation sources. No package action was prepared.")
                            yield ProviderEvent.done()
                            return
                    else:
                        if (
                            spec.vendor
                            and source is SoftwareSource.PACKAGE_REPOSITORY
                            and manager in {"apt", "dnf"}
                            and _search_has_no_package(search_result, package)
                        ):
                            source = SoftwareSource.OFFICIAL_VENDOR
                            record["source"] = source.value
                            yield self._stage("source", "Searching software sources", TroubleshootingStageStatus.COMPLETED, "Using the allowlisted official vendor source")
                        elif _search_has_no_package(search_result, package):
                            alternate = yield from self._find_alternate_candidate(
                                candidates,
                                (manager, package, source),
                                spec,
                                cancel_event,
                            )
                            if alternate is not None:
                                manager, package, source, search_result = alternate
                                record["manager"] = manager
                                record["source"] = source.value
                                record["package"] = package
                                yield self._stage("source", "Searching software sources", TroubleshootingStageStatus.COMPLETED, f"Found {package} in {source.value}")
                            else:
                                yield self._stage("source", "Searching software sources", TroubleshootingStageStatus.FAILED, "Package was not found in the trusted sources")
                                yield ProviderEvent.text_chunk(f"I could not find {spec.display_name} in the trusted repositories or approved installation sources. No package action was prepared.")
                                yield ProviderEvent.done()
                                return
                        else:
                            yield self._stage("source", "Searching software sources", TroubleshootingStageStatus.COMPLETED, f"Checked {source.value}")
            else:
                yield self._stage("source", "Searching software sources", TroubleshootingStageStatus.COMPLETED, f"Using {source.value}")

            if cancel_event.is_set():
                return
            yield self._stage(
                "method",
                "Finding installation method",
                TroubleshootingStageStatus.IN_PROGRESS,
                f"Selecting a trusted {source.value} method for {profile.architecture}",
            )
            yield self._stage(
                "method",
                "Finding installation method",
                TroubleshootingStageStatus.COMPLETED,
                "Trusted package method selected",
            )

            if software_request.operation in {
                SoftwareOperation.SEARCH,
                SoftwareOperation.INSTALLED_VERSION,
                SoftwareOperation.AVAILABLE_VERSION,
                SoftwareOperation.VERIFY,
            }:
                read_only_status = yield from self._stream_read_only_result(
                    software_request, spec, manager, package, source, cancel_event, search_result
                )
                record["status"] = read_only_status or "completed"
                return

            result, manager, package, source = yield from self._execute_with_recovery(
                software_request,
                spec,
                profile,
                candidates,
                manager,
                package,
                source,
                current_version,
                available_version,
                record,
                cancel_event,
            )
            if result is None:
                return
            if software_request.operation is SoftwareOperation.DOWNLOAD:
                path = _result_path(result)
                download_detail = (
                    f"Already downloaded · {_display_path(path)}"
                    if isinstance(result.data, dict) and result.data.get("already_downloaded")
                    else f"Download completed · {_display_path(path)}"
                )
                yield self._stage("download", "Downloading software", TroubleshootingStageStatus.COMPLETED, download_detail)
                if not path:
                    record["status"] = "failed"
                    yield self._stage("verify", "Verifying download", TroubleshootingStageStatus.FAILED, "The package manager did not report a saved file")
                    yield ProviderEvent.text_chunk(f"The {spec.display_name} download finished, but no saved package path was reported.")
                    yield ProviderEvent.done()
                    return
                yield self._stage("verify", "Verifying download", TroubleshootingStageStatus.IN_PROGRESS, "Checking package metadata without executing the file")
                self._set_task_state(
                    task_state=TroubleshootingTaskState.VERIFYING,
                    active_step_id="verify",
                    verification_required=True,
                    verification_complete=False,
                )
                record["verification_required"] = True
                verification = yield from self._run_tool(
                    ToolRequest("software_verify_download", {"path": path}),
                    cancel_event,
                    stage_id="verify",
                    stage_title="Verifying download",
                )
                verified = bool(verification and verification.ok and isinstance(verification.data, dict) and verification.data.get("verified", False))
                record["path"] = path
                record["bytes"] = result.data.get("bytes") if isinstance(result.data, dict) else None
                yield self._stage("verify", "Verifying download", TroubleshootingStageStatus.COMPLETED if verified else TroubleshootingStageStatus.FAILED, "File verified" if verified else "File saved, but package metadata could not be verified")
                if verified:
                    record["status"] = "completed"
                    yield ProviderEvent.text_chunk(f"The {spec.display_name} package was downloaded and verified. It was not installed.\n\nSaved to: {path}")
                else:
                    record["status"] = "failed"
                    yield ProviderEvent.text_chunk(f"The {spec.display_name} package was downloaded to {path}, but package verification needs attention.")
                self._set_task_state(
                    task_state=TroubleshootingTaskState.COMPLETED if verified else TroubleshootingTaskState.FAILED,
                    active_step_id="",
                    verification_complete=True,
                    pending_tool_calls=(),
                )
                yield ProviderEvent.done()
                return
            yield self._stage("install", _action_title(software_request.operation), TroubleshootingStageStatus.COMPLETED, "Package action completed")
            yield self._stage("verify", "Verifying installation", TroubleshootingStageStatus.IN_PROGRESS, "Checking installed package state")
            self._set_task_state(
                task_state=TroubleshootingTaskState.VERIFYING,
                active_step_id="verify",
                verification_required=True,
                verification_complete=False,
            )
            record["verification_required"] = True
            verify_manager, verify_package = manager, package
            verify = yield from self._run_tool(
                ToolRequest("software_verify", {"manager": verify_manager, "package": verify_package, "executable": spec.executable}),
                cancel_event,
            )
            verified = bool(verify and verify.ok and isinstance(verify.data, dict) and verify.data.get("verified", verify.data.get("installed", False)))
            yield self._stage("verify", "Verifying installation", TroubleshootingStageStatus.COMPLETED if verified else TroubleshootingStageStatus.WARNING, "Installed version confirmed" if verified else "Package action finished; verification needs attention")
            record["status"] = "completed" if verified else "failed"
            self._set_task_state(
                task_state=TroubleshootingTaskState.COMPLETED if verified else TroubleshootingTaskState.FAILED,
                active_step_id="",
                verification_complete=True,
                pending_tool_calls=(),
            )
            if verified:
                yield ProviderEvent.text_chunk(f"{spec.display_name} was successfully {_past_tense(software_request.operation)} through the trusted {source.value}.")
            else:
                yield ProviderEvent.text_chunk(_installation_verification_warning(spec.display_name, spec.executable, verify))
            yield ProviderEvent.done()
        finally:
            if cancel_event.is_set():
                self._set_task_state(
                    task_state=TroubleshootingTaskState.CANCELLED,
                    active_step_id="",
                    pending_tool_calls=(),
                    verification_complete=False,
                )
                record["status"] = "cancelled"
            elif record["status"] == "completed":
                self._set_task_state(
                    task_state=TroubleshootingTaskState.COMPLETED,
                    active_step_id="",
                    pending_tool_calls=(),
                    verification_complete=(
                        self.state.verification_complete
                        or not self.state.verification_required
                    ),
                )
            elif record["status"] == "cancelled":
                self._set_task_state(
                    task_state=TroubleshootingTaskState.CANCELLED,
                    active_step_id="",
                    pending_tool_calls=(),
                )
            elif self.state.task_state not in {
                TroubleshootingTaskState.COMPLETED,
                TroubleshootingTaskState.CANCELLED,
            }:
                self._set_task_state(
                    task_state=TroubleshootingTaskState.FAILED,
                    active_step_id="",
                    pending_tool_calls=(),
                )
                record["status"] = "failed"
            state = self.state
            record["task_state"] = state.task_state.value
            record["verification_required"] = state.verification_required
            record["verification_complete"] = state.verification_complete
            record["duration_ms"] = int((time.monotonic() - started) * 1000)
            try:
                self.history.append(record)
            except OSError:
                pass

    def _stream_missing_diagnostic_target(
        self,
        request: str,
        task_id: str,
    ) -> Iterator[ProviderEvent]:
        """Ask for an application instead of claiming a generic check passed."""
        started = time.monotonic()
        record: dict[str, Any] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "request": request.strip(),
            "software": None,
            "operation": "verify",
            "scope": "target_required",
            "status": "waiting_for_target",
            "task_id": task_id,
            "task_state": TroubleshootingTaskState.WAITING_FOR_CONFIRMATION.value,
        }
        try:
            self._set_task_state(
                task_state=TroubleshootingTaskState.WAITING_FOR_CONFIRMATION,
                active_step_id="identify",
            )
            yield self._stage(
                "understand",
                "Understanding request",
                TroubleshootingStageStatus.COMPLETED,
                "Software diagnostic requested",
            )
            yield self._stage(
                "identify",
                "Identifying software",
                TroubleshootingStageStatus.FAILED,
                "No application name was provided",
            )
            yield ProviderEvent.text_chunk(
                "Which software would you like me to check? Please name the application."
            )
            yield ProviderEvent.done()
        finally:
            if self.state.task_state is not TroubleshootingTaskState.CANCELLED:
                self._set_task_state(
                    task_state=TroubleshootingTaskState.WAITING_FOR_CONFIRMATION,
                    active_step_id="identify",
                )
            record["task_state"] = self.state.task_state.value
            record["verification_required"] = self.state.verification_required
            record["duration_ms"] = int((time.monotonic() - started) * 1000)
            try:
                self.history.append(record)
            except OSError:
                pass

    def _stream_overall_diagnostics(
        self,
        request: str,
        cancel_event: threading.Event,
        task_id: str,
    ) -> Iterator[ProviderEvent]:
        """Inspect the package/software environment without making changes."""
        started = time.monotonic()
        results: dict[str, ToolResult | None] = {}
        record: dict[str, Any] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "request": request.strip(),
            "software": "overall software environment",
            "operation": "verify",
            "scope": "overall",
            "status": "in_progress",
            "task_id": task_id,
            "task_state": TroubleshootingTaskState.THINKING.value,
            "diagnostics": [],
        }
        try:
            self._set_task_state(task_state=TroubleshootingTaskState.PLANNING)
            yield self._stage(
                "understand",
                "Understanding request",
                TroubleshootingStageStatus.COMPLETED,
                "Overall software environment check",
            )
            yield self._stage(
                "system",
                "Detecting system",
                TroubleshootingStageStatus.IN_PROGRESS,
                "Checking the local distribution and package managers",
            )
            profile = yield from self._run_tool(
                ToolRequest("software_system_profile", {}), cancel_event
            )
            results["software_system_profile"] = profile
            if profile is None or not profile.ok:
                yield self._stage(
                    "system",
                    "Detecting system",
                    TroubleshootingStageStatus.FAILED,
                    "The Linux software profile could not be read",
                )
                yield ProviderEvent.text_chunk(
                    "I could not safely inspect the software environment, so I cannot determine whether it is healthy."
                )
                record["status"] = "failed"
                yield ProviderEvent.done()
                return
            profile_data = profile.data if isinstance(profile.data, dict) else {}
            manager = str(profile_data.get("package_manager", "") or "unknown")
            yield self._stage(
                "system",
                "Detecting system",
                TroubleshootingStageStatus.COMPLETED,
                f"{profile_data.get('distribution', 'Linux')} · {manager}",
            )

            checks = (
                (
                    "packages",
                    "Checking package health",
                    ToolRequest("package_health", {}),
                ),
                (
                    "updates",
                    "Checking software updates",
                    ToolRequest("package_update_status", {}),
                ),
                (
                    "services",
                    "Checking failed software services",
                    ToolRequest("service_failures", {}),
                ),
                (
                    "failures",
                    "Checking recent software errors",
                    ToolRequest("recent_failures", {}),
                ),
            )
            for stage_id, title, tool_request in checks:
                if cancel_event.is_set():
                    return
                yield self._stage(stage_id, title, TroubleshootingStageStatus.IN_PROGRESS, "Read-only check in progress")
                self._set_task_state(
                    task_state=TroubleshootingTaskState.EXECUTING,
                    active_step_id=stage_id,
                    pending_tool_calls=(tool_request.name,),
                )
                result = yield from self._run_diagnostic_tool(tool_request, cancel_event)
                results[tool_request.name] = result
                if result is not None and isinstance(result.data, dict):
                    record["diagnostics"].append(result.as_dict())
                healthy = _overall_check_health(tool_request.name, result)
                stage_status = TroubleshootingStageStatus.COMPLETED if healthy is not False else TroubleshootingStageStatus.FAILED
                detail = "Check passed" if healthy is not False else "The check reported a problem"
                yield self._stage(stage_id, title, stage_status, detail)
                self._set_task_state(pending_tool_calls=())

            if cancel_event.is_set():
                return
            yield self._stage(
                "verify",
                "Verifying software environment",
                TroubleshootingStageStatus.IN_PROGRESS,
                "Comparing package, update, and service results",
            )
            self._set_task_state(
                task_state=TroubleshootingTaskState.VERIFYING,
                active_step_id="verify",
                verification_required=True,
                verification_complete=False,
            )
            problems = _overall_diagnostic_problems(results)
            if problems:
                detail = "; ".join(problems)
                yield self._stage(
                    "verify",
                    "Verifying software environment",
                    TroubleshootingStageStatus.FAILED,
                    detail,
                )
                yield ProviderEvent.text_chunk(
                    "Problem detected in the software environment.\n\n"
                    + "\n".join(f"- {problem}" for problem in problems)
                    + "\n\nNo system changes were made."
                )
                record["status"] = "failed"
                self._set_task_state(
                    task_state=TroubleshootingTaskState.FAILED,
                    active_step_id="",
                    verification_complete=True,
                )
            else:
                yield self._stage(
                    "verify",
                    "Verifying software environment",
                    TroubleshootingStageStatus.COMPLETED,
                    "No package, update, or failed-service problem was detected",
                )
                yield ProviderEvent.text_chunk(
                    "✓ Software check completed\n\n"
                    "✅ Everything is normal. No problems were detected.\n\n"
                    f"- The detected package manager is {manager}.\n"
                    "- Package dependency health is normal.\n"
                    "- No failed software services were reported."
                )
                record["status"] = "completed"
                self._set_task_state(
                    task_state=TroubleshootingTaskState.COMPLETED,
                    active_step_id="",
                    verification_required=True,
                    verification_complete=True,
                )
            record["task_state"] = self.state.task_state.value
            record["verification_required"] = self.state.verification_required
            record["verification_complete"] = self.state.verification_complete
            yield ProviderEvent.done()
        finally:
            if cancel_event.is_set():
                self._set_task_state(
                    task_state=TroubleshootingTaskState.CANCELLED,
                    active_step_id="",
                    pending_tool_calls=(),
                    verification_complete=False,
                )
                record["status"] = "cancelled"
            elif self.state.task_state not in {
                TroubleshootingTaskState.COMPLETED,
                TroubleshootingTaskState.FAILED,
            }:
                self._set_task_state(
                    task_state=TroubleshootingTaskState.FAILED,
                    active_step_id="",
                    pending_tool_calls=(),
                )
                record["status"] = "failed"
            record["task_state"] = self.state.task_state.value
            record["duration_ms"] = int((time.monotonic() - started) * 1000)
            try:
                self.history.append(record)
            except OSError:
                pass

    def _run_diagnostic_tool(
        self,
        request: ToolRequest,
        cancel_event: threading.Event,
    ) -> Iterator[ProviderEvent]:
        """Run a read-only diagnostic definition through the same registry."""
        return (yield from self._run_tool(request, cancel_event, diagnostic=True, software=False))

    def _run_tool(
        self,
        request: ToolRequest,
        cancel_event: threading.Event,
        *,
        confirmation: bool = False,
        stage_id: str | None = None,
        stage_title: str | None = None,
        diagnostic: bool = False,
        software: bool = True,
    ) -> Iterator[ProviderEvent]:
        result: ToolResult | None = None
        self._set_task_state(
            task_state=TroubleshootingTaskState.EXECUTING,
            active_step_id=stage_id or request.name,
            pending_tool_calls=(request.name,),
        )
        try:
            for event in self.registry.execute_stream(
                request,
                cancel_event,
                confirmation=confirmation,
                diagnostic=diagnostic,
                software=software,
            ):
                if event.result is not None:
                    result = event.result
                if stage_id is not None and event.kind is ToolEventKind.STARTED:
                    yield self._stage(
                        stage_id,
                        stage_title or event.display_name,
                        TroubleshootingStageStatus.IN_PROGRESS,
                        "Command started",
                    )
                if event.progress is not None and stage_id is not None:
                    yield self._stage(
                        stage_id,
                        stage_title or event.display_name,
                        TroubleshootingStageStatus.IN_PROGRESS,
                        event.message,
                    )
                if stage_id is not None and event.kind is ToolEventKind.COMPLETED:
                    yield self._stage(
                        stage_id,
                        stage_title or event.display_name,
                        TroubleshootingStageStatus.COMPLETED,
                        _tool_completed_detail(event.result),
                    )
                elif stage_id is not None and event.kind is ToolEventKind.FAILED:
                    yield self._stage(
                        stage_id,
                        stage_title or event.display_name,
                        TroubleshootingStageStatus.FAILED,
                        _tool_failed_detail(event.result),
                    )
                elif stage_id is not None and event.kind is ToolEventKind.CANCELLED:
                    yield self._stage(
                        stage_id,
                        stage_title or event.display_name,
                        TroubleshootingStageStatus.CANCELLED,
                        "Command cancelled",
                    )
                yield ProviderEvent.tool_update(event)
        finally:
            self._set_task_state(active_step_id="", pending_tool_calls=())
        return result

    def _execute_approved(self, plan: SoftwarePlan, cancel_event: threading.Event) -> Iterator[ProviderEvent]:
        if plan.operation is SoftwareOperation.DOWNLOAD and plan.source is SoftwareSource.OFFICIAL_VENDOR:
            download = yield from self._run_tool(plan.request, cancel_event, confirmation=True, stage_id="download", stage_title="Downloading software")
            return download
        if plan.source is SoftwareSource.OFFICIAL_VENDOR and plan.operation in {SoftwareOperation.INSTALL, SoftwareOperation.REINSTALL}:
            download_request = ToolRequest(
                "software_vendor_download",
                {
                    "vendor": plan.request.arguments["vendor"],
                    "manager": plan.request.arguments["manager"],
                    "destination": str(_download_dir()),
                },
            )
            downloaded = yield from self._run_tool(download_request, cancel_event, confirmation=True, stage_id="download", stage_title="Downloading software")
            if downloaded is None or not downloaded.ok:
                return downloaded
            path = str(downloaded.data.get("path", "")) if isinstance(downloaded.data, dict) else ""
            download_detail = (
                f"Already downloaded · {_display_path(path)}"
                if isinstance(downloaded.data, dict) and downloaded.data.get("already_downloaded")
                else f"Download completed · {_display_path(path)}"
            )
            yield self._stage("download", "Downloading software", TroubleshootingStageStatus.COMPLETED, download_detail)
            verification = yield from self._run_tool(
                ToolRequest("software_verify_download", {"path": path}),
                cancel_event,
                stage_id="verify",
                stage_title="Verifying download",
            )
            if verification is None or not verification.ok or not isinstance(verification.data, dict) or not verification.data.get("verified", False):
                return ToolResult(
                    "software_verify_download",
                    False,
                    data=verification.data if verification is not None else None,
                    error_code="download_verification_failed",
                    error_message="the downloaded package did not pass metadata verification",
                )
            yield self._stage("verify", "Verifying download", TroubleshootingStageStatus.COMPLETED, "File verified")
            yield self._stage("install", "Installing software", TroubleshootingStageStatus.IN_PROGRESS, "Installing the verified vendor package")
            install_request = ToolRequest(
                "software_vendor_install",
                {
                    "vendor": plan.request.arguments["vendor"],
                    "manager": plan.request.arguments["manager"],
                    "path": path,
                },
            )
            return (yield from self._run_tool(install_request, cancel_event, confirmation=True, stage_id="install", stage_title="Installing software"))
        stage_id = "download" if plan.operation is SoftwareOperation.DOWNLOAD else "install"
        return (yield from self._run_tool(plan.request, cancel_event, confirmation=True, stage_id=stage_id, stage_title=_action_title(plan.operation)))

    def _discover_candidate(
        self,
        spec: SoftwareSpec,
        profile: SystemProfile,
        cancel_event: threading.Event,
        excluded: set[tuple[str, str, SoftwareSource]] | None = None,
    ) -> Iterator[ProviderEvent]:
        """Search trusted sources and return an identity reported by them."""
        query = query_for(spec.display_name, explicit_query=spec.search_query)
        excluded = excluded or set()
        had_network_error = False
        for manager in _search_managers(profile):
            result = yield from self._run_tool(
                ToolRequest("software_search", {"manager": manager, "query": query}),
                cancel_event,
            )
            if result is None or not result.ok:
                had_network_error = had_network_error or bool(
                    result and result.error_code == SoftwareErrorCode.NETWORK_ERROR.value
                )
                continue
            data = result.data if isinstance(result.data, dict) else {}
            packages = repository_candidates(
                manager,
                str(data.get("query", query)),
                str(data.get("stdout", "")),
            )
            for package in packages:
                candidate = (manager, package, _source_for_manager(manager))
                if candidate not in excluded:
                    return (candidate, result, had_network_error)
        return (None, None, had_network_error)

    def _find_alternate_candidate(
        self,
        candidates: list[tuple[str, str, SoftwareSource]],
        failed: tuple[str, str, SoftwareSource],
        spec: SoftwareSpec,
        cancel_event: threading.Event,
    ) -> Iterator[ProviderEvent]:
        """Try other catalog identities after a source rejects one package."""
        failed_seen = False
        for manager, package, source in candidates:
            if not failed_seen:
                if (manager, package, source) == failed:
                    failed_seen = True
                continue
            if source is SoftwareSource.OFFICIAL_VENDOR:
                return (manager, package, source, None)
            result = yield from self._run_tool(
                ToolRequest(
                    "software_search",
                    {"manager": manager, "query": query_for(spec.display_name, package=package)},
                ),
                cancel_event,
            )
            if result is None or not result.ok:
                continue
            data = result.data if isinstance(result.data, dict) else {}
            reported = repository_candidates(
                manager,
                str(data.get("query", package)),
                str(data.get("stdout", "")),
            )
            if package.casefold() in {item.casefold() for item in reported}:
                return (manager, package, source, result)
        return None

    def _recover_after_failure(
        self,
        spec: SoftwareSpec,
        profile: SystemProfile,
        candidates: list[tuple[str, str, SoftwareSource]],
        failed: tuple[str, str, SoftwareSource],
        error_code: str,
        cancel_event: threading.Event,
    ) -> Iterator[ProviderEvent]:
        """Select one safe recovery; never repeat an unknown command blindly."""
        if error_code == SoftwareErrorCode.NETWORK_ERROR.value:
            return failed
        if error_code != SoftwareErrorCode.PACKAGE_NOT_FOUND.value:
            return None
        alternate = yield from self._find_alternate_candidate(
            candidates,
            failed,
            spec,
            cancel_event,
        )
        if alternate is not None:
            return alternate[:3]
        discovered, _search_result, _network_error = yield from self._discover_candidate(
            spec,
            profile,
            cancel_event,
            excluded={failed},
        )
        return discovered

    def _execute_with_recovery(
        self,
        request: SoftwareRequest,
        spec: SoftwareSpec,
        profile: SystemProfile,
        candidates: list[tuple[str, str, SoftwareSource]],
        manager: str,
        package: str,
        source: SoftwareSource,
        current_version: str,
        available_version: str,
        record: dict[str, Any],
        cancel_event: threading.Event,
    ) -> Iterator[ProviderEvent]:
        """Confirm, execute, classify, and boundedly recover one package action."""
        attempts = 0
        max_attempts = 2
        while True:
            operation = request.operation
            yield self._stage("prepare", _prepare_title(operation), TroubleshootingStageStatus.IN_PROGRESS, "Building a safe package-manager action")
            plan = _make_plan(
                request,
                spec,
                profile,
                manager,
                package,
                source,
                current_version=current_version,
                available_version=available_version,
            )
            yield self._stage("prepare", _prepare_title(operation), TroubleshootingStageStatus.COMPLETED, "Action is ready for review")
            yield self._stage("permission", "Waiting for permission", TroubleshootingStageStatus.IN_PROGRESS, "Review the proposed software change")
            with self._pending_lock:
                decision_event = threading.Event()
                decision: list[bool | None] = [None]
                self._pending[plan.plan_id] = (decision_event, decision)
            record["plan_id"] = plan.plan_id
            yield ProviderEvent.software_plan_ready(plan)
            approved = self._wait_for_decision(plan.plan_id, cancel_event, decision_event, decision)
            if approved is None:
                yield self._stage("permission", "Waiting for permission", TroubleshootingStageStatus.CANCELLED, "Cancelled")
                if attempts:
                    yield self._stage("retry", "Retrying package action", TroubleshootingStageStatus.CANCELLED, "Cancelled")
                return (None, manager, package, source)
            if not approved:
                record["status"] = "cancelled"
                yield self._stage("permission", "Waiting for permission", TroubleshootingStageStatus.CANCELLED, "Cancelled by user")
                if attempts:
                    yield self._stage("retry", "Retrying package action", TroubleshootingStageStatus.CANCELLED, "Cancelled by user")
                yield ProviderEvent.text_chunk(f"No changes were made. The {spec.display_name} request was cancelled.")
                yield ProviderEvent.done()
                return (None, manager, package, source)

            if attempts:
                yield self._stage("retry", "Retrying package action", TroubleshootingStageStatus.COMPLETED, "Fresh approval received")
            yield self._stage("permission", "Waiting for permission", TroubleshootingStageStatus.COMPLETED, "Permission received")
            yield self._stage("terminal", "Opening terminal execution", TroubleshootingStageStatus.IN_PROGRESS, "Preparing the approved registry command")
            yield self._stage("terminal", "Opening terminal execution", TroubleshootingStageStatus.COMPLETED, "Execution layer ready")
            if operation is SoftwareOperation.DOWNLOAD:
                yield self._stage("download", "Downloading software", TroubleshootingStageStatus.IN_PROGRESS, "Saving the package to the approved downloads directory")
            elif source is SoftwareSource.OFFICIAL_VENDOR and operation in {SoftwareOperation.INSTALL, SoftwareOperation.REINSTALL}:
                yield self._stage("download", "Downloading software", TroubleshootingStageStatus.IN_PROGRESS, "Downloading the fixed official vendor package")
            else:
                yield self._stage("install", _action_title(operation), TroubleshootingStageStatus.IN_PROGRESS, "Running the approved package-manager action")
            result = yield from self._execute_approved(plan, cancel_event)
            if result is not None and result.ok:
                yield self._stage("analyze", "Analyzing result", TroubleshootingStageStatus.IN_PROGRESS, "Checking the package-manager exit status")
                yield self._stage("analyze", "Analyzing result", TroubleshootingStageStatus.COMPLETED, _success_detail(result))
                return (result, manager, package, source)

            cancelled = bool(result and result.error_code == "cancelled") or cancel_event.is_set()
            failure_stage = "download" if operation is SoftwareOperation.DOWNLOAD else "install"
            failure_title = "Downloading software" if failure_stage == "download" else _action_title(operation)
            yield self._stage(failure_stage, failure_title, TroubleshootingStageStatus.CANCELLED if cancelled else TroubleshootingStageStatus.FAILED, "Cancelled" if cancelled else "Package manager reported a failure")
            if cancelled:
                record["status"] = "cancelled"
                yield ProviderEvent.text_chunk(f"The {spec.display_name} operation was cancelled. No further action was attempted.")
                yield ProviderEvent.done()
                return (None, manager, package, source)

            error_code = result.error_code if result and result.error_code else SoftwareErrorCode.INSTALLATION_ERROR.value
            yield self._stage("analyze", "Analyzing result", TroubleshootingStageStatus.IN_PROGRESS, "Reading exit code, stdout, and stderr")
            yield self._stage("analyze", "Analyzing result", TroubleshootingStageStatus.COMPLETED, _error_detail(result, error_code))
            qwen_decision = yield from self._analyze_failure(
                request,
                operation,
                manager,
                package,
                source,
                result,
                candidates,
                cancel_event,
            )
            if cancel_event.is_set():
                return (None, manager, package, source)
            if qwen_decision.action == "stop":
                record["status"] = "failed"
                reason = qwen_decision.reason or "Qwen did not identify a safe recovery."
                yield self._stage(
                    "recover",
                    "Recovering installation",
                    TroubleshootingStageStatus.WARNING,
                    reason,
                )
                yield ProviderEvent.text_chunk(
                    _failure_text(
                        spec.display_name,
                        result,
                        max_attempts=max_attempts,
                        attempts=attempts + 1,
                    )
                )
                yield ProviderEvent.done()
                return (None, manager, package, source)
            if attempts + 1 >= max_attempts:
                record["status"] = "failed"
                yield ProviderEvent.text_chunk(_failure_text(spec.display_name, result, max_attempts=max_attempts, attempts=attempts + 1))
                yield ProviderEvent.done()
                return (None, manager, package, source)
            recovery = yield from self._recover_after_failure(
                spec,
                profile,
                candidates,
                (manager, package, source),
                error_code,
                cancel_event,
            )
            if recovery is None:
                record["status"] = "failed"
                yield self._stage("recover", "Recovering installation", TroubleshootingStageStatus.WARNING, "No safe alternate package or installation method was found")
                yield ProviderEvent.text_chunk(_failure_text(spec.display_name, result, max_attempts=max_attempts, attempts=attempts + 1))
                yield ProviderEvent.done()
                return (None, manager, package, source)
            attempts += 1
            yield self._stage("recover", "Recovering installation", TroubleshootingStageStatus.IN_PROGRESS, _recovery_detail(error_code))
            manager, package, source = recovery
            candidates = [recovery, *[item for item in candidates if item != recovery]]
            record["manager"] = manager
            record["source"] = source.value
            record["package"] = package
            yield self._stage("recover", "Recovering installation", TroubleshootingStageStatus.COMPLETED, f"Prepared trusted alternative: {package} via {source.value}")
            yield self._stage(
                "retry",
                "Retrying package action",
                TroubleshootingStageStatus.IN_PROGRESS,
                "A fresh approval is required before retrying",
            )

    def _analyze_failure(
        self,
        request: SoftwareRequest,
        operation: SoftwareOperation,
        manager: str,
        package: str,
        source: SoftwareSource,
        result: ToolResult | None,
        candidates: list[tuple[str, str, SoftwareSource]],
        cancel_event: threading.Event,
    ) -> Generator[ProviderEvent, None, SoftwareRecoveryDecision]:
        """Ask Qwen for a bounded recovery choice, never for a command."""
        if result is None or self._provider is None:
            return SoftwareRecoveryDecision()
        stream_failure = getattr(self._provider, "stream_software_failure", None)
        if not callable(stream_failure):
            return SoftwareRecoveryDecision()
        alternatives = tuple(
            f"{candidate_manager}:{candidate_package} via {candidate_source.value}"
            for candidate_manager, candidate_package, candidate_source in candidates
            if (candidate_manager, candidate_package, candidate_source)
            != (manager, package, source)
        )
        decision = SoftwareRecoveryDecision()
        try:
            for event in stream_failure(
                request.original_request,
                operation.value,
                f"{manager}:{package} via {source.value}",
                result,
                alternatives,
                cancel_event,
            ):
                if cancel_event.is_set():
                    return decision
                if event.kind == "software_recovery" and event.software_recovery is not None:
                    decision = event.software_recovery
                if event.kind == "status":
                    yield event
            return decision
        except Exception as exc:  # Analysis is advisory; the registry stays authoritative.
            logger.warning("Software failure analysis unavailable: %s", exc)
            return decision

    def _stream_read_only_result(
        self,
        software_request: SoftwareRequest,
        spec: SoftwareSpec,
        manager: str,
        package: str,
        source: SoftwareSource,
        cancel_event: threading.Event,
        search_result: ToolResult | None,
    ) -> Iterator[ProviderEvent]:
        operation = software_request.operation
        if operation is SoftwareOperation.AVAILABLE_VERSION:
            result = yield from self._run_tool(
                ToolRequest("software_available_version", {"manager": manager, "package": package}),
                cancel_event,
            )
            available = str(result.data.get("version", "")) if result and isinstance(result.data, dict) else ""
            good = bool(result and result.ok and available)
            yield self._stage(
                "result",
                "Reading source result",
                TroubleshootingStageStatus.COMPLETED if good else TroubleshootingStageStatus.WARNING,
                f"Available version: {available}" if good else "Source version unavailable",
            )
            yield ProviderEvent.text_chunk(
                f"The available {spec.display_name} version is {available}." if good else f"I could not determine the available version of {spec.display_name}."
            )
            yield ProviderEvent.done()
            return
        if operation is SoftwareOperation.SEARCH:
            data = search_result.data if search_result else None
            if search_result and search_result.ok:
                yield self._stage("result", "Reading source result", TroubleshootingStageStatus.COMPLETED, "Source search completed")
                output = data.get("stdout", "") if isinstance(data, dict) else str(data)
                yield ProviderEvent.text_chunk(f"I checked {source.value} for {spec.display_name}.\n\n{_clean_output(output) or 'No matching package details were returned by the source.'}")
            else:
                yield self._stage("result", "Reading source result", TroubleshootingStageStatus.WARNING, "Source search unavailable")
                yield ProviderEvent.text_chunk(f"I could not read the {source.value} search result for {spec.display_name}.")
            yield ProviderEvent.done()
            return
        if operation is SoftwareOperation.VERIFY:
            self._set_task_state(
                task_state=TroubleshootingTaskState.VERIFYING,
                active_step_id="dependencies",
                verification_required=True,
                verification_complete=False,
            )
            yield self._stage(
                "dependencies",
                "Checking package dependencies",
                TroubleshootingStageStatus.IN_PROGRESS,
                "Running a read-only package health check",
            )
            dependency_result = yield from self._run_diagnostic_tool(
                ToolRequest("package_health", {}), cancel_event
            )
            dependency_data = (
                dependency_result.data
                if dependency_result is not None and isinstance(dependency_result.data, dict)
                else {}
            )
            dependency_health = dependency_data.get("healthy")
            dependency_ok = dependency_health is not False
            self._set_task_state(
                task_state=TroubleshootingTaskState.VERIFYING,
                active_step_id="verify",
                verification_required=True,
            )
            yield self._stage(
                "dependencies",
                "Checking package dependencies",
                TroubleshootingStageStatus.COMPLETED if dependency_ok else TroubleshootingStageStatus.FAILED,
                "Package health is normal" if dependency_ok else "The package manager reported a dependency problem",
            )
            yield self._stage(
                "verify",
                "Verifying application",
                TroubleshootingStageStatus.IN_PROGRESS,
                "Checking the installed package and approved launcher",
            )
            result = yield from self._run_tool(
                ToolRequest(
                    "software_verify",
                    {"manager": manager, "package": package, "executable": spec.executable},
                ),
                cancel_event,
            )
            data = result.data if result is not None and isinstance(result.data, dict) else {}
            installed = bool(result and result.ok and data.get("installed", False))
            verified = bool(
                result
                and result.ok
                and data.get("verified", installed)
                and dependency_ok
            )
            if verified:
                version = str(data.get("version", "") or data.get("version_output", "")).strip()
                launcher = str(data.get("executable_path", "")).strip()
                evidence = [f"{spec.display_name} is installed" + (f" (version {version})" if version else "") + "."]
                if launcher:
                    evidence.append(f"The approved launcher is available at {launcher}.")
                evidence.append("Package dependencies are healthy.")
                message = "✅ Everything is normal. No problems were detected.\n\n" + "\n".join(
                    f"- {line}" for line in evidence
                ) + "\n\n" + f"{spec.display_name} verification passed."
            else:
                reason = str(data.get("verification_reason", "")).strip()
                if dependency_health is False:
                    reason = "the package manager reported a dependency problem"
                elif not installed:
                    reason = "the package is not installed"
                elif not reason:
                    reason = "the installed package or approved launcher could not be verified"
                message = f"Problem detected: {spec.display_name} could not be verified because {reason}."
            yield self._stage(
                "verify",
                "Verifying application",
                TroubleshootingStageStatus.COMPLETED if verified else TroubleshootingStageStatus.FAILED,
                "Application and dependencies verified" if verified else "Application verification failed",
            )
            yield ProviderEvent.text_chunk(message)
            self._set_task_state(
                task_state=TroubleshootingTaskState.COMPLETED if verified else TroubleshootingTaskState.FAILED,
                active_step_id="",
                verification_required=True,
                verification_complete=True,
                pending_tool_calls=(),
            )
            yield ProviderEvent.done()
            return "completed" if verified else "failed"

        arguments = {"manager": manager, "package": package}
        result = yield from self._run_tool(ToolRequest("software_query", arguments), cancel_event)
        installed = bool(result and result.ok and isinstance(result.data, dict) and result.data.get("installed", False))
        version = result.data.get("version_output", "") if result and isinstance(result.data, dict) else ""
        message = (
            f"✅ Everything is normal. No problems were detected.\n\n"
            f"{spec.display_name} is installed. {version}"
            if installed
            else f"{spec.display_name} is not installed through the detected {source.value}."
        )
        status = TroubleshootingStageStatus.COMPLETED if result and result.ok else TroubleshootingStageStatus.WARNING
        yield self._stage("result", "Reading installed state", status, "Result ready")
        yield ProviderEvent.text_chunk(message.strip())
        yield ProviderEvent.done()
        return "completed" if result and result.ok else "failed"

    def _wait_for_decision(self, plan_id: str, cancel_event: threading.Event, event: threading.Event, decision: list[bool | None]) -> bool | None:
        try:
            while not event.wait(0.1):
                if cancel_event.is_set():
                    return None
            return decision[0]
        finally:
            with self._pending_lock:
                self._pending.pop(plan_id, None)

    @staticmethod
    def _stage(stage_id: str, title: str, status: TroubleshootingStageStatus, detail: str) -> ProviderEvent:
        now = time.time()
        finished = status in {
            TroubleshootingStageStatus.COMPLETED,
            TroubleshootingStageStatus.WARNING,
            TroubleshootingStageStatus.FAILED,
            TroubleshootingStageStatus.CANCELLED,
        }
        return ProviderEvent.stage_update(
            TroubleshootingStageEvent(
                stage_id,
                title,
                status,
                detail,
                step_type="software",
                action=detail,
                started_at=now,
                ended_at=now if finished else None,
            )
        )


def _profile_from_result(result: ToolResult) -> SystemProfile:
    data = result.data if isinstance(result.data, dict) else {}
    managers = tuple(str(value) for value in data.get("available_managers", []) if value)
    return SystemProfile(
        distribution=str(data.get("distribution", "")),
        distribution_id=str(data.get("distribution_id", "")),
        version=str(data.get("version", "")),
        architecture=str(data.get("architecture", "")),
        package_manager=str(data.get("package_manager", "")),
        available_managers=managers,
        sources=tuple(str(value) for value in data.get("sources", [])),
    )


def _overall_check_health(tool_name: str, result: ToolResult | None) -> bool | None:
    """Interpret only structured health fields; command success is not health."""
    if result is None or not result.ok or not isinstance(result.data, dict):
        return None
    data = result.data
    if tool_name in {"package_health", "package_update_status"}:
        value = data.get("healthy")
        return value if isinstance(value, bool) else None
    if tool_name == "service_failures":
        count = data.get("failed_count")
        return int(count) == 0 if isinstance(count, (int, float, str)) and str(count).isdigit() else None
    # Journal entries are evidence for review, not proof that the software
    # environment is broken. Keep them informational in the overall result.
    return True


def _overall_diagnostic_problems(
    results: dict[str, ToolResult | None],
) -> list[str]:
    problems: list[str] = []
    package = results.get("package_health")
    package_data = package.data if package is not None and isinstance(package.data, dict) else {}
    if package is None or not package.ok or package_data.get("healthy") is not True:
        problems.append("package or dependency health could not be confirmed")
    updates = results.get("package_update_status")
    update_data = updates.data if updates is not None and isinstance(updates.data, dict) else {}
    if updates is None or not updates.ok or update_data.get("healthy") is not True:
        problems.append("the software update check could not be confirmed")
    services = results.get("service_failures")
    service_data = services.data if services is not None and isinstance(services.data, dict) else {}
    failed_count = service_data.get("failed_count")
    if services is None or not services.ok or not isinstance(failed_count, (int, float, str)):
        problems.append("failed software services could not be confirmed")
    else:
        try:
            failed_total = int(failed_count)
        except (TypeError, ValueError):
            failed_total = -1
        if failed_total < 0:
            problems.append("failed software services could not be confirmed")
        elif failed_total > 0:
            problems.append(f"Linux reports {failed_total} failed system service(s)")
    return problems


def _candidate_options(
    spec: SoftwareSpec,
    profile: SystemProfile,
) -> list[tuple[str, str, SoftwareSource]]:
    """Return only catalog-approved package identities for this host."""
    options: list[tuple[str, str, SoftwareSource]] = []
    manager = profile.package_manager
    if manager and manager in spec.packages:
        options.append((manager, spec.packages[manager], SoftwareSource.PACKAGE_REPOSITORY))
    if "flatpak" in profile.available_managers and spec.flatpak:
        options.append(("flatpak", spec.flatpak, SoftwareSource.FLATPAK))
    if "snap" in profile.available_managers and spec.snap:
        options.append(("snap", spec.snap, SoftwareSource.SNAP))
    if (
        spec.vendor
        and manager in {"apt", "dnf", "yum"}
        and profile.architecture in {"x86_64", "amd64", "AMD64"}
    ):
        options.append(
            (
                manager,
                spec.packages.get(manager, spec.packages.get("dnf", "")),
                SoftwareSource.OFFICIAL_VENDOR,
            )
        )
    seen: set[tuple[str, str, SoftwareSource]] = set()
    unique: list[tuple[str, str, SoftwareSource]] = []
    for option in options:
        if option in seen:
            continue
        seen.add(option)
        unique.append(option)
    return unique


def _search_managers(profile: SystemProfile) -> tuple[str, ...]:
    """Order searches by detected primary manager, then trusted alternatives."""
    available = set(profile.available_managers)
    ordered = [profile.package_manager, "apt", "dnf", "yum", "pacman", "zypper", "flatpak", "snap"]
    return tuple(manager for manager in dict.fromkeys(ordered) if manager in available)


def _source_for_manager(manager: str) -> SoftwareSource:
    return {
        "flatpak": SoftwareSource.FLATPAK,
        "snap": SoftwareSource.SNAP,
    }.get(manager, SoftwareSource.PACKAGE_REPOSITORY)


def _make_plan(
    request: SoftwareRequest,
    spec: SoftwareSpec,
    profile: SystemProfile,
    manager: str,
    package: str,
    source: SoftwareSource,
    *,
    current_version: str = "",
    available_version: str = "",
) -> SoftwarePlan:
    operation = request.operation
    destination = str(_download_dir())
    if source is SoftwareSource.OFFICIAL_VENDOR:
        vendor = spec.vendor
        if operation is SoftwareOperation.DOWNLOAD:
            tool_request = ToolRequest("software_vendor_download", {"vendor": vendor, "manager": _vendor_manager(manager), "destination": destination}, True)
        else:
            tool_request = ToolRequest("software_vendor_install", {"vendor": vendor, "manager": _vendor_manager(manager), "path": f"{destination}/{_vendor_filename(vendor, manager)}"}, True)
    elif operation is SoftwareOperation.DOWNLOAD:
        tool_request = ToolRequest("software_download", {"manager": manager, "package": package, "destination": destination}, True)
    elif operation is SoftwareOperation.REINSTALL:
        tool_request = ToolRequest("software_reinstall", {"manager": manager, "package": package}, True)
    elif operation in {SoftwareOperation.UPDATE, SoftwareOperation.UPGRADE}:
        tool_request = ToolRequest("software_update", {"manager": manager, "package": "all" if request.scope_all else package}, True)
    elif operation is SoftwareOperation.REMOVE:
        tool_request = ToolRequest("software_remove", {"manager": manager, "package": package}, True)
    else:
        tool_request = ToolRequest("software_install", {"manager": manager, "package": package}, True)
    source_text = source.value
    preview = _command_preview(operation, manager, package, source, vendor if source is SoftwareSource.OFFICIAL_VENDOR else "", destination)
    risk = "High" if operation is SoftwareOperation.REMOVE else ("Medium" if operation in {SoftwareOperation.INSTALL, SoftwareOperation.UPDATE, SoftwareOperation.UPGRADE, SoftwareOperation.REINSTALL} else "Low")
    what_will_do = {
        SoftwareOperation.DOWNLOAD: f"Download {spec.display_name} to the approved local downloads directory without installing it.",
        SoftwareOperation.INSTALL: f"Download and install {spec.display_name} and any package-manager dependencies.",
        SoftwareOperation.UPDATE: f"Update the installed {spec.display_name} package from the trusted source.",
        SoftwareOperation.UPGRADE: f"Upgrade the installed {spec.display_name} package from the trusted source.",
        SoftwareOperation.REMOVE: f"Remove {spec.display_name} without purging unrelated packages.",
        SoftwareOperation.REINSTALL: f"Reinstall {spec.display_name} from the trusted source.",
    }.get(operation, f"Run the approved {operation.value} action for {spec.display_name}.")
    return SoftwarePlan(
        plan_id=uuid.uuid4().hex,
        software_name=spec.display_name,
        operation=operation,
        source=source,
        package_name=package,
        manager=manager,
        architecture=profile.architecture,
        dependencies=("Package-manager dependencies, if required",),
        command_preview=preview,
        details=f"Distribution: {profile.distribution or 'Linux'}\nArchitecture: {profile.architecture}\nSource: {source_text}\nNo unrelated packages will be removed.",
        request=tool_request,
        current_version=current_version,
        available_version=available_version,
        risk=risk,
        what_will_do=what_will_do,
    )


def _download_dir() -> Path:
    from config.config import _data_home

    return _data_home / "system-agent" / "downloads"


def _vendor_manager(manager: str) -> str:
    return "dnf" if manager in {"dnf", "yum"} else "apt"


def _vendor_filename(vendor: str, manager: str) -> str:
    return {
        ("google_chrome", "apt"): "google-chrome-stable_current_amd64.deb",
        ("google_chrome", "dnf"): "google-chrome-stable_current_x86_64.rpm",
        ("visual_studio_code", "apt"): "code_latest_amd64.deb",
        ("visual_studio_code", "dnf"): "code_latest_x86_64.rpm",
    }.get((vendor, _vendor_manager(manager)), "installer.deb")


def _action_title(operation: SoftwareOperation) -> str:
    return {
        SoftwareOperation.INSTALL: "Installing software",
        SoftwareOperation.DOWNLOAD: "Downloading software",
        SoftwareOperation.UPDATE: "Updating software",
        SoftwareOperation.UPGRADE: "Upgrading software",
        SoftwareOperation.REMOVE: "Removing software",
        SoftwareOperation.REINSTALL: "Reinstalling software",
    }.get(operation, "Managing software")


def _prepare_title(operation: SoftwareOperation) -> str:
    return {
        SoftwareOperation.DOWNLOAD: "Preparing download",
        SoftwareOperation.INSTALL: "Preparing installation",
        SoftwareOperation.UPDATE: "Preparing update",
        SoftwareOperation.UPGRADE: "Preparing upgrade",
        SoftwareOperation.REMOVE: "Preparing removal",
        SoftwareOperation.REINSTALL: "Preparing reinstallation",
    }.get(operation, "Preparing software action")


def _make_state(
    spec: SoftwareSpec,
    manager: str,
    package: str,
    source: SoftwareSource,
    installed: bool,
    current_version: str,
    available_version: str,
    update_available: bool | None,
    request: SoftwareRequest,
) -> SoftwareState:
    return SoftwareState(
        state_id=uuid.uuid4().hex,
        software_name=spec.display_name,
        package_name=package,
        manager=manager,
        source=source,
        installed=installed,
        current_version=current_version,
        available_version=available_version,
        update_available=update_available,
        actions=_state_actions(request.operation, installed, update_available),
    )


def _state_actions(
    operation: SoftwareOperation,
    installed: bool,
    update_available: bool | None,
) -> tuple[SoftwareOperation, ...]:
    if installed:
        actions: list[SoftwareOperation] = []
        if update_available is True and operation in {
            SoftwareOperation.DOWNLOAD,
            SoftwareOperation.INSTALL,
            SoftwareOperation.UPDATE,
            SoftwareOperation.UPGRADE,
        }:
            actions.append(SoftwareOperation.UPDATE)
        if operation in {SoftwareOperation.DOWNLOAD, SoftwareOperation.INSTALL, SoftwareOperation.UPDATE, SoftwareOperation.UPGRADE}:
            actions.extend((SoftwareOperation.REMOVE, SoftwareOperation.REINSTALL))
        return tuple(actions)
    if operation in {SoftwareOperation.REMOVE, SoftwareOperation.REINSTALL}:
        # Do not offer Install/Download after a removal or reinstallation
        # request when the queried package is absent. Those actions imply a
        # different user intent and made an uninstall result look like an
        # install flow in the UI.
        return ()
    if operation in {SoftwareOperation.UPDATE, SoftwareOperation.UPGRADE}:
        return (SoftwareOperation.INSTALL, SoftwareOperation.DOWNLOAD)
    return ()


def _installed_detail(name: str, installed: bool, version: str) -> str:
    if installed:
        return f"{name} is installed" + (f" · Version {version}" if version else "")
    return f"{name} is not installed"


def _available_detail(version: str, update_available: bool | None) -> str:
    if update_available is True:
        return f"New version available: {version or 'unknown'}"
    if update_available is False:
        return f"Already up to date · {version or 'current version confirmed'}"
    return "Candidate version could not be compared"


def _already_installed_text(name: str, current_version: str, update_available: bool | None) -> str:
    version = f"\n\nVersion: {current_version}" if current_version else ""
    if update_available is True:
        return f"✅ Software is already installed.\n\n{name} is already installed on your system.{version}\n\nA newer version is available. Choose an action below."
    return f"✅ Software is already installed.\n\n{name} is already installed on your system.{version}\n\nNo additional download or installation was performed."


def _is_newer_version(current: str, available: str) -> bool | None:
    if not current or not available:
        return None
    if current == available:
        return False
    current_parts = tuple(int(value) for value in re.findall(r"\d+", current))
    available_parts = tuple(int(value) for value in re.findall(r"\d+", available))
    if current_parts and available_parts and current_parts != available_parts:
        return available_parts > current_parts
    return available != current


def _command_prefix() -> str:
    if shutil.which("pkexec"):
        return "pkexec"
    return "sudo -n"


def _command_preview(
    operation: SoftwareOperation,
    manager: str,
    package: str,
    source: SoftwareSource,
    vendor: str,
    destination: str,
) -> str:
    if source is SoftwareSource.OFFICIAL_VENDOR:
        # Vendor downloads are performed by a fixed HTTPS allowlist handler;
        # exposing that fixed registry call is safer than presenting a
        # misleading arbitrary curl command.
        if operation is SoftwareOperation.DOWNLOAD:
            return f"system-agent software_vendor_download --vendor {vendor} --manager {_vendor_manager(manager)}"
        return f"system-agent software_vendor_download + software_vendor_install --vendor {vendor} --manager {_vendor_manager(manager)}"
    if operation is SoftwareOperation.DOWNLOAD:
        if manager == "apt":
            return f"apt download {shlex.quote(package)}  # saved under {shlex.quote(destination)}"
        return f"{manager} download --destdir {shlex.quote(destination)} {shlex.quote(package)}"
    prefix = _command_prefix()
    if operation is SoftwareOperation.INSTALL:
        if manager == "apt":
            return f"{prefix} apt-get install -y --no-install-recommends {shlex.quote(package)}"
        if manager in {"dnf", "yum"}:
            return f"{prefix} {manager} install -y {shlex.quote(package)}"
        if manager == "pacman":
            return f"{prefix} pacman -S --noconfirm {shlex.quote(package)}"
        if manager == "zypper":
            return f"{prefix} zypper --non-interactive install {shlex.quote(package)}"
        if manager == "flatpak":
            return f"{prefix} flatpak install -y flathub {shlex.quote(package)}"
        return f"{prefix} snap install {shlex.quote(package)}"
    if operation is SoftwareOperation.REINSTALL:
        if manager == "apt":
            return f"{prefix} apt-get install -y --no-install-recommends --reinstall {shlex.quote(package)}"
        return f"{prefix} {manager} reinstall -y {shlex.quote(package)}"
    if operation is SoftwareOperation.REMOVE:
        command = {"apt": "apt-get remove -y", "dnf": "dnf remove -y", "yum": "yum remove -y", "pacman": "pacman -R --noconfirm", "zypper": "zypper --non-interactive remove", "flatpak": "flatpak uninstall -y", "snap": "snap remove"}.get(manager, f"{manager} remove")
        return f"{prefix} {command} {shlex.quote(package)}"
    if operation in {SoftwareOperation.UPDATE, SoftwareOperation.UPGRADE}:
        if manager == "apt":
            command = f"apt-get {'upgrade -y' if package == 'all' else 'install -y --only-upgrade'}"
        elif manager in {"dnf", "yum"}:
            command = f"{manager} upgrade -y"
        elif manager == "pacman":
            command = "pacman -Syu --noconfirm"
        elif manager == "zypper":
            command = "zypper --non-interactive update"
        elif manager == "flatpak":
            command = "flatpak update -y"
        else:
            command = "snap refresh"
        return f"{prefix} {command}" + (f" {shlex.quote(package)}" if package != "all" and manager != "apt" else (f" {shlex.quote(package)}" if package != "all" else ""))
    return f"{manager} {operation.value} {shlex.quote(package)}"


def _failure_text(
    name: str,
    result: ToolResult | None,
    *,
    max_attempts: int = 1,
    attempts: int = 1,
) -> str:
    if result is None:
        return f"I could not complete the {name} package action. No further automatic command was attempted."
    message = result.error_message or "the package manager returned an error"
    code = result.error_code or SoftwareErrorCode.INSTALLATION_ERROR.value
    data = result.data if isinstance(result.data, dict) else {}
    exit_code = data.get("exit_code", "unknown")
    retry_note = (
        f" The bounded recovery limit of {max_attempts} attempts was reached."
        if attempts >= max_attempts
        else ""
    )
    return (
        f"I could not complete the {name} package action ({code}): {message}. "
        f"Exit code: {exit_code}. Command output was captured for analysis."
        f"{retry_note} No further automatic command was attempted."
    )


def _success_detail(result: ToolResult | None) -> str:
    if result is None:
        return "No result was returned"
    data = result.data if isinstance(result.data, dict) else {}
    return f"Command succeeded · exit code {data.get('exit_code', 0)} · {result.duration_ms} ms"


def _error_detail(result: ToolResult | None, error_code: str) -> str:
    data = result.data if result and isinstance(result.data, dict) else {}
    return f"{error_code} · exit code {data.get('exit_code', 'unknown')} · output captured"


def _recovery_detail(error_code: str) -> str:
    if error_code == SoftwareErrorCode.NETWORK_ERROR.value:
        return "Retrying once after a transient repository/network failure"
    if error_code == SoftwareErrorCode.PACKAGE_NOT_FOUND.value:
        return "Searching for another trusted package identity or source"
    return "Evaluating a safe recovery path"


def _installation_verification_warning(
    name: str,
    executable: str,
    result: ToolResult | None,
) -> str:
    """Explain exactly which post-install verification check did not pass."""
    if result is None or not result.ok or not isinstance(result.data, dict):
        return f"The {name} package action completed, but installation status could not be read safely. No further command was attempted."
    data = result.data
    if not bool(data.get("installed", False)):
        return f"The {name} package action completed, but the package is not installed. No further command was attempted."
    if executable and not bool(data.get("executable_available", False)):
        return f"{name} is installed, but its launcher could not be found. Installation verification needs attention; no further command was attempted."
    return f"The {name} package action completed, but installation verification needs attention. No further command was attempted."


def _tool_completed_detail(result: ToolResult | None) -> str:
    if result is None:
        return "Command completed"
    data = result.data if isinstance(result.data, dict) else {}
    exit_code = data.get("exit_code", 0)
    return f"Command completed · exit code {exit_code} · {result.duration_ms} ms"


def _tool_failed_detail(result: ToolResult | None) -> str:
    if result is None:
        return "Command failed"
    message = result.error_message or "Command failed"
    return f"Command failed · {message[:240]}"


def _result_path(result: ToolResult) -> str:
    if not isinstance(result.data, dict):
        return ""
    path = str(result.data.get("path", "")).strip()
    if path:
        return path
    paths = result.data.get("paths", [])
    if isinstance(paths, list) and paths:
        return str(paths[0]).strip()
    return ""


def _display_path(path: str) -> str:
    return path if path else "the approved downloads directory"


def _past_tense(operation: SoftwareOperation) -> str:
    return {
        SoftwareOperation.INSTALL: "installed",
        SoftwareOperation.DOWNLOAD: "downloaded",
        SoftwareOperation.UPDATE: "updated",
        SoftwareOperation.UPGRADE: "upgraded",
        SoftwareOperation.REMOVE: "removed",
        SoftwareOperation.REINSTALL: "reinstalled",
    }.get(operation, operation.value)


def _clean_output(value: str) -> str:
    return "\n".join(line.strip() for line in value.splitlines() if line.strip())[:4_000]


def _search_has_no_package(result: ToolResult, package: str) -> bool:
    if not isinstance(result.data, dict):
        return True
    data = result.data
    manager = str(data.get("manager", ""))
    query = str(data.get("query", package))
    reported = repository_candidates(manager, query, str(data.get("stdout", "")))
    if reported:
        return package.casefold() not in {item.casefold() for item in reported}
    return package.casefold() not in str(data.get("stdout", "")).casefold()
