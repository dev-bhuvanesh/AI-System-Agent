from __future__ import annotations

import threading
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from llm.provider import ProviderEvent, SoftwareRecoveryDecision
from software.catalog import parse_request
from software.contracts import SoftwareErrorCode, SoftwareOperation, SoftwareSource
from software.history import SoftwareHistory
from software.manager import SoftwareManager
from software.resolver import repository_candidates
from tools.contracts import PermissionLevel, ToolDefinition, ToolEvent, ToolEventKind, ToolExecutionError, ToolResult, ToolRequest
from tools.registry import ToolPolicy, ToolRegistry
from tools.linux_tools import create_tool_definitions
from tools.software_tools import (
    _classify_command_error,
    _download,
    _query,
    _vendor_download,
    _verify,
    create_software_tool_definitions,
)
from llm.prompts import parse_software_recovery_response


class SoftwareFeatureTests(unittest.TestCase):
    def test_qwen_recovery_response_is_closed_and_safe(self) -> None:
        retry = parse_software_recovery_response(
            '{"action":"retry_alternative","reason":"Use the trusted Flatpak result."}'
        )
        self.assertEqual(retry.action, "retry_alternative")
        self.assertEqual(retry.reason, "Use the trusted Flatpak result.")

        malformed = parse_software_recovery_response("not JSON")
        self.assertEqual(malformed.action, "stop")

        arbitrary = parse_software_recovery_response(
            '{"action":"run_command","command":"sudo rm -rf /"}'
        )
        self.assertEqual(arbitrary.action, "stop")

    def test_dpkg_config_files_state_is_not_reported_as_installed(self) -> None:
        with patch("tools.software_tools.shutil.which", return_value="/usr/bin/tool"), patch(
            "tools.software_tools._run_argv",
            return_value={
                "exit_code": 0,
                "stdout": "deinstall ok config-files 151.0.7922.173-1",
                "stderr": "",
                "timed_out": False,
            },
        ):
            result = _query({"manager": "apt", "package": "google-chrome-stable"}, threading.Event())
        self.assertFalse(result["installed"])
        self.assertEqual(result["version"], "")

    def test_chrome_verification_accepts_the_stable_launcher_alias(self) -> None:
        def fake_which(name: str) -> str | None:
            return {
                "apt-get": "/usr/bin/apt-get",
                "google-chrome-stable": "/usr/bin/google-chrome-stable",
            }.get(name)

        with patch("tools.software_tools.shutil.which", side_effect=fake_which), patch(
            "tools.software_tools._run_argv",
            return_value={
                "exit_code": 0,
                "stdout": "install ok installed 151.0.7922.173-1",
                "stderr": "",
                "timed_out": False,
            },
        ):
            result = _verify(
                {"manager": "apt", "package": "google-chrome-stable", "executable": "google-chrome"},
                threading.Event(),
            )
        self.assertTrue(result["installed"])
        self.assertTrue(result["verified"])
        self.assertEqual(result["executable_path"], "/usr/bin/google-chrome-stable")

    def test_natural_language_examples_are_classified(self) -> None:
        expected = {
            "I need Google Chrome": "install",
            "Install VLC": "install",
            "Download VS Code": "download",
            "Update Firefox": "update",
            "Remove Docker": "remove",
            "Download Steam": "download",
            "update all my software": "update",
            "Troubleshoot VLC": "verify",
            "Check Firefox": "verify",
        }
        for request, operation in expected.items():
            parsed = parse_request(request)
            self.assertIsNotNone(parsed, request)
            self.assertEqual(parsed[0].operation.value, operation)
        typo = parse_request("Uninstall Google Chrom")
        self.assertIsNotNone(typo)
        self.assertEqual(typo[0].operation, SoftwareOperation.REMOVE)
        self.assertEqual(typo[1].display_name, "Google Chrome")
        self.assertIsNone(parse_request("Check my system"))
        self.assertIsNone(parse_request("Troubleshoot my computer"))
        self.assertIsNone(parse_request("Troubleshoot Wi-Fi"))
        self.assertIsNone(parse_request("Check Bluetooth"))

    def test_unlisted_software_is_kept_as_a_repository_query_until_discovered(self) -> None:
        install = parse_request("Install htop")
        self.assertIsNotNone(install)
        self.assertEqual(install[0].operation, SoftwareOperation.INSTALL)
        self.assertEqual(install[1].display_name, "Htop")
        self.assertEqual(install[1].packages, {})
        self.assertEqual(install[1].search_query, "htop")
        self.assertEqual(install[1].vendor, "")

        natural_install = parse_request("I want to install htop on my Linux")
        self.assertIsNotNone(natural_install)
        self.assertEqual(natural_install[1].search_query, "htop")

        nonexistent = parse_request("Install nonexistent software")
        self.assertIsNotNone(nonexistent)
        self.assertEqual(nonexistent[1].search_query, "nonexistent")

        download = parse_request("Download LibreOffice please")
        self.assertIsNotNone(download)
        self.assertEqual(download[0].operation, SoftwareOperation.DOWNLOAD)
        self.assertEqual(download[1].packages, {})
        self.assertEqual(download[1].search_query, "libreoffice")

        self.assertIsNone(parse_request("Install this file"))

    def test_repository_resolution_uses_only_identities_returned_by_the_source(self) -> None:
        apt_output = "Sorting... Done\nFull Text Search... Done\ngoogle-chrome-stable - The web browser from Google\n"
        self.assertEqual(
            repository_candidates("apt", "Google Chrome", apt_output),
            ("google-chrome-stable",),
        )
        self.assertNotIn("google-chrom", repository_candidates("apt", "Google Chrome", apt_output))
        self.assertEqual(
            repository_candidates(
                "dnf",
                "Firefox",
                "Available Packages\nLast metadata expiration check: 1:00:00 ago\nfirefox.x86_64 128.0-1 updates",
            ),
            ("firefox",),
        )
        self.assertEqual(
            repository_candidates(
                "snap",
                "firefox",
                "Name Version Rev Tracking Publisher Notes\nfirefox 1.0 latest/stable mozilla -",
            ),
            ("firefox",),
        )

    def test_package_manager_failures_are_classified_without_losing_command_evidence(self) -> None:
        cases = {
            "E: Unable to locate package unknown-app": SoftwareErrorCode.PACKAGE_NOT_FOUND.value,
            "Could not resolve host: archive.ubuntu.com": SoftwareErrorCode.NETWORK_ERROR.value,
            "E: Unmet dependencies. Try 'apt --fix-broken install'": SoftwareErrorCode.DEPENDENCY_ERROR.value,
            "E: Could not get lock /var/lib/dpkg/lock-frontend": SoftwareErrorCode.PERMISSION_ERROR.value,
            "package manager exited unexpectedly": SoftwareErrorCode.INSTALLATION_ERROR.value,
        }
        for stderr, expected in cases.items():
            self.assertEqual(
                _classify_command_error(
                    {"exit_code": 100, "stdout": "stdout evidence", "stderr": stderr, "timed_out": False}
                ),
                expected,
            )

        def fail_handler(_args, _cancel_event):
            raise ToolExecutionError(
                SoftwareErrorCode.PACKAGE_NOT_FOUND.value,
                "E: Unable to locate package unknown-app",
                {"exit_code": 100, "stdout": "stdout evidence", "stderr": "E: Unable to locate package unknown-app", "timed_out": False},
            )

        definition = ToolDefinition(
            "test_package_failure",
            "test package failure",
            {"type": "object", "properties": {}, "additionalProperties": False},
            {"type": "object"},
            PermissionLevel.WRITE,
            2,
            fail_handler,
            "Test package failure",
            safe_software=True,
            confirmation_required=True,
        )
        with TemporaryDirectory() as temp:
            events = list(
                ToolRegistry((definition,), ToolPolicy(allowed_roots=(Path(temp),))).execute_stream(
                    ToolRequest("test_package_failure", {}),
                    threading.Event(),
                    confirmation=True,
                    software=True,
                )
            )
        result = events[-1].result
        self.assertIsNotNone(result)
        self.assertEqual(result.error_code, SoftwareErrorCode.PACKAGE_NOT_FOUND.value)
        self.assertEqual(result.data["exit_code"], 100)
        self.assertIn("Unable to locate", result.data["stderr"])

    def test_unknown_application_is_discovered_before_confirmation(self) -> None:
        class DiscoveryRegistry:
            def __init__(self) -> None:
                self.requests: list[ToolRequest] = []

            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False, software=False):
                del cancel_event, confirmation, diagnostic, software
                self.requests.append(request)
                if request.name == "software_system_profile":
                    data = {
                        "distribution": "Test Linux",
                        "distribution_id": "test",
                        "version": "1",
                        "architecture": "x86_64",
                        "package_manager": "apt",
                        "available_managers": ["apt"],
                        "sources": ["distribution repositories"],
                    }
                elif request.name == "software_search":
                    data = {
                        "manager": "apt",
                        "query": request.arguments["query"],
                        "exit_code": 0,
                        "stdout": "libreoffice - office productivity suite",
                        "stderr": "",
                    }
                elif request.name == "software_query":
                    data = {"manager": "apt", "package": request.arguments["package"], "installed": False, "version": ""}
                else:
                    data = {}
                result = ToolResult(request.name, True, data)
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

        with TemporaryDirectory() as temp:
            registry = DiscoveryRegistry()
            manager = SoftwareManager(None, registry, SoftwareHistory(Path(temp) / "history.jsonl"))
            events: list = []
            thread = threading.Thread(
                target=lambda: events.extend(manager.stream("Install Libre Office", threading.Event()))
            )
            thread.start()
            for _ in range(200):
                if any(event.kind == "software_plan" for event in events):
                    break
                threading.Event().wait(0.01)
            plan_event = next(event for event in events if event.kind == "software_plan")
            self.assertEqual(plan_event.software_plan.package_name, "libreoffice")
            self.assertNotEqual(plan_event.software_plan.package_name, "libre-office")
            self.assertTrue(manager.approve(plan_event.software_plan.plan_id, False))
            thread.join(2)
            self.assertFalse(thread.is_alive())
            search_queries = [
                request.arguments["query"]
                for request in registry.requests
                if request.name == "software_search"
            ]
            self.assertEqual(search_queries, ["libre office"])

    def test_chrome_package_failure_recovers_to_allowlisted_official_source(self) -> None:
        class ChromeRegistry:
            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False, software=False):
                del cancel_event, confirmation, diagnostic, software
                if request.name == "software_system_profile":
                    result = ToolResult(
                        request.name,
                        True,
                        {
                            "distribution": "Test Linux",
                            "distribution_id": "test",
                            "version": "1",
                            "architecture": "x86_64",
                            "package_manager": "apt",
                            "available_managers": ["apt"],
                            "sources": ["distribution repositories"],
                        },
                    )
                    yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)
                    return
                if request.name == "software_query":
                    result = ToolResult(request.name, True, {"installed": False, "version": ""})
                    yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)
                    return
                result = ToolResult(
                    request.name,
                    False,
                    {
                        "exit_code": 100,
                        "stdout": "",
                        "stderr": "E: Unable to locate package google-chrome-stable",
                        "timed_out": False,
                    },
                    error_code=SoftwareErrorCode.PACKAGE_NOT_FOUND.value,
                    error_message="E: Unable to locate package google-chrome-stable",
                )
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.FAILED, "failed", result)

        with TemporaryDirectory() as temp:
            manager = SoftwareManager(None, ChromeRegistry(), SoftwareHistory(Path(temp) / "history.jsonl"))
            events: list = []
            thread = threading.Thread(
                target=lambda: events.extend(manager.stream("Install Google Chrome", threading.Event()))
            )
            thread.start()
            for _ in range(200):
                if any(event.kind == "software_plan" for event in events):
                    break
                threading.Event().wait(0.01)
            plan_event = next(event for event in events if event.kind == "software_plan")
            plan = plan_event.software_plan
            self.assertEqual(plan.package_name, "google-chrome-stable")
            self.assertEqual(plan.source, SoftwareSource.OFFICIAL_VENDOR)
            self.assertNotIn("google-chrom'", plan.command_preview)
            self.assertTrue(manager.approve(plan.plan_id, False))
            thread.join(2)
            self.assertFalse(thread.is_alive())

    def test_install_failure_is_analyzed_and_retried_with_a_trusted_alternative(self) -> None:
        class RecoveryRegistry:
            def __init__(self) -> None:
                self.requests: list[ToolRequest] = []
                self.install_attempts = 0

            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False, software=False):
                del cancel_event, confirmation, diagnostic, software
                self.requests.append(request)
                if request.name == "software_system_profile":
                    data = {
                        "distribution": "Test Linux",
                        "distribution_id": "test",
                        "version": "1",
                        "architecture": "x86_64",
                        "package_manager": "apt",
                        "available_managers": ["apt", "flatpak"],
                        "sources": ["distribution repositories", "Flatpak / Flathub"],
                    }
                    result = ToolResult(request.name, True, data)
                    yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)
                    return
                if request.name == "software_query":
                    result = ToolResult(
                        request.name,
                        True,
                        {"installed": False, "version": "", "manager": request.arguments["manager"]},
                    )
                    yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)
                    return
                if request.name == "software_search":
                    if request.arguments["manager"] == "apt":
                        data = {"manager": "apt", "query": request.arguments["query"], "stdout": "vlc - media player"}
                    else:
                        data = {
                            "manager": "flatpak",
                            "query": request.arguments["query"],
                            "stdout": "org.videolan.VLC VLC media player 3.0",
                        }
                    result = ToolResult(request.name, True, {"exit_code": 0, "stderr": "", **data})
                    yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)
                    return
                if request.name == "software_install":
                    self.install_attempts += 1
                    if self.install_attempts == 1:
                        result = ToolResult(
                            request.name,
                            False,
                            {
                                "exit_code": 100,
                                "stdout": "",
                                "stderr": "E: Unable to locate package vlc",
                                "timed_out": False,
                            },
                            error_code=SoftwareErrorCode.PACKAGE_NOT_FOUND.value,
                            error_message="E: Unable to locate package vlc",
                        )
                        yield ToolEvent(request.name, request.name, request.name, ToolEventKind.FAILED, "failed", result)
                        return
                    result = ToolResult(request.name, True, {"exit_code": 0, "stdout": "installed", "stderr": ""})
                    yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)
                    return
                if request.name == "software_verify":
                    result = ToolResult(request.name, True, {"installed": True, "verified": True, "version": "3.0"})
                    yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)
                    return
                result = ToolResult(request.name, True, {})
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

        class RecoveryProvider:
            def __init__(self) -> None:
                self.failures: list[tuple[str, str, str, ToolResult, tuple[str, ...]]] = []

            def stream_software_failure(
                self,
                request,
                operation,
                attempted_action,
                result,
                alternatives,
                cancel_event,
            ):
                self.failures.append((request, operation, attempted_action, result, alternatives))
                if not cancel_event.is_set():
                    yield ProviderEvent.status("Analyzing software failure...")
                    yield ProviderEvent.software_recovery_ready(
                        SoftwareRecoveryDecision("retry_alternative", "The trusted Flatpak result is available.")
                    )

        with TemporaryDirectory() as temp:
            registry = RecoveryRegistry()
            provider = RecoveryProvider()
            manager = SoftwareManager(provider, registry, SoftwareHistory(Path(temp) / "history.jsonl"))
            events: list = []
            thread = threading.Thread(target=lambda: events.extend(manager.stream("Install VLC", threading.Event())))
            thread.start()
            approved: set[str] = set()
            for _ in range(500):
                for event in events:
                    if event.kind != "software_plan" or event.software_plan.plan_id in approved:
                        continue
                    self.assertTrue(manager.approve(event.software_plan.plan_id, True))
                    approved.add(event.software_plan.plan_id)
                if not thread.is_alive():
                    break
                threading.Event().wait(0.01)
            thread.join(2)
            self.assertFalse(thread.is_alive())
            install_requests = [
                (request.arguments["manager"], request.arguments["package"])
                for request in registry.requests
                if request.name == "software_install"
            ]
            self.assertEqual(install_requests, [("apt", "vlc"), ("flatpak", "org.videolan.VLC")])
            stage_ids = [event.stage_event.stage_id for event in events if event.kind == "stage" and event.stage_event]
            self.assertIn("analyze", stage_ids)
            self.assertIn("recover", stage_ids)
            self.assertIn("retry", stage_ids)
            self.assertIn("successfully installed", "".join(event.text for event in events if event.kind == "text"))
            self.assertEqual(len(provider.failures), 1)
            self.assertEqual(provider.failures[0][1], "install")
            self.assertIn("PACKAGE_NOT_FOUND", provider.failures[0][3].error_code)
            self.assertIn("flatpak:org.videolan.VLC via Flatpak / Flathub", provider.failures[0][4])

    def test_update_all_reaches_an_approval_plan(self) -> None:
        class UpdateAllRegistry:
            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False, software=False):
                del cancel_event, confirmation, diagnostic, software
                data = {}
                if request.name == "software_system_profile":
                    data = {
                        "distribution": "Test Linux",
                        "distribution_id": "test",
                        "version": "1",
                        "architecture": "x86_64",
                        "package_manager": "apt",
                        "available_managers": ["apt"],
                        "sources": ["distribution repositories"],
                    }
                result = ToolResult(request.name, True, data)
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

        with TemporaryDirectory() as temp:
            manager = SoftwareManager(
                None,
                UpdateAllRegistry(),
                SoftwareHistory(Path(temp) / "history.jsonl"),
            )
            events: list = []
            thread = threading.Thread(
                target=lambda: events.extend(manager.stream("Update all my software", threading.Event()))
            )
            thread.start()
            while thread.is_alive() and not any(event.kind == "software_plan" for event in events):
                threading.Event().wait(0.01)
            plan_event = next(event for event in events if event.kind == "software_plan")
            self.assertEqual(plan_event.software_plan.package_name, "all")
            self.assertIn("apt-get upgrade -y", plan_event.software_plan.command_preview)
            self.assertTrue(manager.approve(plan_event.software_plan.plan_id, False))
            thread.join(2)
            self.assertFalse(thread.is_alive())

    def test_download_detects_steam_installed_through_snap(self) -> None:
        class SteamRegistry:
            def __init__(self) -> None:
                self.queries: list[tuple[str, str]] = []

            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False, software=False):
                del cancel_event, confirmation, diagnostic, software
                if request.name == "software_system_profile":
                    data = {
                        "distribution": "Ubuntu",
                        "distribution_id": "ubuntu",
                        "version": "26.04",
                        "architecture": "x86_64",
                        "package_manager": "apt",
                        "available_managers": ["apt", "flatpak", "snap"],
                        "sources": ["distribution repositories", "Snap Store"],
                    }
                elif request.name == "software_query":
                    self.queries.append((request.arguments["manager"], request.arguments["package"]))
                    data = {
                        "installed": request.arguments["manager"] == "snap",
                        "version": "1.0.0.85" if request.arguments["manager"] == "snap" else "",
                    }
                elif request.name == "software_available_version":
                    data = {"available": True, "version": "1.0.0.85"}
                else:
                    data = {}
                result = ToolResult(request.name, True, data)
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

        with TemporaryDirectory() as temp:
            registry = SteamRegistry()
            manager = SoftwareManager(
                None,
                registry,
                SoftwareHistory(Path(temp) / "history.jsonl"),
            )
            events = list(manager.stream("Download Steam", threading.Event()))
            state = next(event.software_state for event in events if event.kind == "software_state")
            self.assertTrue(state.installed)
            self.assertEqual(state.manager, "snap")
            self.assertEqual(state.package_name, "steam")
            self.assertEqual(state.current_version, "1.0.0.85")
            self.assertFalse(any(event.kind == "software_plan" for event in events))
            self.assertIn(("apt", "steam-installer"), registry.queries)
            self.assertIn(("snap", "steam"), registry.queries)
            self.assertIn("✅ Software is already installed.", "".join(event.text for event in events if event.kind == "text"))

    def test_software_tools_are_metadata_and_confirmation_gated(self) -> None:
        with TemporaryDirectory() as temp:
            policy = ToolPolicy(allowed_roots=(Path(temp),))
            registry = ToolRegistry(
                (*create_tool_definitions((Path(temp),)), *create_software_tool_definitions((Path(temp),))),
                policy,
            )
            install = ToolRequest("software_install", {"manager": "apt", "package": "vlc"})
            with self.assertRaises(PermissionError):
                registry.validate(install, software=True)
            self.assertEqual(registry.validate(install, confirmation=True, software=True).name, "software_install")

            download = ToolRequest(
                "software_download",
                {"manager": "apt", "package": "vlc", "destination": str(Path(temp) / "downloads")},
            )
            with self.assertRaises(PermissionError):
                registry.validate(download, software=True)
            self.assertEqual(
                registry.validate(download, confirmation=True, software=True).name,
                "software_download",
            )

    def test_registry_emits_bounded_download_progress(self) -> None:
        with TemporaryDirectory() as temp:
            def handler(_args, _cancel_event, report):
                report(512, 1_024, 256.0)
                return {"path": str(Path(temp) / "package.deb")}

            from tools.contracts import PermissionLevel, ToolDefinition

            definition = ToolDefinition(
                "test_download",
                "test",
                {"type": "object", "properties": {}, "additionalProperties": False},
                {"type": "object"},
                PermissionLevel.NETWORK,
                2,
                handler,
                "Test download",
                safe_software=True,
                reports_progress=True,
                confirmation_required=True,
            )
            registry = ToolRegistry((definition,), ToolPolicy(allowed_roots=(Path(temp),)))
            events = list(
                registry.execute_stream(
                    ToolRequest("test_download", {}),
                    threading.Event(),
                    confirmation=True,
                    software=True,
                )
            )
            progress = [event for event in events if event.kind is ToolEventKind.PROGRESS and event.progress]
            self.assertEqual(len(progress), 1)
            self.assertEqual(progress[0].progress.percent, 50.0)
            self.assertIn("50%", progress[0].message)

    def test_apt_download_uses_apt_download_command(self) -> None:
        from tools.linux_tools import PathGuard
        from tools.software_tools import _download

        with TemporaryDirectory() as temp:
            destination = Path(temp) / "downloads"
            captured: list[list[str]] = []

            def fake_run(argv, _cancel_event, _timeout, **_kwargs):
                captured.append(argv)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "vlc_amd64.deb").write_bytes(b"package")
                return {"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False}

            with patch("tools.software_tools.shutil.which", return_value="/usr/bin/apt"), patch(
                "tools.software_tools._run_argv", side_effect=fake_run
            ):
                result = _download(
                    {"manager": "apt", "package": "vlc", "destination": str(destination)},
                    threading.Event(),
                    PathGuard((Path(temp),)),
                )

            self.assertEqual(captured, [["apt", "download", "vlc"]])
            self.assertEqual(result["path"], str(destination / "vlc_amd64.deb"))

    def test_download_reuses_existing_package_without_running_package_manager(self) -> None:
        from tools.linux_tools import PathGuard

        with TemporaryDirectory() as temp:
            destination = Path(temp) / "downloads"
            destination.mkdir()
            package = destination / "vlc_3.0.21_amd64.deb"
            package.write_bytes(b"already downloaded package")
            with patch("tools.software_tools.shutil.which") as which, patch(
                "tools.software_tools._run_argv"
            ) as run:
                result = _download(
                    {"manager": "apt", "package": "vlc", "destination": str(destination)},
                    threading.Event(),
                    PathGuard((Path(temp),)),
                )

            which.assert_not_called()
            run.assert_not_called()
            self.assertTrue(result["already_downloaded"])
            self.assertFalse(result["downloaded"])
            self.assertEqual(result["path"], str(package))

    def test_official_vendor_download_reuses_existing_installer_without_network(self) -> None:
        from tools.linux_tools import PathGuard

        with TemporaryDirectory() as temp:
            destination = Path(temp) / "downloads"
            destination.mkdir()
            installer = destination / "google-chrome-stable_current_amd64.deb"
            installer.write_bytes(b"already downloaded installer")
            with patch("urllib.request.Request") as request, patch("urllib.request.build_opener") as opener:
                result = _vendor_download(
                    {"vendor": "google_chrome", "manager": "apt", "destination": str(destination)},
                    threading.Event(),
                    PathGuard((Path(temp),)),
                )

            request.assert_not_called()
            opener.assert_not_called()
            self.assertTrue(result["already_downloaded"])
            self.assertFalse(result["downloaded"])
            self.assertEqual(result["path"], str(installer))

    def test_plan_waits_for_trusted_decision_and_cancel_runs_no_tool_change(self) -> None:
        class FakeRegistry:
            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False, software=False):
                del cancel_event, confirmation, diagnostic, software
                event_id = request.name
                if request.name == "software_system_profile":
                    data = {
                        "distribution": "Test Linux",
                        "distribution_id": "test",
                        "version": "1",
                        "architecture": "x86_64",
                        "package_manager": "apt",
                        "available_managers": ["apt"],
                        "sources": ["distribution repositories"],
                    }
                elif request.name == "software_search":
                    data = {"exit_code": 0, "stdout": "vlc - media player"}
                else:
                    data = {}
                result = ToolResult(request.name, True, data)
                yield ToolEvent(event_id, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

        with TemporaryDirectory() as temp:
            manager = SoftwareManager(
                None,
                FakeRegistry(),
                SoftwareHistory(Path(temp) / "history.jsonl"),
            )
            events: list = []
            thread = threading.Thread(
                target=lambda: events.extend(manager.stream("Install VLC", threading.Event()))
            )
            thread.start()
            while thread.is_alive() and not any(event.kind == "software_plan" for event in events):
                threading.Event().wait(0.01)
            plan_event = next(event for event in events if event.kind == "software_plan")
            self.assertTrue(manager.approve(plan_event.software_plan.plan_id, False))
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertIn("No changes were made", "".join(event.text for event in events if event.kind == "text"))
            self.assertEqual(manager.history.recent()[0]["status"], "cancelled")

    def test_installed_software_is_not_installed_again_and_exposes_state_actions(self) -> None:
        class InstalledRegistry:
            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False, software=False):
                del cancel_event, confirmation, diagnostic, software
                if request.name == "software_system_profile":
                    data = {
                        "distribution": "Test Linux",
                        "distribution_id": "test",
                        "version": "1",
                        "architecture": "x86_64",
                        "package_manager": "apt",
                        "available_managers": ["apt"],
                        "sources": ["distribution repositories"],
                    }
                elif request.name == "software_query":
                    data = {"installed": True, "version": "3.0.21"}
                elif request.name == "software_available_version":
                    data = {"available": True, "version": "3.0.21"}
                else:
                    data = {}
                result = ToolResult(request.name, True, data)
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

        with TemporaryDirectory() as temp:
            manager = SoftwareManager(
                None,
                InstalledRegistry(),
                SoftwareHistory(Path(temp) / "history.jsonl"),
            )
            events = list(manager.stream("Install VLC", threading.Event()))
            state = next(event.software_state for event in events if event.kind == "software_state")
            self.assertTrue(state.installed)
            self.assertEqual(state.current_version, "3.0.21")
            self.assertEqual([action.value for action in state.actions], ["remove", "reinstall"])
            self.assertFalse(any(event.kind == "software_plan" for event in events))
            text = "".join(event.text for event in events if event.kind == "text")
            self.assertIn("✅ Software is already installed.", text)
            self.assertIn("already installed", text)
            self.assertIn("No additional download or installation", text)

    def test_healthy_software_troubleshooting_reports_normal_without_plan(self) -> None:
        class HealthyRegistry:
            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False, software=False):
                del cancel_event, confirmation, diagnostic, software
                if request.name == "software_system_profile":
                    data = {
                        "distribution": "Test Linux",
                        "distribution_id": "test",
                        "version": "1",
                        "architecture": "x86_64",
                        "package_manager": "apt",
                        "available_managers": ["apt"],
                        "sources": ["distribution repositories"],
                    }
                elif request.name == "software_query":
                    data = {"installed": True, "version": "3.0.21"}
                elif request.name == "software_search":
                    data = {"exit_code": 0, "stdout": "vlc - media player"}
                elif request.name == "software_verify":
                    data = {"installed": True, "verified": True, "version": "3.0.21"}
                else:
                    data = {}
                result = ToolResult(request.name, True, data)
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

        with TemporaryDirectory() as temp:
            manager = SoftwareManager(
                None,
                HealthyRegistry(),
                SoftwareHistory(Path(temp) / "history.jsonl"),
            )
            events = list(manager.stream("Troubleshoot VLC", threading.Event()))
            text = "".join(event.text for event in events if event.kind == "text")
            self.assertIn("✅ Everything is normal. No problems were detected.", text)
            self.assertIn("verification passed", text)
            self.assertFalse(any(event.kind == "software_plan" for event in events))
            self.assertEqual(manager.state.task_state.value, "COMPLETED")
            self.assertTrue(manager.state.verification_complete)
            self.assertEqual(events[-1].kind, "done")

    def test_generic_software_check_asks_for_target_without_running_tools(self) -> None:
        class Registry:
            def __init__(self) -> None:
                self.requests = []

            def execute_stream(self, request, **_kwargs):
                self.requests.append(request)
                raise AssertionError("a missing software target must not run diagnostics")

        with TemporaryDirectory() as temp:
            registry = Registry()
            manager = SoftwareManager(
                None,
                registry,
                SoftwareHistory(Path(temp) / "history.jsonl"),
            )
            events = list(manager.stream("troubleshoot and check the software", threading.Event()))
            text = "".join(event.text for event in events if event.kind == "text")
            self.assertIn("Which software would you like me to check?", text)
            self.assertFalse(registry.requests)
            self.assertEqual(manager.history.recent()[0]["status"], "waiting_for_target")

    def test_named_software_check_fails_when_dependency_health_is_bad(self) -> None:
        class BrokenRegistry:
            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False, software=False):
                del cancel_event, confirmation, diagnostic, software
                data = {
                    "software_system_profile": {
                        "distribution": "Test Linux",
                        "architecture": "x86_64",
                        "package_manager": "apt",
                        "available_managers": ["apt"],
                    },
                    "software_query": {"installed": True, "version": "3.0.21"},
                    "package_health": {"healthy": False, "broken_packages": ["vlc"]},
                    "software_verify": {
                        "installed": True,
                        "verified": True,
                        "executable_path": "/usr/bin/vlc",
                    },
                }.get(request.name, {})
                result = ToolResult(request.name, True, data)
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

        with TemporaryDirectory() as temp:
            manager = SoftwareManager(
                None,
                BrokenRegistry(),
                SoftwareHistory(Path(temp) / "history.jsonl"),
            )
            events = list(manager.stream("Check VLC", threading.Event()))
            text = "".join(event.text for event in events if event.kind == "text")
            self.assertIn("Problem detected", text)
            self.assertIn("dependency problem", text)
            self.assertEqual(manager.history.recent()[0]["status"], "failed")
            verification = [
                event.stage_event
                for event in events
                if event.kind == "stage" and event.stage_event and event.stage_event.stage_id == "verify"
            ]
            self.assertTrue(verification)
            self.assertEqual(verification[-1].status.value, "failed")

    def test_overall_software_check_reports_normal_only_after_scoped_checks(self) -> None:
        class HealthyRegistry:
            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False, software=False):
                del cancel_event, confirmation, diagnostic, software
                data = {
                    "software_system_profile": {
                        "distribution": "Test Linux",
                        "architecture": "x86_64",
                        "package_manager": "apt",
                        "available_managers": ["apt"],
                    },
                    "package_health": {"healthy": True},
                    "package_update_status": {"healthy": True, "updates_available": False},
                    "service_failures": {"failed_count": 0},
                    "recent_failures": {"entry_count": 0},
                }.get(request.name, {})
                result = ToolResult(request.name, True, data)
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

        with TemporaryDirectory() as temp:
            manager = SoftwareManager(
                None,
                HealthyRegistry(),
                SoftwareHistory(Path(temp) / "history.jsonl"),
            )
            events = list(manager.stream("check the overall software environment", threading.Event()))
            text = "".join(event.text for event in events if event.kind == "text")
            self.assertIn("✅ Everything is normal. No problems were detected.", text)
            self.assertIn("Software check completed", text)
            self.assertEqual(manager.history.recent()[0]["status"], "completed")

    def test_update_plan_has_state_check_and_exact_safe_command_preview(self) -> None:
        class UpdateRegistry:
            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False, software=False):
                del cancel_event, confirmation, diagnostic, software
                if request.name == "software_system_profile":
                    data = {
                        "distribution": "Test Linux",
                        "distribution_id": "test",
                        "version": "1",
                        "architecture": "x86_64",
                        "package_manager": "apt",
                        "available_managers": ["apt"],
                        "sources": ["distribution repositories"],
                    }
                elif request.name == "software_query":
                    data = {"installed": True, "version": "3.0.20"}
                elif request.name == "software_available_version":
                    data = {"available": True, "version": "3.0.21"}
                elif request.name == "software_search":
                    data = {"exit_code": 0, "stdout": "vlc - media player"}
                else:
                    data = {}
                result = ToolResult(request.name, True, data)
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

        with TemporaryDirectory() as temp:
            manager = SoftwareManager(
                None,
                UpdateRegistry(),
                SoftwareHistory(Path(temp) / "history.jsonl"),
            )
            events = []
            thread = threading.Thread(
                target=lambda: events.extend(manager.stream("Update VLC", threading.Event()))
            )
            thread.start()
            while thread.is_alive() and not any(event.kind == "software_plan" for event in events):
                threading.Event().wait(0.01)
            plan_event = next(event for event in events if event.kind == "software_plan")
            plan = plan_event.software_plan
            self.assertIn("apt-get install -y --only-upgrade vlc", plan.command_preview)
            self.assertEqual(plan.current_version, "3.0.20")
            self.assertEqual(plan.available_version, "3.0.21")
            self.assertEqual(plan.risk, "Medium")
            self.assertTrue(manager.approve(plan.plan_id, False))
            thread.join(2)
            self.assertFalse(thread.is_alive())

    def test_removal_checks_installed_state_and_waits_for_approval(self) -> None:
        class RemovalRegistry:
            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False, software=False):
                del cancel_event, confirmation, diagnostic, software
                if request.name == "software_system_profile":
                    data = {
                        "distribution": "Test Linux",
                        "distribution_id": "test",
                        "version": "1",
                        "architecture": "x86_64",
                        "package_manager": "apt",
                        "available_managers": ["apt"],
                        "sources": ["distribution repositories"],
                    }
                elif request.name == "software_query":
                    data = {"installed": True, "version": "128.0"}
                else:
                    data = {}
                result = ToolResult(request.name, True, data)
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

        with TemporaryDirectory() as temp:
            manager = SoftwareManager(
                None,
                RemovalRegistry(),
                SoftwareHistory(Path(temp) / "history.jsonl"),
            )
            events: list = []
            thread = threading.Thread(target=lambda: events.extend(manager.stream("Remove Firefox", threading.Event())))
            thread.start()
            for _ in range(200):
                if any(event.kind == "software_plan" for event in events):
                    break
                threading.Event().wait(0.01)
            plan_event = next(event for event in events if event.kind == "software_plan")
            self.assertEqual(plan_event.software_plan.package_name, "firefox")
            self.assertIn("apt-get remove -y firefox", plan_event.software_plan.command_preview)
            self.assertTrue(manager.approve(plan_event.software_plan.plan_id, False))
            thread.join(2)
            self.assertFalse(thread.is_alive())

    def test_removal_of_missing_software_offers_no_install_actions(self) -> None:
        class MissingRemovalRegistry:
            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False, software=False):
                del cancel_event, confirmation, diagnostic, software
                if request.name == "software_system_profile":
                    data = {
                        "distribution": "Test Linux",
                        "distribution_id": "test",
                        "version": "1",
                        "architecture": "x86_64",
                        "package_manager": "apt",
                        "available_managers": ["apt"],
                        "sources": ["distribution repositories"],
                    }
                elif request.name == "software_query":
                    data = {"installed": False, "version": ""}
                else:
                    data = {}
                result = ToolResult(request.name, True, data)
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

        with TemporaryDirectory() as temp:
            manager = SoftwareManager(
                None,
                MissingRemovalRegistry(),
                SoftwareHistory(Path(temp) / "history.jsonl"),
            )
            events = list(manager.stream("Uninstall Google Chrome", threading.Event()))
            state = next(event.software_state for event in events if event.kind == "software_state")
            self.assertFalse(state.installed)
            self.assertEqual(state.actions, ())
            self.assertNotIn("software_plan", [event.kind for event in events])
            text = "".join(event.text for event in events if event.kind == "text")
            self.assertIn("not installed", text)
            self.assertIn("No removal command was run", text)

    def test_download_reports_saved_path_after_verification(self) -> None:
        class DownloadRegistry:
            def __init__(self, path: Path) -> None:
                self.path = path

            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False, software=False):
                del cancel_event, confirmation, diagnostic, software
                data = {}
                if request.name == "software_system_profile":
                    data = {
                        "distribution": "Test Linux",
                        "distribution_id": "test",
                        "version": "1",
                        "architecture": "x86_64",
                        "package_manager": "apt",
                        "available_managers": ["apt"],
                        "sources": ["distribution repositories"],
                    }
                elif request.name == "software_search":
                    data = {"exit_code": 0, "stdout": "vlc - media player"}
                elif request.name == "software_download":
                    data = {"path": str(self.path), "bytes": self.path.stat().st_size}
                elif request.name == "software_verify_download":
                    data = {"path": str(self.path), "verified": True}
                result = ToolResult(request.name, True, data)
                yield ToolEvent(
                    request.name,
                    request.name,
                    request.name,
                    ToolEventKind.COMPLETED,
                    "completed",
                    result,
                )

        with TemporaryDirectory() as temp:
            package = Path(temp) / "vlc_amd64.deb"
            package.write_bytes(b"package")
            manager = SoftwareManager(
                None,
                DownloadRegistry(package),
                SoftwareHistory(Path(temp) / "history.jsonl"),
            )
            events: list = []
            thread = threading.Thread(
                target=lambda: events.extend(manager.stream("Download VLC", threading.Event()))
            )
            thread.start()
            while thread.is_alive() and not any(event.kind == "software_plan" for event in events):
                threading.Event().wait(0.01)
            plan_event = next(event for event in events if event.kind == "software_plan")
            self.assertTrue(manager.approve(plan_event.software_plan.plan_id, True))
            thread.join(2)
            self.assertFalse(thread.is_alive())
            text = "".join(event.text for event in events if event.kind == "text")
            self.assertIn("downloaded and verified", text)
            self.assertIn(str(package), text)
            stage_titles = [event.stage_event.title for event in events if event.kind == "stage" and event.stage_event]
            self.assertIn("Downloading software", stage_titles)
            self.assertIn("Verifying download", stage_titles)


if __name__ == "__main__":
    unittest.main()
