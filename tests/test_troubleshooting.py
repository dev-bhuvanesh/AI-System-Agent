from __future__ import annotations

import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm.provider import ChatProvider, DiagnosticDecision, ProviderEvent
from tools.contracts import PermissionLevel, ToolDefinition, ToolEvent, ToolEventKind, ToolResult, ToolRequest
from tools.registry import ToolPolicy, ToolRegistry
from troubleshooting.engine import TroubleshootingEngine, _assess, _diagnostic_steps, _fix_for, _normal_report
from troubleshooting.history import TroubleshootingHistory
from troubleshooting.contracts import TroubleshootingCategory, TroubleshootingOutcome


class FakeChatProvider(ChatProvider):
    def stream_chat(self, messages, cancel_event):
        yield ProviderEvent.text_chunk("The diagnostic evidence is ready.")
        yield ProviderEvent.done()


class FakeRegistry:
    def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False):
        event_id = request.name + str(len(request.arguments))
        yield ToolEvent(
            event_id,
            request.name,
            request.name,
            ToolEventKind.STARTED,
            f"{request.name} — In progress...",
        )
        yield ToolEvent(
            event_id,
            request.name,
            request.name,
            ToolEventKind.COMPLETED,
            f"{request.name} — Completed",
            ToolResult(request.name, True, {"diagnostic": True}),
        )


class TroubleshootingTests(unittest.TestCase):
    def _healthy_network_results(self):
        return {
            "network_interfaces": {"interfaces": [{"name": "eth0", "operstate": "up", "carrier": "1", "link_up": True}]},
            "routing_info": {"routes": [{"destination_hex": "00000000", "interface": "eth0"}]},
            "gateway_detection": {"gateways": [{"interface": "eth0", "gateway": "192.168.1.1"}]},
            "dns_info": {"nameservers": ["1.1.1.1"]},
            "ping_connectivity": {"reachable": True},
            "service_status": {"active_state": "active"},
        }

    def test_common_problem_categories_are_detected_without_matching_normal_chat(self):
        self.assertEqual(TroubleshootingEngine.classify("my internet is not working"), TroubleshootingCategory.NETWORK)
        self.assertEqual(TroubleshootingEngine.classify("Wi-Fi is disconnected"), TroubleshootingCategory.WIFI)
        self.assertEqual(TroubleshootingEngine.classify("Whfi is disconnected"), TroubleshootingCategory.WIFI)
        self.assertEqual(TroubleshootingEngine.classify("my system is slow"), TroubleshootingCategory.PERFORMANCE)
        self.assertEqual(TroubleshootingEngine.classify("Chrome is slow"), TroubleshootingCategory.PERFORMANCE)
        self.assertEqual(TroubleshootingEngine.classify("troubleshoot my computer"), TroubleshootingCategory.GENERAL)
        self.assertEqual(TroubleshootingEngine.classify("run a diagnostic check"), TroubleshootingCategory.GENERAL)
        self.assertEqual(TroubleshootingEngine.classify("the network cable is loose"), TroubleshootingCategory.PHYSICAL)
        self.assertEqual(TroubleshootingEngine.classify("I have a hardware problem"), TroubleshootingCategory.HARDWARE)
        self.assertEqual(TroubleshootingEngine.classify("install google chrome"), None)

    def test_diagnostic_steps_follow_the_reported_problem(self):
        chrome_titles = [step.title for step in _diagnostic_steps(
            TroubleshootingCategory.PERFORMANCE,
            "Chrome is slow",
        )]
        generic_titles = [step.title for step in _diagnostic_steps(
            TroubleshootingCategory.PERFORMANCE,
            "my system is slow",
        )]
        self.assertIn("Checking browser processes", chrome_titles)
        self.assertIn("Checking running processes", generic_titles)

    def test_wifi_diagnostic_steps_start_with_radio_cause_checks(self):
        steps = _diagnostic_steps(TroubleshootingCategory.WIFI, "troubleshoot wifi")
        self.assertEqual(
            [step.request.name for step in steps[:5]],
            ["network_management_info", "wifi_hardware_info", "wifi_interface_info", "wifi_radio_state", "rfkill_status"],
        )
        self.assertLess(
            next(index for index, step in enumerate(steps) if step.request.name == "wifi_radio_state"),
            next(index for index, step in enumerate(steps) if step.request.name == "routing_info"),
        )

    def test_wifi_off_is_primary_cause_and_missing_network_data_is_secondary(self):
        results = [
            ToolResult("network_management_info", True, {"available_tools": ["nmcli", "rfkill"], "active_manager": "NetworkManager"}),
            ToolResult("wifi_hardware_info", True, {"hardware_detected": True, "interfaces": ["wlo1"]}),
            ToolResult("wifi_interface_info", True, {"interface": "wlo1", "exists": True}),
            ToolResult("wifi_radio_state", True, {"interface": "wlo1", "radio_enabled": False, "software_blocked": True, "hardware_blocked": False}),
            ToolResult("rfkill_status", True, {"software_blocked": True, "hardware_blocked": False}),
            ToolResult("wifi_connection", True, {"interface": "wlo1", "connected": False}),
            ToolResult("wifi_ip_info", True, {"interface": "wlo1", "has_ipv4": False, "ip_address": None}),
            ToolResult("routing_info", True, {"routes": []}),
            ToolResult("gateway_detection", True, {"gateways": []}),
            ToolResult("gateway_connectivity", True, {"reachable": None}),
            ToolResult("dns_info", True, {"nameservers": []}),
            ToolResult("ping_connectivity", True, {"reachable": False}),
        ]
        assessment = _assess(TroubleshootingCategory.WIFI, results)
        self.assertEqual(assessment.primary_cause, "WIFI_DISABLED")
        self.assertEqual(assessment.outcome, TroubleshootingOutcome.SOFTWARE_PROBLEM)
        self.assertIn("consequences", assessment.summary)
        self.assertIn("NO_IP_ADDRESS", assessment.secondary_symptoms)
        self.assertTrue(assessment.automatic_fix_available)
        self.assertEqual(assessment.structured_data["diagnosis"]["primary_cause"], "WIFI_DISABLED")
        self.assertFalse(assessment.structured_data["wifi"]["radio_enabled"])
        self.assertFalse(assessment.structured_data["network"]["default_route"])
        proposal = _fix_for(TroubleshootingCategory.WIFI, "troubleshoot wifi", assessment)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.request.name, "wifi_enable")
        self.assertEqual(proposal.command_preview, "nmcli radio wifi on")

    def test_healthy_wifi_report_contains_only_evidence_bullets(self):
        results = [
            ToolResult("wifi_interface_info", True, {"interface": "wlo1", "exists": True}),
            ToolResult("wifi_connection", True, {"interface": "wlo1", "connected": True}),
            ToolResult("wifi_ip_info", True, {"interface": "wlo1", "ip_address": "192.168.1.20", "has_ipv4": True}),
            ToolResult("gateway_detection", True, {"gateways": [{"interface": "wlo1", "gateway": "192.168.1.1"}]}),
            ToolResult("dns_info", True, {"nameservers": ["127.0.0.53"], "working": True}),
            ToolResult("ping_connectivity", True, {"reachable": True}),
        ]
        assessment = _assess(TroubleshootingCategory.WIFI, [
            ToolResult("network_management_info", True, {"available_tools": ["nmcli"]}),
            ToolResult("wifi_hardware_info", True, {"hardware_detected": True}),
            ToolResult("wifi_interface_info", True, {"interface": "wlo1", "exists": True}),
            ToolResult("wifi_radio_state", True, {"radio_enabled": True}),
            ToolResult("wifi_connection", True, {"interface": "wlo1", "connected": True}),
            ToolResult("wifi_ip_info", True, {"interface": "wlo1", "has_ipv4": True, "ip_address": "192.168.1.20"}),
            ToolResult("routing_info", True, {"routes": [{"interface": "wlo1", "destination_hex": "00000000"}]}),
            ToolResult("gateway_detection", True, {"gateways": [{"interface": "wlo1", "gateway": "192.168.1.1"}]}),
            ToolResult("gateway_connectivity", True, {"reachable": True}),
            ToolResult("dns_info", True, {"nameservers": ["127.0.0.53"], "working": True}),
            ToolResult("ping_connectivity", True, {"reachable": True}),
        ])
        text = _normal_report(TroubleshootingCategory.WIFI, assessment, results)
        self.assertEqual(text.splitlines()[0], "✅ Everything is normal. No problems were detected.")
        self.assertIn("Wi-Fi interface (wlo1) is active and has IP address 192.168.1.20.", text)
        self.assertIn("A default gateway (192.168.1.1) and DNS settings are correctly configured.", text)

    def test_read_only_diagnostics_stream_stages_tools_and_history(self):
        with TemporaryDirectory() as temp:
            history = TroubleshootingHistory(Path(temp) / "history.jsonl")
            engine = TroubleshootingEngine(FakeChatProvider(), FakeRegistry(), history)
            events = list(engine.stream("my system is slow", threading.Event()))
            self.assertTrue(any(event.kind == "stage" for event in events))
            self.assertTrue(any(event.kind == "tool" for event in events))
            self.assertTrue(any(event.kind == "text" for event in events))
            stage_events = [event.stage_event for event in events if event.kind == "stage"]
            self.assertIsNotNone(stage_events[0])
            self.assertNotEqual(stage_events[0].status.value, "pending")
            self.assertEqual(events[-1].kind, "done")
            self.assertEqual(len(history.recent()), 1)
            self.assertEqual(history.recent()[0]["category"], "performance")
            self.assertEqual(engine.state.task_state.value, "COMPLETED")
            self.assertTrue(engine.state.verification_complete)

    def test_qwen_selects_the_next_diagnostic_from_prior_observations(self):
        class AdaptiveProvider(FakeChatProvider):
            def __init__(self):
                self.calls = []

            def stream_diagnostic_decision(
                self,
                request,
                category,
                observations,
                previous_tools,
                cancel_event,
            ):
                self.calls.append((request, category, observations, previous_tools))
                if not observations:
                    yield ProviderEvent.diagnostic_decision_ready(
                        DiagnosticDecision(
                            reason="Start with the approved host probe.",
                            tool_requests=(ToolRequest("safe_probe", {}),),
                        )
                    )
                else:
                    yield ProviderEvent.diagnostic_decision_ready(
                        DiagnosticDecision(done=True, reason="The available evidence is sufficient.")
                    )

        with TemporaryDirectory() as temp:
            definition = ToolDefinition(
                "safe_probe",
                "Safe diagnostic probe",
                {"type": "object", "properties": {}, "additionalProperties": False},
                {"type": "object"},
                PermissionLevel.READ_ONLY,
                2,
                lambda _args, _cancel: {"observed": True},
                "Safe diagnostic probe",
            )
            registry = ToolRegistry(
                (definition,),
                ToolPolicy(allowed_roots=(Path(temp),)),
            )
            provider = AdaptiveProvider()
            engine = TroubleshootingEngine(
                provider,
                registry,
                TroubleshootingHistory(Path(temp) / "history.jsonl"),
                max_steps=4,
                use_dynamic_diagnostics=True,
            )
            events = []
            thread = threading.Thread(
                target=lambda: events.extend(
                    engine.stream("troubleshoot my computer", threading.Event())
                )
            )
            thread.start()
            while thread.is_alive() and not any(event.kind == "fix" for event in events):
                threading.Event().wait(0.01)
            proposal = next(event.fix_proposal for event in events if event.kind == "fix")
            self.assertTrue(engine.choose_fix(proposal.proposal_id, "manual"))
            thread.join(2)
            self.assertFalse(thread.is_alive())

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(provider.calls[0][2], ())
        self.assertEqual(provider.calls[1][3], ("safe_probe",))
        self.assertTrue(any(event.kind == "tool" for event in events))
        self.assertEqual(events[-1].kind, "done")

    def test_fix_waits_for_trusted_user_decision(self):
        with TemporaryDirectory() as temp:
            history = TroubleshootingHistory(Path(temp) / "history.jsonl")
            engine = TroubleshootingEngine(FakeChatProvider(), FakeRegistry(), history)
            events = []

            def run():
                events.extend(engine.stream("my internet is not working", threading.Event()))

            thread = threading.Thread(target=run)
            thread.start()
            while not any(event.kind == "fix" for event in events):
                if not thread.is_alive():
                    break
                threading.Event().wait(0.01)
            proposal = next(event.fix_proposal for event in events if event.kind == "fix")
            self.assertTrue(engine.approve_fix(proposal.proposal_id, False))
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertTrue(any(event.kind == "done" for event in events))
            self.assertIn("cancelled", history.recent()[0]["fix"]["decision"])

    def test_internal_diagnostic_analysis_is_not_shown_before_fix_choice(self):
        class ProviderWithInternalProse(FakeChatProvider):
            def __init__(self):
                self.stream_calls = 0

            def stream_chat(self, messages, cancel_event):
                self.stream_calls += 1
                del messages, cancel_event
                yield ProviderEvent.text_chunk("UNTRUSTED_MODEL_TEXT")
                yield ProviderEvent.done()

        with TemporaryDirectory() as temp:
            events = []
            provider = ProviderWithInternalProse()
            engine = TroubleshootingEngine(
                provider,
                FakeRegistry(),
                TroubleshootingHistory(Path(temp) / "history.jsonl"),
            )
            thread = threading.Thread(
                target=lambda: events.extend(
                    engine.stream("troubleshoot my Wi-Fi", threading.Event())
                )
            )
            thread.start()
            while thread.is_alive() and not any(event.kind == "fix" for event in events):
                threading.Event().wait(0.01)
            proposal = next(event.fix_proposal for event in events if event.kind == "fix")
            text_before_choice = "".join(
                event.text for event in events if event.kind == "text"
            )
            self.assertNotIn("UNTRUSTED_MODEL_TEXT", text_before_choice)
            self.assertEqual(provider.stream_calls, 0)
            self.assertTrue(engine.choose_fix(proposal.proposal_id, "manual"))
            thread.join(2)
            self.assertFalse(thread.is_alive())
            text_after_choice = "".join(
                event.text for event in events if event.kind == "text"
            )
            self.assertIn("Manual troubleshooting", text_after_choice)
            self.assertEqual(events[-1].kind, "done")

    def test_healthy_network_reports_normal_without_fix_action(self):
        class HealthyRegistry:
            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False):
                del cancel_event, confirmation, diagnostic
                data = self._healthy.get(request.name, {})
                result = ToolResult(request.name, True, data)
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

            _healthy = self._healthy_network_results()

        with TemporaryDirectory() as temp:
            engine = TroubleshootingEngine(
                FakeChatProvider(),
                HealthyRegistry(),
                TroubleshootingHistory(Path(temp) / "history.jsonl"),
            )
            events = list(engine.stream("My internet is not working", threading.Event()))
            text = "".join(event.text for event in events if event.kind == "text")
            self.assertIn("Everything is normal", text)
            self.assertIn("✅ Everything is normal. No problems were detected.", text)
            self.assertNotIn("The diagnostic evidence is ready.", text)
            self.assertIn("- A default gateway (192.168.1.1) and DNS settings are correctly configured.", text)
            self.assertNotIn("fix", [event.kind for event in events])
            self.assertEqual(events[-1].kind, "done")
            self.assertEqual(events[-2].kind, "text")

    def test_healthy_general_system_reports_normal_without_fix_action(self):
        class HealthyRegistry:
            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False):
                del cancel_event, confirmation, diagnostic
                data = {
                    "system_info": {"system": "Linux", "release": "6.8", "machine": "x86_64"},
                    "kernel_info": {"release": "6.8", "machine": "x86_64"},
                    "uptime": {"seconds": 3600, "human": "1h 0m 0s"},
                    "cpu_usage": {"usage_percent": 24.5},
                    "ram_usage": {"usage_percent": 42.0},
                    "disk_usage": {"total_bytes": 100, "free_bytes": 50},
                }.get(request.name, {})
                result = ToolResult(request.name, True, data)
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

        with TemporaryDirectory() as temp:
            engine = TroubleshootingEngine(
                FakeChatProvider(),
                HealthyRegistry(),
                TroubleshootingHistory(Path(temp) / "history.jsonl"),
            )
            events = list(engine.stream("Check my system", threading.Event()))
            text = "".join(event.text for event in events if event.kind == "text")
            self.assertIn("✅ Everything is normal. No problems were detected.", text)
            self.assertNotIn("The diagnostic evidence is ready.", text)
            self.assertNotIn("fix", [event.kind for event in events])
            self.assertEqual(events[-1].kind, "done")

    def test_inactive_network_service_requires_action_then_permission_and_verification(self):
        class NetworkRegistry:
            def __init__(self):
                self.fixed = False
                self.confirmed_requests = []

            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False):
                del cancel_event, diagnostic
                if confirmation:
                    self.confirmed_requests.append(request)
                data = dict(HealthyRegistryData.get(request.name, {}))
                if request.name == "service_status":
                    data["active_state"] = "active" if self.fixed else "inactive"
                if request.name == "controlled_terminal":
                    self.fixed = True
                    data = {"exit_code": 0, "stdout": "", "stderr": ""}
                result = ToolResult(request.name, True, data)
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

        HealthyRegistryData = {
            "network_interfaces": {"interfaces": [{"name": "eth0", "operstate": "up", "carrier": "1", "link_up": True}]},
            "routing_info": {"routes": [{"destination_hex": "00000000", "interface": "eth0"}]},
            "gateway_detection": {"gateways": [{"interface": "eth0", "gateway": "192.168.1.1"}]},
            "dns_info": {"nameservers": ["1.1.1.1"]},
            "ping_connectivity": {"reachable": True},
        }
        with TemporaryDirectory() as temp:
            registry = NetworkRegistry()
            engine = TroubleshootingEngine(
                FakeChatProvider(),
                registry,
                TroubleshootingHistory(Path(temp) / "history.jsonl"),
            )
            events = []
            thread = threading.Thread(target=lambda: events.extend(engine.stream("My internet is not working", threading.Event())))
            thread.start()
            while thread.is_alive() and not any(event.kind == "fix" for event in events):
                threading.Event().wait(0.01)
            choice = next(event.fix_proposal for event in events if event.kind == "fix")
            self.assertEqual(choice.action_kind, "automatic")
            self.assertTrue(engine.choose_fix(choice.proposal_id, "automatic"))
            while thread.is_alive() and not any(event.kind == "fix" and event.fix_proposal.mode == "confirmation" for event in events):
                threading.Event().wait(0.01)
            self.assertTrue(engine.approve_fix(choice.proposal_id, True))
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertTrue(any(request.name == "controlled_terminal" for request in registry.confirmed_requests))
            text = "".join(event.text for event in events if event.kind == "text")
            self.assertIn("Problem fixed successfully", text)
            self.assertEqual(events[-1].kind, "done")

    def test_physical_link_finding_offers_manual_and_check_again_without_auto_fix(self):
        results = [
            ToolResult("network_interfaces", True, {"interfaces": [{"name": "eth0", "operstate": "down", "carrier": "0", "link_up": False}]}),
        ]
        assessment = _assess(TroubleshootingCategory.PHYSICAL, results)
        self.assertEqual(assessment.outcome, TroubleshootingOutcome.PHYSICAL_PROBLEM)
        self.assertIn("cable", assessment.summary)
        self.assertFalse(assessment.automatic_fix_available)
        proposal = _fix_for(TroubleshootingCategory.PHYSICAL, "ethernet is disconnected", assessment)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.action_kind, "physical")
        self.assertEqual(proposal.request.name, "manual_troubleshooting")
        self.assertIn("No automatic system change", proposal.command_preview)

    def test_failed_verification_does_not_repeat_the_same_fix(self):
        class StillBrokenRegistry:
            def __init__(self):
                self.fix_attempts = 0

            def execute_stream(self, request, cancel_event, *, confirmation=False, diagnostic=False):
                del cancel_event, diagnostic
                if request.name == "controlled_terminal" and confirmation:
                    self.fix_attempts += 1
                data = dict(HealthyRegistryData.get(request.name, {}))
                if request.name == "service_status":
                    data["active_state"] = "inactive"
                result = ToolResult(request.name, True, data)
                yield ToolEvent(request.name, request.name, request.name, ToolEventKind.COMPLETED, "completed", result)

        HealthyRegistryData = {
            "network_interfaces": {"interfaces": [{"name": "eth0", "operstate": "up", "carrier": "1", "link_up": True}]},
            "routing_info": {"routes": [{"destination_hex": "00000000", "interface": "eth0"}]},
            "gateway_detection": {"gateways": [{"interface": "eth0", "gateway": "192.168.1.1"}]},
            "dns_info": {"nameservers": ["1.1.1.1"]},
            "ping_connectivity": {"reachable": True},
        }
        with TemporaryDirectory() as temp:
            registry = StillBrokenRegistry()
            engine = TroubleshootingEngine(
                FakeChatProvider(),
                registry,
                TroubleshootingHistory(Path(temp) / "history.jsonl"),
            )
            events = []
            thread = threading.Thread(target=lambda: events.extend(engine.stream("My internet is not working", threading.Event())))
            thread.start()
            while thread.is_alive() and not any(event.kind == "fix" for event in events):
                threading.Event().wait(0.01)
            choice = next(event.fix_proposal for event in events if event.kind == "fix")
            self.assertTrue(engine.choose_fix(choice.proposal_id, "automatic"))
            while thread.is_alive() and not any(event.kind == "fix" and event.fix_proposal.mode == "confirmation" for event in events):
                threading.Event().wait(0.01)
            self.assertTrue(engine.approve_fix(choice.proposal_id, True))
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(registry.fix_attempts, 1)
            text = "".join(event.text for event in events if event.kind == "text")
            self.assertIn("did not resolve the problem", text)
            self.assertEqual(engine.state.task_state.value, "FAILED")
            self.assertTrue(engine.state.verification_complete)
            self.assertEqual(events[-1].kind, "done")

    def test_ping_is_only_bypassed_for_explicit_diagnostic_path(self):
        class Policy:
            allow_network = False
            auto_approve_read_only = True
            allow_write = False
            allow_destructive = False
            allow_terminal = False

        from tools.linux_tools import create_tool_definitions
        from tools.registry import ToolRegistry

        registry = ToolRegistry(create_tool_definitions((Path.home(),)), Policy())
        request = ToolRequest("ping_connectivity", {"host": "1.1.1.1"})
        with self.assertRaises(PermissionError):
            registry.validate(request)
        self.assertEqual(registry.validate(request, diagnostic=True).name, "ping_connectivity")


if __name__ == "__main__":
    unittest.main()
