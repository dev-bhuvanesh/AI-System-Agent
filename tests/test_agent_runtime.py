from __future__ import annotations

import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.runtime import AssistantRuntime
from config.config import AgentConfig
from llm.prompts import build_planner_prompt
from llm.provider import parse_plan_response, visible_response
from llm.provider import LLMProvider, Plan, ProviderEvent
from tools.contracts import PermissionLevel, ToolEvent, ToolEventKind, ToolRequest, ToolResult
from tools.registry import ToolPolicy, ToolRegistry
from tools.linux_tools import create_tool_definitions
from tools.network_tools import create_network_tool_definitions


class ToolRegistryTests(unittest.TestCase):
    def make_registry(self, root: Path, **overrides: bool) -> ToolRegistry:
        policy = ToolPolicy(
            allowed_roots=(root,),
            auto_approve_read_only=True,
            allow_network=overrides.get("allow_network", False),
            allow_write=overrides.get("allow_write", False),
            allow_destructive=overrides.get("allow_destructive", False),
            allow_terminal=overrides.get("allow_terminal", False),
        )
        return ToolRegistry((*create_tool_definitions((root,)), *create_network_tool_definitions()), policy)

    def test_all_requested_tools_are_registered(self) -> None:
        with TemporaryDirectory() as temp_dir:
            names = {definition.name for definition in self.make_registry(Path(temp_dir)).definitions()}
        self.assertEqual(
            names,
            {
                "system_info", "cpu_usage", "ram_usage", "disk_usage", "kernel_info", "gpu_info", "uptime",
                "directory_list", "file_read", "file_create", "file_delete", "file_rename", "file_copy",
                "file_move", "directory_create", "directory_delete", "process_list", "process_info",
                "network_interfaces", "bluetooth_info", "usb_info", "routing_info", "gateway_detection", "dns_info", "ping_connectivity",
                "service_status", "controlled_terminal", "network_management_info", "wifi_hardware_info", "wifi_interface_info",
                "wifi_radio_state", "rfkill_status", "wifi_interface_state", "wifi_connection", "wifi_ip_info",
                "gateway_connectivity", "wifi_enable",
            },
        )

    def test_read_only_tool_returns_structured_result(self) -> None:
        with TemporaryDirectory() as temp_dir:
            events = list(self.make_registry(Path(temp_dir)).execute_stream(ToolRequest("system_info", {}), threading.Event()))
        self.assertEqual(events[0].kind, ToolEventKind.STARTED)
        self.assertEqual(events[-1].kind, ToolEventKind.COMPLETED)
        self.assertIsNotNone(events[-1].result)
        self.assertTrue(events[-1].result.ok)
        self.assertIn("system", events[-1].result.data)

    def test_model_cannot_self_approve_write_tool(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = self.make_registry(root, allow_write=True)
            request = ToolRequest("file_create", {"path": "new.txt", "content": "safe"}, requires_confirmation=False)
            events = list(registry.execute_stream(request, threading.Event()))
        self.assertEqual(events[-1].kind, ToolEventKind.BLOCKED)
        self.assertFalse((root / "new.txt").exists())
        self.assertEqual(events[-1].result.error_code, "permission_denied")

    def test_diagnostic_path_accepts_read_only_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = self.make_registry(root, allow_write=True)
            events = list(
                registry.execute_stream(
                    ToolRequest(
                        "file_create",
                        {"path": "blocked.txt", "content": "must not be written"},
                    ),
                    threading.Event(),
                    diagnostic=True,
                )
            )
        self.assertEqual(events[-1].kind, ToolEventKind.BLOCKED)
        self.assertFalse((root / "blocked.txt").exists())

    def test_path_escape_is_rejected_by_schema_boundary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            registry = self.make_registry(Path(temp_dir))
            events = list(registry.execute_stream(ToolRequest("directory_list", {"path": "/etc"}), threading.Event()))
        self.assertEqual(events[-1].kind, ToolEventKind.FAILED)
        self.assertIn("outside", events[-1].result.error_message)

    def test_controlled_terminal_rejects_shell_switch(self) -> None:
        with TemporaryDirectory() as temp_dir:
            registry = self.make_registry(Path(temp_dir), allow_terminal=True)
            events = list(
                registry.execute_stream(
                    ToolRequest("controlled_terminal", {"program": "date", "args": ["-c"]}),
                    threading.Event(),
                    confirmation=True,
                )
            )
        self.assertEqual(events[-1].kind, ToolEventKind.FAILED)
        self.assertFalse(events[-1].result.ok)

    def test_confirmed_constrained_repair_does_not_require_global_terminal_policy(self) -> None:
        with TemporaryDirectory() as temp_dir:
            registry = self.make_registry(Path(temp_dir), allow_terminal=False)
            definition = registry.validate(
                ToolRequest("controlled_terminal", {"program": "date", "args": []}),
                confirmation=True,
            )
        self.assertEqual(definition.name, "controlled_terminal")

    def test_wifi_enable_has_no_model_command_input_and_requires_trusted_confirmation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            registry = self.make_registry(Path(temp_dir), allow_terminal=False)
            request = ToolRequest("wifi_enable", {}, requires_confirmation=False)
            with self.assertRaises(PermissionError):
                registry.validate(request)
            definition = registry.validate(request, confirmation=True)
        self.assertEqual(definition.name, "wifi_enable")
        self.assertEqual(definition.input_schema["additionalProperties"], False)
        self.assertTrue(definition.safe_troubleshooting)


class AgentRuntimeContractTests(unittest.TestCase):
    def test_tool_policy_round_trips_through_config(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "system-agent" / "config.toml"
            config = AgentConfig(
                tool_allowed_roots=(Path(temp_dir),),
                tool_auto_approve_read_only=True,
                tool_allow_network=True,
                tool_allow_write=False,
                tool_allow_destructive=False,
                tool_allow_terminal=False,
            )
            config.save_quick_size(420, 50, config_path)
            restored = AgentConfig.load(config_path)
        self.assertEqual(restored.tool_allowed_roots, (Path(temp_dir),))
        self.assertTrue(restored.tool_allow_network)
        self.assertFalse(restored.tool_allow_write)

    def test_plan_accepts_data_only_tool_requests(self) -> None:
        plan = Plan.from_dict(
            {
                "summary": "Inspect the host",
                "tool_requests": [{"name": "system_info", "arguments": {}, "requires_confirmation": False}],
            },
            "show system info",
        )
        self.assertEqual(plan.tool_requests[0].name, "system_info")
        self.assertEqual(plan.tool_requests[0].arguments, {})

    def test_plan_normalizes_qwen_no_argument_tool_shorthand(self) -> None:
        plan = Plan.from_dict(
            {
                "summary": "Inspect the kernel",
                "tool_requests": ["kernel_info"],
            },
            "show kernel info",
        )
        self.assertEqual(plan.tool_requests, (ToolRequest("kernel_info", {}),))

    def test_prompt_mentions_registry_and_no_hidden_reasoning(self) -> None:
        prompt = build_planner_prompt("check my computer")
        self.assertIn("Never reveal chain-of-thought", prompt)
        self.assertIn("approved", prompt.lower())
        self.assertIn("tool_requests", prompt)

    def test_model_protocol_text_is_filtered_until_assistant_marker(self) -> None:
        self.assertEqual(visible_response("private prompt text"), "")
        self.assertEqual(visible_response("ASSISTANT:\nhello\nPLAN_JSON:{}"), "hello")
        self.assertEqual(visible_response("ASSISTANT:\nhello\nPLAN"), "hello")
        self.assertEqual(
            parse_plan_response(
                '{"summary":"Inspect host","actions":[],"tool_requests":[],"notes":[]}',
                "check host",
                "",
            ).summary,
            "Inspect host",
        )

    def test_definitions_have_security_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            definitions = create_tool_definitions((Path(temp_dir),))
        for definition in definitions:
            self.assertTrue(definition.name)
            self.assertTrue(definition.description)
            self.assertEqual(definition.input_schema["type"], "object")
            self.assertEqual(definition.output_schema["type"], "object")
            self.assertIsInstance(definition.permission_level, PermissionLevel)
            self.assertGreater(definition.timeout_seconds, 0)

    def test_runtime_returns_registry_results_to_provider(self) -> None:
        class FakeProvider(LLMProvider):
            def __init__(self) -> None:
                self.received = ()

            def stream_plan(self, request: str, cancel_event: threading.Event):
                yield ProviderEvent.plan_ready(
                    Plan(
                        summary="Inspect host",
                        tool_requests=(ToolRequest("system_info", {}),),
                    )
                )
                yield ProviderEvent.done()

            def stream_tool_results(self, request, plan, results, cancel_event):
                self.received = results
                yield ProviderEvent.text_chunk("Validated local result")

        with TemporaryDirectory() as temp_dir:
            fake = FakeProvider()
            runtime = AssistantRuntime(fake, self._registry(Path(temp_dir)))
            events = list(runtime.stream("show system information", threading.Event()))
        self.assertTrue(any(event.kind == "tool" for event in events))
        self.assertTrue(any(event.kind == "text" for event in events))
        self.assertEqual(fake.received[0].tool_name, "system_info")

    def test_runtime_keeps_failed_tool_tasks_out_of_completed_state(self) -> None:
        class FakeProvider(LLMProvider):
            def stream_plan(self, request, cancel_event):
                del request, cancel_event
                yield ProviderEvent.plan_ready(
                    Plan(summary="Inspect host", tool_requests=(ToolRequest("system_info", {}),))
                )
                yield ProviderEvent.done()

            def stream_tool_results(self, request, plan, results, cancel_event):
                del request, plan, results, cancel_event
                return
                yield  # Keep this a generator for the provider contract.

        class FailedRegistry:
            def validate(self, request, **_kwargs):
                return request

            def get(self, _name):
                return None

            def execute_stream(self, request, cancel_event, **_kwargs):
                del cancel_event
                result = ToolResult(
                    request.name,
                    False,
                    data={"exit_code": 1, "stdout": "", "stderr": "diagnostic failed"},
                    error_code="diagnostic_failed",
                    error_message="diagnostic failed",
                )
                yield ToolEvent(
                    request.name,
                    request.name,
                    request.name,
                    ToolEventKind.FAILED,
                    "failed",
                    result,
                )

        runtime = AssistantRuntime(FakeProvider(), FailedRegistry())
        events = list(runtime.stream("check host", threading.Event()))
        self.assertTrue(any(event.kind == "tool" for event in events))
        self.assertEqual(runtime.state.status, "failed")
        self.assertEqual(runtime.state.task_state, "FAILED")
        self.assertEqual(runtime.state.verification_status, "needs_attention")

    def test_runtime_requires_trusted_approval_before_write(self) -> None:
        class FakeProvider(LLMProvider):
            def stream_plan(self, request, cancel_event):
                yield ProviderEvent.plan_ready(
                    Plan(
                        summary="Create the requested file",
                        tool_requests=(
                            ToolRequest(
                                "file_create",
                                {"path": "approved.txt", "content": "approved"},
                                requires_confirmation=False,
                            ),
                        ),
                    )
                )
                yield ProviderEvent.done()

            def stream_tool_results(self, request, plan, results, cancel_event):
                yield ProviderEvent.text_chunk("The file was created and verified.")

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy = ToolPolicy(allowed_roots=(root,), allow_write=True)
            runtime = AssistantRuntime(FakeProvider(), ToolRegistry(create_tool_definitions((root,)), policy))
            events: list[ProviderEvent] = []
            worker = threading.Thread(
                target=lambda: events.extend(runtime.stream("create a file", threading.Event()))
            )
            worker.start()
            for _ in range(30):
                approval = next((event.tool_approval for event in events if event.kind == "tool_approval"), None)
                if approval is not None:
                    break
                threading.Event().wait(0.02)
            self.assertIsNotNone(approval)
            self.assertFalse((root / "approved.txt").exists())
            self.assertTrue(runtime.approve_tool(approval.approval_id, True))  # type: ignore[union-attr]
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertTrue((root / "approved.txt").exists())
            self.assertTrue(any(event.kind == "tool" and event.tool_event and event.tool_event.result and event.tool_event.result.data.get("verified") for event in events))
            self.assertEqual(runtime.state.status, "completed")
            self.assertEqual(runtime.state.verification_status, "complete")

    def test_runtime_cancellation_while_waiting_for_approval_runs_no_write(self) -> None:
        class FakeProvider(LLMProvider):
            def stream_plan(self, request, cancel_event):
                yield ProviderEvent.plan_ready(
                    Plan(
                        summary="Delete the requested file",
                        tool_requests=(ToolRequest("file_delete", {"path": "missing.txt"}),),
                    )
                )
                yield ProviderEvent.done()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy = ToolPolicy(allowed_roots=(root,), allow_destructive=True)
            cancel_event = threading.Event()
            runtime = AssistantRuntime(FakeProvider(), ToolRegistry(create_tool_definitions((root,)), policy))
            events: list[ProviderEvent] = []
            worker = threading.Thread(
                target=lambda: events.extend(runtime.stream("delete a file", cancel_event))
            )
            worker.start()
            for _ in range(30):
                if any(event.kind == "tool_approval" for event in events):
                    break
                threading.Event().wait(0.02)
            cancel_event.set()
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(runtime.state.status, "cancelled")
            self.assertFalse(any(event.kind == "tool" and event.tool_event and event.tool_event.kind is ToolEventKind.STARTED for event in events))

    def test_controller_routes_ordinary_requests_through_runtime(self) -> None:
        from agent.controller import AIController

        class FakeProvider(LLMProvider):
            def stream_chat(self, messages, cancel_event):
                raise AssertionError("ordinary requests must use AssistantRuntime")

            def stream_plan(self, request, cancel_event):
                yield ProviderEvent.plan_ready(
                    Plan(summary="Inspect host", tool_requests=(ToolRequest("system_info", {}),))
                )
                yield ProviderEvent.done()

            def stream_tool_results(self, request, plan, results, cancel_event):
                yield ProviderEvent.text_chunk("System information was checked.")

        with TemporaryDirectory() as temp_dir:
            fake = FakeProvider()
            runtime = AssistantRuntime(fake, self._registry(Path(temp_dir)))
            controller = AIController(fake, runtime=runtime)
            events = list(controller.stream_response("check my system", threading.Event()))
        self.assertTrue(any(event.kind == "tool" for event in events))
        self.assertIn("System information was checked.", "".join(event.text for event in events if event.kind == "text"))
        self.assertEqual(controller.history[-1].role, "assistant")

    @staticmethod
    def _registry(root: Path) -> ToolRegistry:
        policy = ToolPolicy(allowed_roots=(root,))
        return ToolRegistry(create_tool_definitions((root,)), policy)


if __name__ == "__main__":
    unittest.main()
