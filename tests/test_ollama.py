from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path

from agent.controller import AIController
from config.config import AgentConfig
from llm.hardware import GiB, HardwareProfile
from llm.ollama_provider import (
    OllamaModelUnavailable,
    OllamaProvider,
)
from llm.provider import ChatMessage, ChatProvider, Plan, ProviderEvent
from tools.contracts import ToolResult


class _FakeResponse:
    fp = None

    def __init__(self) -> None:
        self.lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}).encode() + b"\n",
            json.dumps({"message": {"content": "!"}, "done": True}).encode() + b"\n",
        ]

    def readline(self) -> bytes:
        return self.lines.pop(0) if self.lines else b""

    def close(self) -> None:
        pass


class _FakeConnection:
    def close(self) -> None:
        pass


class OllamaProviderTests(unittest.TestCase):
    def test_defaults_target_local_qwen(self) -> None:
        config = AgentConfig()
        provider = OllamaProvider(config)
        self.assertEqual(provider.config.ollama_model, "qwen2.5:7b")
        self.assertEqual(provider.config.ollama_base_url, "http://localhost:11434")

    def test_effective_options_are_hardware_aware_for_small_integrated_gpu(self) -> None:
        provider = OllamaProvider(AgentConfig())
        provider.hardware = HardwareProfile(
            logical_cpus=12,
            available_cpus=12,
            physical_cores=6,
            ram_bytes=15 * 1024**3,
            available_ram_bytes=8 * 1024**3,
            gpu_vram_bytes=512 * 1024**2,
            gpu_vendor="AMD",
        )
        options = provider.effective_options()
        self.assertEqual(options["num_ctx"], 3968)
        self.assertEqual(options["num_predict"], 512)
        self.assertEqual(options["num_thread"], 11)
        self.assertEqual(options["num_batch"], 512)
        self.assertEqual(options["num_gpu"], 0)
        self.assertTrue(options["low_vram"])

    def test_effective_options_honor_explicit_settings_when_auto_tune_disabled(self) -> None:
        provider = OllamaProvider(
            AgentConfig(
                context_size=8192,
                max_tokens=1024,
                threads=4,
                batch_size=128,
                gpu_layers=12,
                hardware_auto_tune=False,
            )
        )
        options = provider.effective_options()
        self.assertEqual(options["num_ctx"], 8192)
        self.assertEqual(options["num_predict"], 1024)
        self.assertEqual(options["num_thread"], 4)
        self.assertEqual(options["num_batch"], 128)
        self.assertEqual(options["num_gpu"], 12)

    def test_low_memory_profile_reduces_runtime_pressure(self) -> None:
        provider = OllamaProvider(AgentConfig())
        provider.hardware = HardwareProfile(
            logical_cpus=4,
            available_cpus=4,
            physical_cores=2,
            ram_bytes=8 * GiB,
            available_ram_bytes=2 * GiB,
            cpu_model="low-end CPU",
            cpu_load_1m=4.0,
        )
        options = provider.effective_options()
        self.assertEqual(options["num_ctx"], 1920)
        self.assertEqual(options["num_batch"], 64)
        self.assertEqual(options["num_thread"], 2)
        self.assertEqual(options["num_gpu"], 0)
        self.assertTrue(options["low_vram"])

    def test_high_memory_accelerated_profile_uses_available_gpu(self) -> None:
        provider = OllamaProvider(AgentConfig(context_size=8192, max_tokens=1024))
        provider.hardware = HardwareProfile(
            logical_cpus=16,
            available_cpus=16,
            physical_cores=8,
            ram_bytes=64 * GiB,
            available_ram_bytes=32 * GiB,
            gpu_vram_bytes=16 * GiB,
            gpu_vendor="NVIDIA",
            gpu_name="Test GPU",
            acceleration_backend="cuda",
            gpu_vram_used_bytes=1 * GiB,
            cpu_load_1m=0.2,
        )
        provider._model_size_bytes = 4 * GiB
        options = provider.effective_options()
        self.assertEqual(options["num_ctx"], 7936)
        self.assertEqual(options["num_gpu"], 999)
        self.assertEqual(options["num_batch"], 512)
        self.assertFalse(options["low_vram"])

    def test_current_pressure_shrinks_even_a_capable_profile(self) -> None:
        provider = OllamaProvider(AgentConfig(context_size=8192, max_tokens=1024))
        provider.hardware = HardwareProfile(
            logical_cpus=12,
            available_cpus=12,
            physical_cores=6,
            ram_bytes=32 * GiB,
            available_ram_bytes=2 * GiB,
            gpu_vram_bytes=12 * GiB,
            gpu_vendor="NVIDIA",
            acceleration_backend="cuda",
            gpu_vram_used_bytes=11 * GiB,
            cpu_load_1m=14.0,
        )
        options = provider.effective_options()
        self.assertTrue(provider.hardware.under_load)
        self.assertEqual(options["num_ctx"], 1792)
        self.assertEqual(options["num_batch"], 64)
        self.assertEqual(options["num_thread"], 6)
        self.assertEqual(options["num_gpu"], 0)

    def test_planner_stream_hides_protocol_and_returns_validated_plan(self) -> None:
        provider = OllamaProvider(AgentConfig())
        provider._ensure_model = lambda _cancel: None  # type: ignore[method-assign]
        provider._stream_chat = lambda *_args, **_kwargs: iter(  # type: ignore[method-assign]
            (
                ProviderEvent.text_chunk(
                    'ASSISTANT:\nI will inspect the host.\nPLAN_JSON:{"summary":"Inspect host","actions":[],"tool_requests":[{"name":"system_info","arguments":{}}],"notes":[]}'
                ),
            )
        )
        events = list(provider.stream_plan("check my computer", threading.Event()))
        self.assertEqual([event.kind for event in events], ["status", "status", "text", "status", "plan", "done"])
        self.assertEqual("I will inspect the host.", events[2].text)
        self.assertNotIn("PLAN_JSON", events[2].text)
        self.assertEqual(events[4].plan.tool_requests[0].name, "system_info")  # type: ignore[union-attr]

    def test_diagnostic_decision_stream_returns_one_typed_next_tool(self) -> None:
        provider = OllamaProvider(AgentConfig())
        provider._ensure_model = lambda _cancel: None  # type: ignore[method-assign]
        provider._stream_chat = lambda *_args, **_kwargs: iter(  # type: ignore[method-assign]
            (
                ProviderEvent.text_chunk(
                    '{"done":false,"reason":"Inspect the host first.","tool_request":{"name":"system_info","arguments":{}}}'
                ),
            )
        )
        events = list(
            provider.stream_diagnostic_decision(
                "troubleshoot my computer",
                "general",
                (),
                (),
                threading.Event(),
            )
        )
        self.assertEqual([event.kind for event in events], ["status", "diagnostic_decision"])
        decision = events[-1].diagnostic_decision
        self.assertIsNotNone(decision)
        self.assertFalse(decision.done)
        self.assertEqual(decision.tool_requests[0].name, "system_info")

    def test_result_review_streams_plain_user_safe_text(self) -> None:
        provider = OllamaProvider(AgentConfig())
        provider._stream_chat = lambda *_args, **_kwargs: iter(  # type: ignore[method-assign]
            (ProviderEvent.text_chunk("The kernel is Linux 6.8."),)
        )
        events = list(
            provider.stream_tool_results(
                "show kernel information",
                Plan(summary="Inspect the kernel"),
                (ToolResult("kernel_info", True, {"release": "6.8"}),),
                threading.Event(),
            )
        )
        self.assertEqual([event.kind for event in events], ["status", "text"])
        self.assertEqual(events[-1].text, "The kernel is Linux 6.8.")

    def test_legacy_llama_backend_migrates_to_ollama(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.toml"
            path.write_text('[llm]\nbackend = "llama.cpp"\n', encoding="utf-8")
            config = AgentConfig.load(path)
        self.assertEqual(config.llm_backend, "ollama")

    def test_cloud_endpoint_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OllamaProvider(AgentConfig(ollama_base_url="https://example.com"))

    def test_streaming_ndjson_becomes_progressive_text_events(self) -> None:
        provider = OllamaProvider(AgentConfig())
        provider._ensure_model = lambda _cancel: None  # type: ignore[method-assign]
        provider._open_request = lambda *_args, **_kwargs: (_FakeResponse(), _FakeConnection())  # type: ignore[method-assign]
        events = list(provider.stream_chat((ChatMessage("user", "hello"),), threading.Event()))
        self.assertEqual([event.kind for event in events], ["status", "status", "text", "text", "done"])
        self.assertEqual("".join(event.text for event in events if event.kind == "text"), "Hello!")

    def test_missing_model_is_mapped_to_clear_ui_error(self) -> None:
        provider = OllamaProvider(AgentConfig())
        provider._ensure_model = lambda _cancel: (_ for _ in ()).throw(  # type: ignore[method-assign]
            OllamaModelUnavailable("missing qwen2.5:7b")
        )
        events = list(provider.stream_chat((ChatMessage("user", "hello"),), threading.Event()))
        self.assertEqual(events[-1].kind, "error")
        self.assertIn("missing qwen2.5:7b", events[-1].error)

    def test_missing_exact_tag_uses_only_an_installed_quantized_variant(self) -> None:
        provider = OllamaProvider(AgentConfig())
        provider._request_json = lambda method, _path, _body, _cancel: (  # type: ignore[method-assign]
            {
                "models": [
                    {
                        "name": "qwen2.5:7b-q4_K_M",
                        "size": 4 * GiB,
                        "details": {"quantization_level": "Q4_K_M"},
                    }
                ]
            }
            if method == "GET"
            else {"details": {"quantization_level": "Q4_K_M"}}
        )
        provider._ensure_model(threading.Event())
        self.assertEqual(provider._active_model, "qwen2.5:7b-q4_K_M")
        self.assertEqual(provider._model_size_bytes, 4 * GiB)

    def test_model_settings_round_trip_through_config(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.toml"
            config = AgentConfig(
                top_k=32,
                min_p=0.1,
                repeat_penalty=1.12,
                num_keep=512,
                seed=42,
                threads=6,
                batch_size=256,
                gpu_layers=3,
                hardware_auto_tune=False,
            )
            config.save_quick_size(420, 50, path)
            restored = AgentConfig.load(path)
        self.assertEqual(restored.top_k, 32)
        self.assertEqual(restored.min_p, 0.1)
        self.assertEqual(restored.repeat_penalty, 1.12)
        self.assertEqual(restored.num_keep, 512)
        self.assertEqual(restored.seed, 42)
        self.assertEqual(restored.threads, 6)
        self.assertEqual(restored.batch_size, 256)
        self.assertEqual(restored.gpu_layers, 3)
        self.assertFalse(restored.hardware_auto_tune)


class ControllerHistoryTests(unittest.TestCase):
    def test_cancellation_stops_stream_and_does_not_store_partial_response(self) -> None:
        class CancellableProvider(ChatProvider):
            def __init__(self) -> None:
                self.continued_after_cancel = False

            def stream_chat(self, messages, cancel_event):
                del messages
                yield ProviderEvent.text_chunk("partial")
                if cancel_event.is_set():
                    self.continued_after_cancel = True
                    yield ProviderEvent.text_chunk("unexpected")
                yield ProviderEvent.done()

        provider = CancellableProvider()
        controller = AIController(provider)
        cancel_event = threading.Event()
        stream = controller.stream_response("hello", cancel_event)
        self.assertEqual(next(stream).text, "partial")
        cancel_event.set()
        self.assertEqual(list(stream), [])
        self.assertFalse(provider.continued_after_cancel)
        self.assertEqual(
            [(message.role, message.content) for message in controller.history],
            [("user", "hello")],
        )

    def test_controller_preserves_current_conversation(self) -> None:
        class FakeChatProvider(ChatProvider):
            def stream_chat(self, messages, cancel_event):
                yield ProviderEvent.text_chunk(f"messages={len(messages)}")
                yield ProviderEvent.done()

        controller = AIController(FakeChatProvider())
        first = list(controller.stream_response("hello", threading.Event()))
        second = list(controller.stream_response("install google chrome", threading.Event()))
        self.assertEqual("".join(event.text for event in first if event.kind == "text"), "messages=1")
        self.assertEqual("".join(event.text for event in second if event.kind == "text"), "messages=3")
        self.assertEqual([message.role for message in controller.history], ["user", "assistant", "user", "assistant"])


if __name__ == "__main__":
    unittest.main()
