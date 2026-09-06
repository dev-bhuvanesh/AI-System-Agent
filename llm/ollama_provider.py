"""Streaming provider for a local Ollama HTTP API."""

from __future__ import annotations

import http.client
import json
import logging
import queue
import socket
from dataclasses import asdict
from threading import Event, Thread
from typing import Any, Iterator
from urllib.parse import urlparse

from config.config import AgentConfig
from llm.hardware import HardwareProfile, detect_hardware, refresh_runtime_resources
from llm.prompts import (
    QWEN_CHAT_SYSTEM_PROMPT,
    QWEN_DIAGNOSTIC_SYSTEM_PROMPT,
    QWEN_PLANNER_SYSTEM_PROMPT,
    build_diagnostic_decision_prompt,
    build_planner_prompt,
    build_software_failure_prompt,
    build_tool_result_prompt,
    parse_software_recovery_response,
    parse_diagnostic_decision_response,
)
from llm.provider import (
    ChatMessage,
    ChatProvider,
    LLMProvider,
    Plan,
    ProviderEvent,
    DiagnosticDecision,
    SoftwareRecoveryDecision,
    model_safe_result,
    parse_plan_response,
    visible_answer,
    visible_response,
)
from tools.contracts import ToolResult


logger = logging.getLogger(__name__)


def _recommended_quantization(profile: HardwareProfile) -> str:
    """Choose a Qwen quantization target without pulling or modifying weights."""
    tier = profile.performance_tier
    if tier == "low":
        return "Q3_K_M"
    if tier == "high":
        return "Q5_K_M"
    return "Q4_K_M"


class OllamaError(RuntimeError):
    """Expected local Ollama/API failure."""


class OllamaConnectionError(OllamaError):
    """Ollama could not be reached on the configured local endpoint."""


class OllamaModelUnavailable(OllamaError):
    """The configured Ollama model is not installed."""


class OllamaProvider(ChatProvider, LLMProvider):
    """Call Ollama's local ``/api/chat`` endpoint with NDJSON streaming."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.hardware = detect_hardware()
        self._model_checked = False
        self._active_model = config.ollama_model
        self._model_size_bytes = 0
        self._model_quantization = ""
        self.last_metrics: dict[str, int] = {}
        parsed = urlparse(config.ollama_base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Ollama URL must be an HTTP(S) URL")
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Ollama must use a local loopback URL; cloud endpoints are disabled")
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        logger.info(
            "Local LLM hardware: CPU=%s cores=%d threads=%d/%d RAM=%dMiB available=%dMiB GPU=%s backend=%s VRAM=%dMiB available=%dMiB gpu_load=%.1f%% load=%.2f pressure=%.2f tier=%s quantization=%s",
            self.hardware.cpu_model or "unknown",
            self.hardware.physical_cores,
            self.hardware.available_cpus,
            self.hardware.logical_cpus,
            self.hardware.ram_bytes // (1024 * 1024),
            self.hardware.available_ram_bytes // (1024 * 1024),
            self.hardware.gpu_name or self.hardware.gpu_vendor or "none",
            self.hardware.acceleration_backend,
            self.hardware.gpu_vram_bytes // (1024 * 1024),
            self.hardware.available_vram_bytes // (1024 * 1024),
            self.hardware.gpu_utilization_percent,
            self.hardware.cpu_load_1m,
            self.hardware.resource_pressure,
            self.hardware.performance_tier,
            _recommended_quantization(self.hardware),
        )

    def stream_chat(
        self,
        messages: tuple[ChatMessage, ...],
        cancel_event: Event,
    ) -> Iterator[ProviderEvent]:
        logger.info("Ollama chat start: model=%s messages=%d", self._active_model, len(messages))
        if not messages or not messages[-1].content.strip():
            yield ProviderEvent.failure("Please enter a message before sending.")
            return
        try:
            yield ProviderEvent.status("Checking local AI...")
            self._ensure_model(cancel_event)
            if cancel_event.is_set():
                return
            yield ProviderEvent.status("Generating response...")
            for event in self._stream_chat(messages, cancel_event, QWEN_CHAT_SYSTEM_PROMPT):
                yield event
            if not cancel_event.is_set():
                logger.info("Ollama chat complete: model=%s", self._active_model)
                yield ProviderEvent.done()
        except OllamaModelUnavailable as exc:
            logger.warning("Ollama model unavailable: %s", exc)
            yield ProviderEvent.failure(str(exc))
        except OllamaConnectionError:
            logger.warning("Unable to connect to local Ollama")
            yield ProviderEvent.failure("Unable to connect to local AI. Please start Ollama.")
        except OllamaError as exc:
            logger.warning("Ollama request failed: %s", exc)
            yield ProviderEvent.failure(f"Local AI request failed: {exc}")
        except (OSError, socket.timeout) as exc:
            logger.warning("Ollama transport failed: %s", exc)
            yield ProviderEvent.failure("Unable to connect to local AI. Please start Ollama.")

    def stream_plan(
        self,
        request: str,
        cancel_event: Event,
        conversation: tuple[ChatMessage, ...] = (),
    ) -> Iterator[ProviderEvent]:
        """Generate a validated plan without exposing the planner protocol."""
        if not request.strip():
            yield ProviderEvent.failure("Please enter a message before sending.")
            return
        try:
            yield ProviderEvent.status("Understanding request...")
            self._ensure_model(cancel_event)
            if cancel_event.is_set():
                return
            yield ProviderEvent.status("Preparing plan...")
            raw_output = ""
            emitted_text = ""
            prompt = build_planner_prompt(request)
            prior_messages = conversation
            # AIController includes the current user message in its session
            # history. It is represented by the dedicated planner prompt
            # below, so do not send it twice.
            if (
                prior_messages
                and prior_messages[-1].role == "user"
                and prior_messages[-1].content.strip() == request.strip()
            ):
                prior_messages = prior_messages[:-1]
            planner_messages = (*prior_messages[-8:], ChatMessage("user", prompt))
            for event in self._stream_chat(
                planner_messages,
                cancel_event,
                QWEN_PLANNER_SYSTEM_PROMPT,
                response_format="json",
                max_predict=192,
            ):
                if cancel_event.is_set():
                    return
                if event.kind != "text":
                    continue
                raw_output += event.text
                visible = visible_response(raw_output)
                delta = visible[len(emitted_text):] if visible.startswith(emitted_text) else visible
                if delta:
                    emitted_text = visible
                    yield ProviderEvent.text_chunk(delta)
            if cancel_event.is_set():
                return
            yield ProviderEvent.status("Preparing actions...")
            yield ProviderEvent.plan_ready(parse_plan_response(raw_output, request, emitted_text))
            yield ProviderEvent.done()
        except OllamaModelUnavailable as exc:
            yield ProviderEvent.failure(str(exc))
        except OllamaConnectionError:
            yield ProviderEvent.failure("Unable to connect to local AI. Please start Ollama.")
        except OllamaError as exc:
            yield ProviderEvent.failure(f"Local AI planning failed: {exc}")
        except (OSError, socket.timeout):
            yield ProviderEvent.failure("Unable to connect to local AI. Please start Ollama.")

    def stream_tool_results(
        self,
        request: str,
        plan: Plan,
        results: tuple[ToolResult, ...],
        cancel_event: Event,
    ) -> Iterator[ProviderEvent]:
        """Review only structured registry results and stream a safe answer."""
        if cancel_event.is_set():
            return
        yield ProviderEvent.status("Reviewing tool results...")
        result_json = json.dumps(
            [model_safe_result(result.as_dict()) for result in results],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        prompt = build_tool_result_prompt(request, plan.summary, result_json)
        raw_output = ""
        emitted_text = ""
        try:
            for event in self._stream_chat(
                (ChatMessage("user", prompt),),
                cancel_event,
                QWEN_CHAT_SYSTEM_PROMPT,
                # Result reviews are intentionally concise. On CPU-only
                # machines this keeps the final answer responsive after the
                # tool has already completed and been verified.
                max_predict=96,
            ):
                if cancel_event.is_set():
                    return
                if event.kind != "text":
                    continue
                raw_output += event.text
                visible = visible_answer(raw_output)
                delta = visible[len(emitted_text):] if visible.startswith(emitted_text) else visible
                if delta:
                    emitted_text = visible
                    yield ProviderEvent.text_chunk(delta)
        except OllamaModelUnavailable as exc:
            yield ProviderEvent.failure(str(exc))
        except OllamaConnectionError:
            yield ProviderEvent.failure("Unable to connect to local AI. Please start Ollama.")
        except OllamaError as exc:
            yield ProviderEvent.failure(f"Local AI result review failed: {exc}")
        final_visible = visible_answer(raw_output, final=True)
        if final_visible.startswith(emitted_text):
            delta = final_visible[len(emitted_text):]
        else:
            delta = final_visible
        if delta:
            yield ProviderEvent.text_chunk(delta)

    def stream_software_failure(
        self,
        request: str,
        operation: str,
        attempted_action: str,
        result: ToolResult,
        alternatives: tuple[str, ...],
        cancel_event: Event,
    ) -> Iterator[ProviderEvent]:
        """Let Qwen classify a failure without giving it execution authority."""
        if cancel_event.is_set():
            return
        yield ProviderEvent.status("Analyzing software failure...")
        evidence = json.dumps(
            model_safe_result(result.as_dict(), limit=12_000),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        prompt = build_software_failure_prompt(
            request,
            operation,
            attempted_action,
            evidence,
            "\n".join(alternatives),
        )
        raw_output = ""
        try:
            for event in self._stream_chat(
                (ChatMessage("user", prompt),),
                cancel_event,
                QWEN_PLANNER_SYSTEM_PROMPT,
                response_format="json",
                max_predict=96,
            ):
                if cancel_event.is_set():
                    return
                if event.kind == "text":
                    raw_output += event.text
            if not cancel_event.is_set():
                yield ProviderEvent.software_recovery_ready(
                    parse_software_recovery_response(raw_output)
                )
        except (OllamaModelUnavailable, OllamaConnectionError, OllamaError, OSError, socket.timeout) as exc:
            logger.warning("Qwen software failure analysis unavailable: %s", exc)
            yield ProviderEvent.software_recovery_ready(
                SoftwareRecoveryDecision("unavailable", "Qwen failure analysis was unavailable.")
            )

    def stream_diagnostic_decision(
        self,
        request: str,
        category: str,
        observations: tuple[ToolResult, ...],
        previous_tools: tuple[str, ...],
        cancel_event: Event,
    ) -> Iterator[ProviderEvent]:
        """Choose one next read-only diagnostic request from real evidence."""
        if cancel_event.is_set():
            return
        yield ProviderEvent.status("Choosing the next diagnostic check...")
        evidence = json.dumps(
            [model_safe_result(result.as_dict(), limit=4_000) for result in observations],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        prompt = build_diagnostic_decision_prompt(
            request,
            category,
            evidence,
            previous_tools,
        )
        raw_output = ""
        try:
            self._ensure_model(cancel_event)
            if cancel_event.is_set():
                return
            for event in self._stream_chat(
                (ChatMessage("user", prompt),),
                cancel_event,
                QWEN_DIAGNOSTIC_SYSTEM_PROMPT,
                response_format="json",
                max_predict=128,
            ):
                if cancel_event.is_set():
                    return
                if event.kind == "text":
                    raw_output += event.text
            if not cancel_event.is_set():
                yield ProviderEvent.diagnostic_decision_ready(
                    parse_diagnostic_decision_response(raw_output)
                )
        except (OllamaModelUnavailable, OllamaConnectionError, OllamaError, OSError, socket.timeout) as exc:
            logger.warning("Qwen diagnostic planning unavailable: %s", exc)
            # The troubleshooting controller will use its bounded compatibility
            # catalog when the local model cannot make a decision.
            yield ProviderEvent.diagnostic_decision_ready(
                DiagnosticDecision(
                    done=True,
                    reason="Local diagnostic planning was unavailable.",
                    available=False,
                )
            )

    def _ensure_model(self, cancel_event: Event) -> None:
        if self._model_checked:
            return
        data = self._request_json("GET", "/api/tags", None, cancel_event)
        models = data.get("models", []) if isinstance(data, dict) else []
        available = {
            str(item.get("name", ""))
            for item in models
            if isinstance(item, dict)
        }
        if self.config.ollama_model in available:
            self._active_model = self.config.ollama_model
        else:
            fallback = self._installed_quantized_variant(models)
            if fallback is None:
                raise OllamaModelUnavailable(
                    f"Ollama model {self.config.ollama_model} is unavailable. "
                    f"Install it locally with: ollama pull {self.config.ollama_model}"
                )
            self._active_model = fallback
            logger.info(
                "Configured model missing; using installed compatible variant=%s",
                self._active_model,
            )
        model_info = next(
            (
                item
                for item in models
                if isinstance(item, dict)
                and str(item.get("name", "")) == self._active_model
            ),
            {},
        )
        try:
            self._model_size_bytes = max(0, int(model_info.get("size", 0)))
        except (TypeError, ValueError):
            self._model_size_bytes = 0
        # /api/show is optional across Ollama versions. Metadata is useful for
        # diagnostics, but it must never prevent an otherwise valid request.
        try:
            metadata = self._request_json(
                "POST",
                "/api/show",
                {"name": self._active_model},
                cancel_event,
            )
            if isinstance(metadata, dict):
                details = metadata.get("details", {})
                if isinstance(details, dict):
                    self._model_quantization = str(details.get("quantization_level", ""))
                    logger.info(
                        "Ollama model=%s parameters=%s quantization=%s recommended=%s",
                        self._active_model,
                        details.get("parameter_size", "unknown"),
                        self._model_quantization or "unknown",
                        _recommended_quantization(self.hardware),
                    )
        except OllamaError as exc:
            logger.debug("Ollama model metadata unavailable: %s", exc)
        self._model_checked = True

    def _installed_quantized_variant(self, models: Any) -> str | None:
        """Choose a matching installed quantization without pulling weights."""
        target = self.config.ollama_model
        family, separator, variant = target.partition(":")
        if not separator:
            return None
        base_variant = variant.split("-", 1)[0]
        prefix = f"{family}:{base_variant}"
        desired = _recommended_quantization(self.hardware).lower().replace("_", "")
        candidates: list[tuple[int, str]] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            if not name.startswith(prefix) or name == target:
                continue
            suffix = name[len(prefix):]
            if suffix and suffix[0] not in {"-", ":"}:
                continue
            details = item.get("details", {})
            quantization = ""
            if isinstance(details, dict):
                quantization = str(details.get("quantization_level", ""))
            haystack = f"{name} {quantization}".lower().replace("_", "")
            score = 100 if desired and desired in haystack else 0
            # Prefer the highest-quality installed fallback when the exact
            # recommendation is not available, but never invent a tag.
            for rank, marker in enumerate(("q8", "q6", "q5", "q4", "q3", "q2"), start=1):
                if marker in haystack:
                    score += 20 - rank
                    break
            candidates.append((score, name))
        return max(candidates, default=(0, ""))[1] or None

    def _stream_chat(
        self,
        messages: tuple[ChatMessage, ...],
        cancel_event: Event,
        system_prompt: str = QWEN_CHAT_SYSTEM_PROMPT,
        response_format: str | dict[str, Any] | None = None,
        max_predict: int | None = None,
    ) -> Iterator[ProviderEvent]:
        self.hardware = refresh_runtime_resources(self.hardware)
        options = self.effective_options()
        if max_predict is not None:
            options["num_predict"] = min(
                int(options["num_predict"]),
                max(32, int(max_predict)),
            )
            options["num_keep"] = min(
                int(options["num_keep"]),
                max(0, int(options["num_ctx"]) - int(options["num_predict"]) - 64),
            )
        self.last_metrics = {}
        body = {
            "model": self._active_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                *[asdict(message) for message in self._fit_messages(messages, options)],
            ],
            "stream": bool(self.config.ollama_stream),
            "options": options,
            # Ollama unloads an idle model after this TTL. Release it after
            # the request when current RAM/VRAM pressure is already high.
            "keep_alive": 0 if self.hardware.under_load else self.config.ollama_keep_alive,
        }
        if response_format is not None:
            body["format"] = response_format
        response, connection = self._open_request(
            "POST", "/api/chat", body, cancel_event, streaming=self.config.ollama_stream
        )
        request_finished = Event()
        cancel_watcher = Thread(
            target=_close_on_cancel,
            args=(cancel_event, request_finished, response, connection),
            name="ollama-cancel-watcher",
            daemon=True,
        )
        cancel_watcher.start()
        try:
            if self.config.ollama_stream:
                while True:
                    if cancel_event.is_set():
                        return
                    try:
                        line = response.readline()
                    except socket.timeout:
                        continue
                    if not line:
                        break
                    event, done = self._decode_line(line)
                    if event is not None:
                        yield event
                    if done:
                        break
            else:
                payload = response.read(2_000_000)
                event, _done = self._decode_line(payload)
                if event is not None:
                    yield event
        except (OSError, http.client.IncompleteRead, AttributeError) as exc:
            if cancel_event.is_set():
                return
            raise OllamaConnectionError(str(exc)) from exc
        finally:
            request_finished.set()
            response.close()
            connection.close()

    def _decode_line(self, line: bytes) -> tuple[ProviderEvent | None, bool]:
        try:
            payload = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaError(f"invalid response from Ollama: {exc}") from exc
        if not isinstance(payload, dict):
            return None, False
        if payload.get("error"):
            raise OllamaError(str(payload["error"]))
        done = bool(payload.get("done", False))
        if done:
            self.last_metrics = {
                key: int(payload[key])
                for key in (
                    "total_duration",
                    "load_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                )
                if isinstance(payload.get(key), (int, float))
            }
        message = payload.get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""
        if content:
            return ProviderEvent.text_chunk(str(content)), done
        return None, done

    def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        cancel_event: Event,
    ) -> dict[str, Any]:
        response, connection = self._open_request(method, path, body, cancel_event)
        try:
            raw = response.read(2_000_000)
        except socket.timeout as exc:
            raise OllamaConnectionError("Ollama request timed out") from exc
        finally:
            response.close()
            connection.close()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaError(f"invalid JSON from Ollama: {exc}") from exc
        if not isinstance(payload, dict):
            raise OllamaError("Ollama returned an invalid JSON object")
        return payload

    def _open_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        cancel_event: Event,
        streaming: bool = False,
    ) -> tuple[http.client.HTTPResponse, http.client.HTTPConnection]:
        if cancel_event.is_set():
            raise OllamaError("request cancelled")
        connection_type = http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
        connection = connection_type(self._host, self._port, timeout=self.config.ollama_timeout_seconds)
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        try:
            connection.request(
                method,
                path,
                body=encoded,
                headers={"Content-Type": "application/json", "Accept": "application/x-ndjson, application/json"},
            )
        except (ConnectionRefusedError, ConnectionResetError, socket.timeout, OSError) as exc:
            connection.close()
            raise OllamaConnectionError(str(exc)) from exc

        # Model loading can delay the HTTP headers. Keep getresponse off the
        # UI/worker cancellation path so Escape can close a queued request too.
        response_queue: queue.Queue[tuple[http.client.HTTPResponse | None, BaseException | None]] = queue.Queue(maxsize=1)

        def receive_response() -> None:
            try:
                response_queue.put((connection.getresponse(), None))
            except BaseException as exc:
                response_queue.put((None, exc))

        Thread(target=receive_response, name="ollama-response-wait", daemon=True).start()
        while True:
            if cancel_event.is_set():
                connection.close()
                raise OllamaError("request cancelled")
            try:
                response, response_error = response_queue.get(timeout=0.1)
                break
            except queue.Empty:
                continue
        if response_error is not None:
            connection.close()
            if isinstance(response_error, (ConnectionRefusedError, ConnectionResetError, socket.timeout, OSError)):
                raise OllamaConnectionError(str(response_error)) from response_error
            raise OllamaError(str(response_error)) from response_error
        assert response is not None
        if response.fp is not None and not streaming:
            try:
                response.fp.raw._sock.settimeout(0.25)  # type: ignore[attr-defined]
            except AttributeError:
                pass
        status = response.status
        if status >= 400:
            raw = response.read(16_000)
            response.close()
            connection.close()
            detail = raw.decode("utf-8", errors="replace").strip()
            if status == 404 and method == "POST":
                raise OllamaModelUnavailable(
                    f"Ollama model {self.config.ollama_model} is unavailable. "
                    f"Install it locally with: ollama pull {self.config.ollama_model}"
                )
            raise OllamaError(f"Ollama HTTP {status}: {detail[:500]}")
        return response, connection

    def effective_options(self) -> dict[str, Any]:
        """Return the bounded Ollama options selected for this machine."""
        if self.config.hardware_auto_tune:
            context = self.hardware.recommended_context(
                self.config.context_size, self.config.max_tokens
            )
            batch = self.hardware.recommended_batch(self.config.batch_size)
            threads = self.config.threads or self.hardware.recommended_threads()
            gpu_layers = (
                999
                if self.hardware.gpu_offload_recommended(self._model_size_bytes)
                else 0
            ) if self.config.gpu_layers == 0 else self.config.gpu_layers
        else:
            context = self.config.context_size
            batch = self.config.batch_size
            threads = self.config.threads or max(1, self.hardware.available_cpus - 1)
            gpu_layers = self.config.gpu_layers

        if self.config.hardware_auto_tune and self.hardware.under_load:
            context = min(context, 2048)
            batch = min(batch, 128)
            threads = min(threads, max(1, self.hardware.available_cpus // 2))

        max_tokens = min(self.config.max_tokens, max(32, context - 256))
        num_keep = min(self.config.num_keep, max(0, context - max_tokens - 64))
        options: dict[str, Any] = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
            "repeat_penalty": self.config.repeat_penalty,
            "num_ctx": context,
            "num_predict": max_tokens,
            "num_keep": num_keep,
            "num_thread": max(1, threads),
            "num_batch": batch,
            "num_gpu": 999 if gpu_layers < 0 else gpu_layers,
            "use_mmap": True,
            "use_mlock": False,
            "low_vram": self.hardware.low_memory or self.hardware.available_vram_bytes < 2 * 1024 ** 3,
        }
        if self.config.seed > 0:
            options["seed"] = self.config.seed
        return options

    def _fit_messages(
        self,
        messages: tuple[ChatMessage, ...],
        options: dict[str, Any],
    ) -> tuple[ChatMessage, ...]:
        """Keep recent conversation turns within the configured context budget."""
        context = int(options["num_ctx"])
        max_tokens = int(options["num_predict"])
        # A conservative character estimate prevents oversized prompts while
        # leaving room for the generated answer and tokenizer variation.
        budget = max(2_048, (context - max_tokens - 128) * 4)
        selected: list[ChatMessage] = []
        used = 0
        for message in reversed(messages):
            content = message.content
            cost = len(content) + 32
            if selected and used + cost > budget:
                break
            if not selected and cost > budget:
                keep = max(256, budget - 32)
                if len(content) > keep:
                    # Keep the safety/instruction prefix and the newest
                    # observation tail when a single result envelope is big.
                    head = keep // 2
                    tail = keep - head
                    content = content[:head] + "\n… [context trimmed] …\n" + content[-tail:]
                cost = len(content) + 32
            selected.append(ChatMessage(message.role, content))
            used += cost
        return tuple(reversed(selected))


def _close_on_cancel(
    cancel_event: Event,
    request_finished: Event,
    response: http.client.HTTPResponse,
    connection: http.client.HTTPConnection,
) -> None:
    """Interrupt a blocked streaming read when Escape/cancel is pressed."""
    while not request_finished.wait(0.1):
        if cancel_event.is_set():
            try:
                response.close()
            finally:
                connection.close()
            return
