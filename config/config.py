"""Configuration loading for the local System Agent and its tool policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import tomllib
from typing import Any


_config_home = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
_data_home = Path(
    os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
)
DEFAULT_CONFIG_PATH = _config_home / "system-agent" / "config.toml"


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(slots=True)
class AgentConfig:
    """User-configurable values used by the overlay."""

    # Expanded response size; the compact size remains the default idle view.
    window_width: int = 430
    window_height: int = 360
    # Compact overlay preference.
    quick_width: int = 420
    quick_height: int = 50
    min_width: int = 420
    min_height: int = 50
    max_width: int = 1180
    max_height: int = 900
    shortcut: str = "<Super>space"
    llm_backend: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_stream: bool = True
    ollama_timeout_seconds: float = 180.0
    ollama_keep_alive: str = "5m"
    context_size: int = 4096
    temperature: float = 0.2
    max_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 40
    min_p: float = 0.05
    repeat_penalty: float = 1.05
    num_keep: int = 768
    seed: int = 0
    threads: int = 0
    batch_size: int = 512
    # 0 lets the provider choose CPU/all-fitting-GPU layers from hardware;
    # positive values force an Ollama layer count, and -1 means all layers.
    gpu_layers: int = 0
    hardware_auto_tune: bool = True
    # Tool execution is deny-by-default except for safe local inspection.
    tool_allowed_roots: tuple[Path, ...] = field(default_factory=lambda: (Path.home(),))
    tool_auto_approve_read_only: bool = True
    tool_allow_network: bool = False
    tool_allow_write: bool = False
    tool_allow_destructive: bool = False
    tool_allow_terminal: bool = False
    tool_max_concurrent: int = 1

    def save_window_size(self, width: int, height: int, path: Path = DEFAULT_CONFIG_PATH) -> None:
        """Persist the expanded preferred size without storing transient state."""
        self.window_width = max(self.min_width, min(int(width), self.max_width))
        self.window_height = max(self.min_height, min(int(height), self.max_height))
        self._write(path)

    def save_quick_size(self, width: int, height: int, path: Path = DEFAULT_CONFIG_PATH) -> None:
        """Persist the compact overlay dimensions."""
        self.quick_width = max(self.min_width, min(int(width), self.max_width))
        self.quick_height = max(self.min_height, min(int(height), self.max_height))
        self._write(path)

    def _write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    f'shortcut = "{self.shortcut}"',
                    "",
                    "[llm]",
                    f'backend = "{self.llm_backend}"',
                    f'base_url = "{self.ollama_base_url}"',
                    f'model = "{self.ollama_model}"',
                    f"stream = {str(self.ollama_stream).lower()}",
                    f"timeout_seconds = {self.ollama_timeout_seconds}",
                    f'keep_alive = "{self.ollama_keep_alive}"',
                    f"context_size = {self.context_size}",
                    f"temperature = {self.temperature}",
                    f"max_tokens = {self.max_tokens}",
                    f"top_p = {self.top_p}",
                    f"top_k = {self.top_k}",
                    f"min_p = {self.min_p}",
                    f"repeat_penalty = {self.repeat_penalty}",
                    f"num_keep = {self.num_keep}",
                    f"seed = {self.seed}",
                    f"threads = {self.threads}",
                    f"batch_size = {self.batch_size}",
                    f"gpu_layers = {self.gpu_layers}",
                    f"hardware_auto_tune = {str(self.hardware_auto_tune).lower()}",
                    "",
                    "[window]",
                    f"width = {self.window_width}",
                    f"height = {self.window_height}",
                    f"quick_width = {self.quick_width}",
                    f"quick_height = {self.quick_height}",
                    f"min_width = {self.min_width}",
                    f"min_height = {self.min_height}",
                    f"max_width = {self.max_width}",
                    f"max_height = {self.max_height}",
                    "",
                    "[tools]",
                    "allowed_roots = [" + ", ".join(f'"{root}"' for root in self.tool_allowed_roots) + "]",
                    f"auto_approve_read_only = {str(self.tool_auto_approve_read_only).lower()}",
                    f"allow_network = {str(self.tool_allow_network).lower()}",
                    f"allow_write = {str(self.tool_allow_write).lower()}",
                    f"allow_destructive = {str(self.tool_allow_destructive).lower()}",
                    f"allow_terminal = {str(self.tool_allow_terminal).lower()}",
                    f"max_concurrent = {self.tool_max_concurrent}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "AgentConfig":
        """Load a TOML config, retaining safe defaults for missing values."""

        if not path.is_file():
            return cls()

        try:
            with path.open("rb") as config_file:
                raw: dict[str, Any] = tomllib.load(config_file)
        except (OSError, tomllib.TOMLDecodeError):
            return cls()

        defaults = cls()
        window = raw.get("window", {})
        llm = raw.get("llm", {})
        tools = raw.get("tools", {})
        quick_width = int(window.get("quick_width", defaults.quick_width))
        quick_height = int(window.get("quick_height", defaults.quick_height))
        min_width = int(window.get("min_width", defaults.min_width))
        min_height = int(window.get("min_height", defaults.min_height))
        window_width = int(window.get("width", defaults.window_width))
        window_height = int(window.get("height", defaults.window_height))

        # Migrate obsolete overlay dimensions that do not fit the current
        # compact layout.
        if (quick_width, quick_height) in ((620, 72), (737, 100), (739, 94)):
            quick_width, quick_height = defaults.quick_width, defaults.quick_height
            min_width, min_height = defaults.min_width, defaults.min_height
        if (window_width, window_height) in ((700, 182), (733, 615)):
            window_width, window_height = defaults.window_width, defaults.window_height

        shortcut = str(raw.get("shortcut", defaults.shortcut))
        if shortcut.startswith("{"):
            shortcut = defaults.shortcut
        raw_roots = tools.get("allowed_roots", [str(root) for root in defaults.tool_allowed_roots])
        if not isinstance(raw_roots, list):
            raw_roots = [str(root) for root in defaults.tool_allowed_roots]
        allowed_roots = tuple(Path(str(root)).expanduser() for root in raw_roots if str(root).strip())
        if not allowed_roots:
            allowed_roots = defaults.tool_allowed_roots
        configured_backend = str(llm.get("backend", defaults.llm_backend)).strip().lower()
        # Keep older llama.cpp settings on the active local Ollama backend.
        if configured_backend in {"llama.cpp", "llama_cpp", "llama-cpp"}:
            configured_backend = defaults.llm_backend
        return cls(
            window_width=window_width,
            window_height=window_height,
            quick_width=quick_width,
            quick_height=quick_height,
            min_width=min_width,
            min_height=min_height,
            max_width=int(window.get("max_width", defaults.max_width)),
            max_height=int(window.get("max_height", defaults.max_height)),
            shortcut=shortcut,
            llm_backend=configured_backend,
            ollama_base_url=str(llm.get("base_url", defaults.ollama_base_url)),
            ollama_model=str(llm.get("model", defaults.ollama_model)),
            ollama_stream=_as_bool(llm.get("stream", defaults.ollama_stream)),
            ollama_timeout_seconds=max(
                5.0, min(900.0, float(llm.get("timeout_seconds", defaults.ollama_timeout_seconds)))
            ),
            ollama_keep_alive=str(llm.get("keep_alive", defaults.ollama_keep_alive)),
            context_size=max(512, min(32_768, int(llm.get("context_size", defaults.context_size)))),
            temperature=max(0.0, min(2.0, float(llm.get("temperature", defaults.temperature)))),
            max_tokens=max(32, min(8_192, int(llm.get("max_tokens", defaults.max_tokens)))),
            top_p=max(0.0, min(1.0, float(llm.get("top_p", defaults.top_p)))),
            top_k=max(1, min(200, int(llm.get("top_k", defaults.top_k)))),
            min_p=max(0.0, min(1.0, float(llm.get("min_p", defaults.min_p)))),
            repeat_penalty=max(0.8, min(2.0, float(llm.get("repeat_penalty", defaults.repeat_penalty)))),
            num_keep=max(0, int(llm.get("num_keep", defaults.num_keep))),
            seed=max(0, int(llm.get("seed", defaults.seed))),
            threads=max(0, min(256, int(llm.get("threads", defaults.threads)))),
            batch_size=max(32, min(4_096, int(llm.get("batch_size", defaults.batch_size)))),
            gpu_layers=max(-1, min(9_999, int(llm.get("gpu_layers", defaults.gpu_layers)))),
            hardware_auto_tune=_as_bool(llm.get("hardware_auto_tune", defaults.hardware_auto_tune)),
            tool_allowed_roots=allowed_roots,
            tool_auto_approve_read_only=bool(
                tools.get("auto_approve_read_only", defaults.tool_auto_approve_read_only)
            ),
            tool_allow_network=bool(tools.get("allow_network", defaults.tool_allow_network)),
            tool_allow_write=bool(tools.get("allow_write", defaults.tool_allow_write)),
            tool_allow_destructive=bool(
                tools.get("allow_destructive", defaults.tool_allow_destructive)
            ),
            tool_allow_terminal=bool(tools.get("allow_terminal", defaults.tool_allow_terminal)),
            tool_max_concurrent=max(
                1, min(4, int(tools.get("max_concurrent", defaults.tool_max_concurrent)))
            ),
        )
