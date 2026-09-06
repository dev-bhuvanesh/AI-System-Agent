"""Small, read-only hardware probe used to tune local inference.

The probe never accepts model input and never runs a model-provided command.
It only reads kernel files and, when present, invokes the fixed ``nvidia-smi``
query for VRAM reporting. The probe does not load or configure a model.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path


GiB = 1024 ** 3


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    logical_cpus: int
    available_cpus: int
    physical_cores: int
    ram_bytes: int
    available_ram_bytes: int
    gpu_vram_bytes: int = 0
    gpu_vendor: str = ""
    gpu_name: str = ""
    cpu_model: str = ""
    acceleration_backend: str = "cpu"
    gpu_vram_used_bytes: int = 0
    gpu_utilization_percent: float = 0.0
    cpu_load_1m: float = 0.0

    @property
    def available_vram_bytes(self) -> int:
        if self.gpu_vram_bytes <= 0:
            return 0
        return max(0, self.gpu_vram_bytes - self.gpu_vram_used_bytes)

    @property
    def low_memory(self) -> bool:
        return self.ram_bytes < 12 * GiB or self.available_ram_bytes < 4 * GiB

    @property
    def cpu_pressure(self) -> float:
        return min(1.0, self.cpu_load_1m / max(1.0, float(self.available_cpus)))

    @property
    def memory_pressure(self) -> float:
        if self.ram_bytes <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - self.available_ram_bytes / self.ram_bytes))

    @property
    def gpu_pressure(self) -> float:
        if self.gpu_vram_bytes <= 0:
            return 0.0
        return max(0.0, min(1.0, self.gpu_vram_used_bytes / self.gpu_vram_bytes))

    @property
    def resource_pressure(self) -> float:
        """Return the highest current pressure across CPU, RAM, and VRAM."""
        return max(self.cpu_pressure, self.memory_pressure, self.gpu_pressure)

    @property
    def under_load(self) -> bool:
        """Whether reducing inference pressure is safer for the desktop."""
        return (
            self.cpu_load_1m >= max(1.0, self.available_cpus * 0.85)
            or self.available_ram_bytes < max(2 * GiB, self.ram_bytes // 6)
            or self.gpu_pressure >= 0.90
            or self.gpu_utilization_percent >= 95.0
        )

    @property
    def performance_tier(self) -> str:
        if self.ram_bytes < 8 * GiB or self.available_ram_bytes < 3 * GiB:
            return "low"
        if (
            self.ram_bytes >= 24 * GiB
            and self.available_ram_bytes >= 10 * GiB
            and self.available_vram_bytes >= 8 * GiB
            and self.acceleration_backend != "cpu"
        ):
            return "high"
        return "mid"

    def recommended_threads(self) -> int:
        """Leave one logical CPU available for GTK/Ollama I/O."""
        if self.cpu_pressure >= 0.75:
            return max(1, self.available_cpus // 2)
        return max(1, self.available_cpus - 1)

    def recommended_context(self, requested: int, max_tokens: int) -> int:
        """Bound KV-cache growth while preserving the configured preference."""
        if self.ram_bytes < 8 * GiB or self.available_ram_bytes < 3 * GiB:
            ceiling = 2048
        elif self.ram_bytes < 12 * GiB or self.available_ram_bytes < 6 * GiB:
            ceiling = 3072
        elif self.performance_tier == "high":
            ceiling = 8192
        else:
            ceiling = 4096
        if self.under_load:
            ceiling = min(ceiling, 2048)
        # Leave room for the generated answer and the model runtime itself.
        ceiling = max(1024, ceiling - min(max_tokens, 1024) // 4)
        return max(512, min(int(requested), ceiling))

    def recommended_batch(self, requested: int) -> int:
        if self.available_ram_bytes < 3 * GiB:
            ceiling = 64
        elif self.low_memory or self.available_ram_bytes < 6 * GiB:
            ceiling = 256
        elif self.performance_tier == "high" and not self.under_load:
            ceiling = 1024
        else:
            ceiling = 512
        if self.under_load:
            ceiling = min(ceiling, 128)
        return max(32, min(int(requested), ceiling))

    def gpu_offload_recommended(self, model_size_bytes: int = 0) -> bool:
        """Whether current accelerator capacity safely favors GPU offload."""
        if self.acceleration_backend == "cpu" or self.under_load:
            return False
        # Keep headroom for the runtime and KV cache. The active backend maps
        # this capability to its own offload setting.
        required = int(model_size_bytes * 1.20) if model_size_bytes > 0 else 6 * GiB
        if self.available_vram_bytes < max(2 * GiB, required // 3):
            return False
        return True


def detect_hardware() -> HardwareProfile:
    logical = max(1, int(os.cpu_count() or 1))
    try:
        available = max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        available = logical
    physical = _physical_core_count() or min(logical, available)
    cpu_model = _cpu_model()
    total, available_ram = _memory_bytes()
    vram, vendor, name, backend, vram_used, gpu_utilization = _gpu_details()
    return HardwareProfile(
        logical_cpus=logical,
        available_cpus=available,
        physical_cores=max(1, physical),
        ram_bytes=total,
        available_ram_bytes=available_ram,
        gpu_vram_bytes=vram,
        gpu_vendor=vendor,
        gpu_name=name,
        cpu_model=cpu_model,
        acceleration_backend=backend,
        gpu_vram_used_bytes=vram_used,
        gpu_utilization_percent=gpu_utilization,
        cpu_load_1m=_load_average(),
    )


def refresh_runtime_resources(profile: HardwareProfile) -> HardwareProfile:
    """Refresh volatile resource values without repeating hardware discovery."""
    total, available = _memory_bytes()
    used, gpu_utilization = _gpu_runtime(profile)
    return replace(
        profile,
        ram_bytes=total or profile.ram_bytes,
        available_ram_bytes=available or profile.available_ram_bytes,
        gpu_vram_used_bytes=used,
        gpu_utilization_percent=gpu_utilization,
        cpu_load_1m=_load_average(),
    )


def _memory_bytes() -> tuple[int, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                values[fields[0].rstrip(":")] = int(fields[1]) * 1024
    except OSError:
        pass
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", values.get("MemFree", 0))
    return total, available


def _load_average() -> float:
    try:
        return max(0.0, float(os.getloadavg()[0]))
    except (AttributeError, OSError):
        return 0.0


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() in {"model name", "hardware"}:
                return value.strip()
    except OSError:
        pass
    return ""


def _physical_core_count() -> int:
    pairs: set[tuple[str, str]] = set()
    try:
        current: dict[str, str] = {}
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
            if line.strip():
                key, separator, value = line.partition(":")
                if separator and key.strip() in {"physical id", "core id"}:
                    current[key.strip()] = value.strip()
                continue
            if "physical id" in current and "core id" in current:
                pairs.add((current["physical id"], current["core id"]))
            current = {}
    except OSError:
        return 0
    return len(pairs)


def _gpu_details() -> tuple[int, str, str, str, int, float]:
    best_vram = 0
    best_used = 0
    best_utilization = 0.0
    vendor = ""
    name = ""
    for path in sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_total")):
        try:
            value = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if value > best_vram:
            best_vram = value
            try:
                best_used = int(
                    path.with_name("mem_info_vram_used").read_text(
                        encoding="utf-8"
                    ).strip()
                )
            except (OSError, ValueError):
                best_used = 0
            vendor = _pci_vendor(path.parent / "vendor")
            name = _pci_name(path.parent)

    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        try:
            result = subprocess.run(
                [
                    nvidia,
                    "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            )
            for line in result.stdout.splitlines():
                fields = [item.strip() for item in line.split(",")]
                if len(fields) != 4 or not fields[1].isdigit():
                    continue
                value = int(fields[1]) * 1024 ** 2
                if value > best_vram:
                    best_vram, name, vendor = value, fields[0], "NVIDIA"
                    best_used = int(fields[2]) * 1024 ** 2 if fields[2].isdigit() else 0
                    try:
                        best_utilization = float(fields[3].rstrip(" %"))
                    except ValueError:
                        best_utilization = 0.0
        except (OSError, subprocess.SubprocessError):
            pass
    backend = _acceleration_backend(vendor)
    return best_vram, vendor, name, backend, best_used, best_utilization


def _gpu_runtime(profile: HardwareProfile) -> tuple[int, float]:
    if profile.gpu_vram_bytes <= 0:
        return 0, 0.0
    if profile.gpu_vendor == "NVIDIA":
        nvidia = shutil.which("nvidia-smi")
        if nvidia:
            try:
                result = subprocess.run(
                    [
                        nvidia,
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1,
                    env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
                )
                values: list[tuple[int, float]] = []
                for line in result.stdout.splitlines():
                    fields = [item.strip() for item in line.split(",")]
                    if not fields or not fields[0].isdigit():
                        continue
                    try:
                        utilization = float(fields[1].rstrip(" %")) if len(fields) > 1 else 0.0
                    except ValueError:
                        utilization = 0.0
                    values.append((int(fields[0]) * 1024 ** 2, utilization))
                return max(values, default=(profile.gpu_vram_used_bytes, profile.gpu_utilization_percent))
            except (OSError, ValueError, subprocess.SubprocessError):
                return profile.gpu_vram_used_bytes, profile.gpu_utilization_percent
    values: list[int] = []
    for path in sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")):
        try:
            values.append(int(path.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            continue
    return max(values, default=profile.gpu_vram_used_bytes), profile.gpu_utilization_percent


def _acceleration_backend(vendor: str) -> str:
    if vendor == "NVIDIA" and shutil.which("nvidia-smi"):
        return "cuda"
    if vendor == "AMD" and (
        shutil.which("rocminfo") or shutil.which("rocm-smi")
    ):
        return "rocm"
    if vendor in {"AMD", "Intel", "NVIDIA"} and shutil.which("vulkaninfo"):
        return "vulkan"
    return "cpu"


def _pci_vendor(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return ""
    return {"0x1002": "AMD", "0x10de": "NVIDIA", "0x8086": "Intel"}.get(value, value)


def _pci_name(device_path: Path) -> str:
    lspci = shutil.which("lspci")
    if lspci:
        try:
            result = subprocess.run(
                [lspci, "-s", device_path.resolve().name],
                check=False,
                capture_output=True,
                text=True,
                timeout=1,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            )
            line = result.stdout.strip()
            for marker in (
                "VGA compatible controller:",
                "3D controller:",
                "Display controller:",
            ):
                if marker in line:
                    return line.split(marker, 1)[1].strip()
            if line:
                return line
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        text = (device_path / "uevent").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    match = re.search(r"^PCI_ID=([^\n]+)", text, re.MULTILINE)
    return match.group(1) if match else ""
