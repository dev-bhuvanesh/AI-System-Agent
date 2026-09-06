"""Approved Linux tools.

Every handler in this module is reached through :class:`ToolRegistry`. There
is no shell parser here: subprocess-backed tools receive an argv list and run
with ``shell=False``. File tools are constrained to configured safe roots.
"""

from __future__ import annotations

import os
import platform
import json
import re
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from tools.contracts import PermissionLevel, ToolDefinition


class ToolCancelled(RuntimeError):
    """Raised by a handler when the user cancels an operation."""


class PathGuard:
    """Resolve user paths and prevent symlink/path traversal outside roots."""

    def __init__(self, roots: Iterable[Path]) -> None:
        resolved = tuple(path.expanduser().resolve() for path in roots)
        self.roots = resolved or (Path.home().resolve(),)

    def resolve(self, raw: str, *, allow_root: bool = True) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("path must be a non-empty string")
        raw_path = Path(raw).expanduser()
        if not raw_path.is_absolute():
            raw_path = self.roots[0] / raw_path
        candidate = raw_path.resolve(strict=False)
        if not any(candidate == root or root in candidate.parents for root in self.roots):
            raise PermissionError("path is outside the configured tool roots")
        if not allow_root and candidate in self.roots:
            raise PermissionError("the configured tool root cannot be modified")
        return candidate


def _check_cancel(cancel_event: Any) -> None:
    if cancel_event.is_set():
        raise ToolCancelled("operation cancelled")


def _read_text(path: Path, max_bytes: int = 1_000_000) -> str:
    with path.open("rb") as stream:
        return stream.read(max_bytes).decode("utf-8", errors="replace")


def _proc_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def _run_argv(
    argv: list[str],
    cancel_event: Any,
    timeout_seconds: float,
    max_output: int = 64_000,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Run fixed argv without a shell and with cancellation support."""
    _check_cancel(cancel_event)
    safe_env = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"}
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=safe_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        shell=False,
        start_new_session=True,
    )
    started = time.monotonic()
    try:
        while process.poll() is None:
            if cancel_event.is_set():
                _terminate_process(process)
                raise ToolCancelled("operation cancelled")
            if time.monotonic() - started >= timeout_seconds:
                _terminate_process(process)
                return {
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "command timed out",
                    "timed_out": True,
                }
            time.sleep(0.03)
        stdout, stderr = process.communicate()
        return {
            "exit_code": process.returncode,
            "stdout": stdout[-max_output:],
            "stderr": stderr[-max_output:],
            "timed_out": False,
        }
    finally:
        if process.poll() is None:
            _terminate_process(process)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass


def _system_info(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    uname = platform.uname()
    distro: dict[str, str] = {}
    for line in _proc_text("/etc/os-release").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        distro[key] = value.strip().strip('"')
    memory = _proc_text("/proc/meminfo").splitlines()
    memory_total = next(
        (line.split()[1] for line in memory if line.startswith("MemTotal:") and len(line.split()) > 1),
        "",
    )
    return {
        "hostname": socket.gethostname(),
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "distribution": distro.get("PRETTY_NAME", distro.get("NAME", "")),
        "distribution_id": distro.get("ID", ""),
        "desktop": os.environ.get("XDG_CURRENT_DESKTOP", os.environ.get("DESKTOP_SESSION", "")),
        "session_type": os.environ.get("XDG_SESSION_TYPE", ""),
        "cpu_model": platform.processor(),
        "logical_cpus": os.cpu_count() or 1,
        "memory_total_kib": int(memory_total) if memory_total.isdigit() else None,
        "python": platform.python_version(),
        "user_id": os.getuid() if hasattr(os, "getuid") else None,
    }


def _cpu_usage(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    def snapshot() -> tuple[int, int]:
        fields = _proc_text("/proc/stat").splitlines()
        if not fields:
            return 0, 0
        values = fields[0].split()[1:]
        numbers = [int(item) for item in values if item.isdigit()]
        idle = (numbers[3] if len(numbers) > 3 else 0) + (numbers[4] if len(numbers) > 4 else 0)
        return sum(numbers), idle

    total_a, idle_a = snapshot()
    deadline = time.monotonic() + 0.12
    while time.monotonic() < deadline:
        _check_cancel(cancel_event)
        time.sleep(0.02)
    total_b, idle_b = snapshot()
    total_delta, idle_delta = total_b - total_a, idle_b - idle_a
    usage = 0.0 if total_delta <= 0 else (1.0 - idle_delta / total_delta) * 100.0
    return {
        "usage_percent": round(max(0.0, min(100.0, usage)), 2),
        "logical_cpus": os.cpu_count() or 1,
        "load_average": list(os.getloadavg()),
    }


def _ram_usage(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    values: dict[str, int] = {}
    for line in _proc_text("/proc/meminfo").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            values[parts[0].rstrip(":")] = int(parts[1]) * 1024
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", values.get("MemFree", 0))
    used = max(0, total - available)
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": used,
        "usage_percent": round(used / total * 100, 2) if total else None,
    }


def _disk_usage(args: dict[str, Any], cancel_event: Any, guard: PathGuard) -> dict[str, Any]:
    _check_cancel(cancel_event)
    path = guard.resolve(str(args.get("path", "~")))
    usage = shutil.disk_usage(path)
    return {"path": str(path), "total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def _kernel_info(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    uname = platform.uname()
    return {"release": uname.release, "version": uname.version, "machine": uname.machine}


def _gpu_info(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    cards: list[dict[str, str]] = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        _check_cancel(cancel_event)
        device = (card / "device").resolve()
        vendor = _read_optional(device / "vendor")
        device_id = _read_optional(device / "device")
        cards.append({"name": card.name, "vendor_id": vendor, "device_id": device_id})
    return {"devices": cards, "count": len(cards)}


def _read_optional(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii", errors="replace").strip()
    except (OSError, UnicodeError):
        return ""


def _uptime(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    raw = _proc_text("/proc/uptime").split()
    seconds = float(raw[0]) if raw else 0.0
    return {"seconds": seconds, "human": _format_duration(seconds)}


def _format_duration(seconds: float) -> str:
    days, remainder = divmod(int(seconds), 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    chunks = []
    if days:
        chunks.append(f"{days}d")
    if hours or days:
        chunks.append(f"{hours}h")
    if minutes or hours or days:
        chunks.append(f"{minutes}m")
    chunks.append(f"{secs}s")
    return " ".join(chunks)


def _directory_list(args: dict[str, Any], cancel_event: Any, guard: PathGuard) -> dict[str, Any]:
    path = guard.resolve(str(args.get("path", "~")))
    if not path.is_dir():
        raise NotADirectoryError(str(path))
    limit = max(1, min(int(args.get("limit", 100)), 200))
    entries: list[dict[str, Any]] = []
    for entry in sorted(path.iterdir(), key=lambda item: item.name.lower())[:limit]:
        _check_cancel(cancel_event)
        try:
            kind = "directory" if entry.is_dir() else "file" if entry.is_file() else "other"
            size = entry.stat().st_size if entry.is_file() else None
        except OSError:
            kind, size = "unavailable", None
        entries.append({"name": entry.name, "type": kind, "size_bytes": size})
    return {"path": str(path), "entries": entries, "truncated": len(list(path.iterdir())) > limit}


def _file_read(args: dict[str, Any], cancel_event: Any, guard: PathGuard) -> dict[str, Any]:
    _check_cancel(cancel_event)
    path = guard.resolve(str(args["path"]))
    if not path.is_file():
        raise FileNotFoundError(str(path))
    max_bytes = max(1, min(int(args.get("max_bytes", 100_000)), 1_000_000))
    content = _read_text(path, max_bytes)
    return {"path": str(path), "content": content, "truncated": path.stat().st_size > max_bytes}


def _file_create(args: dict[str, Any], cancel_event: Any, guard: PathGuard) -> dict[str, Any]:
    _check_cancel(cancel_event)
    path = guard.resolve(str(args["path"]), allow_root=False)
    content = str(args.get("content", ""))
    content_bytes = content.encode("utf-8")
    if len(content_bytes) > 1_000_000:
        raise ValueError("content exceeds the 1 MiB tool limit")
    if path.exists() and not bool(args.get("overwrite", False)):
        raise FileExistsError(str(path))
    with path.open("w", encoding="utf-8") as stream:
        stream.write(content)
    verified = path.is_file() and path.stat().st_size == len(content_bytes)
    return {"path": str(path), "bytes_written": len(content_bytes), "verified": verified}


def _file_delete(args: dict[str, Any], cancel_event: Any, guard: PathGuard) -> dict[str, Any]:
    _check_cancel(cancel_event)
    path = guard.resolve(str(args["path"]), allow_root=False)
    if path.is_dir():
        raise IsADirectoryError("use directory_delete for directories")
    path.unlink()
    return {"path": str(path), "deleted": True, "verified": not path.exists()}


def _file_rename(args: dict[str, Any], cancel_event: Any, guard: PathGuard) -> dict[str, Any]:
    _check_cancel(cancel_event)
    source = guard.resolve(str(args["source"]), allow_root=False)
    destination = guard.resolve(str(args["destination"]), allow_root=False)
    if destination.exists():
        raise FileExistsError(str(destination))
    source.rename(destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "verified": not source.exists() and destination.exists(),
    }


def _file_copy(args: dict[str, Any], cancel_event: Any, guard: PathGuard) -> dict[str, Any]:
    _check_cancel(cancel_event)
    source = guard.resolve(str(args["source"]), allow_root=False)
    destination = guard.resolve(str(args["destination"]), allow_root=False)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    if destination.exists() and not bool(args.get("overwrite", False)):
        raise FileExistsError(str(destination))
    shutil.copy2(source, destination)
    source_size = source.stat().st_size
    destination_size = destination.stat().st_size
    return {
        "source": str(source),
        "destination": str(destination),
        "bytes": destination_size,
        "verified": destination.exists() and destination_size == source_size,
    }


def _file_move(args: dict[str, Any], cancel_event: Any, guard: PathGuard) -> dict[str, Any]:
    _check_cancel(cancel_event)
    source = guard.resolve(str(args["source"]), allow_root=False)
    destination = guard.resolve(str(args["destination"]), allow_root=False)
    if destination.exists() and not bool(args.get("overwrite", False)):
        raise FileExistsError(str(destination))
    if destination.exists() and destination.is_file():
        destination.unlink()
    shutil.move(str(source), str(destination))
    return {
        "source": str(source),
        "destination": str(destination),
        "verified": not source.exists() and destination.exists(),
    }


def _directory_create(args: dict[str, Any], cancel_event: Any, guard: PathGuard) -> dict[str, Any]:
    _check_cancel(cancel_event)
    path = guard.resolve(str(args["path"]), allow_root=False)
    path.mkdir(parents=bool(args.get("parents", True)), exist_ok=bool(args.get("exist_ok", False)))
    return {"path": str(path), "created": True, "verified": path.is_dir()}


def _directory_delete(args: dict[str, Any], cancel_event: Any, guard: PathGuard) -> dict[str, Any]:
    _check_cancel(cancel_event)
    path = guard.resolve(str(args["path"]), allow_root=False)
    if not path.is_dir():
        raise NotADirectoryError(str(path))
    if bool(args.get("recursive", False)):
        shutil.rmtree(path)
    else:
        path.rmdir()
    return {
        "path": str(path),
        "deleted": True,
        "recursive": bool(args.get("recursive", False)),
        "verified": not path.exists(),
    }


def _process_list(args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    limit = max(1, min(int(args.get("limit", 100)), 200))
    processes: list[dict[str, Any]] = []
    for entry in sorted(Path("/proc").glob("[0-9]*"), key=lambda item: int(item.name))[:limit]:
        _check_cancel(cancel_event)
        comm = _proc_text(str(entry / "comm")).strip()
        if comm:
            processes.append({"pid": int(entry.name), "name": comm})
    return {"processes": processes, "truncated": len(list(Path("/proc").glob("[0-9]*"))) > limit}


def _process_info(args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    pid = int(args["pid"])
    if pid <= 0:
        raise ValueError("pid must be positive")
    root = Path("/proc") / str(pid)
    if not root.is_dir():
        raise ProcessLookupError(str(pid))
    status: dict[str, str] = {}
    for line in _proc_text(str(root / "status")).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()
    cmdline = _proc_text(str(root / "cmdline")).replace("\x00", " ").strip()
    return {"pid": pid, "name": status.get("Name", ""), "state": status.get("State", ""), "memory": status.get("VmRSS", ""), "cmdline": cmdline[:1_000]}


def _network_interfaces(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    interfaces: list[dict[str, Any]] = []
    ip_details: dict[str, dict[str, Any]] = {}
    if shutil.which("ip"):
        result = _run_argv(["ip", "-j", "address", "show"], cancel_event, 5, max_output=100_000)
        if result["exit_code"] == 0:
            try:
                raw_details = json.loads(result["stdout"])
            except json.JSONDecodeError:
                raw_details = []
            if isinstance(raw_details, list):
                for item in raw_details:
                    if isinstance(item, dict) and item.get("ifname"):
                        ip_details[str(item["ifname"])] = item
    for _index, name in socket.if_nameindex():
        _check_cancel(cancel_event)
        base = Path("/sys/class/net") / name
        detail = ip_details.get(name, {})
        addresses = [
            {
                "family": str(address.get("family", "")),
                "address": str(address.get("local", "")),
                "prefix_length": address.get("prefixlen"),
            }
            for address in detail.get("addr_info", [])
            if isinstance(address, dict) and address.get("local")
        ]
        carrier = _read_optional(base / "carrier")
        interfaces.append({
            "name": name,
            "operstate": _read_optional(base / "operstate"),
            "carrier": carrier,
            "link_up": carrier == "1" or _read_optional(base / "operstate") == "up",
            "mac": _read_optional(base / "address"),
            "addresses": addresses,
            "has_ipv4": any(item["family"] == "inet" for item in addresses),
        })
    return {"interfaces": interfaces}


def _usb_info(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    """Read USB devices exposed by sysfs without changing device state."""
    devices: list[dict[str, str]] = []
    for device in sorted(Path("/sys/bus/usb/devices").glob("*/idVendor")):
        _check_cancel(cancel_event)
        parent = device.parent
        vendor = _read_optional(parent / "idVendor")
        product_id = _read_optional(parent / "idProduct")
        if not vendor or not product_id:
            continue
        devices.append({
            "sysfs_name": parent.name,
            "vendor_id": vendor,
            "product_id": product_id,
            "manufacturer": _read_optional(parent / "manufacturer"),
            "product": _read_optional(parent / "product"),
        })
    return {"devices": devices, "count": len(devices)}


def _bluetooth_info(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    """Read Bluetooth adapters exposed by the kernel."""
    adapters: list[dict[str, str]] = []
    for adapter in sorted(Path("/sys/class/bluetooth").glob("hci*")):
        _check_cancel(cancel_event)
        adapters.append({
            "name": adapter.name,
            "address": _read_optional(adapter / "address"),
            "type": _read_optional(adapter / "type"),
        })
    return {"adapters": adapters, "count": len(adapters)}


def _routing_info(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    routes = _read_routes(cancel_event)
    return {"routes": routes}


def _read_routes(cancel_event: Any) -> list[dict[str, Any]]:
    lines = _proc_text("/proc/net/route").splitlines()
    routes: list[dict[str, Any]] = []
    for line in lines[1:]:
        _check_cancel(cancel_event)
        fields = line.split()
        if len(fields) < 8:
            continue
        routes.append({"interface": fields[0], "destination_hex": fields[1], "gateway": _hex_ipv4(fields[2]), "mask_hex": fields[7], "metric": int(fields[6]) if fields[6].isdigit() else None})
    return routes


def _hex_ipv4(value: str) -> str:
    try:
        raw = int(value, 16).to_bytes(4, "little")
        return ".".join(str(part) for part in raw)
    except (ValueError, OverflowError):
        return ""


def _gateway_detection(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    routes = _read_routes(cancel_event)
    defaults = [route for route in routes if route.get("destination_hex") == "00000000"]
    return {"gateways": [{"interface": route["interface"], "gateway": route["gateway"]} for route in defaults]}


def _dns_info(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    nameservers: list[str] = []
    search: list[str] = []
    for line in _proc_text("/etc/resolv.conf").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "nameserver":
            nameservers.append(parts[1])
        elif len(parts) >= 2 and parts[0] == "search":
            search.extend(parts[1:])
    working: bool | None = None
    if nameservers and shutil.which("getent"):
        result = _run_argv(["getent", "ahostsv4", "example.com"], cancel_event, 5, max_output=8_000)
        working = result["exit_code"] == 0 and bool(result["stdout"].strip())
    elif nameservers:
        # A configured resolver is useful evidence when no bounded resolver
        # probe is available, but do not claim that it was tested.
        working = None
    return {"nameservers": nameservers, "search_domains": search, "working": working}


def _ping_connectivity(args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    host = str(args["host"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]{0,252}", host) or host.startswith("-"):
        raise ValueError("host contains unsupported characters")
    count = max(1, min(int(args.get("count", 1)), 3))
    timeout = max(1, min(int(args.get("timeout_seconds", 5)), 10))
    result = _run_argv(["ping", "-c", str(count), "-W", str(timeout), host], cancel_event, timeout + 1)
    return {"host": host, "reachable": result["exit_code"] == 0, **result}


def _service_status(args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    service = str(args["service"])
    if not re.fullmatch(r"[A-Za-z0-9_.@:-]{1,128}", service):
        raise ValueError("invalid service name")
    command = ["systemctl", "show", service, "--no-pager", "--property=Id,LoadState,ActiveState,SubState,UnitFileState"]
    result = _run_argv(command, cancel_event, 5)
    if result["exit_code"] != 0 and service in {"pipewire", "wireplumber"}:
        command = ["systemctl", "--user", "show", service, "--no-pager", "--property=Id,LoadState,ActiveState,SubState,UnitFileState"]
        result = _run_argv(command, cancel_event, 5)
    properties: dict[str, str] = {}
    for line in result["stdout"].splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = value
    return {
        "service": service,
        "scope": "user" if "--user" in command else "system",
        "active_state": properties.get("ActiveState", ""),
        "sub_state": properties.get("SubState", ""),
        "load_state": properties.get("LoadState", ""),
        "unit_file_state": properties.get("UnitFileState", ""),
        **result,
    }


_SAFE_TERMINAL_PROGRAMS = frozenset({"date", "df", "free", "id", "ip", "ps", "systemctl", "uname", "uptime", "whoami"})
_FORBIDDEN_TERMINAL_ARGS = frozenset({"-c", "--command", "--exec", "--execute"})


def _controlled_terminal(args: dict[str, Any], cancel_event: Any, guard: PathGuard) -> dict[str, Any]:
    program = str(args["program"])
    if program not in _SAFE_TERMINAL_PROGRAMS:
        raise PermissionError("program is not in the controlled terminal allowlist")
    raw_args = args.get("args", [])
    if not isinstance(raw_args, list) or len(raw_args) > 16:
        raise ValueError("args must contain at most 16 items")
    command_args: list[str] = []
    for value in raw_args:
        value = str(value)
        if not value or len(value) > 160 or "\n" in value or "\x00" in value or value in _FORBIDDEN_TERMINAL_ARGS:
            raise ValueError("terminal argument is not allowed")
        if any(char in value for char in (";", "|", "&", ">", "<", "`")):
            raise ValueError("shell syntax is not allowed")
        command_args.append(value)
    if program == "ip":
        safe_ip_operations = {"address", "addr", "link", "route", "rule", "neigh", "-details", "-brief", "-4", "-6"}
        if any(value in {"add", "del", "delete", "set", "flush", "replace", "change", "append"} for value in command_args):
            raise PermissionError("network-changing ip operations are not allowed")
        if command_args and command_args[0].startswith("-") and command_args[0] not in safe_ip_operations:
            raise PermissionError("unsupported ip diagnostic option")
    if program == "systemctl":
        valid_system = len(command_args) == 2 and command_args[0] == "restart"
        valid_user = len(command_args) == 3 and command_args[:2] == ["--user", "restart"]
        if not valid_system and not valid_user:
            raise PermissionError("controlled systemctl only permits restarting one named service")
        service_name = command_args[-1]
        if re.fullmatch(r"[A-Za-z0-9_.@:-]{1,128}", service_name) is None:
            raise ValueError("invalid service name")
    cwd = None
    if args.get("cwd"):
        cwd = str(guard.resolve(str(args["cwd"])))
        if not Path(cwd).is_dir():
            raise NotADirectoryError(cwd)
    execution_argv = [program, *command_args]
    if program == "systemctl" and command_args[0] == "restart":
        if shutil.which("pkexec"):
            execution_argv = ["pkexec", *execution_argv]
        elif shutil.which("sudo"):
            execution_argv = ["sudo", "-n", *execution_argv]
        else:
            raise FileNotFoundError("pkexec or sudo is required to restart a system service")
    result = _run_argv(execution_argv, cancel_event, 30 if program == "systemctl" else 5, cwd=cwd)
    response = {"program": program, "args": command_args, "execution": execution_argv, **result}
    if program == "systemctl" and result["exit_code"] == 0:
        verify_argv = ["systemctl", "is-active", service_name]
        if valid_user:
            verify_argv.insert(1, "--user")
        verification = _run_argv(verify_argv, cancel_event, 5)
        response["verified"] = verification["exit_code"] == 0 and verification["stdout"].strip() == "active"
        response["verification"] = verification
    return response


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


_OUTPUT = {
    "type": "object",
    "description": "A JSON object containing the tool-specific observation or change result.",
    "additionalProperties": True,
}
_PATH = {"type": "string", "minLength": 1, "maxLength": 4_096}


def create_tool_definitions(roots: Iterable[Path]) -> tuple[ToolDefinition, ...]:
    """Build all initial Linux tools with their metadata and safe handlers."""
    guard = PathGuard(roots)
    string = {"type": "string", "minLength": 1, "maxLength": 4_096}
    no_args = _object_schema({})
    return (
        ToolDefinition("system_info", "Read basic host and operating-system information.", no_args, _OUTPUT, PermissionLevel.READ_ONLY, 3, _system_info, "System information"),
        ToolDefinition("cpu_usage", "Read a short CPU usage sample and load average.", no_args, _OUTPUT, PermissionLevel.READ_ONLY, 4, _cpu_usage, "CPU usage"),
        ToolDefinition("ram_usage", "Read memory totals and current usage from the local kernel view.", no_args, _OUTPUT, PermissionLevel.READ_ONLY, 3, _ram_usage, "RAM usage"),
        ToolDefinition("disk_usage", "Read filesystem capacity for an approved path.", _object_schema({"path": _PATH}), _OUTPUT, PermissionLevel.READ_ONLY, 3, lambda a, c: _disk_usage(a, c, guard), "Disk usage"),
        ToolDefinition("kernel_info", "Read Linux kernel release and architecture information.", no_args, _OUTPUT, PermissionLevel.READ_ONLY, 3, _kernel_info, "Kernel information"),
        ToolDefinition("gpu_info", "Read locally exposed GPU device identifiers from sysfs.", no_args, _OUTPUT, PermissionLevel.READ_ONLY, 3, _gpu_info, "GPU information"),
        ToolDefinition("uptime", "Read system uptime from the local kernel view.", no_args, _OUTPUT, PermissionLevel.READ_ONLY, 3, _uptime, "Uptime"),
        ToolDefinition("directory_list", "List entries in an approved directory.", _object_schema({"path": _PATH, "limit": {"type": "integer", "minimum": 1, "maximum": 200}}), _OUTPUT, PermissionLevel.READ_ONLY, 5, lambda a, c: _directory_list(a, c, guard), "Directory listing"),
        ToolDefinition("file_read", "Read UTF-8 text from an approved file with a byte limit.", _object_schema({"path": _PATH, "max_bytes": {"type": "integer", "minimum": 1, "maximum": 1_000_000}}, ["path"]), _OUTPUT, PermissionLevel.READ_ONLY, 5, lambda a, c: _file_read(a, c, guard), "Read file"),
        ToolDefinition("file_create", "Create or explicitly overwrite a file inside an approved root.", _object_schema({"path": _PATH, "content": {"type": "string", "maxLength": 1_000_000}, "overwrite": {"type": "boolean"}}, ["path"]), _OUTPUT, PermissionLevel.WRITE, 5, lambda a, c: _file_create(a, c, guard), "Create file"),
        ToolDefinition("file_delete", "Delete one file inside an approved root.", _object_schema({"path": _PATH}, ["path"]), _OUTPUT, PermissionLevel.DESTRUCTIVE, 5, lambda a, c: _file_delete(a, c, guard), "Delete file"),
        ToolDefinition("file_rename", "Rename one filesystem entry inside an approved root.", _object_schema({"source": _PATH, "destination": _PATH}, ["source", "destination"]), _OUTPUT, PermissionLevel.WRITE, 5, lambda a, c: _file_rename(a, c, guard), "Rename file"),
        ToolDefinition("file_copy", "Copy one file to another approved path.", _object_schema({"source": _PATH, "destination": _PATH, "overwrite": {"type": "boolean"}}, ["source", "destination"]), _OUTPUT, PermissionLevel.WRITE, 8, lambda a, c: _file_copy(a, c, guard), "Copy file"),
        ToolDefinition("file_move", "Move one filesystem entry to another approved path.", _object_schema({"source": _PATH, "destination": _PATH, "overwrite": {"type": "boolean"}}, ["source", "destination"]), _OUTPUT, PermissionLevel.WRITE, 8, lambda a, c: _file_move(a, c, guard), "Move file"),
        ToolDefinition("directory_create", "Create a directory inside an approved root.", _object_schema({"path": _PATH, "parents": {"type": "boolean"}, "exist_ok": {"type": "boolean"}}, ["path"]), _OUTPUT, PermissionLevel.WRITE, 5, lambda a, c: _directory_create(a, c, guard), "Create directory"),
        ToolDefinition("directory_delete", "Delete an empty or explicitly recursive directory inside an approved root.", _object_schema({"path": _PATH, "recursive": {"type": "boolean"}}, ["path"]), _OUTPUT, PermissionLevel.DESTRUCTIVE, 10, lambda a, c: _directory_delete(a, c, guard), "Delete directory"),
        ToolDefinition("process_list", "List local process IDs and names from procfs.", _object_schema({"limit": {"type": "integer", "minimum": 1, "maximum": 200}}), _OUTPUT, PermissionLevel.READ_ONLY, 5, _process_list, "Process list"),
        ToolDefinition("process_info", "Read safe summary fields for one local process.", _object_schema({"pid": {"type": "integer", "minimum": 1}}, ["pid"]), _OUTPUT, PermissionLevel.READ_ONLY, 5, _process_info, "Process information"),
        ToolDefinition("network_interfaces", "Read local interface names, state, and MAC addresses.", no_args, _OUTPUT, PermissionLevel.READ_ONLY, 3, _network_interfaces, "Network interfaces"),
        ToolDefinition("bluetooth_info", "Read Bluetooth adapters exposed by the local kernel.", no_args, _OUTPUT, PermissionLevel.READ_ONLY, 3, _bluetooth_info, "Bluetooth adapters"),
        ToolDefinition("usb_info", "Read USB devices exposed by the local kernel without changing them.", no_args, _OUTPUT, PermissionLevel.READ_ONLY, 5, _usb_info, "USB devices"),
        ToolDefinition("routing_info", "Read local IPv4 routing table entries.", no_args, _OUTPUT, PermissionLevel.READ_ONLY, 3, _routing_info, "Routing information"),
        ToolDefinition("gateway_detection", "Detect default gateways from the local routing table.", no_args, _OUTPUT, PermissionLevel.READ_ONLY, 3, _gateway_detection, "Gateway detection"),
        ToolDefinition("dns_info", "Read configured DNS nameservers and search domains.", no_args, _OUTPUT, PermissionLevel.READ_ONLY, 3, _dns_info, "DNS information"),
        ToolDefinition("ping_connectivity", "Ping one validated host for connectivity testing.", _object_schema({"host": string, "count": {"type": "integer", "minimum": 1, "maximum": 3}, "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 10}}, ["host"]), _OUTPUT, PermissionLevel.NETWORK, 12, _ping_connectivity, "Ping connectivity", True),
        ToolDefinition("service_status", "Read status for one validated systemd service name.", _object_schema({"service": {"type": "string", "pattern": r"^[A-Za-z0-9_.@:-]{1,128}$"}}, ["service"]), _OUTPUT, PermissionLevel.READ_ONLY, 7, _service_status, "Service status"),
        ToolDefinition("controlled_terminal", "Run a short read-only diagnostic or one fixed service restart from a constrained executable allowlist; never a shell.", _object_schema({"program": {"type": "string", "enum": sorted(_SAFE_TERMINAL_PROGRAMS)}, "args": {"type": "array", "items": {"type": "string", "maxLength": 160}, "maxItems": 16}, "cwd": _PATH}, ["program"]), _OUTPUT, PermissionLevel.TERMINAL, 8, lambda a, c: _controlled_terminal(a, c, guard), "Controlled terminal", safe_troubleshooting=True),
    )
