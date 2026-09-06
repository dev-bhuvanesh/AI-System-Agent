"""Read-only Linux diagnostics exposed through the Tool Registry.

These handlers collect observations only. They never change packages, services,
firewall rules, files, devices, or user permissions.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from tools.contracts import PermissionLevel, ToolDefinition
from tools.linux_tools import PathGuard, _check_cancel, _proc_text, _run_argv


_NO_ARGS = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}
_OUTPUT = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "exit_code": {"type": ["integer", "null"]},
        "stdout": {"type": "string"},
        "stderr": {"type": "string"},
        "error_type": {"type": "string"},
        "data": {"type": "object"},
    },
    "required": ["success", "exit_code", "stdout", "stderr", "error_type"],
    "additionalProperties": True,
}


def _command(
    program: str,
    args: list[str],
    cancel_event: Any,
    *,
    timeout: float = 5,
    max_output: int = 40_000,
) -> dict[str, Any]:
    executable = shutil.which(program)
    if executable is None:
        return {
            "available": False,
            "command": [program, *args],
            "success": False,
            "exit_code": None,
            "stdout": "",
            "stderr": f"{program} is not installed",
            "error_type": "COMMAND_NOT_AVAILABLE",
        }
    result = _run_argv([executable, *args], cancel_event, timeout, max_output=max_output)
    return {
        "available": True,
        "command": [executable, *args],
        "success": result.get("exit_code") == 0,
        "error_type": "" if result.get("exit_code") == 0 else "COMMAND_FAILED",
        **result,
    }


def _mounts(cancel_event: Any) -> list[dict[str, str]]:
    mounts: list[dict[str, str]] = []
    for line in _proc_text("/proc/mounts").splitlines():
        _check_cancel(cancel_event)
        parts = line.split()
        if len(parts) >= 3:
            mounts.append({"source": parts[0], "target": parts[1], "filesystem": parts[2]})
    return mounts[:200]


def _storage_status(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    mounts = _mounts(cancel_event)
    usage = os.statvfs("/")
    total = usage.f_blocks * usage.f_frsize
    free = usage.f_bavail * usage.f_frsize
    errors = _command("journalctl", ["-k", "-p", "0..3", "-b", "-n", "40", "--no-pager"], cancel_event)
    return {
        "mounts": mounts,
        "root_filesystem": {
            "path": "/",
            "total_bytes": total,
            "available_bytes": free,
            "used_bytes": max(0, total - free),
            "usage_percent": round((total - free) * 100 / total, 2) if total else None,
        },
        "filesystem_errors": errors,
    }


def _drive_health(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    result = _command(
        "lsblk",
        ["--json", "--bytes", "--output", "NAME,TYPE,SIZE,MODEL,ROTA,TRAN,MOUNTPOINTS"],
        cancel_event,
    )
    devices: list[dict[str, Any]] = []
    if result.get("success"):
        try:
            decoded = json.loads(str(result.get("stdout", "")))
            devices = decoded.get("blockdevices", []) if isinstance(decoded, dict) else []
        except json.JSONDecodeError:
            pass
    return {
        "devices": devices,
        "smart_check": "not_run",
        "smartctl_available": shutil.which("smartctl") is not None,
        "command_result": result,
    }


def _audio_status(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    if shutil.which("wpctl"):
        server = _command("wpctl", ["status"], cancel_event)
        command = "wpctl"
    else:
        server = _command("pactl", ["info"], cancel_event)
        command = "pactl"
    sinks = _command("pactl", ["list", "short", "sinks"], cancel_event) if shutil.which("pactl") else {
        "success": False, "exit_code": None, "stdout": "", "stderr": "pactl is not installed",
        "error_type": "COMMAND_NOT_AVAILABLE", "available": False,
    }
    sources = _command("pactl", ["list", "short", "sources"], cancel_event) if shutil.which("pactl") else {
        "success": False, "exit_code": None, "stdout": "", "stderr": "pactl is not installed",
        "error_type": "COMMAND_NOT_AVAILABLE", "available": False,
    }
    raw = str(server.get("stdout", ""))
    return {
        "server": command,
        "server_running": bool(server.get("success")),
        "outputs_detected": bool(sinks.get("stdout", "").strip()) or "sink" in raw.casefold(),
        "inputs_detected": bool(sources.get("stdout", "").strip()) or "source" in raw.casefold(),
        "server_result": server,
        "outputs": sinks,
        "inputs": sources,
    }


def _display_status(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    session_type = os.environ.get("XDG_SESSION_TYPE", "")
    if shutil.which("wlr-randr"):
        result = _command("wlr-randr", [], cancel_event)
        command = "wlr-randr"
    elif shutil.which("xrandr"):
        result = _command("xrandr", ["--query"], cancel_event)
        command = "xrandr"
    else:
        result = {
            "available": False, "success": False, "exit_code": None, "stdout": "",
            "stderr": "No supported display query tool is installed",
            "error_type": "COMMAND_NOT_AVAILABLE",
        }
        command = ""
    lines = str(result.get("stdout", "")).splitlines()
    connected = [line.strip() for line in lines if " connected" in f" {line}" or line.startswith("connected")]
    return {
        "session_type": session_type,
        "query_command": command,
        "monitors": connected,
        "monitor_count": len(connected),
        "query_result": result,
    }


def _recent_failures(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    result = _command("journalctl", ["-p", "0..3", "-b", "-n", "60", "--no-pager"], cancel_event)
    return {
        "available": result.get("available", False),
        "entries": str(result.get("stdout", "")).splitlines()[-60:],
        "entry_count": len(str(result.get("stdout", "")).splitlines()),
        "command_result": result,
    }


def _service_failures(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    failed = _command("systemctl", ["--failed", "--no-legend", "--no-pager"], cancel_event)
    return {
        "failed_units": [line.strip() for line in str(failed.get("stdout", "")).splitlines() if line.strip()],
        "failed_count": len([line for line in str(failed.get("stdout", "")).splitlines() if line.strip()]),
        "command_result": failed,
    }


def _package_health(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    audit = _command("dpkg", ["--audit"], cancel_event)
    check = _command("apt-get", ["--simulate", "check"], cancel_event, timeout=12)
    return {
        "package_manager": "apt/dpkg" if shutil.which("apt-get") and shutil.which("dpkg") else "unknown",
        "broken_packages": str(audit.get("stdout", "")).splitlines(),
        "audit": audit,
        "dependency_check": check,
        "healthy": bool(audit.get("success")) and bool(check.get("success")),
    }


def _package_update_status(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    result = _command("apt-get", ["--simulate", "upgrade"], cancel_event, timeout=15, max_output=60_000)
    output = f"{result.get('stdout', '')}\n{result.get('stderr', '')}".strip()
    return {
        "package_manager": "apt",
        "updates_available": result.get("success") and any(
            line.startswith(("Inst ", "The following packages will be upgraded"))
            for line in str(result.get("stdout", "")).splitlines()
        ),
        "healthy": result.get("success") is True,
        "command_result": result,
        "output": output,
    }


def _permission_info(args: dict[str, Any], cancel_event: Any, guard: PathGuard) -> dict[str, Any]:
    _check_cancel(cancel_event)
    path = guard.resolve(str(args.get("path", "~")))
    stat = path.stat()
    return {
        "user": os.environ.get("USER", ""),
        "uid": getattr(os, "getuid", lambda: None)(),
        "groups": _command("id", ["-nG"], cancel_event),
        "path": str(path),
        "mode": oct(stat.st_mode & 0o777),
        "owner_uid": stat.st_uid,
        "owner_gid": stat.st_gid,
        "readable": os.access(path, os.R_OK),
        "writable": os.access(path, os.W_OK),
        "executable": os.access(path, os.X_OK),
    }


def _printer_status(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    service = _command("systemctl", ["is-active", "cups"], cancel_event)
    printers = _command("lpstat", ["-p", "-d", "-o"], cancel_event)
    return {
        "cups_active": service.get("success") and str(service.get("stdout", "")).strip() == "active",
        "printers_detected": bool(str(printers.get("stdout", "")).strip()),
        "service": service,
        "printers": printers,
    }


def _battery_status(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    batteries: list[dict[str, str]] = []
    for battery in sorted(Path("/sys/class/power_supply").glob("BAT*")):
        _check_cancel(cancel_event)
        batteries.append({
            "name": battery.name,
            "status": (battery / "status").read_text(errors="replace").strip() if (battery / "status").exists() else "",
            "capacity": (battery / "capacity").read_text(errors="replace").strip() if (battery / "capacity").exists() else "",
            "health": (battery / "health").read_text(errors="replace").strip() if (battery / "health").exists() else "",
        })
    ac_online = []
    for supply in sorted(Path("/sys/class/power_supply").glob("AC*")):
        _check_cancel(cancel_event)
        if (supply / "online").exists():
            ac_online.append({"name": supply.name, "online": (supply / "online").read_text(errors="replace").strip()})
    return {"batteries": batteries, "ac_adapters": ac_online, "battery_detected": bool(batteries)}


def _security_status(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    firewall = _command("ufw", ["status", "verbose"], cancel_event) if shutil.which("ufw") else {
        "available": False, "success": False, "exit_code": None, "stdout": "",
        "stderr": "ufw is not installed", "error_type": "COMMAND_NOT_AVAILABLE",
    }
    firewalld = _command("firewall-cmd", ["--state"], cancel_event) if shutil.which("firewall-cmd") else {
        "available": False, "success": False, "exit_code": None, "stdout": "",
        "stderr": "firewall-cmd is not installed", "error_type": "COMMAND_NOT_AVAILABLE",
    }
    sockets = _command("ss", ["-lntup"], cancel_event)
    return {
        "firewall": firewall,
        "firewalld": firewalld,
        "listening_sockets": sockets,
        "security_change": "none",
    }


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def create_diagnostic_tool_definitions(roots: Iterable[Path]) -> tuple[ToolDefinition, ...]:
    """Return bounded, read-only diagnostic tools for specialist categories."""
    guard = PathGuard(roots)
    path = {"type": "string", "minLength": 1, "maxLength": 4_096}
    safe = dict(
        permission_level=PermissionLevel.READ_ONLY,
        safe_diagnostic=True,
    )
    return (
        ToolDefinition("storage_status", "Inspect mounts, root filesystem usage, and recent kernel filesystem errors.", _NO_ARGS, _OUTPUT, timeout_seconds=8, handler=_storage_status, display_name="Storage status", **safe),
        ToolDefinition("drive_health", "Inspect block devices and expose whether SMART tooling is available; does not run repairs.", _NO_ARGS, _OUTPUT, timeout_seconds=8, handler=_drive_health, display_name="Drive health", **safe),
        ToolDefinition("audio_status", "Inspect PipeWire/PulseAudio availability and detected input/output devices.", _NO_ARGS, _OUTPUT, timeout_seconds=8, handler=_audio_status, display_name="Audio status", **safe),
        ToolDefinition("display_status", "Inspect session type and connected display outputs using the available local display query.", _NO_ARGS, _OUTPUT, timeout_seconds=8, handler=_display_status, display_name="Display status", **safe),
        ToolDefinition("recent_failures", "Read recent boot errors from the local journal without modifying logs or services.", _NO_ARGS, _OUTPUT, timeout_seconds=8, handler=_recent_failures, display_name="Recent failures", **safe),
        ToolDefinition("service_failures", "List failed systemd units without starting, stopping, or changing any service.", _NO_ARGS, _OUTPUT, timeout_seconds=8, handler=_service_failures, display_name="Failed services", **safe),
        ToolDefinition("package_health", "Check dpkg audit state and simulate APT dependency checks without changing packages.", _NO_ARGS, _OUTPUT, timeout_seconds=18, handler=_package_health, display_name="Package health", **safe),
        ToolDefinition("package_update_status", "Simulate APT upgrades to inspect update availability and repository errors without changing packages.", _NO_ARGS, _OUTPUT, timeout_seconds=20, handler=_package_update_status, display_name="Update status", **safe),
        ToolDefinition("permission_info", "Inspect the current user's groups and permissions for an approved path without changing ownership or modes.", _object_schema({"path": path}), _OUTPUT, timeout_seconds=5, handler=lambda a, c: _permission_info(a, c, guard), display_name="Permission status", **safe),
        ToolDefinition("printer_status", "Inspect CUPS status, printers, and print queues without modifying them.", _NO_ARGS, _OUTPUT, timeout_seconds=8, handler=_printer_status, display_name="Printer status", **safe),
        ToolDefinition("battery_status", "Inspect battery, charging, and AC-adapter state from sysfs.", _NO_ARGS, _OUTPUT, timeout_seconds=5, handler=_battery_status, display_name="Battery status", **safe),
        ToolDefinition("security_status", "Inspect available firewall state and listening sockets without weakening security controls.", _NO_ARGS, _OUTPUT, timeout_seconds=10, handler=_security_status, display_name="Security status", **safe),
    )

