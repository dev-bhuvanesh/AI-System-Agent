"""Curated Wi-Fi diagnostics and one permission-gated recovery action.

The troubleshooting engine uses these tools to determine *why* Wi-Fi is not
working. In particular, a disabled radio is diagnosed before missing IP,
route, or ping results are interpreted. The only modifying operation in this
module is ``wifi_enable``; it has no model-controlled command or path input.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from tools.contracts import PermissionLevel, ToolDefinition
from tools.linux_tools import (
    _check_cancel,
    _network_interfaces,
    _read_optional,
    _read_routes,
    _run_argv,
)


_OUTPUT = {
    "type": "object",
    "description": "A structured observation from the local Wi-Fi or network stack.",
    "additionalProperties": True,
}
_NO_ARGS = {"type": "object", "properties": {}, "additionalProperties": False}


def _wireless_interfaces(cancel_event: Any) -> list[str]:
    """Find wireless interfaces without assuming a distro-specific name."""
    names: set[str] = set()
    net_root = Path("/sys/class/net")
    try:
        entries = sorted(net_root.iterdir(), key=lambda item: item.name)
    except OSError:
        entries = []
    for entry in entries:
        _check_cancel(cancel_event)
        if (entry / "wireless").exists():
            names.add(entry.name)

    if shutil.which("iw"):
        result = _run_argv(["iw", "dev"], cancel_event, 3, max_output=16_000)
        for line in result.get("stdout", "").splitlines():
            match = re.match(r"^\s*Interface\s+(\S+)", line)
            if match:
                names.add(match.group(1))
    return sorted(names)


def _first_wireless_interface(cancel_event: Any) -> str | None:
    return next(iter(_wireless_interfaces(cancel_event)), None)


def _network_management_info(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    available_tools = [name for name in ("nmcli", "rfkill", "iw", "ip") if shutil.which(name)]
    nmcli_available = shutil.which("nmcli") is not None
    nm_running: bool | None = None
    if nmcli_available:
        result = _run_argv(["nmcli", "-t", "-f", "RUNNING", "general"], cancel_event, 3)
        if result.get("exit_code") == 0:
            nm_running = result.get("stdout", "").strip().casefold() == "running"
    elif shutil.which("systemctl"):
        result = _run_argv(["systemctl", "is-active", "NetworkManager"], cancel_event, 3)
        if result.get("exit_code") is not None:
            nm_running = result.get("stdout", "").strip().casefold() == "active"

    managers: list[str] = []
    if nm_running is True:
        managers.append("NetworkManager")
    elif nmcli_available:
        managers.append("NetworkManager")
    if shutil.which("networkctl"):
        managers.append("systemd-networkd")
    return {
        "active_manager": "NetworkManager" if nm_running is True else None,
        "available_managers": managers,
        "available_tools": available_tools,
        "nmcli_available": nmcli_available,
        "networkmanager_running": nm_running,
    }


def _wifi_hardware_info(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    interfaces = _wireless_interfaces(cancel_event)
    return {"hardware_detected": bool(interfaces), "interfaces": interfaces, "source": "sysfs/iw"}


def _wifi_interface_info(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    interfaces = _wireless_interfaces(cancel_event)
    selected = interfaces[0] if interfaces else None
    return {"interface": selected, "exists": selected is not None, "candidates": interfaces}


def _rfkill_snapshot(cancel_event: Any) -> dict[str, Any]:
    if not shutil.which("rfkill"):
        return {"available": False, "devices": [], "software_blocked": None, "hardware_blocked": None}
    result = _run_argv(["rfkill", "list"], cancel_event, 3, max_output=16_000)
    devices: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in result.get("stdout", "").splitlines():
        header = re.match(r"^\s*(\d+)\s*:\s*([^:]+):\s*(.*)$", line)
        if header:
            current = {
                "index": int(header.group(1)),
                "name": header.group(2).strip(),
                "type": header.group(3).strip(),
                "soft_blocked": None,
                "hard_blocked": None,
            }
            devices.append(current)
            continue
        if current is None:
            continue
        blocked = re.match(r"^\s*(Soft|Hard) blocked:\s*(yes|no)\s*$", line, re.IGNORECASE)
        if blocked:
            key = "soft_blocked" if blocked.group(1).casefold() == "soft" else "hard_blocked"
            current[key] = blocked.group(2).casefold() == "yes"

    wireless = [
        item for item in devices
        if any(word in f"{item['name']} {item['type']}".casefold() for word in ("wireless", "wifi", "wlan", "802.11"))
    ]
    relevant = wireless or devices
    soft_values = [item["soft_blocked"] for item in relevant if item["soft_blocked"] is not None]
    hard_values = [item["hard_blocked"] for item in relevant if item["hard_blocked"] is not None]
    return {
        "available": result.get("exit_code") == 0,
        "devices": wireless,
        "software_blocked": any(soft_values) if soft_values else None,
        "hardware_blocked": any(hard_values) if hard_values else None,
    }


def _rfkill_status(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    return _rfkill_snapshot(cancel_event)


def _wifi_radio_state(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    interface = _first_wireless_interface(cancel_event)
    radio_enabled: bool | None = None
    source = "unknown"
    manager = None
    if shutil.which("nmcli"):
        result = _run_argv(["nmcli", "radio", "wifi"], cancel_event, 3)
        value = result.get("stdout", "").strip().casefold()
        if result.get("exit_code") == 0 and value in {"enabled", "disabled"}:
            radio_enabled = value == "enabled"
            source = "nmcli"
            manager = "NetworkManager"
    rfkill = _rfkill_snapshot(cancel_event)
    if radio_enabled is None and rfkill.get("software_blocked") is not None:
        radio_enabled = not bool(rfkill.get("software_blocked") or rfkill.get("hardware_blocked"))
        source = "rfkill"
    return {
        "interface": interface,
        "radio_enabled": radio_enabled,
        "software_blocked": rfkill.get("software_blocked"),
        "hardware_blocked": rfkill.get("hardware_blocked"),
        "manager": manager,
        "source": source,
    }


def _wifi_interface_state(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    interface = _first_wireless_interface(cancel_event)
    if interface is None:
        return {"interface": None, "exists": False, "operstate": None, "carrier": None, "interface_up": None}
    base = Path("/sys/class/net") / interface
    operstate = _read_optional(base / "operstate")
    carrier = _read_optional(base / "carrier")
    return {
        "interface": interface,
        "exists": True,
        "operstate": operstate,
        "carrier": carrier,
        "interface_up": carrier == "1" or operstate == "up",
    }


def _wifi_connection(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    interface = _first_wireless_interface(cancel_event)
    if interface is None:
        return {"interface": None, "connected": None, "state": "not_detected", "connection_name": None, "access_point": None}
    if shutil.which("nmcli"):
        result = _run_argv(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status"], cancel_event, 4)
        for line in result.get("stdout", "").splitlines():
            fields = line.split(":", 3)
            if len(fields) < 3 or fields[0] != interface:
                continue
            state = fields[2].strip()
            return {
                "interface": interface,
                "connected": state.casefold().startswith("connected"),
                "state": state,
                "connection_name": fields[3] if len(fields) > 3 else None,
                "access_point": None,
                "manager": "NetworkManager",
            }
    if shutil.which("iw"):
        result = _run_argv(["iw", "dev", interface, "link"], cancel_event, 4)
        connected = result.get("stdout", "").lstrip().casefold().startswith("connected to")
        ssid = None
        for line in result.get("stdout", "").splitlines():
            if line.strip().startswith("SSID:"):
                ssid = line.split(":", 1)[1].strip()
                break
        return {
            "interface": interface,
            "connected": connected,
            "state": "connected" if connected else "disconnected",
            "connection_name": ssid,
            "access_point": None,
            "manager": None,
        }
    return {"interface": interface, "connected": None, "state": "unknown", "connection_name": None, "access_point": None}


def _wifi_ip_info(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    interface = _first_wireless_interface(cancel_event)
    all_interfaces = _network_interfaces({}, cancel_event).get("interfaces", [])
    item = next((entry for entry in all_interfaces if isinstance(entry, dict) and entry.get("name") == interface), None)
    addresses = item.get("addresses", []) if isinstance(item, dict) else []
    ipv4 = next((address.get("address") for address in addresses if isinstance(address, dict) and address.get("family") == "inet"), None)
    return {"interface": interface, "ip_address": ipv4, "addresses": addresses, "has_ipv4": ipv4 is not None}


def _gateway_connectivity(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    routes = _read_routes(cancel_event)
    default = next((route for route in routes if route.get("destination_hex") == "00000000" and route.get("gateway")), None)
    if not default:
        return {"gateway": None, "interface": None, "reachable": None, "tested": False}
    gateway = str(default["gateway"])
    if not shutil.which("ping"):
        return {"gateway": gateway, "interface": default.get("interface"), "reachable": None, "tested": False}
    result = _run_argv(["ping", "-c", "1", "-W", "3", gateway], cancel_event, 4)
    return {
        "gateway": gateway,
        "interface": default.get("interface"),
        "reachable": result.get("exit_code") == 0,
        "tested": True,
        "exit_code": result.get("exit_code"),
    }


def _wifi_enable(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    """Enable Wi-Fi through a fixed operation selected by trusted code."""
    rfkill = _rfkill_snapshot(cancel_event)
    if rfkill.get("hardware_blocked") is True:
        raise PermissionError("Wi-Fi is hardware-blocked; use the physical switch or BIOS/UEFI settings")
    if shutil.which("nmcli"):
        result = _run_argv(["nmcli", "radio", "wifi", "on"], cancel_event, 12)
        if result.get("exit_code") != 0:
            raise RuntimeError((result.get("stderr") or result.get("stdout") or "NetworkManager could not enable Wi-Fi").strip())
        return {"action": "enable_wifi", "manager": "NetworkManager", "radio_enabled": True, "command": "nmcli radio wifi on"}
    if shutil.which("rfkill"):
        result = _run_argv(["rfkill", "unblock", "wifi"], cancel_event, 12)
        if result.get("exit_code") != 0:
            raise RuntimeError((result.get("stderr") or result.get("stdout") or "rfkill could not unblock Wi-Fi").strip())
        return {"action": "unblock_wifi", "manager": "rfkill", "radio_enabled": True, "command": "rfkill unblock wifi"}
    raise FileNotFoundError("No supported Wi-Fi manager was found (nmcli or rfkill)")


def create_network_tool_definitions() -> tuple[ToolDefinition, ...]:
    """Return the dynamic Wi-Fi/network tools used by troubleshooting."""
    return (
        ToolDefinition("network_management_info", "Detect the active network manager and available local network tools.", _NO_ARGS, _OUTPUT, PermissionLevel.READ_ONLY, 5, _network_management_info, "Network management", safe_diagnostic=True),
        ToolDefinition("wifi_hardware_info", "Detect wireless adapters exposed by Linux without changing device state.", _NO_ARGS, _OUTPUT, PermissionLevel.READ_ONLY, 5, _wifi_hardware_info, "Wi-Fi hardware", safe_diagnostic=True),
        ToolDefinition("wifi_interface_info", "Find the current wireless interface dynamically instead of assuming a name.", _NO_ARGS, _OUTPUT, PermissionLevel.READ_ONLY, 5, _wifi_interface_info, "Wi-Fi interface", safe_diagnostic=True),
        ToolDefinition("wifi_radio_state", "Read whether the Wi-Fi radio is enabled and whether software or hardware blocks are present.", _NO_ARGS, _OUTPUT, PermissionLevel.READ_ONLY, 5, _wifi_radio_state, "Wi-Fi radio", safe_diagnostic=True),
        ToolDefinition("rfkill_status", "Read rfkill software and hardware block state for wireless devices.", _NO_ARGS, _OUTPUT, PermissionLevel.READ_ONLY, 4, _rfkill_status, "rfkill status", safe_diagnostic=True),
        ToolDefinition("wifi_interface_state", "Read the selected wireless interface operational state and carrier.", _NO_ARGS, _OUTPUT, PermissionLevel.READ_ONLY, 4, _wifi_interface_state, "Wi-Fi interface state", safe_diagnostic=True),
        ToolDefinition("wifi_connection", "Read whether the wireless interface is connected and its connection state.", _NO_ARGS, _OUTPUT, PermissionLevel.READ_ONLY, 5, _wifi_connection, "Wi-Fi connection", safe_diagnostic=True),
        ToolDefinition("wifi_ip_info", "Read IPv4 addresses currently assigned to the detected wireless interface.", _NO_ARGS, _OUTPUT, PermissionLevel.READ_ONLY, 5, _wifi_ip_info, "Wi-Fi IP address", safe_diagnostic=True),
        ToolDefinition("gateway_connectivity", "Test reachability of the detected default gateway.", _NO_ARGS, _OUTPUT, PermissionLevel.NETWORK, 8, _gateway_connectivity, "Gateway connectivity", safe_diagnostic=True),
        ToolDefinition("wifi_enable", "Enable Wi-Fi using the detected NetworkManager or rfkill operation; no command input is accepted.", _NO_ARGS, _OUTPUT, PermissionLevel.TERMINAL, 15, _wifi_enable, "Enable Wi-Fi", confirmation_required=True, safe_troubleshooting=True),
    )
