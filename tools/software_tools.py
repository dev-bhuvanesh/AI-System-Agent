"""Approved package-management tools used by the software agent.

The handlers below construct every command from a small manager allowlist and
validated package identifiers. They never accept a model-supplied shell line,
URL, executable, or command flag. Mutating handlers are marked
``safe_software`` and can only run through a trusted confirmation call in the
registry.
"""

from __future__ import annotations

import platform
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from software.contracts import SoftwareErrorCode
from tools.contracts import PermissionLevel, ToolDefinition, ToolExecutionError
from tools.linux_tools import PathGuard, _check_cancel, _run_argv


_MANAGERS = ("apt", "dnf", "yum", "pacman", "zypper", "flatpak", "snap")
_PACKAGE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9+_.:@/-]{0,127}$"
_PACKAGE_RE = re.compile(_PACKAGE_PATTERN)
_VENDOR_NAMES = ("google_chrome", "visual_studio_code")
_VENDOR_DOWNLOADS = {
    "google_chrome": {
        "apt": {
            "host": "dl.google.com",
            "redirect_hosts": (),
            "url": "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb",
            "filename": "google-chrome-stable_current_amd64.deb",
            "package": "google-chrome-stable",
            "format": "deb",
        },
        "dnf": {
            "host": "dl.google.com",
            "redirect_hosts": (),
            "url": "https://dl.google.com/linux/direct/google-chrome-stable_current_x86_64.rpm",
            "filename": "google-chrome-stable_current_x86_64.rpm",
            "package": "google-chrome-stable",
            "format": "rpm",
        },
    },
    "visual_studio_code": {
        "apt": {
            "host": "update.code.visualstudio.com",
            "redirect_hosts": ("vscode.download.prss.microsoft.com",),
            "url": "https://update.code.visualstudio.com/latest/linux-deb-x64/stable",
            "filename": "code_latest_amd64.deb",
            "package": "code",
            "format": "deb",
        },
        "dnf": {
            "host": "update.code.visualstudio.com",
            "redirect_hosts": ("vscode.download.prss.microsoft.com",),
            "url": "https://update.code.visualstudio.com/latest/linux-rpm-x64/stable",
            "filename": "code_latest_x86_64.rpm",
            "package": "code",
            "format": "rpm",
        },
    },
}
_ProgressCallback = Callable[[int, int | None, float | None], None]


def _existing_download(destination: Path, package: str) -> Path | None:
    """Return a non-empty package file already saved for ``package``."""
    pattern = re.compile(rf"^{re.escape(package)}(?:[_-].+)?\.(?:deb|rpm)$", re.IGNORECASE)
    try:
        candidates = [
            path
            for path in destination.iterdir()
            if path.is_file() and path.stat().st_size > 0 and pattern.fullmatch(path.name)
        ]
    except OSError:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns, default=None)


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


_OUTPUT = {
    "type": "object",
    "description": "Structured package-manager result with exit code and bounded output.",
    "additionalProperties": True,
}
_MANAGER = {"type": "string", "enum": list(_MANAGERS)}
_PACKAGE = {"type": "string", "pattern": _PACKAGE_PATTERN}
_SEARCH_QUERY = {
    "type": "string",
    "pattern": r"^[A-Za-z0-9][A-Za-z0-9+_.:@/ -]{0,95}$",
}


def _manager_command(manager: str) -> str:
    if manager not in _MANAGERS:
        raise ValueError("unsupported package manager")
    executable = {"apt": "apt-get", "dnf": "dnf", "yum": "yum", "pacman": "pacman", "zypper": "zypper", "flatpak": "flatpak", "snap": "snap"}[manager]
    if shutil.which(executable) is None:
        raise FileNotFoundError(f"{executable} is not installed")
    return executable


def _package_name(value: Any) -> str:
    package = str(value)
    if _PACKAGE_RE.fullmatch(package) is None or package.startswith("-"):
        raise ValueError("package identifier contains unsupported characters")
    return package


def _search_query(value: Any) -> str:
    query = " ".join(str(value).strip().split())
    if not query or len(query) > 96 or query.startswith("-"):
        raise ValueError("search query is invalid")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9+_.:@/ -]{0,95}", query) is None:
        raise ValueError("search query contains unsupported characters")
    return query


def _profile(_args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    release: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                release[key] = value.strip().strip('"')
    except OSError:
        pass
    executable_map = {
        "apt": "apt-get", "dnf": "dnf", "yum": "yum", "pacman": "pacman",
        "zypper": "zypper", "flatpak": "flatpak", "snap": "snap",
    }
    available = tuple(name for name, executable in executable_map.items() if shutil.which(executable))
    primary = next((name for name in ("apt", "dnf", "yum", "pacman", "zypper") if name in available), "")
    return {
        "distribution": release.get("PRETTY_NAME", release.get("NAME", "")),
        "distribution_id": release.get("ID", ""),
        "version": release.get("VERSION_ID", ""),
        "architecture": platform.machine(),
        "package_manager": primary,
        "available_managers": list(available),
        "sources": ["distribution repositories"] if primary else [],
    }


def _search(args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    manager = str(args["manager"])
    query = _search_query(args["query"])
    executable = _manager_command(manager)
    if manager == "apt":
        if shutil.which("apt-cache") is None:
            raise FileNotFoundError("apt-cache is not installed")
        executable = "apt-cache"
    commands = {
        "apt": [executable, "search", query],
        "dnf": [executable, "list", "--available", query],
        "yum": [executable, "list", "available", query],
        "pacman": [executable, "-Ss", query],
        "zypper": [executable, "--non-interactive", "search", query],
        "flatpak": [executable, "search", "--columns=application,name,version", query],
        "snap": [executable, "find", query],
    }
    result = _run_argv(commands[manager], cancel_event, 30)
    if result.get("timed_out") or result.get("exit_code") != 0:
        if not result.get("stdout") and not result.get("stderr"):
            raise ToolExecutionError(
                SoftwareErrorCode.PACKAGE_NOT_FOUND.value,
                f"no package matches the repository query {query!r}",
                result,
            )
        _raise_command_failure(result)
    return {"manager": manager, "query": query, "matches": True, **result}


def _query(args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    manager = str(args["manager"])
    package = _package_name(args["package"])
    executable = _manager_command(manager)
    commands = {
        "apt": ["dpkg-query", "-W", "-f=${Status} ${Version}", package],
        "dnf": ["rpm", "-q", "--qf", "%{NAME} %{VERSION}-%{RELEASE}", package],
        "yum": ["rpm", "-q", "--qf", "%{NAME} %{VERSION}-%{RELEASE}", package],
        "pacman": [executable, "-Q", package],
        "zypper": ["rpm", "-q", "--qf", "%{NAME} %{VERSION}-%{RELEASE}", package],
        "flatpak": [executable, "info", package],
        "snap": [executable, "list", package],
    }
    result = _run_argv(commands[manager], cancel_event, 12)
    version_output = result["stdout"].strip()
    if manager == "apt":
        # dpkg-query exits successfully for packages in states such as
        # ``deinstall ok config-files``. Only the canonical installed state
        # means that the package is actually present and configured.
        installed = bool(
            result["exit_code"] == 0
            and re.search(r"(?:^|\s)install ok installed(?:\s|$)", version_output, re.IGNORECASE)
        )
    else:
        installed = result["exit_code"] == 0
    return {
        "manager": manager,
        "package": package,
        "installed": installed,
        "version_output": version_output,
        "version": _extract_installed_version(manager, result["stdout"], package) if installed else "",
        **result,
    }


def _available_version(args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    """Read the candidate version from the selected trusted source."""
    _check_cancel(cancel_event)
    manager = str(args["manager"])
    package = _package_name(args["package"])
    executable = _manager_command(manager)
    if manager == "apt":
        if shutil.which("apt-cache") is None:
            raise FileNotFoundError("apt-cache is not installed")
        command = ["apt-cache", "policy", package]
    elif manager in {"dnf", "yum"}:
        command = [executable, "info", package]
    elif manager == "pacman":
        command = [executable, "-Si", package]
    elif manager == "zypper":
        command = [executable, "--non-interactive", "info", package]
    elif manager == "flatpak":
        command = [executable, "remote-info", "flathub", package]
    else:
        command = [executable, "info", package]
    result = _run_argv(command, cancel_event, 30, max_output=40_000)
    if result.get("timed_out") or result.get("exit_code") != 0:
        _raise_command_failure(result)
    version = _extract_available_version(manager, result["stdout"])
    return {
        "manager": manager,
        "package": package,
        "available": bool(version) and result["exit_code"] == 0,
        "version": version,
        **result,
    }


def _extract_version(output: str) -> str:
    tokens = re.findall(r"(?<![A-Za-z])\d[0-9A-Za-z.+:~_-]*", output or "")
    return tokens[-1].strip(".,") if tokens else ""


def _extract_installed_version(manager: str, output: str, package: str) -> str:
    if manager == "snap":
        for line in (output or "").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0].casefold() == package.casefold():
                return fields[1]
    if manager == "flatpak":
        match = re.search(r"^Version:\s*(\S+)", output or "", re.I | re.M)
        if match:
            return match.group(1)
    if manager in {"pacman", "zypper"}:
        match = re.search(r"^Version\s*:\s*(\S+)", output or "", re.I | re.M)
        if match:
            return match.group(1)
    return _extract_version(output)


def _extract_available_version(manager: str, output: str) -> str:
    for line in (output or "").splitlines():
        if manager == "apt" and line.strip().lower().startswith("candidate:"):
            return line.split(":", 1)[1].strip()
        if manager in {"dnf", "yum", "pacman", "zypper", "flatpak", "snap"}:
            match = re.match(r"\s*(?:version|version\s*:|latest/stable)\s*:?\s*(\S+)", line, re.I)
            if match:
                return match.group(1).strip()
    return _extract_version(output)


def _privileged(command: list[str]) -> list[str]:
    if shutil.which("pkexec"):
        return ["pkexec", *command]
    if shutil.which("sudo"):
        return ["sudo", "-n", *command]
    raise FileNotFoundError("pkexec or sudo is required for package changes")


def _install_command(manager: str, package: str, *, reinstall: bool = False) -> list[str]:
    executable = _manager_command(manager)
    if manager == "apt":
        return _privileged([executable, "install", "-y", "--no-install-recommends", *(["--reinstall"] if reinstall else []), package])
    if manager in {"dnf", "yum"}:
        return _privileged([executable, "reinstall" if reinstall else "install", "-y", package])
    if manager == "pacman":
        return _privileged([executable, "-S", "--noconfirm", package])
    if manager == "zypper":
        return _privileged([executable, "--non-interactive", "install", package])
    if manager == "flatpak":
        return _privileged([executable, "install", "-y", "flathub", package])
    return _privileged([executable, "install", package])


def _install(args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    manager = str(args["manager"])
    package = _package_name(args["package"])
    result = _run_argv(_install_command(manager, package), cancel_event, 900, max_output=120_000)
    _ensure_command_success(result)
    return {"manager": manager, "package": package, "action": "install", **result}


def _reinstall(args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    manager = str(args["manager"])
    package = _package_name(args["package"])
    result = _run_argv(_install_command(manager, package, reinstall=True), cancel_event, 900, max_output=120_000)
    _ensure_command_success(result)
    return {"manager": manager, "package": package, "action": "reinstall", **result}


def _remove(args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    manager = str(args["manager"])
    package = _package_name(args["package"])
    executable = _manager_command(manager)
    commands = {
        "apt": [executable, "remove", "-y", package],
        "dnf": [executable, "remove", "-y", package],
        "yum": [executable, "remove", "-y", package],
        "pacman": [executable, "-R", "--noconfirm", package],
        "zypper": [executable, "--non-interactive", "remove", package],
        "flatpak": [executable, "uninstall", "-y", package],
        "snap": [executable, "remove", package],
    }
    result = _run_argv(_privileged(commands[manager]), cancel_event, 900, max_output=120_000)
    _ensure_command_success(result)
    return {"manager": manager, "package": package, "action": "remove", **result}


def _update(args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    _check_cancel(cancel_event)
    manager = str(args["manager"])
    package = str(args.get("package", "all"))
    executable = _manager_command(manager)
    if package != "all":
        package = _package_name(package)
    if manager == "apt":
        command = [executable, "install", "-y", "--only-upgrade", package] if package != "all" else [executable, "upgrade", "-y"]
    elif manager in {"dnf", "yum"}:
        command = [executable, "upgrade", "-y", *([] if package == "all" else [package])]
    elif manager == "pacman":
        command = [executable, "-Syu", "--noconfirm"] if package == "all" else [executable, "-S", "--noconfirm", package]
    elif manager == "zypper":
        command = [executable, "--non-interactive", "update", *([] if package == "all" else [package])]
    elif manager == "flatpak":
        command = [executable, "update", "-y", *([] if package == "all" else [package])]
    else:
        command = [executable, "refresh"] if package == "all" else [executable, "refresh", package]
    result = _run_argv(_privileged(command), cancel_event, 900, max_output=120_000)
    _ensure_command_success(result)
    return {"manager": manager, "package": package, "action": "update", **result}


def _download(args: dict[str, Any], cancel_event: Any, guard: PathGuard, report: _ProgressCallback | None = None) -> dict[str, Any]:
    _check_cancel(cancel_event)
    manager = str(args["manager"])
    package = _package_name(args["package"])
    destination = guard.resolve(str(args["destination"]), allow_root=False)
    destination.mkdir(parents=True, exist_ok=True)
    existing = _existing_download(destination, package)
    if existing is not None:
        size = existing.stat().st_size
        if report is not None:
            report(size, size, None)
        return {
            "manager": manager,
            "package": package,
            "destination": str(destination),
            "path": str(existing),
            "paths": [str(existing)],
            "action": "download",
            "already_downloaded": True,
            "downloaded": False,
            "bytes": size,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
        }
    if manager == "apt":
        # ``download`` is provided by apt itself; apt-get has no download
        # subcommand. Keep the executable fixed and still validate that the
        # distro's apt tooling is present before starting.
        if shutil.which("apt") is None:
            raise FileNotFoundError("apt is not installed")
        command = ["apt", "download", package]
    elif manager == "dnf":
        executable = _manager_command(manager)
        command = [executable, "download", "--destdir", str(destination), package]
    else:
        raise ValueError("download is supported only for apt and dnf sources")
    before = {path for path in destination.iterdir() if path.is_file()}
    result = _run_argv(command, cancel_event, 900, max_output=120_000, cwd=str(destination))
    after = {path for path in destination.iterdir() if path.is_file()}
    paths = sorted(
        (path for path in after - before if not path.name.endswith(".part")),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    ) if result.get("exit_code") == 0 else []
    if not paths and result.get("exit_code") == 0:
        # A package can already exist in the approved destination. Return the
        # newest matching package so the UI still reports its exact location.
        paths = sorted(
            (path for path in after if package.casefold() in path.name.casefold()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    path = paths[0] if paths else None
    if report is not None and path is not None:
        report(path.stat().st_size, None, None)
    _ensure_command_success(result)
    return {
        "manager": manager,
        "package": package,
        "destination": str(destination),
        "path": str(path) if path is not None else "",
        "paths": [str(item) for item in paths[:8]],
        "action": "download",
        **result,
    }


def _vendor_download(
    args: dict[str, Any],
    cancel_event: Any,
    guard: PathGuard,
    report: _ProgressCallback | None = None,
) -> dict[str, Any]:
    # This handler is deliberately an allowlist, not a generic URL downloader.
    import urllib.error
    import urllib.parse
    import urllib.request

    _check_cancel(cancel_event)
    vendor = str(args["vendor"])
    manager = str(args.get("manager", "apt"))
    metadata = _VENDOR_DOWNLOADS.get(vendor, {}).get(manager)
    if metadata is None:
        raise ValueError("vendor source or package format is not approved")
    destination = guard.resolve(str(args["destination"]), allow_root=False)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / metadata["filename"]
    partial = target.with_name(target.name + ".part")
    if target.is_file() and target.stat().st_size > 0:
        size = target.stat().st_size
        if report is not None:
            report(size, size, None)
        return {
            "vendor": vendor,
            "manager": manager,
            "url_host": metadata["host"],
            "resolved_host": metadata["host"],
            "path": str(target),
            "package": metadata["package"],
            "format": metadata["format"],
            "bytes": size,
            "already_downloaded": True,
            "downloaded": False,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
        }
    request = urllib.request.Request(metadata["url"], headers={"User-Agent": "system-agent/1.0"})
    total = 0
    started = time.monotonic()
    class _ApprovedRedirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            host = urllib.parse.urlparse(newurl).hostname
            if host not in {metadata["host"], *metadata.get("redirect_hosts", ())}:
                raise ValueError("official vendor download redirected to an unapproved host")
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(_ApprovedRedirects())
    try:
        with opener.open(request, timeout=60) as response, partial.open("wb") as output:
            final_host = urllib.parse.urlparse(response.geturl()).hostname
            if final_host not in {metadata["host"], *metadata.get("redirect_hosts", ())}:
                raise ValueError("official vendor download resolved to an unapproved host")
            content_length = response.headers.get("Content-Length")
            expected = int(content_length) if content_length and content_length.isdigit() else None
            if expected is not None and expected > 500 * 1024 * 1024:
                raise ValueError("approved installer exceeds the 500 MiB limit")
            while True:
                _check_cancel(cancel_event)
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > 500 * 1024 * 1024:
                    raise ValueError("approved installer exceeds the 500 MiB limit")
                output.write(chunk)
                if report is not None:
                    elapsed = max(0.001, time.monotonic() - started)
                    report(total, expected, total / elapsed)
        if expected is not None and total != expected:
            raise ValueError("official installer download was incomplete")
        partial.replace(target)
    except (urllib.error.URLError, TimeoutError) as exc:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        reason = getattr(exc, "reason", exc)
        timed_out = isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError)
        raise ToolExecutionError(
            SoftwareErrorCode.NETWORK_ERROR.value,
            f"official vendor download failed: {reason}",
            {
                "exit_code": None,
                "stdout": "",
                "stderr": str(reason),
                "timed_out": timed_out,
            },
        ) from exc
    except BaseException:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return {
        "vendor": vendor,
        "manager": manager,
        "url_host": metadata["host"],
        "resolved_host": final_host,
        "path": str(target),
        "package": metadata["package"],
        "format": metadata["format"],
        "bytes": total,
        "already_downloaded": False,
        "downloaded": True,
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
    }


def _vendor_install(args: dict[str, Any], cancel_event: Any, guard: PathGuard) -> dict[str, Any]:
    _check_cancel(cancel_event)
    vendor = str(args["vendor"])
    manager = str(args.get("manager", "apt"))
    metadata = _VENDOR_DOWNLOADS.get(vendor, {}).get(manager)
    if metadata is None:
        raise ValueError("vendor source or package format is not approved")
    path = guard.resolve(str(args["path"]), allow_root=False)
    if path.name != metadata["filename"] or not path.is_file():
        raise ValueError("installer path does not match the approved vendor package")
    if manager == "apt":
        if shutil.which("apt-get") is None:
            raise FileNotFoundError("apt-get is required for the approved Debian installer")
        command = ["apt-get", "install", "-y", str(path)]
    elif manager in {"dnf", "yum"}:
        command = [_manager_command(manager), "install", "-y", str(path)]
    else:
        raise ValueError("approved vendor installation is supported only for apt, dnf, and yum")
    result = _run_argv(_privileged(command), cancel_event, 900, max_output=120_000)
    _ensure_command_success(result)
    return {"vendor": vendor, "manager": manager, "path": str(path), "package": metadata["package"], "action": "install", **result}


def _ensure_command_success(result: dict[str, Any]) -> None:
    if result.get("exit_code") == 0:
        return
    _raise_command_failure(result)


def _raise_command_failure(result: dict[str, Any]) -> None:
    detail = str(result.get("stderr") or result.get("stdout") or "command failed").strip()
    raise ToolExecutionError(_classify_command_error(result), detail[-500:], result)


def _classify_command_error(result: dict[str, Any]) -> str:
    text = " ".join(str(result.get(key, "")) for key in ("stderr", "stdout")).casefold()
    if result.get("timed_out") or any(
        marker in text
        for marker in (
            "could not resolve", "temporary failure resolving", "network is unreachable",
            "failed to fetch", "connection timed out", "connection reset",
            "unable to connect", "network error",
        )
    ):
        return SoftwareErrorCode.NETWORK_ERROR.value
    if any(
        marker in text
        for marker in (
            "unable to locate package", "no installation candidate", "package not found",
            "no match for argument", "target not found", "not found in repository",
            "no package matches", "no matching package", "no matches found",
        )
    ):
        return SoftwareErrorCode.PACKAGE_NOT_FOUND.value
    if any(
        marker in text
        for marker in (
            "permission denied", "not permitted", "must be root", "could not get lock",
            "unable to acquire the dpkg frontend lock", "authentication is required",
        )
    ):
        return SoftwareErrorCode.PERMISSION_ERROR.value
    if any(
        marker in text
        for marker in (
            "unmet dependencies", "dependency problems", "held broken packages",
            "requires:", "conflicts with", "broken dependencies",
        )
    ):
        return SoftwareErrorCode.DEPENDENCY_ERROR.value
    return SoftwareErrorCode.INSTALLATION_ERROR.value


def _verify_download(args: dict[str, Any], cancel_event: Any, guard: PathGuard) -> dict[str, Any]:
    """Verify a downloaded package file without executing the file."""
    _check_cancel(cancel_event)
    path = guard.resolve(str(args["path"]), allow_root=False)
    if not path.is_file():
        return {"path": str(path), "verified": False, "reason": "file does not exist"}
    size = path.stat().st_size
    if size <= 0 or size > 500 * 1024 * 1024:
        return {"path": str(path), "bytes": size, "verified": False, "reason": "file size is outside the approved range"}
    if path.name.endswith(".deb") and shutil.which("dpkg-deb"):
        inspect = _run_argv(["dpkg-deb", "--info", str(path)], cancel_event, 15, max_output=20_000)
    elif path.name.endswith(".rpm") and shutil.which("rpm"):
        inspect = _run_argv(["rpm", "--query", "--package", "--info", str(path)], cancel_event, 15, max_output=20_000)
    else:
        return {
            "path": str(path),
            "bytes": size,
            "format": path.suffix.lstrip("."),
            "verified": False,
            "reason": "only .deb and .rpm package metadata can be verified",
        }
    verified = inspect["exit_code"] == 0
    return {
        "path": str(path),
        "bytes": size,
        "format": path.suffix.lstrip("."),
        "verified": verified,
        "inspection": inspect,
    }


def _verify(args: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
    result = _query(args, cancel_event)
    executable = str(args.get("executable", ""))
    if executable and not re.fullmatch(r"[A-Za-z0-9._+-]{1,80}", executable):
        raise ValueError("invalid executable name")
    result["executable"] = executable
    candidates = _executable_candidates(executable)
    available_paths = {
        candidate: shutil.which(candidate)
        for candidate in candidates
        if shutil.which(candidate)
    }
    result["executable_candidates"] = list(candidates)
    result["executable_available"] = bool(available_paths) if executable else None
    result["executable_path"] = next(iter(available_paths.values()), None)
    result["verified"] = bool(result["installed"]) and (not executable or bool(available_paths))
    if not result["installed"]:
        result["verification_reason"] = "package is not installed"
    elif executable and not available_paths:
        result["verification_reason"] = "package is installed, but no approved launcher was found"
    else:
        result["verification_reason"] = "package and launcher are available"
    return result


def _executable_candidates(executable: str) -> tuple[str, ...]:
    """Return fixed launcher aliases for packages with distro-specific names."""
    aliases = {
        "google-chrome": ("google-chrome", "google-chrome-stable"),
        "code": ("code", "code-insiders"),
    }
    return aliases.get(executable, (executable,)) if executable else ()


def create_software_tool_definitions(roots: Iterable[Path]) -> tuple[ToolDefinition, ...]:
    guard = PathGuard(roots)
    return (
        ToolDefinition("software_system_profile", "Detect Linux distribution, architecture, package managers, and local software sources.", _object_schema({}), _OUTPUT, PermissionLevel.READ_ONLY, 5, _profile, "System profile", safe_software=True),
        ToolDefinition("software_search", "Search one detected package manager or trusted software source for an application or package name.", _object_schema({"manager": _MANAGER, "query": _SEARCH_QUERY}, ["manager", "query"]), _OUTPUT, PermissionLevel.NETWORK, 40, _search, "Search software sources", safe_software=True),
        ToolDefinition("software_query", "Check whether a package is installed and return its package-manager version output.", _object_schema({"manager": _MANAGER, "package": _PACKAGE}, ["manager", "package"]), _OUTPUT, PermissionLevel.READ_ONLY, 15, _query, "Check installed version", safe_software=True),
        ToolDefinition("software_available_version", "Check the newest candidate version from the selected trusted package source.", _object_schema({"manager": _MANAGER, "package": _PACKAGE}, ["manager", "package"]), _OUTPUT, PermissionLevel.NETWORK, 35, _available_version, "Check available version", safe_software=True),
        ToolDefinition("software_install", "Install one allowlisted package through the detected package manager; no arbitrary command is accepted.", _object_schema({"manager": _MANAGER, "package": _PACKAGE}, ["manager", "package"]), _OUTPUT, PermissionLevel.WRITE, 920, _install, "Install software", safe_software=True),
        ToolDefinition("software_reinstall", "Reinstall one allowlisted package through the detected package manager.", _object_schema({"manager": _MANAGER, "package": _PACKAGE}, ["manager", "package"]), _OUTPUT, PermissionLevel.WRITE, 920, _reinstall, "Reinstall software", safe_software=True),
        ToolDefinition("software_remove", "Remove one package through the detected package manager without purging unrelated dependencies.", _object_schema({"manager": _MANAGER, "package": _PACKAGE}, ["manager", "package"]), _OUTPUT, PermissionLevel.DESTRUCTIVE, 920, _remove, "Remove software", safe_software=True),
        ToolDefinition("software_update", "Update one package or all packages through the detected package manager.", _object_schema({"manager": _MANAGER, "package": {"type": "string", "pattern": r"^(all|[A-Za-z0-9][A-Za-z0-9+_.:@/-]{0,127})$"}}, ["manager"]), _OUTPUT, PermissionLevel.WRITE, 920, _update, "Update software", safe_software=True),
        ToolDefinition("software_download", "Download a package into an approved local directory using apt or dnf; does not install it.", _object_schema({"manager": {"type": "string", "enum": ["apt", "dnf"]}, "package": _PACKAGE, "destination": {"type": "string", "minLength": 1, "maxLength": 4_096}}, ["manager", "package", "destination"]), _OUTPUT, PermissionLevel.NETWORK, 920, lambda a, c, p: _download(a, c, guard, p), "Download software", safe_software=True, reports_progress=True, confirmation_required=True),
        ToolDefinition("software_vendor_download", "Download only one fixed official vendor installer from the built-in HTTPS allowlist.", _object_schema({"vendor": {"type": "string", "enum": list(_VENDOR_NAMES)}, "manager": {"type": "string", "enum": ["apt", "dnf"]}, "destination": {"type": "string", "minLength": 1, "maxLength": 4_096}}, ["vendor", "manager", "destination"]), _OUTPUT, PermissionLevel.NETWORK, 920, lambda a, c, p: _vendor_download(a, c, guard, p), "Download official installer", safe_software=True, reports_progress=True, confirmation_required=True),
        ToolDefinition("software_vendor_install", "Install one downloaded allowlisted vendor package through the detected package manager.", _object_schema({"vendor": {"type": "string", "enum": list(_VENDOR_NAMES)}, "manager": {"type": "string", "enum": ["apt", "dnf"]}, "path": {"type": "string", "minLength": 1, "maxLength": 4_096}}, ["vendor", "manager", "path"]), _OUTPUT, PermissionLevel.WRITE, 920, _vendor_install, "Install official software", safe_software=True, confirmation_required=True),
        ToolDefinition("software_verify_download", "Verify that a downloaded allowlisted package file exists and has valid package metadata; never executes it.", _object_schema({"path": {"type": "string", "minLength": 1, "maxLength": 4_096}}, ["path"]), _OUTPUT, PermissionLevel.READ_ONLY, 15, lambda a, c: _verify_download(a, c, guard), "Verify downloaded package", safe_software=True),
        ToolDefinition("software_verify", "Verify package installation and optionally its known executable.", _object_schema({"manager": _MANAGER, "package": _PACKAGE, "executable": {"type": "string", "pattern": r"^[A-Za-z0-9._+-]{1,80}$"}}, ["manager", "package"]), _OUTPUT, PermissionLevel.READ_ONLY, 15, _verify, "Verify installation", safe_software=True),
    )
