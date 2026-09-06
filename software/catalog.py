"""Small allowlist of recognizable software and trusted source metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass

from software.contracts import SoftwareOperation, SoftwareRequest


_GENERIC_STOP_WORDS = {
    "software",
    "package",
    "application",
    "app",
    "program",
    "file",
    "document",
    "image",
    "video",
    "something",
    "anything",
    "it",
    "this",
    "that",
    "system",
    "computer",
    "laptop",
    "desktop",
    "linux",
    "pc",
    # These are Linux system/device concepts, not generic package names.
    # Keeping them out of the fallback parser lets requests such as
    # “troubleshoot Wi-Fi” reach the troubleshooting engine.
    "wifi",
    "wi-fi",
    "whfi",
    "ethernet",
    "network",
    "internet",
    "bluetooth",
    "audio",
    "sound",
    "display",
    "screen",
    "monitor",
    "gpu",
    "graphics",
    "usb",
}


@dataclass(frozen=True, slots=True)
class SoftwareSpec:
    key: str
    display_name: str
    aliases: tuple[str, ...]
    packages: dict[str, str]
    flatpak: str = ""
    snap: str = ""
    vendor: str = ""
    executable: str = ""
    # Generic entries retain only the user's search text until a trusted
    # package manager reports an actual package identity.
    search_query: str = ""


CATALOG: tuple[SoftwareSpec, ...] = (
    SoftwareSpec(
        "google_chrome", "Google Chrome", ("google chrome", "google chrom", "chrome"),
        {"apt": "google-chrome-stable", "dnf": "google-chrome-stable"},
        vendor="google_chrome", executable="google-chrome",
    ),
    SoftwareSpec(
        "vlc", "VLC Media Player", ("vlc", "vlc media player"),
        {"apt": "vlc", "dnf": "vlc", "yum": "vlc", "pacman": "vlc", "zypper": "vlc"},
        flatpak="org.videolan.VLC", snap="vlc", executable="vlc",
    ),
    SoftwareSpec(
        "visual_studio_code", "Visual Studio Code", ("visual studio code", "vs code", "vscode", "code"),
        {"apt": "code", "dnf": "code", "yum": "code", "pacman": "code"},
        flatpak="com.visualstudio.code", snap="code", vendor="visual_studio_code", executable="code",
    ),
    SoftwareSpec(
        "python", "Python", ("python", "python 3", "python3"),
        {"apt": "python3", "dnf": "python3", "yum": "python3", "pacman": "python", "zypper": "python3"},
        executable="python3",
    ),
    SoftwareSpec(
        "firefox", "Mozilla Firefox", ("firefox",),
        {"apt": "firefox", "dnf": "firefox", "yum": "firefox", "pacman": "firefox", "zypper": "MozillaFirefox"},
        flatpak="org.mozilla.firefox", snap="firefox", executable="firefox",
    ),
    SoftwareSpec(
        "docker", "Docker", ("docker", "docker desktop"),
        {"apt": "docker.io", "dnf": "docker", "yum": "docker", "pacman": "docker", "zypper": "docker"},
        executable="docker",
    ),
    SoftwareSpec(
        "git", "Git", ("git",),
        {"apt": "git", "dnf": "git", "yum": "git", "pacman": "git", "zypper": "git"},
        executable="git",
    ),
    SoftwareSpec(
        "steam", "Steam", ("steam", "steam client"),
        {
            "apt": "steam-installer",
            "dnf": "steam",
            "yum": "steam",
            "pacman": "steam",
            "zypper": "steam",
        },
        flatpak="com.valvesoftware.Steam",
        snap="steam",
        executable="steam",
    ),
)


_OPERATION_PATTERNS: tuple[tuple[SoftwareOperation, tuple[str, ...]], ...] = (
    (SoftwareOperation.REINSTALL, (r"\bre[- ]?install\b",)),
    (SoftwareOperation.REMOVE, (r"\b(remove|uninstall|un-install|delete)\b",)),
    (SoftwareOperation.UPGRADE, (r"\bupgrade\b",)),
    (SoftwareOperation.UPDATE, (r"\b(update|updates|upgrade all)\b",)),
    (SoftwareOperation.SEARCH, (r"\b(search|find|look for)\b",)),
    (SoftwareOperation.INSTALLED_VERSION, (r"\b(installed version|is installed|currently installed)\b",)),
    (SoftwareOperation.AVAILABLE_VERSION, (r"\b(available version|latest version|newest version)\b",)),
    (SoftwareOperation.VERIFY, (r"\b(verify|verification|check installation|troubleshoot|troubleshooting|diagnose|diagnostic|check)\b",)),
    (SoftwareOperation.DOWNLOAD, (r"\b(download|get the installer)\b",)),
    (SoftwareOperation.INSTALL, (r"\b(install|setup|set up|need|want|get)\b",)),
)


def find_spec(name: str) -> SoftwareSpec | None:
    normalized = " ".join(name.casefold().split())
    matches = [
        spec for spec in CATALOG
        if any(alias in normalized for alias in spec.aliases)
    ]
    return max(matches, key=lambda item: max(len(alias) for alias in item.aliases), default=None)


def parse_request(request: str) -> tuple[SoftwareRequest, SoftwareSpec] | None:
    normalized = " ".join(request.casefold().split())
    operation = next(
        (candidate for candidate, patterns in _OPERATION_PATTERNS
         if any(re.search(pattern, normalized) for pattern in patterns)),
        None,
    )
    if operation is None:
        return None
    spec = find_spec(normalized)
    if operation in {SoftwareOperation.UPDATE, SoftwareOperation.UPGRADE} and re.search(r"\b(all|everything|my packages|my software)\b", normalized):
        return SoftwareRequest(operation, "all software", request.strip(), True), spec or SoftwareSpec("all", "all software", (), {})
    if spec is None:
        spec = _generic_package_spec(normalized, operation)
        if spec is None:
            return None
    return SoftwareRequest(operation, spec.display_name, request.strip()), spec


def _generic_package_spec(normalized: str, operation: SoftwareOperation) -> SoftwareSpec | None:
    """Build an unresolved repository query without inventing a package ID."""
    if operation not in {
        SoftwareOperation.INSTALL,
        SoftwareOperation.DOWNLOAD,
        SoftwareOperation.UPDATE,
        SoftwareOperation.UPGRADE,
        SoftwareOperation.REMOVE,
        SoftwareOperation.REINSTALL,
        SoftwareOperation.SEARCH,
        SoftwareOperation.INSTALLED_VERSION,
        SoftwareOperation.AVAILABLE_VERSION,
        SoftwareOperation.VERIFY,
    }:
        return None
    patterns = {
        SoftwareOperation.INSTALL: (
            r"\b(?:install|setup|set up)\b\s+(.+)$",
            r"\b(?:need|want|get)\b\s+(?:(?:to|the|a|an)\s+)?(.+)$",
        ),
        SoftwareOperation.DOWNLOAD: (r"\b(?:download|get the installer)\b\s+(.+)$",),
        SoftwareOperation.UPDATE: (r"\b(?:update|upgrade)\b\s+(.+)$",),
        SoftwareOperation.REMOVE: (r"\b(?:remove|uninstall|un-install|delete)\b\s+(.+)$",),
        SoftwareOperation.REINSTALL: (r"\bre[- ]?install\b\s+(.+)$",),
        SoftwareOperation.SEARCH: (r"\b(?:search|find|look for)\b\s+(.+)$",),
        SoftwareOperation.INSTALLED_VERSION: (r"\b(?:installed version|is installed|currently installed)\b\s*(.*)$",),
        SoftwareOperation.AVAILABLE_VERSION: (r"\b(?:available version|latest version|newest version)\b\s*(.*)$",),
        SoftwareOperation.VERIFY: (
            r"\b(?:verify|verification|check installation)\b\s+(.+)$",
            r"\b(?:troubleshoot|troubleshooting|diagnose|diagnostic|check)\b\s+(?:(?:the|my|this)\s+)?(.+)$",
        ),
    }.get(operation, ())
    match = None
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match is not None:
            break
    if match is None:
        return None
    name = match.group(1).strip(" .,:;!?\"'")
    for _ in range(2):
        name = re.sub(r"^(?:the|a|an)\s+", "", name)
        name = re.sub(r"^(?:software|package|application|app)\s+", "", name)
    # Keep natural requests such as “install nonexistent software” useful by
    # treating the trailing noun as a descriptor, not as the package name.
    name = re.sub(r"\s+(?:software|package|application|app)$", "", name)
    name = re.sub(r"\s+(?:please|for me|on my (?:system|computer|linux))$", "", name)
    if not name or len(name) > 96:
        return None
    words = name.split()
    if len(words) > 3 or any(word in _GENERIC_STOP_WORDS for word in words):
        return None
    display_name = " ".join(word.capitalize() for word in words)
    return SoftwareSpec(
        key=f"search:{name.casefold()}",
        display_name=display_name,
        aliases=(name,),
        packages={},
        search_query=name,
    )
