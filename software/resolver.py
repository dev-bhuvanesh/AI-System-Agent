"""Resolve user software names only from trusted package-manager output."""

from __future__ import annotations

import re


_PACKAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9+_.:@/-]{0,127}$")
_ARCHITECTURES = {
    "amd64", "arm64", "aarch64", "armhf", "i386", "i686", "noarch",
    "ppc64le", "riscv64", "s390x", "x86_64",
}


def query_for(display_name: str, package: str = "", explicit_query: str = "") -> str:
    """Return search text, never a guessed package name."""
    return (explicit_query or package or display_name).strip()[:96]


def repository_candidates(manager: str, query: str, output: str) -> tuple[str, ...]:
    """Extract actual package identities reported by one trusted source."""
    tokens = tuple(re.findall(r"[a-z0-9]+", query.casefold()))
    ranked: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for index, line in enumerate((output or "").splitlines()):
        package = _package_from_line(manager, line)
        if not package or package.casefold() in seen:
            continue
        seen.add(package.casefold())
        haystack = re.findall(r"[a-z0-9]+", line.casefold())
        normalized = re.sub(r"[^a-z0-9]", "", package.casefold())
        score = sum(3 for token in tokens if token in haystack)
        compact_query = "".join(tokens)
        if compact_query and compact_query in normalized:
            score += 8
        if tokens and normalized == compact_query:
            score += 20
        # Package-manager output can contain headers and repository notices.
        # Only a query-related returned identity may become a candidate.
        if tokens and score == 0:
            continue
        ranked.append((score, -index, package))
    ranked.sort(reverse=True)
    return tuple(package for _score, _index, package in ranked[:8])


def _package_from_line(manager: str, line: str) -> str:
    text = line.strip()
    if not text:
        return ""
    lowered = text.casefold()
    if lowered.startswith((
        "available packages",
        "installed packages",
        "last metadata expiration check",
        "metadata expiration",
        "name version",
        "application name",
        "results",
    )):
        return ""
    if manager == "apt":
        match = re.match(r"^([^\s]+)\s+-\s+", text)
        return _valid(match.group(1) if match else "")
    if manager in {"dnf", "yum"}:
        package = text.split()[0] if text.split() else ""
        parts = package.rsplit(".", 1)
        if len(parts) == 2 and parts[1].casefold() in _ARCHITECTURES:
            package = parts[0]
        return _valid(package)
    if manager == "pacman":
        package = text.split()[0] if text.split() else ""
        return _valid(package.rsplit("/", 1)[-1])
    if manager == "zypper" and "|" in text:
        columns = [part.strip() for part in text.split("|")]
        return _valid(columns[1] if len(columns) > 1 else "")
    if manager == "snap" and text.split() and text.split()[0].casefold() in {
        "name", "version", "rev", "tracking", "publisher", "notes",
    }:
        return ""
    # Flatpak, Snap, and the conservative fallback use the first field.
    return _valid(text.split()[0] if text.split() else "")


def _valid(package: str) -> str:
    return package if _PACKAGE_PATTERN.fullmatch(package) else ""
