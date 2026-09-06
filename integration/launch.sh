#!/usr/bin/env bash
set -euo pipefail

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "$script_path")/.." && pwd)"

# GNOME Wayland does not let a normal Wayland client choose its screen
# position. Prefer the native Shell bridge when it is active; otherwise use
# the XWayland compatibility backend so the app can apply the same measured
# top-center geometry without changing GNOME's global extension policy.
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" && "${SYSTEM_AGENT_NATIVE_WAYLAND:-auto}" != "1" ]]; then
    extension_active=false
    if command -v gnome-extensions >/dev/null 2>&1 &&
        gnome-extensions list --enabled 2>/dev/null | grep -Fxq "system-agent@local"; then
        extension_active=true
    fi
    if [[ "$extension_active" != true ]]; then
        export GDK_BACKEND=x11
    fi
fi

python_bin="${SYSTEM_AGENT_PYTHON:-/usr/bin/python3}"
if [[ -x "$project_root/.venv/bin/python" && -z "${SYSTEM_AGENT_PYTHON:-}" ]] &&
    "$project_root/.venv/bin/python" -c 'import gi' >/dev/null 2>&1; then
    python_bin="$project_root/.venv/bin/python"
fi

exec "$python_bin" "$project_root/app/main.py" "$@"
