#!/usr/bin/env bash
set -euo pipefail

extension_uuid="system-agent@local"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher_path="${XDG_BIN_HOME:-${HOME}/.local/bin}/system-agent-toggle"

if command -v gnome-extensions >/dev/null 2>&1; then
  gnome-extensions disable "$extension_uuid" 2>/dev/null || true
  gnome-extensions uninstall "$extension_uuid" 2>/dev/null || true
fi
if [[ -L "$launcher_path" && "$(readlink "$launcher_path")" == "$project_root/integration/launch.sh" ]]; then
  rm "$launcher_path"
fi
echo "Removed the System Agent GNOME Shell overlay integration."
