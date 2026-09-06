#!/usr/bin/env bash
set -euo pipefail

schema="org.gnome.settings-daemon.plugins.media-keys"
agent_path="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/system-agent/"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher_path="${XDG_BIN_HOME:-${HOME}/.local/bin}/system-agent-toggle"

if ! command -v gsettings >/dev/null 2>&1; then
  echo "gsettings is required to remove the GNOME shortcut." >&2
  exit 1
fi

current="$(gsettings get "$schema" custom-keybindings)"
if [[ "$current" == "@as []" || "$current" != *"$agent_path"* ]]; then
  echo "System Agent shortcut is not installed."
  exit 0
fi

updated="$(printf '%s' "$current" | sed "s#'$agent_path', ##; s#, '$agent_path'##; s#'$agent_path'##")"
gsettings set "$schema" custom-keybindings "$updated"
if [[ -L "$launcher_path" && "$(readlink "$launcher_path")" == "$project_root/integration/launch.sh" ]]; then
  rm "$launcher_path"
fi
echo "Removed System Agent from Super+Space."
