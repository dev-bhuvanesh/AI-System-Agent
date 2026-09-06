#!/usr/bin/env bash
set -euo pipefail

# GNOME's settings-daemon custom keybindings are Wayland-compatible and keep
# the activation layer independent from the UI/AI process.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
schema="org.gnome.settings-daemon.plugins.media-keys"
base_path="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings"
agent_path="$base_path/system-agent/"
key_path="$schema.custom-keybinding:$agent_path"
launcher_dir="${XDG_BIN_HOME:-${HOME}/.local/bin}"
launcher_path="$launcher_dir/system-agent-toggle"

if ! command -v gsettings >/dev/null 2>&1; then
  echo "gsettings is required to install the GNOME shortcut." >&2
  exit 1
fi

current="$(gsettings get "$schema" custom-keybindings)"

# Do not silently shadow a shortcut the user already configured. The current
# host, for example, may have a different app on Super+Space.
if [[ "$current" != "@as []" ]]; then
  for quoted_path in $current; do
    existing_path="$(printf '%s' "$quoted_path" | tr -d "[]'")"
    if [[ -z "$existing_path" || "$existing_path" == "$agent_path" ]]; then
      continue
    fi
    existing_binding="$(gsettings get "$schema.custom-keybinding:$existing_path" binding 2>/dev/null || true)"
    if [[ "$existing_binding" == "'<Super>space'" ]]; then
      existing_name="$(gsettings get "$schema.custom-keybinding:$existing_path" name 2>/dev/null || true)"
      existing_command="$(gsettings get "$schema.custom-keybinding:$existing_path" command 2>/dev/null || true)"
      echo "Super+Space is already assigned to ${existing_name:-another command}." >&2
      echo "Command: ${existing_command:-unknown}" >&2
      echo "No GNOME settings were changed; disable that binding first, then rerun this installer." >&2
      exit 2
    fi
  done
fi

if [[ -e "$launcher_path" && ! -L "$launcher_path" ]]; then
  echo "Cannot install launcher: $launcher_path already exists and is not a symlink." >&2
  exit 1
fi
mkdir -p "$launcher_dir"
ln -sfn "$project_root/integration/launch.sh" "$launcher_path"

if [[ "$current" == *"$agent_path"* ]]; then
  updated="$current"
elif [[ "$current" == "@as []" ]]; then
  updated="['$agent_path']"
else
  updated="${current%]}, '$agent_path']"
fi
gsettings set "$schema" custom-keybindings "$updated"

gsettings set "$key_path" name "System Agent"
# Launch through bash with the absolute project path. This keeps the binding
# working when the project directory contains spaces and avoids GNOME's media
# key service rejecting a renamed or symlinked launcher as not being in PATH.
gsettings set "$key_path" command "/bin/bash '$project_root/integration/launch.sh' --toggle"
gsettings set "$key_path" binding '<Super>space'

echo "Installed System Agent on Super+Space."
echo "Project: $project_root"
