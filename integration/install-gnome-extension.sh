#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
extension_dir="$project_root/integration/gnome-shell-extension"
extension_uuid="system-agent@local"
schema="org.gnome.settings-daemon.plugins.media-keys"
agent_path="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/system-agent/"
agent_key_path="$schema.custom-keybinding:$agent_path"
launcher_dir="${XDG_BIN_HOME:-${HOME}/.local/bin}"
launcher_path="$launcher_dir/system-agent-toggle"
schema_file="$extension_dir/schemas/org.gnome.shell.extensions.system-agent.gschema.xml"

if ! command -v gnome-extensions >/dev/null 2>&1; then
  echo "gnome-extensions is required on GNOME Wayland." >&2
  exit 1
fi
if ! command -v gsettings >/dev/null 2>&1; then
  echo "gsettings is required to check shortcut conflicts." >&2
  exit 1
fi
if ! command -v glib-compile-schemas >/dev/null 2>&1; then
  echo "glib-compile-schemas is required to package the GNOME extension." >&2
  exit 1
fi

# The custom keybinding handles activation. The Shell extension is only the
# compositor bridge that positions the GTK window on Wayland. Avoid competing
# with another binding, while allowing our own binding to be updated safely.
current="$(gsettings get "$schema" custom-keybindings)"
if [[ "$current" != "@as []" ]]; then
  for quoted_path in $current; do
    existing_path="$(printf '%s' "$quoted_path" | tr -d "[]'")"
    if [[ -z "$existing_path" || "$existing_path" == "$agent_path" ]]; then
      continue
    fi
    existing_binding="$(gsettings get "$schema.custom-keybinding:$existing_path" binding 2>/dev/null || true)"
    if [[ "$existing_binding" == "'<Super>space'" ]]; then
      existing_name="$(gsettings get "$schema.custom-keybinding:$existing_path" name 2>/dev/null || true)"
      echo "Super+Space is already assigned to ${existing_name:-another command}." >&2
      echo "Disable that binding first; no settings were changed." >&2
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
gsettings set "$agent_key_path" name "System Agent"
# Use bash plus the absolute path so GNOME can launch the project reliably
# after a directory rename and when the path contains spaces.
gsettings set "$agent_key_path" command "/bin/bash '$project_root/integration/launch.sh' --toggle"
gsettings set "$agent_key_path" binding '<Super>space'

bundle_work_dir="$(mktemp -d)"
trap 'rm -rf "$bundle_work_dir"' EXIT
bundle_source_dir="$bundle_work_dir/source"
bundle_out_dir="$bundle_work_dir/out"
mkdir -p "$bundle_source_dir" "$bundle_out_dir"
cp -a "$extension_dir/." "$bundle_source_dir/"
glib-compile-schemas "$bundle_source_dir/schemas"
gnome-extensions pack "$bundle_source_dir" \
  --force \
  --out-dir "$bundle_out_dir" >/dev/null
bundle="$(find "$bundle_out_dir" -maxdepth 1 -type f -name '*.zip' -print -quit)"
if [[ -z "$bundle" ]]; then
  echo "GNOME extension bundle was not created." >&2
  exit 1
fi
gnome-extensions install --force "$bundle"

extensions_disabled="$(gsettings get org.gnome.shell disable-user-extensions 2>/dev/null || printf 'false')"
if [[ "$extensions_disabled" == "true" ]]; then
  echo "Installed the Wayland positioning bridge, but GNOME user extensions are disabled."
  echo "Super+Space is active through the safe GNOME keybinding fallback."
elif gnome-extensions enable "$extension_uuid" 2>/dev/null &&
    gnome-extensions list --enabled | grep -Fxq "$extension_uuid"; then
  echo "Enabled the Wayland positioning bridge in the current GNOME session."
else
  # On GNOME Wayland, a newly installed user extension may not be visible to
  # the running Shell extension manager until the next login. Preserve all
  # existing extensions and register this one for that next session.
  enabled="$(gsettings get org.gnome.shell enabled-extensions)"
  if [[ "$enabled" == "@as []" ]]; then
    updated="['$extension_uuid']"
  elif [[ "$enabled" == *"$extension_uuid"* ]]; then
    updated="$enabled"
  else
    updated="${enabled%]}, '$extension_uuid']"
  fi
  gsettings set org.gnome.shell enabled-extensions "$updated"
  echo "GNOME Shell will load the positioning bridge after you log out and log in again."
fi

echo "Installed System Agent Overlay integration."
echo "Super+Space now toggles the System Agent overlay."
