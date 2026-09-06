# System Agent

## Frozen product definition

This section is the product source of truth. It is frozen: future changes must
support this definition directly, and any change to product scope, the core
interaction model, safety model, or UI philosophy requires an explicit product
review first.

System Agent is a privacy-first, local AI Linux desktop assistant. It uses a
locally running Qwen 8B model to understand natural-language requests,
diagnose Linux problems, execute approved operations through controlled tools,
inspect results, recover from failures, and verify completion. It is a native
background desktop application, not a website: `Super+Space` opens a compact
floating input pill which expands smoothly into the lightweight minimalist
chat interface.

The UI keeps the current dark theme, rounded container, subtle blue glow,
streaming responses, and only the functional `Stop` and `Copy` response
actions. It must remain responsive and avoid unnecessary controls, frameworks,
cloud AI dependencies, unrelated features, and autonomous background actions.

The agent architecture is modular and keeps the desktop UI, Agent Core, Qwen
LLM interface, Tool Router, Security/Policy layer, Linux tools, diagnostics
engine, and session state separate. Qwen never receives unrestricted operating
system access. Every requested action passes through structured tool routing,
validation, security policy, permission checks, and the appropriate approval
step. Read-only diagnostics normally run without approval; ordinary user-level
changes follow policy; package installation, driver changes, and service
changes require confirmation when appropriate; destructive operations such as
important-data deletion, disk operations, shutdown, and reboot always require
explicit confirmation.

All agent tasks follow:

```text
observe → diagnose → plan → request permission → execute →
inspect result → verify → report
```

The agent must never claim success without checking the actual command result
or resulting system state. Supported system-agent areas include system
inspection, controlled terminal diagnostics, package/application management,
filesystem and process operations, systemd services, networking, Bluetooth,
audio, GPU and hardware diagnostics, configuration files, and troubleshooting.

The MVP is limited to the desktop popup, local Qwen 8B integration, controlled
terminal execution, system information, package installation, one complete
troubleshooting workflow, Stop/Copy controls, permission confirmation, result
verification, and `Super+Space` activation. Proposed work that does not
directly strengthen a safe, fast, reliable, local Linux System Agent is out of
scope.

The current active build is a native GTK4/Libadwaita desktop overlay for
GNOME Wayland. It keeps the compact top-center input and the existing expanded
chat surface unchanged, while using a local Ollama server for conversation.

Press `Super+Space` to toggle the overlay, type a request, and press Enter or
the paper-plane button. `Shift+Enter` inserts a newline. Escape hides the
overlay and cancels an in-flight response.

## Local Ollama backend

The active flow is:

```text
chat UI → AIController → OllamaProvider → localhost:11434 → qwen2.5:7b
       → streamed response → AIController → chat UI
```

No cloud API key or cloud provider is used. `OllamaProvider` uses the local
`/api/tags` endpoint to verify the model and `/api/chat` with NDJSON streaming.
Conversation history is retained for the current resident application session.

The configured defaults are:

```text
URL:   http://localhost:11434
Model: qwen2.5:7b
```

Start Ollama and install the model locally if needed:

```bash
ollama serve
ollama pull qwen2.5:7b
```

The application reads optional settings from
`~/.config/system-agent/config.toml`. See
[config/config.example.toml](config/config.example.toml) for the local URL,
model, stream mode, timeout, keep-alive, context, temperature, sampling,
thread, batch, GPU-layer, and maximum-token settings. With
`hardware_auto_tune = true` (the default), startup discovery records the CPU
model, physical/logical cores, RAM, GPU vendor/model/VRAM, and detected
CUDA/ROCm/Vulkan capability. Each request refreshes available RAM, CPU load,
and GPU memory pressure, then chooses bounded context/KV retention, CPU
threads, batch size, GPU offload, and low-VRAM behavior. High-pressure
machines automatically use a smaller context, batch, thread count, and
CPU-only execution when needed. The configured `keep_alive` value keeps the
model warm between normal requests; under current memory/VRAM pressure the
request asks Ollama to unload it after generation.

Quantization is part of an Ollama model tag, so the application never edits
weights or downloads a model automatically. It records a low/mid/high-device
recommendation (Q3/Q4/Q5) and prefers the exact configured `qwen2.5:7b` tag;
if that exact tag is absent, it can use only a matching quantized Qwen variant
that is already installed. Otherwise it reports the missing model normally.

The Qwen agent prompt is specialized for Linux intent, approved-tool planning,
validated observations, failure interpretation, and verification. Planner
output is parsed into the typed plan contract; only the trusted controller
can submit a request to the Tool Registry. Registry results are bounded before
they return to Qwen, and recent conversation turns are retained within the
effective context window. The model never receives a callable, subprocess
handle, or permission grant.

If Ollama is stopped, the UI reports:

```text
Unable to connect to local AI. Please start Ollama.
```

If the model is missing, it reports the model name and the local `ollama pull`
command needed to install it. The frontend never executes that command.

## Troubleshooting

Problem reports such as “my internet is not working”, “Wi-Fi is disconnected”,
“my system is slow”, “there is no sound”, or “Chrome is not opening” enter the
separate troubleshooting engine. It detects the category, runs a fixed set of
read-only checks through the Tool Registry, streams each stage and structured
result into the existing response view, and asks Qwen to explain the findings.
Generic requests such as “troubleshoot my computer” and “run a system check”
also enter this diagnostic path instead of ordinary chat.

The engine does not accept shell commands from the model. Any proposed service
restart is shown with its exact command and effect first. For a software/configuration
fault, the response offers **Automatically Troubleshoot & Fix** and **Manual Fix**;
the automatic path then shows a separate **Allow** confirmation before any
modifying action. A healthy result reports that everything is normal. If checks
indicate a loose cable/wire, missing physical link, or hardware problem, the
agent does not offer an automatic system change and instead gives physical
inspection instructions plus a check-again option. The default policy still
blocks modifying operations unless the corresponding local policy is explicitly
enabled. After an approved fix, the relevant checks run again and the result is
recorded in `~/.local/share/system-agent/troubleshooting/history.jsonl`.

## Software management

Requests such as “Install VLC”, “Download Steam”, “I need Google Chrome”, “Download VS Code”,
“Update Firefox”, “Remove Docker”, or “update all my software” use the local
software manager. Curated applications use their known package, Snap, Flatpak,
or official-vendor identity. Other explicit package names (for example, `htop`)
are resolved only through the detected trusted Linux package repositories; the
agent does not guess a vendor URL, installer, or shell command for an unknown app.
It detects the distribution, architecture, available package managers, trusted
source, installed state, and available version before
presenting a plan in the existing chat UI. If software is already installed,
an install/download request stops without repeating the operation and shows
only state-appropriate actions such as Update, Delete, and Reinstall & repair. If an
update is unavailable, no update command is prepared. Read-only checks run
automatically; downloads, installs, updates, removals, and reinstalls always
wait for the trusted action button. The approval card shows the fixed command,
risk, and impact before execution. All package operations use fixed
allowlisted registry tools and `shell=False`; arbitrary model commands and URLs
are rejected.

After approval the process card reports permission received, terminal
execution, the selected operation, real download progress when available,
exit status, execution time, and verification. Captured command output stays
structured in the registry result/history rather than creating a separate raw
output panel in the compact chat UI.

Package actions follow a bounded
`Understand → Discover → Plan → Confirm → Execute → Analyze Result → Recover → Retry → Verify → Report`
flow. Repository searches return only package identities printed by the
detected trusted source; an application name is never converted into a
hyphenated package guess. Failed commands preserve exit code, stdout, stderr,
and a stable category (`PACKAGE_NOT_FOUND`, `NETWORK_ERROR`,
`PERMISSION_ERROR`, `DEPENDENCY_ERROR`, or `INSTALLATION_ERROR`). Recovery can
select another trusted source or retry a transient network failure, and each
retry requires a new user approval.

Software history is stored at
`~/.local/share/system-agent/software/history.jsonl`. Approved vendor
downloads are limited to the built-in HTTPS allowlist for Google Chrome and
Visual Studio Code and are saved under
`~/.local/share/system-agent/downloads/`. A download-only request never
installs the package; after approval it reports the exact saved path and runs
package metadata verification. Vendor packages use the detected Debian/Ubuntu
`.deb` or Fedora-compatible `.rpm` source when available. Long-running vendor
downloads emit bounded bytes/percentage/speed progress to the live process
stage, and cancellation removes the partial file.

## Run

From the project root:

```bash
./integration/launch.sh
```

To toggle an existing resident instance:

```bash
./integration/launch.sh --toggle
```

Install the GNOME shortcut and Wayland positioning bridge with:

```bash
./integration/install-gnome-extension.sh
```

Remove it with:

```bash
./integration/uninstall-gnome-extension.sh
```

The normal chat path still does not execute Linux commands from model output.
Ordinary requests enter the Agent Engine; troubleshooting and software
requests keep their specialized controllers, and every operating-system
operation is routed through the validated Tool Registry.

## Architecture

- [agent/controller.py](agent/controller.py) validates non-empty messages,
  owns conversation history, serializes requests, routes ordinary requests
  through the Agent Engine, and stores completed local assistant responses.
- [agent/runtime.py](agent/runtime.py) owns task/session state and the
  observe → plan → approval → execution → result-analysis → verification
  lifecycle. It accepts only data-only model tool requests, deduplicates them,
  bounds the loop, and exposes trusted approval/cancellation entry points.
- [tools/registry.py](tools/registry.py) is the only OS execution gateway.
  It validates strict schemas, enforces permission policy, limits concurrent
  work, applies timeouts, captures structured results, and streams lifecycle
  events. The model's `requires_confirmation` hint is never trusted.
- [tools/contracts.py](tools/contracts.py) defines the backend-neutral tool,
  result, event, and trusted UI approval contracts.
- [llm/ollama_provider.py](llm/ollama_provider.py) performs loopback-only
  Ollama requests, model detection/metadata logging, hardware-aware inference
  tuning, structured-plan parsing, streaming, and error mapping.
- [llm/hardware.py](llm/hardware.py) performs a read-only CPU/RAM/VRAM probe
  used to select safe local inference settings.
- [llm/prompts.py](llm/prompts.py) contains the concise Linux-specialist chat,
  planner, and verified-result prompts.
- [llm/provider.py](llm/provider.py) contains backend-neutral chat/event types.
- [ui/overlay.py](ui/overlay.py) keeps the existing input, animation, focus,
  cancellation, and response layout.

## Development checks

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q app agent config llm tools ui tests
bash -n integration/*.sh
```
