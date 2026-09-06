#!/usr/bin/env python3
"""System Agent desktop overlay entry point.

The process is a resident GTK application so the GNOME shortcut can toggle
the same overlay instance without creating a terminal window.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gi

gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gdk, Gio, Gtk

from agent.controller import AIController
from agent.runtime import AssistantRuntime
from config.config import AgentConfig
from llm import create_chat_provider
from tools.registry import create_default_registry
from software.manager import SoftwareManager
from troubleshooting.engine import TroubleshootingEngine
from ui.overlay import AgentWindow


class SystemAgentApplication(Adw.Application):
    def __init__(self) -> None:
        super().__init__(
            application_id="com.systemagent.Desktop",
            flags=Gio.ApplicationFlags.HANDLES_COMMAND_LINE,
        )
        self.config = AgentConfig.load()
        self.provider = create_chat_provider(self.config)
        self.registry = create_default_registry(self.config)
        self.runtime = AssistantRuntime(self.provider, self.registry)
        self.software_manager = SoftwareManager(self.provider, self.registry)
        self.troubleshooter = TroubleshootingEngine(self.provider, self.registry)
        self.controller = AIController(
            self.provider,
            self.troubleshooter,
            self.software_manager,
            self.runtime,
        )
        self.window: AgentWindow | None = None

    def do_startup(self) -> None:
        Adw.Application.do_startup(self)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        Adw.StyleManager.get_default().set_color_scheme(
            Adw.ColorScheme.FORCE_DARK
        )
        self._install_css()
        # Keep the application resident while the overlay is hidden. A GNOME
        # keybinding can then reach the existing application over D-Bus.
        self.hold()

    def do_activate(self) -> None:
        self._ensure_window().present_overlay()

    def do_command_line(self, command_line: Gio.ApplicationCommandLine) -> int:
        args = command_line.get_arguments()[1:]
        if "--quit" in args:
            self._quit_application()
            return 0
        if "--hide" in args:
            self._ensure_window().hide_overlay()
            return 0
        if "--toggle" in args:
            self._ensure_window().toggle()
            return 0
        self._ensure_window().present_overlay()
        return 0

    def _ensure_window(self) -> AgentWindow:
        if self.window is None:
            self.window = AgentWindow(self, self.config, self.controller)
        return self.window

    def _install_css(self) -> None:
        css_path = PROJECT_ROOT / "ui" / "style.css"
        provider = Gtk.CssProvider()
        provider.load_from_path(str(css_path))
        display = Gdk.Display.get_default()
        if display is not None:
            Gtk.StyleContext.add_provider_for_display(
                display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )

    def _quit_application(self) -> None:
        if self.window is not None:
            self.window._save_preferred_size()
            self.window.destroy()
        self.release()
        self.quit()


def main(argv: list[str] | None = None) -> int:
    app = SystemAgentApplication()
    return app.run(argv if argv is not None else sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
