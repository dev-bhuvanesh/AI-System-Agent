"""Compact multiline prompt input."""

from __future__ import annotations

from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, GLib, Gtk


class PromptInput(Gtk.Box):
    """A native text editor with a compact, chat-style action row."""

    def __init__(self, on_submit, on_cancel=None) -> None:
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.add_css_class("prompt-row")
        self._on_submit = on_submit
        self._on_cancel = on_cancel
        self._processing = False
        self._stopping = False
        self._cancellable = True

        self.plus_button = Gtk.Button(label="+")
        self.plus_button.set_tooltip_text("Add context")
        self.plus_button.add_css_class("icon-button")
        self.plus_button.set_valign(Gtk.Align.CENTER)
        self.append(self.plus_button)

        self.divider = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self.divider.set_margin_top(20)
        self.divider.set_margin_bottom(20)
        self.divider.add_css_class("prompt-divider")
        self.append(self.divider)

        self._editor_overlay = Gtk.Overlay()
        self._editor_overlay.add_css_class("prompt-editor-area")
        self._editor_overlay.set_hexpand(True)
        self._editor_overlay.set_vexpand(True)
        self._editor_overlay.set_valign(Gtk.Align.CENTER)

        self.editor = Gtk.TextView()
        self.editor.set_focusable(True)
        self.editor.set_can_focus(True)
        self.editor.set_editable(True)
        self.editor.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.editor.set_accepts_tab(False)
        self.editor.set_top_margin(2)
        self.editor.set_bottom_margin(2)
        self.editor.set_left_margin(8)
        self.editor.set_right_margin(0)
        self.editor.set_hexpand(True)
        self.editor.set_vexpand(False)
        self.editor.set_cursor_visible(True)
        self.editor.add_css_class("prompt-editor")
        self._editor_overlay.set_child(self.editor)

        self.placeholder = Gtk.Label(label="Ask system agent...", xalign=0)
        self.placeholder.add_css_class("prompt-placeholder")
        self.placeholder.set_can_target(False)
        self.placeholder.set_halign(Gtk.Align.FILL)
        self.placeholder.set_valign(Gtk.Align.CENTER)
        self.placeholder.set_margin_start(10)
        self._editor_overlay.add_overlay(self.placeholder)
        focus_gesture = Gtk.GestureClick()
        focus_gesture.connect("pressed", self._on_editor_pressed)
        self._editor_overlay.add_controller(focus_gesture)
        self.append(self._editor_overlay)

        self.send_button = Gtk.Button()
        self._paper_plane_path = Path(__file__).resolve().parents[1] / "assets" / "paper-plane.svg"
        self._stop_icon_path = Path(__file__).resolve().parents[1] / "assets" / "stop.svg"
        self._set_action_icon(processing=False)
        self.send_button.set_tooltip_text("Send (Enter)")
        self.send_button.add_css_class("send-button")
        self.send_button.set_valign(Gtk.Align.CENTER)
        self.send_button.connect("clicked", self._submit_from_button)
        self.append(self.send_button)

        self.editor.get_buffer().connect("changed", self._on_text_changed)
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.editor.add_controller(key_controller)

    def _on_text_changed(self, _buffer: Gtk.TextBuffer) -> None:
        self.placeholder.set_visible(not bool(self.text.strip()))

    def _on_editor_pressed(
        self,
        _gesture: Gtk.GestureClick,
        _n_press: int,
        _x: float,
        _y: float,
    ) -> None:
        self.editor.grab_focus()

    def _on_key_pressed(
        self,
        _controller: Gtk.EventControllerKey,
        keyval: int,
        _keycode: int,
        state: Gdk.ModifierType,
    ) -> bool:
        if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
            if state & Gdk.ModifierType.SHIFT_MASK:
                return False
            self.submit()
            return True
        return False

    def _submit_from_button(self, _button: Gtk.Button) -> None:
        if self._processing:
            self._request_cancel()
            return
        self.submit()

    def _request_cancel(self) -> None:
        """Invoke the shared cancellation path exactly once per turn."""
        if not self._processing or not self._cancellable or self._stopping or self._on_cancel is None:
            return
        self._stopping = True
        self.send_button.set_sensitive(False)
        self.send_button.set_tooltip_text("Stopping…")
        self._on_cancel()

    @property
    def text(self) -> str:
        buffer = self.editor.get_buffer()
        return buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), False)

    def submit(self) -> None:
        # Enter and the send button are disabled logically while a turn is
        # active. The same button is the Stop control in that state.
        if self._processing:
            return
        text = self.text.strip()
        if text:
            self._on_submit(text)

    @property
    def processing(self) -> bool:
        return self._processing

    def set_processing(self, processing: bool, *, cancellable: bool = True) -> None:
        """Update the send control, showing Stop only for cancellable tasks."""
        self._processing = bool(processing)
        self._stopping = False
        self._cancellable = bool(cancellable) if self._processing else True
        self._set_action_icon(self._processing and self._cancellable)
        self.send_button.set_sensitive(not self._processing or self._cancellable)
        self.send_button.set_tooltip_text(
            "Stop response" if self._processing and self._cancellable
            else "Generating response…" if self._processing
            else "Send (Enter)"
        )

    def mark_stopping(self) -> None:
        """Keep Stop visible while the underlying request is being aborted."""
        if not self._processing or not self._cancellable:
            return
        self._stopping = True
        self.send_button.set_sensitive(False)
        self.send_button.set_tooltip_text("Stopping…")

    def _set_action_icon(self, processing: bool) -> None:
        icon_path = self._stop_icon_path if processing else self._paper_plane_path
        self.send_button.set_child(Gtk.Image.new_from_file(str(icon_path)))
        if processing:
            self.send_button.add_css_class("stop-button")
        else:
            self.send_button.remove_css_class("stop-button")

    def clear(self) -> None:
        self.editor.get_buffer().set_text("")

    def set_text(self, text: str) -> None:
        self.editor.get_buffer().set_text(text)
        self.editor.get_buffer().place_cursor(self.editor.get_buffer().get_end_iter())

    def focus(self) -> None:
        """Focus the editor after the overlay has become the active window."""
        self.editor.set_cursor_visible(True)
        self.editor.grab_focus()

        # A Wayland/GNOME activation is asynchronous.  The immediate focus
        # request can otherwise run before the compositor has activated the
        # window, leaving a visible overlay that does not accept typing.
        GLib.idle_add(self._focus_editor)
        GLib.timeout_add(80, self._focus_editor)
        GLib.timeout_add(240, self._focus_editor)
        GLib.timeout_add(500, self._focus_editor)

    def _focus_editor(self) -> bool:
        self.editor.grab_focus()
        return GLib.SOURCE_REMOVE
