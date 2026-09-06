"""Native floating System Agent overlay."""

from __future__ import annotations

import ctypes
import re
import threading

import gi

gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
gi.require_version("GdkX11", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gdk, GLib, Gtk

from config.config import AgentConfig
from agent.classifier import RequestType, classify_request
from agent.controller import AIController
from llm.provider import ProviderEvent
from software.contracts import SoftwareOperation, SoftwareState
from ui.quick_input import PromptInput
from ui.response_view import ResponseView


class AgentWindow(Adw.ApplicationWindow):
    """Compact quick-input window with an asynchronous local-LLM response."""

    # Matches the reference images: the GNOME panel occupies roughly the top
    # 40px and the capsule begins another ~40px below it.
    _X11_TOP_OFFSET = 40

    def __init__(
        self,
        application: Adw.Application,
        config: AgentConfig,
        controller: AIController,
    ) -> None:
        super().__init__(application=application)
        self.config = config
        self.controller = controller
        self._generation_active = False
        self.add_css_class("agent-window")
        self.set_title("System Agent")
        self.set_decorated(False)
        self.set_resizable(True)
        self.set_default_size(*self._quick_size())
        self.set_size_request(config.min_width, config.min_height)
        self.set_hide_on_close(True)

        self._fade_source: int | None = None
        self._save_source: int | None = None
        self._hide_source: int | None = None
        self._fallback_position_source: int | None = None
        self._resize_source: int | None = None
        self._generation_id = 0
        self._cancel_event: threading.Event | None = None
        self._generation_thread: threading.Thread | None = None
        self._cancelling_generation_id: int | None = None
        self._retry_request: str | None = None
        self._mode = "quick"

        self._shell = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._shell.add_css_class("overlay-shell")
        self._shell.add_css_class("overlay-quick")
        if self._monitor_width() < 1450:
            self._shell.add_css_class("overlay-small")
        self.set_content(self._shell)

        self.prompt = PromptInput(self._on_submit, self._cancel_from_response)
        self.prompt.set_vexpand(True)
        self._shell.append(self.prompt)

        self.response = ResponseView(
            self._edit_message,
            self._on_fix_decision,
            self._on_software_decision,
            self._on_software_action,
            self._on_fix_choice,
            self._on_tool_decision,
        )
        self.response_revealer = Gtk.Revealer()
        self.response_revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
        self.response_revealer.set_transition_duration(220)
        # Do not let the hidden response tree participate in compact-mode
        # measurement. Long labels and conversation history otherwise become
        # the minimum width of the idle launcher.
        self.response_revealer.set_child(None)
        self.response_revealer.set_reveal_child(False)
        self.response_revealer.set_vexpand(True)
        response_width, response_height = self._response_size()
        self.response.set_viewport_limits(
            max(1, response_width - 2),
            max(1, response_height - 60),
        )
        self._shell.append(self.response_revealer)

        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)
        self.connect("close-request", self._on_close_request)
        self.connect("notify::width", self._on_dimension_changed)
        self.connect("notify::height", self._on_dimension_changed)

    def _on_close_request(self, _window: Gtk.Window) -> bool:
        self.hide_overlay()
        return True

    def _on_key_pressed(
        self,
        _controller: Gtk.EventControllerKey,
        keyval: int,
        _keycode: int,
        _state: Gdk.ModifierType,
    ) -> bool:
        if keyval == Gdk.KEY_Escape:
            self.hide_overlay()
            return True
        return False

    def toggle(self) -> None:
        if self.is_visible():
            self.hide_overlay()
        else:
            self.present_overlay()

    def present_overlay(self) -> None:
        if self._hide_source is not None:
            GLib.source_remove(self._hide_source)
            self._hide_source = None
        if self._mode == "quick":
            self.set_default_size(*self._quick_size())
        self.set_opacity(0.0)
        self.present()
        self.set_focus(self.prompt.editor)
        self.prompt.focus()
        # GNOME's global-keybinding activation and the Wayland compositor can
        # finish presenting the window after the first focus request. Repeat
        # the window-level focus request once the surface is mapped.
        GLib.idle_add(self._focus_prompt)
        GLib.timeout_add(120, self._focus_prompt)
        GLib.timeout_add(420, self._focus_prompt)
        self._queue_fallback_position()
        self._start_fade_in()

    def _focus_prompt(self) -> bool:
        if self.is_visible():
            self.set_focus(self.prompt.editor)
            self.prompt.editor.grab_focus()
            # The launcher uses an XWayland fallback when the GNOME Shell
            # positioning bridge is unavailable. In that mode GTK focus can
            # succeed before the compositor activates the window, so repeat
            # the native focus request along with the GTK request.
            self._queue_fallback_position()
        return GLib.SOURCE_REMOVE

    def hide_overlay(self) -> None:
        was_generating = self._generation_active
        self._cancel_generation()
        self.prompt.set_processing(False)
        # A response can be temporarily measured before its fixed viewport is
        # laid out. Do not remember that automatic/content-driven measurement
        # as the user's preferred size when closing an active request.
        if not (was_generating and self._mode == "response"):
            self._save_preferred_size()
        if self._mode == "response":
            self._mode = "quick"
            self.prompt.set_vexpand(True)
            self.response_revealer.set_reveal_child(False)
            self.response_revealer.set_child(None)
            self._shell.remove_css_class("overlay-expanded")
            self._shell.add_css_class("overlay-quick")
            self._animate_resize(*self._quick_size())
        self._finish_hide()

    def _finish_hide(self) -> bool:
        self._hide_source = None
        self.hide()
        return GLib.SOURCE_REMOVE

    def _start_fade_in(self) -> None:
        if self._fade_source is not None:
            GLib.source_remove(self._fade_source)
        frame = 0

        def fade_frame() -> bool:
            nonlocal frame
            frame += 1
            self.set_opacity(min(1.0, frame / 10))
            if frame >= 10:
                self._fade_source = None
                return GLib.SOURCE_REMOVE
            return GLib.SOURCE_CONTINUE

        self._fade_source = GLib.timeout_add(16, fade_frame)

    def _on_dimension_changed(self, *_args) -> None:
        self._clamp_window_size()
        self._queue_fallback_position()

    def _queue_fallback_position(self) -> None:
        """Position XWayland fallback windows without blocking GTK."""
        if self._fallback_position_source is not None:
            GLib.source_remove(self._fallback_position_source)
        self._fallback_position_source = GLib.timeout_add(30, self._position_x11)

    def _position_x11(self) -> bool:
        self._fallback_position_source = None
        surface = self.get_surface()
        if surface is None or not hasattr(surface, "get_xid"):
            return GLib.SOURCE_REMOVE

        try:
            xdisplay = surface.get_display().get_xdisplay()
            xid = int(surface.get_xid())
            display = Gdk.Display.get_default()
            monitors = display.get_monitors() if display is not None else None
            monitor = monitors.get_item(0) if monitors is not None else None
            if not xdisplay or not xid or monitor is None:
                return GLib.SOURCE_REMOVE
            geometry = monitor.get_geometry()
            width = self.get_width()
            if width <= 0:
                return GLib.SOURCE_REMOVE
            x = round(geometry.x + (geometry.width - width) / 2)
            y = geometry.y + self._X11_TOP_OFFSET

            xlib = ctypes.CDLL("libX11.so.6")
            # PyGObject exposes Xlib's Display as a boxed object without an
            # integer conversion. Its stable repr contains the underlying
            # pointer; use that only for this XWayland compatibility path.
            pointer_match = re.search(r"\(void at (0x[0-9a-fA-F]+)\)", repr(xdisplay))
            if pointer_match is None:
                return GLib.SOURCE_REMOVE
            display_ptr = ctypes.c_void_p(int(pointer_match.group(1), 16))
            window = ctypes.c_ulong(xid)
            xlib.XMoveWindow.argtypes = [
                ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int
            ]
            xlib.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            xlib.XSetInputFocus.argtypes = [
                ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong
            ]
            xlib.XFlush.argtypes = [ctypes.c_void_p]
            xlib.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
            xlib.XInternAtom.restype = ctypes.c_ulong
            xlib.XChangeProperty.argtypes = [
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
            ]

            xlib.XMoveWindow(display_ptr, window, x, y)
            net_wm_state = xlib.XInternAtom(display_ptr, b"_NET_WM_STATE", 0)
            net_wm_state_above = xlib.XInternAtom(display_ptr, b"_NET_WM_STATE_ABOVE", 0)
            xa_atom = xlib.XInternAtom(display_ptr, b"ATOM", 0)
            above = ctypes.c_ulong(net_wm_state_above)
            xlib.XChangeProperty(
                display_ptr,
                window,
                net_wm_state,
                xa_atom,
                32,
                0,
                ctypes.byref(above),
                1,
            )
            xlib.XRaiseWindow(display_ptr, window)
            # Reclaim keyboard focus for the prompt after the Shell shortcut
            # hands control back to the application. RevertToParent=2 and
            # CurrentTime=0 are the standard X11 values for this request.
            xlib.XSetInputFocus(display_ptr, window, 2, 0)
            xlib.XFlush(display_ptr)
        except (AttributeError, OSError, TypeError, ValueError, OverflowError):
            # Native Wayland or a compositor without XWayland needs the
            # GNOME Shell bridge; failure here must never affect the UI loop.
            pass
        return GLib.SOURCE_REMOVE

    def _on_submit(self, request: str) -> None:
        if (
            self._generation_active
            or self.prompt.processing
            or self._cancelling_generation_id is not None
        ):
            return
        entering_response_mode = self._mode != "response"
        classification = classify_request(request)
        is_system_task = classification.type is RequestType.SYSTEM_TASK
        self._cancel_generation()
        self._generation_active = True
        # Conversation turns remain ordinary chat: no task Stop control,
        # process log, plan card, or task loading row is created.
        self.prompt.set_processing(True, cancellable=is_system_task)
        self.prompt.clear()
        self._mode = "response"
        self.prompt.set_vexpand(False)
        self._shell.remove_css_class("overlay-quick")
        self._shell.add_css_class("overlay-expanded")
        # Keep earlier turns visible while the controller keeps the matching
        # conversation history for Ollama.  The first turn clears the empty
        # response surface; later turns are appended in order.
        self.response.start_turn(request, system_task=is_system_task)
        # Troubleshooting gets a visible process card immediately. The worker
        # then updates this same card with the real diagnostic stages instead
        # of leaving a blank gap while the first event is being scheduled.
        if is_system_task and (
            self.controller.troubleshooter is not None
            and self.controller.troubleshooter.matches(request)
        ):
            self.response.show_initial_process()
        elif is_system_task and (
            self.controller.software_manager is not None
            and self.controller.software_manager.matches(request)
        ):
            self.response.show_initial_process(
                "Understanding request",
                "Identifying the requested software operation",
            )
        if self.response_revealer.get_child() is None:
            self.response_revealer.set_child(self.response)
        self.response_revealer.set_reveal_child(True)
        # Expand only when switching from the compact launcher to the chat
        # view.  Subsequent turns must use the current user-selected geometry;
        # resetting it on every submit looks like the chatbot is growing by
        # itself and also defeats manual resizing.
        if entering_response_mode:
            self._animate_resize(*self._response_size())
        self.prompt.focus()
        self._start_generation(request)

    def _edit_message(self, request: str) -> None:
        """Put a previous user message back into the prompt for editing."""
        if self._generation_active:
            self._cancel_from_response()
        self.prompt.set_text(request)
        self.prompt.focus()

    def _on_fix_decision(self, proposal_id: str, approved: bool) -> None:
        if not self.controller.approve_troubleshooting_fix(proposal_id, approved):
            self.response.show_error("This troubleshooting decision is no longer active.")

    def _on_fix_choice(self, proposal_id: str, action: str, original_request: str = "") -> None:
        if not self.controller.choose_troubleshooting_action(proposal_id, action):
            self.response.show_error("This troubleshooting action is no longer active.")
            return
        if action == "check_again" and original_request:
            self._retry_request = original_request

    def _on_software_decision(self, plan_id: str, approved: bool) -> None:
        if not self.controller.approve_software_action(plan_id, approved):
            self.response.show_error("This software plan is no longer active.")

    def _on_tool_decision(self, approval_id: str, approved: bool) -> None:
        if not self.controller.approve_tool(approval_id, approved):
            self.response.show_error("This system action is no longer active.")

    def _on_software_action(self, state: SoftwareState, operation: SoftwareOperation) -> None:
        """Turn a state-card action into a new guarded software request."""
        if self._generation_active:
            return
        labels = {
            SoftwareOperation.INSTALL: "Install",
            SoftwareOperation.DOWNLOAD: "Download",
            SoftwareOperation.UPDATE: "Update",
            SoftwareOperation.UPGRADE: "Upgrade",
            SoftwareOperation.REMOVE: "Delete",
            SoftwareOperation.REINSTALL: "Reinstall & repair",
        }
        self._on_submit(f"{labels.get(operation, operation.value.title())} {state.software_name}")

    def _cancel_from_response(self) -> None:
        if not self._generation_active:
            return
        self.prompt.mark_stopping()
        self.response.begin_cancel()
        self._cancel_generation(show_cancelled=True)

    def _finish_cancelled(self, generation_id: int) -> bool:
        """Finish the UI transition only after the worker has really exited."""
        if self._cancelling_generation_id != generation_id:
            return GLib.SOURCE_REMOVE
        self._cancelling_generation_id = None
        self._generation_thread = None
        self.response.finish_cancelled()
        self.prompt.set_processing(False)
        self.prompt.focus()
        return GLib.SOURCE_REMOVE

    def _start_generation(self, request: str) -> None:
        self._generation_id += 1
        generation_id = self._generation_id
        cancel_event = threading.Event()
        self._cancel_event = cancel_event

        def worker() -> None:
            stream = self.controller.stream_response(request, cancel_event)
            try:
                for event in stream:
                    if cancel_event.is_set():
                        return
                    GLib.idle_add(self._deliver_provider_event, generation_id, event)
            except Exception as exc:  # Keep worker errors inside the UI boundary.
                if not cancel_event.is_set():
                    GLib.idle_add(
                        self._deliver_provider_event,
                        generation_id,
                        ProviderEvent.failure(f"Local provider error: {exc}"),
                    )
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
                if cancel_event.is_set():
                    GLib.idle_add(self._finish_cancelled, generation_id)

        self._generation_thread = threading.Thread(
            target=worker,
            name="system-agent-llm",
            daemon=True,
        )
        self._generation_thread.start()

    def _deliver_provider_event(self, generation_id: int, event: ProviderEvent) -> bool:
        if generation_id != self._generation_id or self._mode != "response":
            return GLib.SOURCE_REMOVE
        if event.kind == "status":
            self.response.set_status(event.text)
        elif event.kind == "text":
            self.response.append_text(event.text)
        elif event.kind == "plan" and event.plan is not None:
            self.response.show_plan(event.plan)
        elif event.kind == "stage" and event.stage_event is not None:
            self.response.show_stage(event.stage_event)
        elif event.kind == "fix" and event.fix_proposal is not None:
            self.response.show_fix_proposal(event.fix_proposal)
        elif event.kind == "software_plan" and event.software_plan is not None:
            self.response.show_software_plan(event.software_plan)
        elif event.kind == "software_state" and event.software_state is not None:
            self.response.show_software_state(event.software_state)
        elif event.kind == "tool" and event.tool_event is not None:
            self.response.show_tool_event(event.tool_event)
        elif event.kind == "tool_approval" and event.tool_approval is not None:
            self.response.show_tool_approval(event.tool_approval)
        elif event.kind == "error":
            self.response.show_error(event.error)
            self.response.finish_response()
            self._generation_active = False
            self._cancelling_generation_id = None
            self.prompt.set_processing(False)
            self.prompt.focus()
        elif event.kind == "done":
            self.response.finish_response()
            if not self.response.has_error:
                self.response.set_status("Response ready")
            self._generation_active = False
            self._cancelling_generation_id = None
            self.prompt.set_processing(False)
            self.prompt.focus()
            if self._retry_request:
                retry_request = self._retry_request
                self._retry_request = None
                GLib.idle_add(self._on_submit, retry_request)
        return GLib.SOURCE_REMOVE

    def _cancel_generation(self, *, show_cancelled: bool = False) -> None:
        active_generation_id = self._generation_id
        if show_cancelled and self._cancel_event is not None:
            self._cancelling_generation_id = active_generation_id
        elif not show_cancelled:
            self._cancelling_generation_id = None
        self._generation_id += 1
        self._generation_active = False
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._cancel_event = None
        self._generation_thread = None

    def _monitor_width(self) -> int:
        display = Gdk.Display.get_default()
        monitors = display.get_monitors() if display is not None else None
        monitor = monitors.get_item(0) if monitors is not None else None
        if monitor is None:
            return 1536
        return monitor.get_geometry().width

    def _quick_size(self) -> tuple[int, int]:
        return (
            max(self.config.min_width, self.config.quick_width),
            max(self.config.min_height, self.config.quick_height),
        )

    def _response_size(self) -> tuple[int, int]:
        return (
            max(self.config.min_width, self.config.window_width),
            max(self.config.min_height, self.config.window_height),
        )

    def _animate_resize(self, target_width: int, target_height: int) -> None:
        if self._resize_source is not None:
            GLib.source_remove(self._resize_source)
        current_width, current_height = self.get_default_size()
        if current_width <= 0:
            current_width, current_height = self._quick_size()
        start_width, start_height = current_width, current_height
        frame = 0
        total_frames = 16

        def resize_frame() -> bool:
            nonlocal frame
            frame += 1
            progress = frame / total_frames
            eased = 1 - (1 - progress) ** 3
            self.set_default_size(
                round(start_width + (target_width - start_width) * eased),
                round(start_height + (target_height - start_height) * eased),
            )
            if frame >= total_frames:
                self._resize_source = None
                return GLib.SOURCE_REMOVE
            return GLib.SOURCE_CONTINUE

        self._resize_source = GLib.timeout_add(16, resize_frame)

    def _clamp_window_size(self, *_args) -> None:
        """Best-effort max-size guard; Wayland compositors own final geometry."""
        width = self.get_width()
        height = self.get_height()
        clamped_width = min(max(width, self.config.min_width), self.config.max_width)
        clamped_height = min(max(height, self.config.min_height), self.config.max_height)
        if (width, height) != (clamped_width, clamped_height) and width > 0 and height > 0:
            self.set_default_size(clamped_width, clamped_height)
        # Only persist a size after automatic resize/layout work has settled
        # and no generation is active. This keeps a transient natural-width
        # measurement from becoming the next response window's size, while
        # still remembering real drag-resizes made by the user.
        if (
            self.is_visible()
            and width > 0
            and height > 0
            and self._resize_source is None
            and not self._generation_active
        ):
            self._queue_save_preferred_size()

    def _queue_save_preferred_size(self) -> None:
        if self._save_source is not None:
            GLib.source_remove(self._save_source)
        self._save_source = GLib.timeout_add(850, self._save_preferred_size)

    def _save_preferred_size(self) -> bool:
        self._save_source = None
        if self.get_width() > 0 and self.get_height() > 0:
            try:
                if self._mode == "response":
                    self.config.save_window_size(self.get_width(), self.get_height())
                else:
                    self.config.save_quick_size(self.get_width(), self.get_height())
            except OSError:
                # A read-only or sandboxed config directory must not affect UI.
                pass
        return GLib.SOURCE_REMOVE
