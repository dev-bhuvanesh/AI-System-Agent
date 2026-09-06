"""Streaming local-model response surface."""

from __future__ import annotations

import json

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, GLib, Gtk, Pango

from llm.provider import Plan
from software.contracts import SoftwareOperation, SoftwarePlan, SoftwareState
from troubleshooting.contracts import (
    FixProposal,
    TroubleshootingStageEvent,
    TroubleshootingStageStatus,
)
from ui.process_card import ProcessCard, Stage, StageStatus
from tools.contracts import ToolApproval, ToolEvent, ToolEventKind


class ResponseView(Gtk.Box):
    """Conversation-style view with safe status text and a structured plan."""

    def __init__(self, on_edit=None, on_fix_decision=None, on_software_decision=None, on_software_action=None, on_fix_choice=None, on_tool_decision=None) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add_css_class("response-view")
        self.set_hexpand(True)
        self.set_vexpand(True)
        self._on_edit = on_edit
        self._on_fix_decision = on_fix_decision
        self._on_software_decision = on_software_decision
        self._on_software_action = on_software_action
        self._on_fix_choice = on_fix_choice
        self._on_tool_decision = on_tool_decision

        self._follow_output = True
        self._new_content_pending = False
        self._scroll_idle_source: int | None = None
        self._scroll_animation_source: int | None = None

        self.scroller = Gtk.ScrolledWindow()
        self.scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        # The conversation is intentionally viewport-based.  If the
        # scrolled window propagates the natural height of its child, every
        # streamed token, process row, or additional chat turn can make the
        # top-level overlay grow.  Keep the window at its remembered size and
        # let the conversation scroll inside it instead.
        self.scroller.set_propagate_natural_height(False)
        self.scroller.set_propagate_natural_width(False)
        self.scroller.set_min_content_height(0)
        self.scroller.set_min_content_width(0)
        self.scroller.set_vexpand(True)
        self.scroller.add_css_class("response-scroll")
        adjustment = self.scroller.get_vadjustment()
        adjustment.connect("value-changed", self._on_scroll_changed)

        self._scroll_overlay = Gtk.Overlay()
        self._scroll_overlay.set_child(self.scroller)
        self._jump_button = Gtk.Button()
        latest_icon = Gtk.Image.new_from_icon_name("go-down-symbolic")
        latest_icon.set_pixel_size(18)
        latest_icon.set_halign(Gtk.Align.CENTER)
        latest_icon.set_valign(Gtk.Align.CENTER)
        self._jump_button.set_child(latest_icon)
        self._jump_button.set_tooltip_text("Go to latest")
        self._jump_button.add_css_class("scroll-down-button")
        self._jump_button.set_size_request(38, 38)
        self._jump_button.set_halign(Gtk.Align.END)
        self._jump_button.set_valign(Gtk.Align.END)
        self._jump_button.set_margin_end(14)
        self._jump_button.set_margin_bottom(12)
        self._jump_button.set_visible(False)
        self._jump_button.connect("clicked", self._jump_to_latest)
        self._scroll_overlay.add_overlay(self._jump_button)

        self.content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
        self.content.set_hexpand(True)
        self.content.set_margin_top(8)
        self.content.set_margin_bottom(12)
        self.content.set_margin_start(24)
        self.content.set_margin_end(24)
        self.scroller.set_child(self.content)
        self.append(self._scroll_overlay)

        self._response_text = ""
        self._has_conversation = False
        self._show_task_controls = True
        self._status_row: Gtk.Box | None = None
        self._status_icon: Gtk.Label | Gtk.Spinner | None = None
        self._status_label: Gtk.Label | None = None
        self._agent_label: Gtk.Label | None = None
        self._plan_box: Gtk.Box | None = None
        self._actions_box: Gtk.Box | None = None
        self._response_copy_button: Gtk.Button | None = None
        self._response_active = False
        self._copy_feedback_source: int | None = None
        self._process_card: ProcessCard | None = None
        self._fix_box: Gtk.Box | None = None
        self._fix_buttons: tuple[Gtk.Button, Gtk.Button] | None = None
        self._software_plan_box: Gtk.Box | None = None
        self._software_buttons: tuple[Gtk.Button, Gtk.Button] | None = None
        self._software_state_box: Gtk.Box | None = None
        self._software_state_buttons: dict[SoftwareOperation, Gtk.Button] = {}
        self._tool_approval_box: Gtk.Box | None = None
        self._tool_approval_buttons: tuple[Gtk.Button, Gtk.Button] | None = None
        self._active_fix_request = ""
        self._has_error = False

    def set_viewport_limits(self, width: int, height: int) -> None:
        """Bound natural measurement while retaining manual window resizing."""
        # These are preferred-size limits for the scroller's child, not a
        # window resize lock.  The user can still drag the window larger or
        # smaller; long conversation content remains inside the viewport.
        self.scroller.set_max_content_width(max(1, int(width)))
        self.scroller.set_max_content_height(max(1, int(height)))

    def reset(self, request: str, *, system_task: bool = True) -> None:
        """Start a fresh visible conversation and append its first turn."""
        self._clear_content()
        self.start_turn(request, system_task=system_task)

    def start_turn(self, request: str, *, system_task: bool = True) -> None:
        """Append one user turn without removing earlier conversation turns."""
        if not self._has_conversation:
            self._clear_content()
        # A cancelled/error turn can still have its transient status row in
        # the conversation. Remove it before starting the next turn so it
        # cannot appear beside the new troubleshooting process card.
        if self._status_row is not None:
            self._stop_status_spinner()
            self.content.remove(self._status_row)
        self._has_conversation = True
        self._show_task_controls = bool(system_task)
        self._response_text = ""
        self._status_row = None
        self._status_icon = None
        self._status_label = None
        self._agent_label = None
        self._plan_box = None
        self._actions_box = None
        self._response_copy_button = None
        self._response_active = True
        self._cancel_copy_feedback()
        self._process_card = None
        self._fix_box = None
        self._fix_buttons = None
        self._software_plan_box = None
        self._software_buttons = None
        self._software_state_box = None
        self._software_state_buttons = {}
        if self._tool_approval_box is not None:
            self.content.remove(self._tool_approval_box)
        self._tool_approval_box = None
        self._tool_approval_buttons = None
        self._has_error = False

        self._append_user_bubble(request)
        self._append_status_row(
            "Preparing…" if self._show_task_controls else "Thinking…"
        )
        # Sending a new turn is an intentional navigation to the latest
        # message, even if the user had been reading older history.
        self.scroll_to_bottom(force=True)

    def _clear_content(self) -> None:
        while child := self.content.get_first_child():
            self.content.remove(child)
        self._response_text = ""
        self._has_conversation = False
        self._show_task_controls = True
        self._status_row = None
        self._status_icon = None
        self._status_label = None
        self._agent_label = None
        self._plan_box = None
        self._actions_box = None
        self._response_copy_button = None
        self._response_active = False
        self._cancel_copy_feedback()
        self._process_card = None
        self._fix_box = None
        self._fix_buttons = None
        self._software_plan_box = None
        self._software_buttons = None
        self._software_state_box = None
        self._software_state_buttons = {}
        self._tool_approval_box = None
        self._tool_approval_buttons = None
        self._has_error = False
        self._follow_output = True
        self._new_content_pending = False
        self._cancel_scroll_sources()

    def _append_user_bubble(self, request: str) -> None:
        wrapper = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        wrapper.set_halign(Gtk.Align.END)
        bubble = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        bubble.add_css_class("user-bubble")
        user_label = Gtk.Label(label=request, xalign=0)
        user_label.set_wrap(True)
        user_label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        user_label.set_max_width_chars(38)
        user_label.set_selectable(True)
        user_label.add_css_class("user-bubble-text")
        bubble.append(user_label)
        wrapper.append(bubble)

        user_actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        user_actions.set_halign(Gtk.Align.END)
        user_actions.add_css_class("user-actions")

        copy_button = Gtk.Button(label="⧉")
        copy_button.set_tooltip_text("Copy message")
        copy_button.add_css_class("user-action-button")
        copy_button.connect("clicked", lambda _button, text=request: self._copy_text(text))
        user_actions.append(copy_button)

        edit_button = Gtk.Button(label="✎")
        edit_button.set_tooltip_text("Edit message")
        edit_button.add_css_class("user-action-button")
        edit_button.connect("clicked", lambda _button, text=request: self._edit_message(text))
        user_actions.append(edit_button)
        wrapper.append(user_actions)
        self.content.append(wrapper)

    def _copy_text(self, text: str) -> None:
        display = Gdk.Display.get_default()
        if display is not None:
            display.get_clipboard().set(text)

    def _copy_response(self) -> None:
        """Copy the complete streamed response as plain text."""
        if self._response_active or not self._response_text.strip():
            return
        display = Gdk.Display.get_default()
        if display is None:
            return
        display.get_clipboard().set(self._response_text)
        if self._response_copy_button is None:
            return
        self._cancel_copy_feedback()
        self._response_copy_button.set_label("Copied")
        self._response_copy_button.set_tooltip_text("Response copied")
        self._response_copy_button.set_sensitive(False)
        self._response_copy_button.add_css_class("response-copy-confirmed")
        self._copy_feedback_source = GLib.timeout_add(1_200, self._restore_copy_button)

    def _restore_copy_button(self) -> bool:
        self._copy_feedback_source = None
        if self._response_copy_button is not None:
            self._response_copy_button.set_label("Copy")
            self._response_copy_button.set_tooltip_text("Copy response")
            self._response_copy_button.set_sensitive(not self._response_active)
            self._response_copy_button.remove_css_class("response-copy-confirmed")
        return GLib.SOURCE_REMOVE

    def _cancel_copy_feedback(self) -> None:
        if self._copy_feedback_source is not None:
            GLib.source_remove(self._copy_feedback_source)
            self._copy_feedback_source = None

    def _sync_action_visibility(self) -> None:
        self.set_response_active(self._response_active)

    def set_response_active(self, active: bool) -> None:
        """Keep the final Copy control available only after completion."""
        self._response_active = bool(active)
        if self._response_copy_button is not None:
            can_copy = not self._response_active and bool(self._response_text.strip())
            self._response_copy_button.set_visible(can_copy)
            self._response_copy_button.set_sensitive(can_copy)
    def _edit_message(self, text: str) -> None:
        if self._on_edit is not None:
            self._on_edit(text)

    def _append_status_row(self, text: str = "Thinking…") -> None:
        self._status_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self._status_row.add_css_class("thinking-row")
        self._status_icon = Gtk.Spinner()
        self._status_icon.set_size_request(18, 18)
        self._status_icon.add_css_class("thinking-icon")
        self._status_icon.start()
        self._status_row.append(self._status_icon)
        self._status_label = Gtk.Label(label=text, xalign=0)
        self._status_label.add_css_class("thinking-label")
        self._status_label.set_hexpand(True)
        self._status_row.append(self._status_label)
        self.content.append(self._status_row)
        self._sync_action_visibility()

    def set_status(self, text: str) -> None:
        if self._status_label is None:
            return
        # Keep ordinary chat's activity wording stable while Ollama emits
        # transport-specific statuses such as "Checking local AI...". Task
        # rows, on the other hand, expose the real current stage.
        self._status_label.set_text(text if self._show_task_controls else "Thinking…")

    def _set_activity_from_stage(self, event: TroubleshootingStageEvent) -> None:
        """Update the single compact activity row from a real task event."""
        if self._status_label is None or not self._show_task_controls:
            return
        if event.status is TroubleshootingStageStatus.IN_PROGRESS:
            self._status_label.set_text(f"⟳ {event.title}…")
        elif event.status is TroubleshootingStageStatus.COMPLETED:
            self._status_label.set_text(f"✓ {event.title}")
        elif event.status is TroubleshootingStageStatus.CANCELLED:
            self._status_label.set_text("Task cancelled")
        elif event.status in {TroubleshootingStageStatus.FAILED, TroubleshootingStageStatus.WARNING}:
            self._status_label.set_text(f"✕ {event.title}")

    def show_initial_process(
        self,
        title: str = "Understanding request",
        detail: str = "Classifying the request",
    ) -> None:
        """Show the live process card before the first worker event arrives."""
        self.show_stage(
            TroubleshootingStageEvent(
                "understand",
                title,
                TroubleshootingStageStatus.IN_PROGRESS,
                detail,
            )
        )

    @property
    def has_error(self) -> bool:
        return self._has_error

    def append_text(self, text: str) -> None:
        if not text:
            return
        self._response_text += text
        if self._status_row is not None:
            self._stop_status_spinner()
            self.content.remove(self._status_row)
            self._status_row = None
        if self._agent_label is None:
            self._agent_label = Gtk.Label(xalign=0)
            self._agent_label.set_wrap(True)
            self._agent_label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
            self._agent_label.set_selectable(True)
            # Keep streamed text inside the remembered response viewport.
            # A long unbroken model token/command must wrap instead of
            # becoming the minimum width of the top-level window.
            self._agent_label.set_width_chars(1)
            self._agent_label.set_max_width_chars(52)
            self._agent_label.set_hexpand(True)
            self._agent_label.add_css_class("agent-message")
            self.content.append(self._agent_label)
            self._append_actions()
        self._agent_label.set_text(self._response_text)
        self.scroll_to_bottom()

    def _stop_status_spinner(self) -> None:
        if isinstance(self._status_icon, Gtk.Spinner):
            self._status_icon.stop()

    def _append_actions(self) -> None:
        if self._actions_box is not None:
            return
        self._actions_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=7)
        self._actions_box.set_halign(Gtk.Align.START)
        self._actions_box.add_css_class("response-action-bar")

        self._response_copy_button = Gtk.Button(label="Copy")
        self._response_copy_button.set_tooltip_text("Copy response")
        self._response_copy_button.add_css_class("response-copy-button")
        self._response_copy_button.connect("clicked", lambda _button: self._copy_response())
        self._actions_box.append(self._response_copy_button)
        self.content.append(self._actions_box)
        self._sync_action_visibility()

    def finish_response(self) -> None:
        """Reveal the final Copy control after the shared input stops."""
        if self._agent_label is not None:
            self._append_actions()
        self.set_response_active(False)
        if self._process_card is not None:
            if self._process_card.has_failures:
                self._process_card.set_final_result("Failed — needs attention", ok=False)
            else:
                self._process_card.set_final_result("Task completed", ok=True)

    def show_stage(self, event: TroubleshootingStageEvent) -> None:
        """Create/update the live troubleshooting process card."""
        # Keep the compact activity row in place. The process log below it is
        # the expandable history; the row is the one continuously changing
        # Compact indicator for the current real operation.
        self._set_activity_from_stage(event)
        status = {
            TroubleshootingStageStatus.PENDING: StageStatus.PENDING,
            TroubleshootingStageStatus.IN_PROGRESS: StageStatus.RUNNING,
            TroubleshootingStageStatus.COMPLETED: StageStatus.COMPLETED,
            TroubleshootingStageStatus.WARNING: StageStatus.WARNING,
            TroubleshootingStageStatus.FAILED: StageStatus.FAILED,
            TroubleshootingStageStatus.CANCELLED: StageStatus.CANCELLED,
        }.get(event.status, StageStatus.WARNING)
        stage = Stage(
            event.stage_id,
            event.title,
            event.detail,
            status,
            step_type=event.step_type or "troubleshooting",
            action=event.action or event.detail,
            output=event.output,
            error=event.error,
            started_at=event.started_at,
            ended_at=event.ended_at,
        )
        if self._process_card is None:
            self._process_card = ProcessCard([stage])
            self.content.append(self._process_card)
        else:
            self._process_card.add_stage(stage)
        self._sync_action_visibility()
        self.scroll_to_bottom()

    def show_tool_event(self, event: ToolEvent) -> None:
        """Show generic registry execution in the same compact process card."""
        if self._status_label is not None and self._show_task_controls:
            if event.kind in {ToolEventKind.STARTED, ToolEventKind.PROGRESS}:
                self._status_label.set_text(f"⟳ {event.display_name}…")
            elif event.kind is ToolEventKind.COMPLETED:
                self._status_label.set_text(f"✓ {event.display_name}")
            elif event.kind is ToolEventKind.CANCELLED:
                self._status_label.set_text("Task cancelled")
            elif event.kind in {ToolEventKind.FAILED, ToolEventKind.BLOCKED}:
                self._status_label.set_text(f"✕ {event.display_name}")
        status = {
            ToolEventKind.STARTED: StageStatus.RUNNING,
            ToolEventKind.PROGRESS: StageStatus.RUNNING,
            ToolEventKind.COMPLETED: StageStatus.COMPLETED,
            ToolEventKind.BLOCKED: StageStatus.WARNING,
            ToolEventKind.FAILED: StageStatus.FAILED,
            ToolEventKind.CANCELLED: StageStatus.CANCELLED,
        }.get(event.kind, StageStatus.WARNING)
        stage_id = f"tool:{event.event_id}"
        result = event.result
        data = result.data if result is not None else None
        action = event.action or event.message
        input_text = (
            json.dumps(event.input_data, ensure_ascii=False, default=str)
            if event.input_data is not None
            else ""
        )
        output = ""
        exit_code = None
        error = result.error_message if result is not None else ""
        if isinstance(data, dict):
            execution = data.get("execution")
            if isinstance(execution, list):
                action = " ".join(str(part) for part in execution)
            elif data.get("command"):
                action = str(data["command"])
            exit_code = data.get("exit_code")
            stdout = str(data.get("stdout", ""))
            stderr = str(data.get("stderr", ""))
            if stdout or stderr:
                output = stdout
                if stderr:
                    output = f"{output}\nstderr: {stderr}" if output else f"stderr: {stderr}"
            else:
                output = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        elif data is not None:
            output = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        stage = Stage(
            stage_id,
            event.display_name or event.tool_name,
            event.message,
            status,
            step_type=event.tool_name,
            action=action,
            input_text=input_text,
            output=output,
            exit_code=exit_code if isinstance(exit_code, int) else None,
            error=error,
            started_at=event.started_at,
            ended_at=event.ended_at,
        )
        if self._process_card is None:
            self._process_card = ProcessCard([stage])
            self.content.append(self._process_card)
        else:
            self._process_card.add_stage(stage)
        self._sync_action_visibility()
        self.scroll_to_bottom()

    def show_tool_approval(self, approval: ToolApproval) -> None:
        """Show a trusted permission prompt for a validated registry request."""
        if self._tool_approval_box is not None:
            self.content.remove(self._tool_approval_box)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.add_css_class("fix-card")
        heading = Gtk.Label(label="CONFIRM SYSTEM ACTION", xalign=0)
        heading.add_css_class("section-label")
        box.append(heading)

        title = Gtk.Label(label=approval.display_name, xalign=0)
        title.add_css_class("fix-title")
        box.append(title)
        description = Gtk.Label(label=approval.description, xalign=0)
        description.set_wrap(True)
        description.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        description.set_width_chars(1)
        description.set_max_width_chars(52)
        description.add_css_class("fix-rationale")
        box.append(description)
        permission = Gtk.Label(
            label=f"Permission: {approval.permission_level.value.replace('_', ' ').title()}",
            xalign=0,
        )
        permission.add_css_class("fix-effect")
        box.append(permission)
        arguments = json.dumps(
            approval.request.arguments,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if len(arguments) > 600:
            arguments = arguments[:600] + "…"
        details = Gtk.Label(label=f"Requested input: {arguments}", xalign=0)
        details.set_wrap(True)
        details.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        details.set_width_chars(1)
        details.set_max_width_chars(52)
        details.set_selectable(True)
        details.add_css_class("fix-command")
        box.append(details)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        actions.add_css_class("fix-actions")
        allow = Gtk.Button(label="Allow")
        allow.add_css_class("fix-button")
        deny = Gtk.Button(label="Cancel")
        deny.add_css_class("fix-cancel")
        allow.connect("clicked", lambda _button: self._decide_tool(approval.approval_id, True))
        deny.connect("clicked", lambda _button: self._decide_tool(approval.approval_id, False))
        actions.append(allow)
        actions.append(deny)
        box.append(actions)
        self._tool_approval_box = box
        self._tool_approval_buttons = (allow, deny)
        self.content.append(box)
        self.scroll_to_bottom()

    def _decide_tool(self, approval_id: str, approved: bool) -> None:
        if self._tool_approval_buttons is not None:
            allow, deny = self._tool_approval_buttons
            allow.set_sensitive(False)
            deny.set_sensitive(False)
            allow.set_label("Allowed" if approved else "Allow")
            deny.set_label("Cancel" if approved else "Cancelled")
        if self._on_tool_decision is not None:
            self._on_tool_decision(approval_id, approved)

    def show_fix_proposal(self, proposal: FixProposal) -> None:
        """Show the exact proposed change and wait for a trusted click."""
        if self._fix_box is not None:
            self.content.remove(self._fix_box)
        self._active_fix_request = proposal.original_request

        fix_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        fix_box.add_css_class("fix-card")
        is_confirmation = proposal.mode == "confirmation"
        is_hardware = proposal.action_kind in {"hardware", "physical"}
        manual_only = proposal.action_kind == "manual_only"
        heading = Gtk.Label(
            label="CONFIRM AUTOMATIC FIX" if is_confirmation else "TROUBLESHOOTING ACTION",
            xalign=0,
        )
        heading.add_css_class("section-label")
        fix_box.append(heading)

        title = Gtk.Label(label=proposal.title, xalign=0)
        title.add_css_class("fix-title")
        fix_box.append(title)
        rationale = Gtk.Label(label=proposal.rationale, xalign=0)
        rationale.set_wrap(True)
        rationale.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        rationale.set_width_chars(1)
        rationale.set_max_width_chars(52)
        rationale.add_css_class("fix-rationale")
        fix_box.append(rationale)
        if is_confirmation:
            command = Gtk.Label(label=proposal.command_preview, xalign=0)
            command.set_selectable(True)
            command.set_wrap(True)
            command.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
            command.set_width_chars(1)
            command.set_max_width_chars(52)
            command.add_css_class("fix-command")
            fix_box.append(command)
        effect = Gtk.Label(label=proposal.effect, xalign=0)
        effect.set_wrap(True)
        effect.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        effect.set_width_chars(1)
        effect.set_max_width_chars(52)
        effect.add_css_class("fix-effect")
        fix_box.append(effect)

        if proposal.technical_details:
            details_button = Gtk.Button(label="Technical Details ▾")
            details_button.set_halign(Gtk.Align.START)
            details_button.add_css_class("technical-details-button")
            details_label = Gtk.Label(
                label="\n".join(f"• {detail}" for detail in proposal.technical_details),
                xalign=0,
            )
            details_label.set_wrap(True)
            details_label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
            details_label.set_width_chars(1)
            details_label.set_max_width_chars(52)
            details_label.add_css_class("technical-details")
            details_revealer = Gtk.Revealer()
            details_revealer.set_reveal_child(False)
            details_revealer.set_child(details_label)
            details_button.connect(
                "clicked",
                lambda _button: details_revealer.set_reveal_child(
                    not details_revealer.get_reveal_child()
                ),
            )
            fix_box.append(details_button)
            fix_box.append(details_revealer)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        actions.add_css_class("fix-actions")
        if is_confirmation:
            approve = Gtk.Button(label="Allow")
            approve.add_css_class("fix-button")
            cancel = Gtk.Button(label="Cancel")
            cancel.add_css_class("fix-cancel")
            approve.connect("clicked", lambda _button: self._decide_fix(proposal.proposal_id, True))
            cancel.connect("clicked", lambda _button: self._decide_fix(proposal.proposal_id, False))
        else:
            # Keep the two action choices consistent. Hardware findings show
            # the automatic option as unavailable rather than pretending a
            # software command can repair a physical limitation.
            primary_label = "🔧 Fix Automatically"
            secondary_label = "📖 Fix Manually"
            primary_action = "automatic"
            secondary_action = "manual"
            approve = Gtk.Button(label=primary_label)
            approve.add_css_class("fix-button")
            cancel = Gtk.Button(label=secondary_label)
            cancel.add_css_class("fix-cancel")
            if is_hardware or manual_only:
                approve.set_sensitive(False)
                approve.set_tooltip_text(
                    "Automatic repair is not available for this finding"
                )
            approve.connect("clicked", lambda _button, action=primary_action: self._choose_fix(proposal.proposal_id, action))
            cancel.connect("clicked", lambda _button, action=secondary_action: self._choose_fix(proposal.proposal_id, action))
        actions.append(approve)
        actions.append(cancel)
        fix_box.append(actions)
        self._fix_box = fix_box
        self._fix_buttons = (approve, cancel)
        self.content.append(fix_box)
        self.scroll_to_bottom()

    def show_software_plan(self, plan: SoftwarePlan) -> None:
        """Show a confirmation-gated software action inside the chat."""
        if self._software_plan_box is not None:
            self.content.remove(self._software_plan_box)

        plan_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=7)
        plan_box.add_css_class("software-plan-card")
        heading = Gtk.Label(label="SOFTWARE PLAN", xalign=0)
        heading.add_css_class("section-label")
        plan_box.append(heading)

        title = Gtk.Label(label=plan.software_name, xalign=0)
        title.add_css_class("software-plan-title")
        plan_box.append(title)
        summary = Gtk.Label(
            label=f"Action: {plan.operation.value.title()}  ·  Source: {plan.source.value}",
            xalign=0,
        )
        summary.set_wrap(True)
        summary.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        summary.set_width_chars(1)
        summary.set_max_width_chars(52)
        summary.add_css_class("software-plan-summary")
        plan_box.append(summary)
        metadata = Gtk.Label(
            label=f"Package: {plan.package_name}  ·  Architecture: {plan.architecture}\n"
            f"Dependencies: {', '.join(plan.dependencies) or 'None reported'}\n"
            f"Risk: {plan.risk}"
            + (f"  ·  Current: {plan.current_version}" if plan.current_version else "")
            + (f"  ·  Available: {plan.available_version}" if plan.available_version else ""),
            xalign=0,
        )
        metadata.set_wrap(True)
        metadata.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        metadata.set_width_chars(1)
        metadata.set_max_width_chars(52)
        metadata.add_css_class("software-plan-meta")
        plan_box.append(metadata)

        command = Gtk.Label(label=f"Command: {plan.command_preview}", xalign=0)
        command.set_wrap(True)
        command.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        command.set_width_chars(1)
        command.set_max_width_chars(52)
        command.set_selectable(True)
        command.add_css_class("software-plan-command")
        plan_box.append(command)

        impact = Gtk.Label(label=f"What this will do: {plan.what_will_do}", xalign=0)
        impact.set_wrap(True)
        impact.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        impact.set_width_chars(1)
        impact.set_max_width_chars(52)
        impact.add_css_class("software-plan-impact")
        plan_box.append(impact)

        details_revealer = Gtk.Revealer()
        details_revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
        details = Gtk.Label(label=plan.details, xalign=0)
        details.set_wrap(True)
        details.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        details.set_width_chars(1)
        details.set_max_width_chars(52)
        details.set_selectable(True)
        details.add_css_class("software-plan-details")
        details_revealer.set_child(details)
        details_revealer.set_reveal_child(False)
        plan_box.append(details_revealer)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=7)
        actions.add_css_class("software-plan-actions")
        action_labels = {
            "install": "Install",
            "download": "Download",
            "update": "Update",
            "upgrade": "Upgrade",
            "remove": "Delete",
            "reinstall": "Reinstall & repair",
        }
        action_name = action_labels.get(plan.operation.value, "Apply")
        approve = Gtk.Button(label=f"Allow & {action_name}")
        approve.add_css_class("software-approve-button")
        cancel = Gtk.Button(label="Cancel")
        cancel.add_css_class("software-cancel-button")
        show_details = Gtk.Button(label="Details")
        show_details.add_css_class("software-details-button")
        show_details.connect(
            "clicked",
            lambda _button: details_revealer.set_reveal_child(
                not details_revealer.get_reveal_child()
            ),
        )
        approve.connect("clicked", lambda _button: self._decide_software(plan.plan_id, True))
        cancel.connect("clicked", lambda _button: self._decide_software(plan.plan_id, False))
        actions.append(approve)
        actions.append(cancel)
        actions.append(show_details)
        plan_box.append(actions)
        self._software_plan_box = plan_box
        self._software_buttons = (approve, cancel)
        self.content.append(plan_box)
        self.scroll_to_bottom()

    def show_software_state(self, state: SoftwareState) -> None:
        """Show current package state and only the actions valid for that state."""
        if self._software_state_box is not None:
            self.content.remove(self._software_state_box)

        state_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=7)
        state_box.add_css_class("software-state-card")
        heading = Gtk.Label(label="SOFTWARE STATUS", xalign=0)
        heading.add_css_class("section-label")
        state_box.append(heading)

        title = Gtk.Label(label=state.software_name, xalign=0)
        title.add_css_class("software-plan-title")
        state_box.append(title)

        if state.installed:
            status = f"{state.software_name} is already installed"
            if state.current_version:
                status += f"\nVersion: {state.current_version}"
            if state.update_available is True and state.available_version:
                status += f"\nNew version available: {state.available_version}"
        else:
            status = f"{state.software_name} is not installed"
        status_label = Gtk.Label(label=status, xalign=0)
        status_label.set_wrap(True)
        status_label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        status_label.set_width_chars(1)
        status_label.set_max_width_chars(52)
        status_label.add_css_class("software-state-summary")
        state_box.append(status_label)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=7)
        actions.add_css_class("software-plan-actions")
        labels = {
            SoftwareOperation.INSTALL: "Install",
            SoftwareOperation.DOWNLOAD: "Download",
            SoftwareOperation.UPDATE: "Update",
            SoftwareOperation.UPGRADE: "Upgrade",
            SoftwareOperation.REMOVE: "Delete",
            SoftwareOperation.REINSTALL: "Reinstall & repair",
        }
        self._software_state_buttons = {}
        for operation in state.actions:
            button = Gtk.Button(label=labels.get(operation, operation.value.title()))
            button.add_css_class("software-state-action")
            button.connect("clicked", lambda _button, selected=operation: self._decide_software_action(state, selected))
            actions.append(button)
            self._software_state_buttons[operation] = button
        if state.actions:
            state_box.append(actions)
        self._software_state_box = state_box
        self.content.append(state_box)
        self.scroll_to_bottom()

    def _decide_software(self, plan_id: str, approved: bool) -> None:
        if self._software_buttons is not None:
            approve, cancel = self._software_buttons
            approve.set_sensitive(False)
            cancel.set_sensitive(False)
            approve.set_label("Approved" if approved else "Install")
            cancel.set_label("Cancelled" if not approved else "Cancel")
        if self._on_software_decision is not None:
            self._on_software_decision(plan_id, approved)

    def _decide_software_action(self, state: SoftwareState, operation: SoftwareOperation) -> None:
        button = self._software_state_buttons.get(operation)
        if button is not None:
            button.set_sensitive(False)
            button.set_label("Preparing…")
        if self._on_software_action is not None:
            self._on_software_action(state, operation)

    def _decide_fix(self, proposal_id: str, approved: bool) -> None:
        if self._fix_buttons is not None:
            approve, cancel = self._fix_buttons
            approve.set_sensitive(False)
            cancel.set_sensitive(False)
            approve.set_label("Approved" if approved else "Fix Problem")
            cancel.set_label("Cancelled" if not approved else "Cancel")
        if self._on_fix_decision is not None:
            self._on_fix_decision(proposal_id, approved)

    def _choose_fix(self, proposal_id: str, action: str) -> None:
        if self._fix_buttons is not None:
            for button in self._fix_buttons:
                button.set_sensitive(False)
            if self._fix_buttons[0].get_label() is not None:
                self._fix_buttons[0].set_label("Selected")
        if self._on_fix_choice is not None:
            self._on_fix_choice(proposal_id, action, self._active_fix_request)

    def begin_cancel(self) -> None:
        """Show the stopping state while the worker aborts the real task."""
        if self._status_label is not None:
            self._status_label.set_text("Stopping…")
        elif self._process_card is None and self._agent_label is not None:
            self._append_terminal_status("Stopping…")
        if self._process_card is not None:
            self._process_card.begin_cancel()

    def finish_cancelled(self) -> None:
        """Render the final cancellation state after the worker has exited."""
        if self._process_card is not None:
            self._process_card.cancel_pending()
        if self._status_label is not None:
            self._status_label.set_text("■ Task cancelled by user")
        elif self._process_card is None and self._agent_label is not None:
            self._append_terminal_status("■ Task cancelled by user")
        self.set_response_active(False)

    def _append_terminal_status(self, text: str) -> None:
        """Append a non-interactive status row for a stopped chat turn."""
        if self._status_row is not None:
            if self._status_label is not None:
                self._status_label.set_text(text)
            return
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        row.add_css_class("completion-status")
        icon = Gtk.Label(label="•", xalign=0.5)
        icon.add_css_class("completion-status-icon")
        row.append(icon)
        label = Gtk.Label(label=text, xalign=0)
        label.add_css_class("completion-status-label")
        row.append(label)
        self.content.append(row)
        self._status_row = row
        self._status_icon = icon
        self._status_label = label

    def show_plan(self, plan: Plan) -> None:
        if self._plan_box is not None:
            self.content.remove(self._plan_box)

        plan_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        plan_box.add_css_class("plan-card")
        heading = Gtk.Label(label="PLAN", xalign=0)
        heading.add_css_class("section-label")
        plan_box.append(heading)

        summary = Gtk.Label(label=plan.summary, xalign=0)
        summary.set_wrap(True)
        summary.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        summary.set_width_chars(1)
        summary.set_max_width_chars(52)
        summary.add_css_class("plan-summary")
        plan_box.append(summary)

        for action in plan.actions:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=9)
            row.add_css_class("plan-action")
            marker = Gtk.Label(label="•", xalign=0.5)
            marker.add_css_class("plan-marker")
            row.append(marker)
            text = Gtk.Label(label=action.description, xalign=0)
            text.set_wrap(True)
            text.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
            text.set_width_chars(1)
            text.set_max_width_chars(52)
            text.set_hexpand(True)
            text.add_css_class("plan-action-text")
            row.append(text)
            plan_box.append(row)

        self._plan_box = plan_box
        self.content.append(plan_box)
        self.scroll_to_bottom()

    def show_error(self, message: str) -> None:
        self._has_error = True
        self.set_status("Unable to respond")
        self.append_text(message)
        if self._process_card is not None:
            self._process_card.set_final_result("Unable to complete task", ok=False)

    def scroll_to_top(self) -> None:
        self._follow_output = False
        self._new_content_pending = False
        GLib.idle_add(self._scroll_to_top)

    def scroll_to_bottom(self, *, force: bool = False) -> None:
        """Follow new content only while the user is already near the end."""
        if force:
            self._follow_output = True
        if not self._follow_output and not self._is_near_bottom():
            self._new_content_pending = True
            self._jump_button.set_visible(True)
            return
        self._new_content_pending = False
        self._follow_output = True
        if self._scroll_idle_source is None:
            self._scroll_idle_source = GLib.idle_add(self._scroll_to_bottom)

    def _on_scroll_changed(self, _adjustment: Gtk.Adjustment) -> None:
        """Track manual scrolling without fighting GTK's content relayout."""
        if self._scroll_idle_source is not None:
            return
        if self._is_near_bottom():
            self._follow_output = True
            self._new_content_pending = False
            self._jump_button.set_visible(False)
        elif self._new_content_pending:
            self._follow_output = False
            self._jump_button.set_visible(True)
        else:
            self._follow_output = False
            # The latest control is useful when new output is waiting. Keep
            # the conversation unobstructed when the user is simply reading
            # older content and nothing new has arrived yet.
            self._jump_button.set_visible(False)

    def _is_near_bottom(self) -> bool:
        adjustment = self.scroller.get_vadjustment()
        distance = adjustment.get_upper() - adjustment.get_page_size() - adjustment.get_value()
        return distance <= 36.0

    def _scroll_to_top(self) -> bool:
        self.scroller.get_vadjustment().set_value(0.0)
        return GLib.SOURCE_REMOVE

    def _scroll_to_bottom(self) -> bool:
        self._scroll_idle_source = None
        adjustment = self.scroller.get_vadjustment()
        adjustment.set_value(max(0.0, adjustment.get_upper() - adjustment.get_page_size()))
        self._follow_output = True
        self._new_content_pending = False
        self._jump_button.set_visible(False)
        return GLib.SOURCE_REMOVE

    def _jump_to_latest(self, _button: Gtk.Button) -> None:
        """Smoothly move to the newest output after an explicit user click."""
        self._follow_output = True
        self._new_content_pending = False
        self._jump_button.set_visible(False)
        if self._scroll_idle_source is not None:
            GLib.source_remove(self._scroll_idle_source)
            self._scroll_idle_source = None
        if self._scroll_animation_source is not None:
            GLib.source_remove(self._scroll_animation_source)
        self._scroll_animation_source = GLib.idle_add(self._animate_to_latest)

    def _animate_to_latest(self) -> bool:
        self._scroll_animation_source = None
        adjustment = self.scroller.get_vadjustment()
        start = adjustment.get_value()
        target = max(0.0, adjustment.get_upper() - adjustment.get_page_size())
        if target <= start + 1.0:
            adjustment.set_value(target)
            return GLib.SOURCE_REMOVE
        frame = 0
        frames = 10

        def step() -> bool:
            nonlocal frame
            frame += 1
            progress = frame / frames
            eased = 1.0 - (1.0 - progress) ** 3
            adjustment.set_value(start + (target - start) * eased)
            if frame >= frames:
                self._scroll_animation_source = None
                self._follow_output = True
                return GLib.SOURCE_REMOVE
            return GLib.SOURCE_CONTINUE

        self._scroll_animation_source = GLib.timeout_add(16, step)
        return GLib.SOURCE_REMOVE

    def _cancel_scroll_sources(self) -> None:
        if self._scroll_idle_source is not None:
            GLib.source_remove(self._scroll_idle_source)
            self._scroll_idle_source = None
        if self._scroll_animation_source is not None:
            GLib.source_remove(self._scroll_animation_source)
            self._scroll_animation_source = None
