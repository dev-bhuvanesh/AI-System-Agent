"""Lightweight inline process log for live agent/tool events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Pango


class StageStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    WARNING = "WARNING"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class Stage:
    """UI-neutral process event data, including expandable execution details."""

    stage_id: str
    title: str
    detail: str = "Pending"
    status: StageStatus = StageStatus.PENDING
    step_type: str = ""
    action: str = ""
    input_text: str = ""
    output: str = ""
    exit_code: int | None = None
    error: str = ""
    started_at: float | None = None
    ended_at: float | None = None


@dataclass(slots=True)
class _StepRow:
    stage: Stage
    container: Gtk.Box
    toggle: Gtk.Button
    static_icon: Gtk.Image
    spinner: Gtk.Spinner
    title: Gtk.Label
    detail: Gtk.Label
    chevron: Gtk.Label
    revealer: Gtk.Revealer
    details: Gtk.Box


_ICON_NAMES = {
    "terminal": "utilities-terminal-symbolic",
    "search": "system-search-symbolic",
    "file": "text-x-generic-symbolic",
    "folder": "folder-symbolic",
    "system": "emblem-system-symbolic",
    "package": "package-x-generic-symbolic",
    "network": "network-wired-symbolic",
    "warning": "dialog-warning-symbolic",
    "cancelled": "process-stop-symbolic",
}


class ProcessCard(Gtk.Box):
    """Chronological, independently scrollable, inline process event log.

    The historical class name is kept so existing ResponseView wiring and
    callers remain compatible. It is no longer a card or a global checklist:
    every step has its own compact row and inline details revealer.
    """

    def __init__(self, stages: list[Stage], on_cancel=None) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add_css_class("process-log")
        self.set_vexpand(False)
        self.set_valign(Gtk.Align.START)
        self._stop_button: Gtk.Button | None = None
        self._rows: dict[str, _StepRow] = {}
        self._statuses: dict[str, StageStatus] = {}
        self._follow_latest = True
        self._scroll_idle_source: int | None = None
        self._final_label: Gtk.Label | None = None

        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        toolbar.add_css_class("process-log-toolbar")
        self._toolbar_label = Gtk.Label(label="Process", xalign=0)
        self._toolbar_label.add_css_class("process-log-label")
        self._toolbar_label.set_hexpand(True)
        toolbar.append(self._toolbar_label)
        if on_cancel is not None:
            self._stop_button = Gtk.Button(label="Stop")
            self._stop_button.set_tooltip_text("Stop task")
            self._stop_button.add_css_class("response-stop-button")
            self._stop_button.connect("clicked", lambda _button: on_cancel())
            toolbar.append(self._stop_button)
        self.append(toolbar)

        self._log_scroller = Gtk.ScrolledWindow()
        self._log_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._log_scroller.set_propagate_natural_height(False)
        self._log_scroller.set_propagate_natural_width(False)
        self._log_scroller.set_min_content_height(0)
        self._log_scroller.set_max_content_height(240)
        self._log_scroller.set_vexpand(False)
        self._log_scroller.set_valign(Gtk.Align.START)
        self._log_scroller.add_css_class("process-log-scroll")
        self._log_scroller.get_vadjustment().connect("value-changed", self._on_scroll_changed)
        self._stage_list = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._stage_list.add_css_class("process-step-list")
        self._log_scroller.set_child(self._stage_list)
        self.append(self._log_scroller)

        for stage in stages:
            self.add_stage(stage)

    def _on_scroll_changed(self, _adjustment: Gtk.Adjustment) -> None:
        self._follow_latest = self._is_near_bottom()

    def _is_near_bottom(self) -> bool:
        adjustment = self._log_scroller.get_vadjustment()
        distance = adjustment.get_upper() - adjustment.get_page_size() - adjustment.get_value()
        return distance <= 28.0

    def _queue_latest(self) -> None:
        if not self._follow_latest or self._scroll_idle_source is not None:
            return
        self._scroll_idle_source = self._log_scroller.add_tick_callback(
            self._scroll_latest_on_tick,
            None,
        )

    def _scroll_latest_on_tick(self, _widget: Gtk.Widget, _frame_clock: Any, _data: Any) -> bool:
        adjustment = self._log_scroller.get_vadjustment()
        adjustment.set_value(max(0.0, adjustment.get_upper() - adjustment.get_page_size()))
        self._scroll_idle_source = None
        return False

    def _toggle_step(self, row: _StepRow) -> None:
        expanded = not row.revealer.get_reveal_child()
        row.revealer.set_reveal_child(expanded)
        row.chevron.set_text("⌃" if expanded else "⌄")

    def _step_kind(self, stage: Stage) -> str:
        haystack = " ".join((stage.step_type, stage.title, stage.action)).lower()
        if stage.status in (StageStatus.FAILED, StageStatus.WARNING):
            return "warning"
        if stage.status is StageStatus.CANCELLED:
            return "cancelled"
        if any(word in haystack for word in ("install", "update", "remove", "delete package", "download", "package")):
            return "package"
        if any(word in haystack for word in ("search", "discover", "find", "query", "repository")):
            return "search"
        if any(word in haystack for word in ("directory", "folder")):
            return "folder"
        if any(word in haystack for word in ("file", "read", "write", "copy", "move", "rename")):
            return "file"
        if any(word in haystack for word in ("network", "wifi", "ethernet", "dns", "gateway", "ping", "route")):
            return "network"
        if any(word in haystack for word in ("terminal", "command", "process", "service", "shell")):
            return "terminal"
        return "system"

    def _set_static_icon(self, row: _StepRow) -> None:
        row.static_icon.set_from_icon_name(_ICON_NAMES[self._step_kind(row.stage)])

    @staticmethod
    def _format_time(value: float | None) -> str:
        if value is None:
            return ""
        return datetime.fromtimestamp(value).strftime("%H:%M:%S")

    def _set_details(self, row: _StepRow) -> None:
        while child := row.details.get_first_child():
            row.details.remove(child)
        stage = row.stage
        fields: list[tuple[str, str]] = [("Status", stage.detail or stage.status.value.title())]
        if stage.action:
            fields.append(("Action", stage.action))
        if stage.input_text:
            fields.append(("Input", stage.input_text))
        if stage.output:
            fields.append(("Output", stage.output))
        if stage.exit_code is not None:
            fields.append(("Exit code", str(stage.exit_code)))
        if stage.error:
            fields.append(("Error", stage.error))
        if started := self._format_time(stage.started_at):
            fields.append(("Started", started))
        if ended := self._format_time(stage.ended_at):
            fields.append(("Finished", ended))
        for name, value in fields:
            line = Gtk.Label(label=f"{name}: {value}", xalign=0)
            line.set_wrap(True)
            line.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
            line.set_width_chars(1)
            line.set_selectable(True)
            line.add_css_class("process-detail-line")
            if name in ("Action", "Input", "Output", "Error"):
                line.add_css_class("process-detail-code")
            row.details.append(line)

    def _append_stage(self, stage: Stage) -> _StepRow:
        container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        container.add_css_class("process-step")

        toggle = Gtk.Button()
        toggle.add_css_class("process-step-toggle")
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header.set_hexpand(True)
        static_icon = Gtk.Image()
        static_icon.set_pixel_size(16)
        static_icon.add_css_class("process-step-icon")
        header.append(static_icon)
        spinner = Gtk.Spinner()
        spinner.set_size_request(16, 16)
        spinner.add_css_class("process-step-spinner")
        header.append(spinner)
        title = Gtk.Label(label=stage.title, xalign=0)
        title.set_width_chars(1)
        title.set_max_width_chars(56)
        title.set_ellipsize(Pango.EllipsizeMode.END)
        title.set_hexpand(True)
        title.add_css_class("process-step-title")
        header.append(title)
        detail = Gtk.Label(label=stage.detail, xalign=1)
        detail.set_width_chars(1)
        detail.set_max_width_chars(28)
        detail.set_ellipsize(Pango.EllipsizeMode.END)
        detail.add_css_class("process-step-detail")
        header.append(detail)
        chevron = Gtk.Label(label="⌄", xalign=0.5)
        chevron.add_css_class("process-step-chevron")
        header.append(chevron)
        toggle.set_child(header)
        # GTK's signal callback resolves row at click time, after row creation.
        row = _StepRow(stage, container, toggle, static_icon, spinner, title, detail, chevron, Gtk.Revealer(), Gtk.Box())
        toggle.connect("clicked", lambda _button: self._toggle_step(row))
        container.append(toggle)

        row.details.add_css_class("process-step-details")
        row.revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
        row.revealer.set_transition_duration(140)
        row.revealer.set_child(row.details)
        container.append(row.revealer)
        self._stage_list.append(container)

        self._rows[stage.stage_id] = row
        self._apply_stage(row)
        return row

    def _apply_stage(self, row: _StepRow) -> None:
        stage = row.stage
        row.title.set_text(stage.title)
        row.detail.set_text(stage.detail)
        self._set_static_icon(row)
        running = stage.status is StageStatus.RUNNING
        row.spinner.set_visible(running)
        row.static_icon.set_visible(not running)
        if running:
            row.spinner.start()
        else:
            row.spinner.stop()
        for css_class in (
            "process-step-pending",
            "process-step-running",
            "process-step-completed",
            "process-step-warning",
            "process-step-failed",
            "process-step-cancelled",
        ):
            row.container.remove_css_class(css_class)
        row.container.add_css_class(f"process-step-{stage.status.value.lower()}")
        self._set_details(row)
        self._statuses[stage.stage_id] = stage.status
        if running:
            self._toolbar_label.set_text("Processing…")
        self._queue_latest()

    def add_stage(self, stage: Stage) -> None:
        """Insert or update one row by its stable event/step ID."""
        row = self._rows.get(stage.stage_id)
        if row is None:
            self._append_stage(stage)
        else:
            row.stage = stage
            self._apply_stage(row)

    def update(self, stage_id: str, status: StageStatus, detail: str) -> None:
        """Compatibility update for callers that only have status text."""
        row = self._rows.get(stage_id)
        if row is None:
            return
        row.stage.status = status
        row.stage.detail = detail
        self._apply_stage(row)

    def set_stop_active(self, active: bool) -> None:
        if self._stop_button is not None:
            self._stop_button.set_visible(bool(active))
            self._stop_button.set_sensitive(bool(active))

    def begin_cancel(self) -> None:
        self.set_stop_active(False)
        self._toolbar_label.set_text("Stopping…")

    def cancel_pending(self) -> None:
        """Mark unfinished rows cancelled after the real operation stops."""
        for stage_id, status in tuple(self._statuses.items()):
            if status in (StageStatus.PENDING, StageStatus.RUNNING):
                self.update(stage_id, StageStatus.CANCELLED, "Cancelled")
        self.set_stop_active(False)
        self.set_final_result("■ Task cancelled by user", ok=False)

    @property
    def has_failures(self) -> bool:
        return any(status in (StageStatus.FAILED, StageStatus.WARNING) for status in self._statuses.values())

    def set_final_result(self, message: str, *, ok: bool) -> None:
        """Show one compact task result while retaining every process row."""
        if self._final_label is None:
            self._final_label = Gtk.Label(xalign=0)
            self._final_label.add_css_class("process-final")
            self._stage_list.append(self._final_label)
        prefix = "" if message.startswith(("■", "✓", "✕")) else ("✓ " if ok else "✕ ")
        self._final_label.set_text(prefix + message)
        self._final_label.remove_css_class("process-final-success")
        self._final_label.remove_css_class("process-final-failure")
        self._final_label.add_css_class("process-final-success" if ok else "process-final-failure")
        self._toolbar_label.set_text("Completed" if ok else "Needs attention")
        self.set_stop_active(False)
        self._queue_latest()
