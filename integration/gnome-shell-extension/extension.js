/* GNOME Shell bridge for exact Wayland overlay geometry. */

import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import GLib from 'gi://GLib';

const APP_ID = 'com.systemagent.Desktop';
const WINDOW_TITLE = 'System Agent';
// GNOME's work area begins below the top panel. The extra gap matches the
// reference screenshot: the capsule floats roughly 40px below that panel.
const TOP_GAP = 0;

export default class SystemAgentExtension extends Extension {
    enable() {
        this._trackedWindow = null;
        this._windowSignals = [];
        this._placing = false;

        this._windowCreatedId = global.display.connect(
            'window-created', (_display, window) => {
                GLib.idle_add(GLib.PRIORITY_DEFAULT_IDLE, () => {
                    this._trackWindow(window);
                    return GLib.SOURCE_REMOVE;
                });
            });
        this._monitorsChangedId = Main.layoutManager.connect(
            'monitors-changed', () => this._placeWindow());

        for (const actor of global.get_window_actors())
            this._trackWindow(actor.get_meta_window());
    }

    disable() {
        if (this._windowCreatedId)
            global.display.disconnect(this._windowCreatedId);
        if (this._monitorsChangedId)
            Main.layoutManager.disconnect(this._monitorsChangedId);
        this._untrackWindow();
    }

    _isAgentWindow(window) {
        if (!window)
            return false;
        const appId = window.get_gtk_application_id?.();
        const title = window.get_title?.();
        return appId === APP_ID || title === WINDOW_TITLE;
    }

    _trackWindow(window) {
        if (!this._isAgentWindow(window))
            return;
        if (this._trackedWindow === window) {
            this._placeWindow();
            return;
        }

        this._untrackWindow();
        this._trackedWindow = window;
        this._windowSignals = [
            window.connect('size-changed', () => this._placeWindow()),
            window.connect('unmanaged', () => {
                if (this._trackedWindow === window)
                    this._untrackWindow();
            }),
        ];
        this._placeWindow();
    }

    _untrackWindow() {
        if (this._trackedWindow) {
            for (const signalId of this._windowSignals)
                this._trackedWindow.disconnect(signalId);
        }
        this._windowSignals = [];
        this._trackedWindow = null;
    }

    _placeWindow() {
        const window = this._trackedWindow;
        if (!window || this._placing || window.minimized)
            return;

        const windowMonitor = window.get_monitor();
        const monitor = Main.layoutManager.monitors[windowMonitor] ??
            Main.layoutManager.primaryMonitor;
        if (!monitor)
            return;

        const workArea = Main.layoutManager.getWorkAreaForMonitor(monitor.index);
        const frame = window.get_frame_rect();
        const x = Math.round(workArea.x + (workArea.width - frame.width) / 2);
        // The work area starts just below GNOME's top panel, so expansion keeps
        // the same top edge while the GTK window grows downward.
        const y = workArea.y + TOP_GAP;

        try {
            this._placing = true;
            if (!window.is_above?.())
                window.make_above();
            window.move_frame(false, x, y);
        } finally {
            this._placing = false;
        }
    }
}
