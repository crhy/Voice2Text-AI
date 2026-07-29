from __future__ import annotations

import gi

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gio, Gtk  # noqa: E402

from . import APP_ID, APP_NAME, APP_VERSION  # noqa: E402
from .window import MainWindow  # noqa: E402


class Voice2TextApplication(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.DEFAULT_FLAGS)
        self.window: MainWindow | None = None
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", lambda *_: self.quit())
        self.add_action(quit_action)
        self.set_accels_for_action("app.quit", ["<Control>q"])

        about_action = Gio.SimpleAction.new("about", None)
        about_action.connect("activate", self._show_about)
        self.add_action(about_action)

    def do_activate(self) -> None:
        if self.window is None:
            self.window = MainWindow(self)
        self.window.present()

    def _show_about(self, *_args) -> None:
        about = Adw.AboutDialog(
            application_name=APP_NAME,
            application_icon=APP_ID,
            developer_name="crhy",
            version=APP_VERSION,
            website="https://github.com/crhy/Voice2Text-AI",
            issue_url="https://github.com/crhy/Voice2Text-AI/issues",
            license_type=Gtk.License.MIT_X11,
            comments="Fast native Linux dictation, local AI prompts, and natural speech output.",
        )
        about.add_credit_section("Built with", ["GTK 4", "libadwaita", "Faster Whisper", "GStreamer", "Ollama", "Edge TTS", "eSpeak NG"])
        about.present(self.window)
