from __future__ import annotations

import threading
from collections.abc import Callable

import gi

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango  # noqa: E402

from .audio import AudioCapture, AudioDevice  # noqa: E402
from .config import ConfigStore  # noqa: E402
from .dictation import DictationController  # noqa: E402
from .ollama import OllamaClient, OllamaError  # noqa: E402
from .speech import SpeechService  # noqa: E402
from .transcription import WhisperService  # noqa: E402

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "turbo"]
APPEARANCE_VALUES = ["system", "light", "dark"]
APPEARANCE_LABELS = ["System", "Light", "Dark"]
TTS_VOICES = [
    ("Aria — US female", "en-US-AriaNeural"),
    ("Jenny — US female", "en-US-JennyNeural"),
    ("Guy — US male", "en-US-GuyNeural"),
    ("Sonia — UK female", "en-GB-SoniaNeural"),
    ("Ryan — UK male", "en-GB-RyanNeural"),
]


def idle(callback: Callable, *args) -> None:
    GLib.idle_add(callback, *args)


def string_item_factory(*, wrap: bool, width_chars: int) -> Gtk.SignalListItemFactory:
    """Create readable labels for long Gtk.StringList entries."""
    factory = Gtk.SignalListItemFactory()

    def setup(_factory, list_item) -> None:
        label = Gtk.Label(xalign=0)
        label.set_hexpand(True)
        label.set_halign(Gtk.Align.FILL)
        label.set_width_chars(width_chars)
        label.set_max_width_chars(90)
        label.set_wrap(wrap)
        if wrap:
            label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        else:
            label.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
        list_item.set_child(label)

    def bind(_factory, list_item) -> None:
        item = list_item.get_item()
        label = list_item.get_child()
        text = item.get_string() if item is not None else ""
        label.set_text(text)
        label.set_tooltip_text(text)

    factory.connect("setup", setup)
    factory.connect("bind", bind)
    return factory


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, application: Adw.Application) -> None:
        super().__init__(application=application)
        self.set_title("Voice2Text AI")
        self.set_default_size(1080, 720)
        self.set_size_request(700, 520)

        self.config_store = ConfigStore()
        self.settings = self.config_store.load()
        self.style_manager = Adw.StyleManager.get_default()
        self._apply_appearance()
        self.whisper = WhisperService()
        self.audio = AudioCapture()
        self.speech = SpeechService()
        self.dictation: DictationController | None = None
        self.listening = False
        self.query_cancel = threading.Event()
        self.devices: list[AudioDevice] = []
        self.ollama_models: list[str] = []
        self._progress_source = 0
        self._level_source = 0
        self._latest_level = 0.0
        self._query_generation = 0

        self._build_ui()
        self._install_actions()
        self._refresh_devices()
        self._refresh_ollama_models()
        self._load_whisper(self.settings.whisper_model)

    def _build_ui(self) -> None:
        self.toast_overlay = Adw.ToastOverlay()
        toolbar = Adw.ToolbarView()
        self.set_content(toolbar)

        header = Adw.HeaderBar()
        header.set_title_widget(Adw.WindowTitle(title="Voice2Text AI", subtitle="Dictation and local AI"))
        toolbar.add_top_bar(header)

        self.record_button = Gtk.ToggleButton(label="Dictate")
        self.record_button.set_tooltip_text("Start or stop dictation (Ctrl+R)")
        self.record_button.connect("toggled", self._on_record_toggled)
        header.pack_start(self.record_button)

        menu = Gio.Menu()
        menu.append("Preferences", "win.preferences")
        menu.append("Keyboard Shortcuts", "win.shortcuts")
        menu.append("About Voice2Text AI", "app.about")
        menu_button = Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu)
        header.pack_end(menu_button)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.toast_overlay.set_child(root)
        toolbar.set_content(self.toast_overlay)

        self.progress = Gtk.ProgressBar()
        self.progress.set_visible(False)
        root.append(self.progress)

        self.status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.status_box.add_css_class("status-strip")
        self.status_box.set_margin_top(10)
        self.status_box.set_margin_bottom(8)
        self.status_box.set_margin_start(18)
        self.status_box.set_margin_end(18)
        self.status_spinner = Adw.Spinner()
        self.status_spinner.set_visible(False)
        self.status_label = Gtk.Label(label="Starting…", xalign=0)
        self.status_label.set_hexpand(True)
        self.level = Gtk.LevelBar()
        self.level.set_min_value(0)
        self.level.set_max_value(4000)
        self.level.set_value(0)
        self.level.set_size_request(150, -1)
        self.level.set_tooltip_text("Microphone level")
        self.status_box.append(self.status_spinner)
        self.status_box.append(self.status_label)
        self.status_box.append(self.level)
        root.append(self.status_box)
        self._install_status_css()

        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_wide_handle(True)
        paned.set_position(525)
        paned.set_vexpand(True)
        paned.set_start_child(self._build_editor("Transcript", editable=True, transcript=True))
        paned.set_end_child(self._build_editor("AI response", editable=False, transcript=False))
        root.append(paned)

        action_bar = Gtk.ActionBar()
        action_bar.set_revealed(True)
        toolbar.add_bottom_bar(action_bar)

        self.copy_button = Gtk.Button(label="Copy")
        self.copy_button.connect("clicked", lambda *_: self.copy_transcript())
        action_bar.pack_start(self.copy_button)

        self.clear_button = Gtk.Button(label="Clear")
        self.clear_button.connect("clicked", lambda *_: self.clear_all())
        action_bar.pack_start(self.clear_button)

        self.ask_button = Gtk.Button(label="Ask AI")
        self.ask_button.add_css_class("suggested-action")
        self.ask_button.connect("clicked", lambda *_: self.ask_ai())
        action_bar.pack_end(self.ask_button)

        self.speak_button = Gtk.Button(label="Speak")
        self.speak_button.connect("clicked", lambda *_: self.speak_response())
        action_bar.pack_end(self.speak_button)

        self.stop_button = Gtk.Button(label="Stop")
        self.stop_button.connect("clicked", lambda *_: self.stop_current_work())
        action_bar.pack_end(self.stop_button)

    def _build_editor(self, title: str, *, editable: bool, transcript: bool) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_top(6)
        box.set_margin_bottom(12)
        box.set_margin_start(12)
        box.set_margin_end(12)

        heading = Gtk.Label(label=title, xalign=0)
        heading.add_css_class("heading")
        box.append(heading)

        view = Gtk.TextView()
        view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        view.set_editable(editable)
        view.set_cursor_visible(editable)
        view.set_top_margin(12)
        view.set_bottom_margin(12)
        view.set_left_margin(12)
        view.set_right_margin(12)
        view.add_css_class("card")
        view.add_css_class("document")
        view.set_vexpand(True)
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_child(view)
        scroller.set_vexpand(True)
        box.append(scroller)

        if transcript:
            self.transcript_view = view
        else:
            self.response_view = view
        return box

    def _install_actions(self) -> None:
        actions = {
            "preferences": self.show_preferences,
            "shortcuts": self.show_shortcuts,
            "record": self.toggle_recording,
            "ask": self.ask_ai,
            "copy": self.copy_transcript,
            "clear": self.clear_all,
        }
        for name, callback in actions.items():
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", lambda _a, _p, cb=callback: cb())
            self.add_action(action)
        app = self.get_application()
        app.set_accels_for_action("win.record", ["<Control>r"])
        app.set_accels_for_action("win.ask", ["<Control>Return"])
        app.set_accels_for_action("win.copy", ["<Control><Shift>c"])
        app.set_accels_for_action("win.clear", ["<Control>l"])
        app.set_accels_for_action("win.preferences", ["<Control>comma"])

    def _install_status_css(self) -> None:
        display = Gdk.Display.get_default()
        if display is None:
            return
        provider = Gtk.CssProvider()
        provider.load_from_data(b".status-strip { background-color: @window_bg_color; }")
        Gtk.StyleContext.add_provider_for_display(
            display,
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )
        self._status_css_provider = provider

    def _set_status(self, text: str, busy: bool = False) -> None:
        if self.status_label.get_text() != text:
            self.status_label.set_text(text)
        if self.status_spinner.get_visible() != busy:
            self.status_spinner.set_visible(busy)
        self.status_box.queue_draw()

    def _toast(self, text: str) -> None:
        self.toast_overlay.add_toast(
            Adw.Toast(title=GLib.markup_escape_text(text), timeout=4)
        )

    def _start_progress(self) -> None:
        self.progress.set_visible(True)
        if self._progress_source:
            GLib.source_remove(self._progress_source)
        self._progress_source = GLib.timeout_add(100, self._pulse_progress)

    def _pulse_progress(self) -> bool:
        self.progress.pulse()
        return True

    def _stop_progress(self) -> None:
        if self._progress_source:
            GLib.source_remove(self._progress_source)
            self._progress_source = 0
        self.progress.set_fraction(0)
        self.progress.set_visible(False)

    def _refresh_devices(self) -> None:
        try:
            self.devices = self.audio.list_devices()
            if self.devices:
                selected = next(
                    (device for device in self.devices if device.identifier == self.settings.microphone_id),
                    None,
                )
                if selected is None and self.settings.microphone_name:
                    saved_name = self.settings.microphone_name.casefold()
                    selected = next(
                        (device for device in self.devices if device.name.casefold() == saved_name),
                        None,
                    )
                if selected is None:
                    selected = self.devices[0]
                if (
                    self.settings.microphone_id != selected.identifier
                    or self.settings.microphone_name != selected.name
                ):
                    self.settings.microphone_id = selected.identifier
                    self.settings.microphone_name = selected.name
                    self.config_store.save(self.settings)
        except Exception as exc:  # noqa: BLE001 - platform boundary
            self._toast(f"Microphone scan failed: {exc}")

    def _refresh_ollama_models(self) -> None:
        def worker() -> None:
            try:
                models = OllamaClient(self.settings.ollama_url).list_models()
                idle(self._apply_ollama_models, models)
            except OllamaError as exc:
                idle(self._set_status, "Ollama is offline. Dictation is still available.")
                idle(self._toast, str(exc))

        threading.Thread(target=worker, name="ollama-models", daemon=True).start()

    def _apply_ollama_models(self, models: list[str]) -> bool:
        self.ollama_models = models
        if models and self.settings.ollama_model not in models:
            self.settings.ollama_model = models[0]
            self.config_store.save(self.settings)
        return False

    def _load_whisper(self, model_name: str) -> None:
        self.settings.whisper_model = model_name
        self.config_store.save(self.settings)
        self._set_status(f"Loading Whisper {model_name}…", busy=True)
        self._start_progress()
        self.whisper.load_async(
            model_name,
            lambda name, backend: idle(self._on_whisper_ready, name, backend),
            lambda error: idle(self._on_whisper_error, error),
        )

    def _on_whisper_ready(self, name: str, backend: str) -> bool:
        self._stop_progress()
        self._set_status(f"Ready — Whisper {name} on {backend}")
        return False

    def _on_whisper_error(self, error: str) -> bool:
        self._stop_progress()
        self._set_status("Whisper could not be loaded.")
        self._toast(error)
        return False

    def _on_record_toggled(self, button: Gtk.ToggleButton) -> None:
        if button.get_active() and not self.listening:
            self.start_recording()
        elif not button.get_active() and self.listening:
            self.stop_recording()

    def toggle_recording(self) -> None:
        self.record_button.set_active(not self.record_button.get_active())

    def start_recording(self) -> None:
        if not self.whisper.ready:
            self.record_button.set_active(False)
            self._toast("Whisper is still loading.")
            return
        if not self.devices:
            self.record_button.set_active(False)
            self._toast("No microphone is available.")
            return

        self.dictation = DictationController(
            self.whisper,
            language=self.settings.language,
            threshold=self.settings.voice_threshold,
            silence_ms=self.settings.silence_ms,
            max_segment_seconds=self.settings.max_segment_seconds,
            on_text=lambda text: idle(self._append_transcript, text),
            on_status=lambda text: idle(self._on_dictation_status, text),
            on_auto_stop=lambda: idle(self._auto_stop_recording),
            on_error=lambda text: idle(self._toast, text),
        )
        self.dictation.start()
        try:
            self.audio.start(
                self.settings.microphone_id,
                self.dictation.feed,
                self._queue_level,
                lambda error: idle(self._capture_error, error),
            )
        except Exception as exc:  # noqa: BLE001 - platform boundary
            self.dictation.stop()
            self.dictation = None
            self.record_button.set_active(False)
            self._toast(str(exc))
            return

        self.listening = True
        self._latest_level = 0.0
        self._start_level_updates()
        self.record_button.set_label("Stop")
        self.record_button.add_css_class("destructive-action")
        self._set_status("Listening…", busy=True)

    def stop_recording(self) -> None:
        self.audio.stop()
        if self.dictation is not None:
            self.dictation.stop()
            self.dictation = None
        self.listening = False
        self._stop_level_updates()
        self.record_button.set_label("Dictate")
        self.record_button.remove_css_class("destructive-action")
        if self.record_button.get_active():
            self.record_button.set_active(False)
        self.level.set_value(0)
        self._set_status("Ready")

    def _auto_stop_recording(self) -> bool:
        if self.listening:
            self.stop_recording()
        return False

    def _capture_error(self, error: str) -> bool:
        self.stop_recording()
        self._toast(f"Microphone error: {error}")
        return False

    def _on_dictation_status(self, text: str) -> bool:
        # Keep the label stable while recording. Rapid label replacement caused
        # stale glyph fragments with GTK 4 on some NVIDIA/X11/Compiz desktops.
        if self.listening and text == "Transcribing…":
            return False
        self._set_status(text, busy=self.listening)
        return False

    def _queue_level(self, value: float) -> None:
        # GStreamer's callback runs outside the GTK main loop. Store only the
        # newest level instead of adding an unbounded series of idle callbacks.
        self._latest_level = max(0.0, min(4000.0, value))

    def _start_level_updates(self) -> None:
        if not self._level_source:
            self._level_source = GLib.timeout_add(50, self._flush_level)

    def _flush_level(self) -> bool:
        if not self.listening:
            self._level_source = 0
            return False
        self.level.set_value(self._latest_level)
        self.status_box.queue_draw()
        return True

    def _stop_level_updates(self) -> None:
        if self._level_source:
            GLib.source_remove(self._level_source)
            self._level_source = 0
        self._latest_level = 0.0
        self.level.set_value(0)
        self.status_box.queue_draw()

    def _append_transcript(self, text: str) -> bool:
        buffer = self.transcript_view.get_buffer()
        end = buffer.get_end_iter()
        prefix = "" if buffer.get_char_count() == 0 else " "
        buffer.insert(end, prefix + text.strip())
        self._set_status("Listening…" if self.listening else "Ready", busy=self.listening)
        return False

    def _get_text(self, view: Gtk.TextView) -> str:
        buffer = view.get_buffer()
        return buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True).strip()

    def _set_text(self, view: Gtk.TextView, text: str) -> None:
        view.get_buffer().set_text(text)

    def _stop_dictation_for_action(self) -> None:
        """Stop microphone capture before actions that consume or replace text."""
        if self.listening or self.record_button.get_active():
            self.stop_recording()

    def copy_transcript(self) -> None:
        self._stop_dictation_for_action()
        text = self._get_text(self.transcript_view)
        if not text:
            self._toast("There is no transcript to copy.")
            return
        display = Gdk.Display.get_default()
        if display is None:
            self._toast("The clipboard is unavailable.")
            return
        display.get_clipboard().set(text)
        self._toast("Transcript copied.")

    def clear_all(self) -> None:
        self._stop_dictation_for_action()
        self.stop_current_work()
        self._set_text(self.transcript_view, "")
        self._set_text(self.response_view, "")
        self._set_status("Ready")

    def ask_ai(self) -> None:
        self._stop_dictation_for_action()
        prompt = self._get_text(self.transcript_view)
        if not prompt:
            self._toast("Speak or type something first.")
            return
        if not self.settings.ollama_model:
            self._refresh_ollama_models()
            self._toast("No Ollama model is selected.")
            return

        self.query_cancel.set()
        cancel_event = threading.Event()
        self.query_cancel = cancel_event
        self._query_generation += 1
        generation = self._query_generation
        model = self.settings.ollama_model
        endpoint = self.settings.ollama_url
        self._set_text(self.response_view, "")
        self.ask_button.set_sensitive(False)
        self._set_status(f"Asking {model}…", busy=True)

        def worker() -> None:
            try:
                client = OllamaClient(endpoint)
                answer = client.generate_stream(
                    model=model,
                    prompt=prompt,
                    cancel_event=cancel_event,
                    on_chunk=lambda chunk: idle(self._append_response, chunk, generation, cancel_event),
                )
                idle(self._on_query_finished, answer, generation, cancel_event)
            except OllamaError as exc:
                idle(self._on_query_error, str(exc), generation, cancel_event)

        threading.Thread(target=worker, name=f"ollama-query-{generation}", daemon=True).start()

    def _query_is_current(self, generation: int, cancel_event: threading.Event) -> bool:
        return generation == self._query_generation and cancel_event is self.query_cancel

    def _append_response(self, chunk: str, generation: int, cancel_event: threading.Event) -> bool:
        if not self._query_is_current(generation, cancel_event) or cancel_event.is_set():
            return False
        buffer = self.response_view.get_buffer()
        buffer.insert(buffer.get_end_iter(), chunk)
        return False

    def _on_query_finished(
        self, answer: str, generation: int, cancel_event: threading.Event
    ) -> bool:
        if not self._query_is_current(generation, cancel_event):
            return False
        self.ask_button.set_sensitive(True)
        if cancel_event.is_set():
            self._set_status("AI request stopped.")
            return False
        self._set_status("AI response complete.")
        if answer and self.settings.auto_speak:
            self.speak_response()
        return False

    def _on_query_error(
        self, error: str, generation: int, cancel_event: threading.Event
    ) -> bool:
        if not self._query_is_current(generation, cancel_event) or cancel_event.is_set():
            return False
        self.ask_button.set_sensitive(True)
        self._set_status("AI request failed.")
        self._toast(error)
        return False

    def speak_response(self) -> None:
        text = self._get_text(self.response_view) or self._get_text(self.transcript_view)
        if not text:
            self._toast("There is no text to speak.")
            return
        self._set_status("Starting speech…", busy=True)
        self.speech.speak(
            text,
            self.settings.tts_rate,
            self.settings.tts_voice,
            on_started=lambda: idle(self._set_status, "Speaking…", True),
            on_done=lambda: idle(self._set_status, "Ready"),
            on_error=lambda error: idle(self._speech_error, error),
        )

    def _speech_error(self, error: str) -> bool:
        self._set_status("Speech playback failed.")
        self._toast(error)
        return False

    def stop_current_work(self) -> None:
        if self.listening:
            self.stop_recording()
        self.query_cancel.set()
        self._query_generation += 1
        self.speech.stop()
        self.ask_button.set_sensitive(True)
        self._set_status("Stopped.")

    @staticmethod
    def _gtk_theme_prefers_dark() -> bool:
        gtk_settings = Gtk.Settings.get_default()
        if gtk_settings is None:
            return False
        if gtk_settings.find_property("gtk-application-prefer-dark-theme") is not None:
            if bool(gtk_settings.get_property("gtk-application-prefer-dark-theme")):
                return True
        if gtk_settings.find_property("gtk-theme-name") is not None:
            theme_name = str(gtk_settings.get_property("gtk-theme-name") or "")
            return "dark" in theme_name.casefold()
        return False

    def _apply_appearance(self) -> None:
        appearance = self.settings.appearance
        if appearance == "dark":
            scheme = Adw.ColorScheme.FORCE_DARK
        elif appearance == "light":
            scheme = Adw.ColorScheme.FORCE_LIGHT
        elif self.style_manager.get_system_supports_color_schemes():
            scheme = Adw.ColorScheme.PREFER_LIGHT
        elif self._gtk_theme_prefers_dark():
            scheme = Adw.ColorScheme.PREFER_DARK
        else:
            scheme = Adw.ColorScheme.PREFER_LIGHT
        self.style_manager.set_color_scheme(scheme)

    def show_preferences(self) -> None:
        dialog = Adw.PreferencesDialog()
        dialog.set_title("Preferences")
        if hasattr(dialog, "set_content_width"):
            dialog.set_content_width(760)
        else:
            dialog.set_size_request(760, -1)
        page = Adw.PreferencesPage()
        dialog.add(page)

        appearance_group = Adw.PreferencesGroup(title="Appearance")
        page.add(appearance_group)
        appearance_row = Adw.ComboRow(
            title="Color scheme",
            subtitle="Follow the desktop or choose an explicit light or dark appearance",
        )
        appearance_row.set_model(Gtk.StringList.new(APPEARANCE_LABELS))
        appearance_row.set_selected(APPEARANCE_VALUES.index(self.settings.appearance))
        appearance_group.add(appearance_row)

        speech_group = Adw.PreferencesGroup(title="Speech recognition")
        page.add(speech_group)

        whisper_row = Adw.ComboRow(title="Whisper model", subtitle="Smaller models use less memory and start faster")
        whisper_model = Gtk.StringList.new(WHISPER_MODELS)
        whisper_row.set_model(whisper_model)
        try:
            whisper_row.set_selected(WHISPER_MODELS.index(self.settings.whisper_model))
        except ValueError:
            whisper_row.set_selected(1)
        speech_group.add(whisper_row)

        mic_names = [device.name for device in self.devices] or ["Default microphone"]
        selected_mic = next(
            (index for index, device in enumerate(self.devices) if device.identifier == self.settings.microphone_id),
            0,
        )
        mic_row = Adw.ComboRow(
            title="Microphone source",
            subtitle=mic_names[selected_mic],
        )
        mic_row.set_model(Gtk.StringList.new(mic_names))
        mic_row.set_factory(string_item_factory(wrap=False, width_chars=42))
        mic_row.set_list_factory(string_item_factory(wrap=True, width_chars=68))
        mic_row.set_selected(selected_mic)
        mic_row.set_tooltip_text(mic_names[selected_mic])

        def update_mic_description(row, _property) -> None:
            index = min(row.get_selected(), len(mic_names) - 1)
            full_name = mic_names[index]
            row.set_subtitle(full_name)
            row.set_tooltip_text(full_name)

        mic_row.connect("notify::selected", update_mic_description)
        speech_group.add(mic_row)

        ai_group = Adw.PreferencesGroup(title="Local AI")
        page.add(ai_group)
        model_names = self.ollama_models or ["No models found"]
        ai_row = Adw.ComboRow(title="Ollama model")
        ai_row.set_model(Gtk.StringList.new(model_names))
        if self.settings.ollama_model in model_names:
            ai_row.set_selected(model_names.index(self.settings.ollama_model))
        ai_group.add(ai_row)

        endpoint_row = Adw.EntryRow(title="Ollama address")
        endpoint_row.set_text(self.settings.ollama_url)
        ai_group.add(endpoint_row)

        auto_speak_row = Adw.SwitchRow(title="Speak AI responses automatically")
        auto_speak_row.set_active(self.settings.auto_speak)
        ai_group.add(auto_speak_row)

        voice_group = Adw.PreferencesGroup(title="Speech output")
        page.add(voice_group)
        voice_row = Adw.ComboRow(
            title="Voice",
            subtitle="Natural online voice with automatic offline fallback",
        )
        voice_row.set_model(Gtk.StringList.new([label for label, _voice in TTS_VOICES]))
        voice_ids = [voice_id for _label, voice_id in TTS_VOICES]
        voice_row.set_selected(
            voice_ids.index(self.settings.tts_voice) if self.settings.tts_voice in voice_ids else 0
        )
        voice_group.add(voice_row)
        rate_row = Adw.SpinRow.new_with_range(80, 350, 5)
        rate_row.set_title("Speaking rate")
        rate_row.set_value(self.settings.tts_rate)
        voice_group.add(rate_row)

        dialog.connect(
            "closed",
            self._save_preferences,
            appearance_row,
            whisper_row,
            mic_row,
            ai_row,
            endpoint_row,
            auto_speak_row,
            voice_row,
            rate_row,
        )
        dialog.present(self)

    def _save_preferences(
        self,
        _dialog,
        appearance_row,
        whisper_row,
        mic_row,
        ai_row,
        endpoint_row,
        auto_speak_row,
        voice_row,
        rate_row,
    ) -> None:
        new_whisper = WHISPER_MODELS[whisper_row.get_selected()]
        if self.devices:
            device = self.devices[min(mic_row.get_selected(), len(self.devices) - 1)]
            self.settings.microphone_id = device.identifier
            self.settings.microphone_name = device.name
        if self.ollama_models:
            self.settings.ollama_model = self.ollama_models[min(ai_row.get_selected(), len(self.ollama_models) - 1)]
        self.settings.ollama_url = endpoint_row.get_text().strip()
        self.settings.auto_speak = auto_speak_row.get_active()
        self.settings.appearance = APPEARANCE_VALUES[appearance_row.get_selected()]
        self.settings.tts_voice = TTS_VOICES[voice_row.get_selected()][1]
        self.settings.tts_rate = int(rate_row.get_value())
        self.config_store.save(self.settings)
        self._apply_appearance()
        if new_whisper != self.settings.whisper_model:
            self._load_whisper(new_whisper)
        self._refresh_ollama_models()

    def show_shortcuts(self) -> None:
        builder = Gtk.Builder.new_from_string(
            """
            <interface>
              <object class="GtkShortcutsWindow" id="shortcuts">
                <property name="modal">true</property>
                <child>
                  <object class="GtkShortcutsSection">
                    <property name="section-name">general</property>
                    <property name="title">General</property>
                    <child>
                      <object class="GtkShortcutsGroup">
                        <property name="title">Actions</property>
                        <child><object class="GtkShortcutsShortcut"><property name="title">Start or stop dictation</property><property name="accelerator">&lt;Control&gt;r</property></object></child>
                        <child><object class="GtkShortcutsShortcut"><property name="title">Ask AI</property><property name="accelerator">&lt;Control&gt;Return</property></object></child>
                        <child><object class="GtkShortcutsShortcut"><property name="title">Copy transcript</property><property name="accelerator">&lt;Control&gt;&lt;Shift&gt;c</property></object></child>
                        <child><object class="GtkShortcutsShortcut"><property name="title">Clear</property><property name="accelerator">&lt;Control&gt;l</property></object></child>
                        <child><object class="GtkShortcutsShortcut"><property name="title">Preferences</property><property name="accelerator">&lt;Control&gt;comma</property></object></child>
                      </object>
                    </child>
                  </object>
                </child>
              </object>
            </interface>
            """,
            -1,
        )
        window = builder.get_object("shortcuts")
        window.set_transient_for(self)
        window.present()

    def do_close_request(self) -> bool:
        self.stop_current_work()
        self.audio.stop()
        self.config_store.save(self.settings)
        return False
