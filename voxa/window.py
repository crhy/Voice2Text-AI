from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

import gi

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango  # noqa: E402

from . import APP_DISPLAY_NAME  # noqa: E402
from .audio import AudioCapture, AudioDevice  # noqa: E402
from .catalog import CatalogUnavailable, load_catalog, refresh_and_cache, refresh_due  # noqa: E402
from .config import ConfigStore  # noqa: E402
from .conversation import ConversationController, finalize_conversation_turn  # noqa: E402
from .dictation import DictationController  # noqa: E402
from .hardware import GpuUsage, detect_available_model_memory_gb, sample_gpu_usage, suggest_models  # noqa: E402
from .installer import InstallerError, install_ollama  # noqa: E402
from .ollama import OllamaClient, OllamaError, strip_reasoning  # noqa: E402
from .speech import SpeechService  # noqa: E402
from .tasks import TaskStore  # noqa: E402
from .transcription import WhisperService  # noqa: E402
from .ui.attachment import attach_files  # noqa: E402
from .ui.shell import AssistantShell  # noqa: E402
from .ui.state import AssistantState  # noqa: E402

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "turbo"]
# Conversation mode's wake-word phase runs continuously in the background, so
# it always uses this small model instead of whichever (possibly much
# larger) model the user picked for real dictation — that one only has to
# run once per turn, after the wake word is actually heard.
WAKE_WHISPER_MODEL = "tiny"
GPU_POLL_INTERVAL_SECONDS = 0.75
APPEARANCE_VALUES = ["system", "light", "dark"]
APPEARANCE_LABELS = ["System", "Light", "Dark"]
TTS_VOICES = [
    ("Aria — US female", "en-US-AriaNeural"),
    ("Jenny — US female", "en-US-JennyNeural"),
    ("Guy — US male", "en-US-GuyNeural"),
    ("Sonia — UK female", "en-GB-SoniaNeural"),
    ("Ryan — UK male", "en-GB-RyanNeural"),
]
# A few turns of history keeps the model aware of what was just said without
# letting a long hands-free session grow the prompt unboundedly.
CONVERSATION_HISTORY_MESSAGES = 24
CONVERSATION_SYSTEM_PROMPT = (
    "You are Voxa, a hands-free voice assistant on the user's desktop. Keep "
    "answers short and conversational — one or two sentences — since they are spoken aloud."
)
# Barge-in: loud sustained speech while a reply is being read interrupts it.
# A short grace period ignores the TTS itself starting, and the streak
# requirement keeps a cough from killing the reply.
BARGE_IN_GRACE_SECONDS = 0.6
BARGE_IN_STREAK = 3


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
        self.set_title(APP_DISPLAY_NAME)
        self.set_default_size(1200, 760)
        self.set_size_request(850, 600)

        self.config_store = ConfigStore()
        self.settings = self.config_store.load()
        self.task_store = TaskStore()
        self._assistant_state = AssistantState.READY
        # True once the user presses OFFLINE: stops everything and no flow
        # may leave this state except via the ACTIVE button.
        self._offline = True
        # Headless backing buffers: the transcript/response editors left the
        # main view (issue #5) but dictation, Ask AI, copy, and speech still
        # consume and produce this text via shortcuts, menus, and voice.
        self._transcript_buffer = Gtk.TextBuffer()
        self._response_buffer = Gtk.TextBuffer()
        self.style_manager = Adw.StyleManager.get_default()
        self._apply_appearance()
        self.whisper = WhisperService()
        self.wake_whisper = WhisperService()
        self.audio = AudioCapture()
        self.speech = SpeechService()
        self.dictation: DictationController | None = None
        self.listening = False
        self.conversation: ConversationController | None = None
        self.conversation_active = False
        # Earlier turns of the current hands-free conversation, sent to the
        # model as a chat message list so it can follow the thread.
        self._conversation_history: deque[dict[str, str]] = deque(
            maxlen=CONVERSATION_HISTORY_MESSAGES
        )
        # Barge-in bookkeeping: when the reply started being spoken (0 = not
        # speaking) and how many recent level ticks were loud enough to count.
        self._speaking_since = 0.0
        self._barge_in_streak = 0
        # The conversation turn whose user message was just pushed, so a stale
        # finish/error callback can't untangle history built by a newer turn.
        self._pending_user_generation: int | None = None
        self.query_cancel = threading.Event()
        self.devices: list[AudioDevice] = []
        self.ollama_models: list[str] = []
        self._progress_source = 0
        self._level_source = 0
        self._latest_level = 0.0
        self._query_generation = 0
        self._hardware_summary = "Detecting your hardware…"
        self._suggested_models: list[str] = []
        self._install_cancel = threading.Event()
        self._installing = False
        self._has_gpu = False
        self._gpu_poll_stop: threading.Event | None = None
        # Set from do_close_request so late idle callbacks (e.g. hardware
        # detection) stop touching a window that is being disposed.
        self._closing = False

        self._build_ui()
        self._install_actions()
        self._refresh_devices()
        self._refresh_ollama_models()
        self._detect_hardware_async()
        self._load_whisper(self.settings.whisper_model)
        self._load_wake_whisper()

    def _load_wake_whisper(self) -> None:
        # Runs quietly in the background: conversation mode falls back to the
        # main model (see start_conversation_mode) if this isn't ready yet,
        # so a slow or failed load here should never block anything.
        self.wake_whisper.load_async(
            WAKE_WHISPER_MODEL,
            lambda _name, _backend: None,
            lambda error: idle(self._toast, f"Wake-word model failed to load: {error}"),
        )

    def _build_ui(self) -> None:
        self.toast_overlay = Adw.ToastOverlay()
        toolbar = Adw.ToolbarView()
        self.set_content(toolbar)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.toast_overlay.set_child(root)
        toolbar.set_content(self.toast_overlay)

        self.progress = Gtk.ProgressBar()
        self.progress.set_visible(False)
        root.append(self.progress)

        self.shell = AssistantShell(
            self.task_store,
            on_active=self._on_shell_active,
            on_offline=self._on_shell_offline,
            on_model_selected=self._on_shell_model_selected,
            on_attach=self._on_shell_attach,
        )
        self.shell.set_vexpand(True)
        root.append(self.shell)
        self.shell.set_agent_state(running=False, offline=True)
        # The assistant starts disabled: listening begins only via ACTIVE.
        self.set_assistant_state(AssistantState.OFFLINE, "Offline")

    def _on_shell_active(self) -> None:
        """ACTIVE button: leave OFFLINE and enter wake-word conversation mode."""
        self._activate_conversation()

    def _activate_conversation(self) -> None:
        """Start conversation mode and restore OFFLINE if startup fails."""
        if self.conversation_active:
            return
        self._offline = False
        self.start_conversation_mode()
        if not self.conversation_active:
            # start_conversation_mode bailed (no mic, model loading): stay OFFLINE.
            self._offline = True
            self.shell.set_agent_state(running=False, offline=True)

    def _on_shell_offline(self) -> None:
        """OFFLINE button: stop listening, capture, speech, and tasks."""
        self._offline = True
        self.stop_current_work()
        # Explicit caption (not detail=None): a late worker idle callback
        # must never repaint this to "Transcribing…" — _set_status drops
        # stale worker statuses while offline.
        self.set_assistant_state(AssistantState.OFFLINE, "Offline")

    def _on_shell_model_selected(self, model: str) -> None:
        if model != self.settings.ollama_model:
            self.settings.ollama_model = model
            self.config_store.save(self.settings)
            self._set_status(f"Asking with {model} from now on.")

    def _on_shell_attach(self) -> None:
        attach_files(self, self.task_store, self._toast)

    def _install_actions(self) -> None:
        actions = {
            "preferences": self.show_preferences,
            "shortcuts": self.show_shortcuts,
            "models": self._show_model_manager,
            "record": self.toggle_recording,
            "conversation": self.toggle_conversation,
            "ask": self.ask_ai,
            "speak": self.speak_response,
            "copy": self.copy_transcript,
            "copy-response": self.copy_response,
            "clear": self.clear_all,
        }
        for name, callback in actions.items():
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", lambda _a, _p, cb=callback: cb())
            self.add_action(action)
        app = self.get_application()
        app.set_accels_for_action("win.record", ["<Control>r"])
        app.set_accels_for_action("win.conversation", ["<Control><Shift>r"])
        app.set_accels_for_action("win.ask", ["<Control>Return"])
        app.set_accels_for_action("win.copy", ["<Control><Shift>c"])
        app.set_accels_for_action("win.copy-response", ["<Control><Shift>v"])
        app.set_accels_for_action("win.clear", ["<Control>l"])
        app.set_accels_for_action("win.preferences", ["<Control>comma"])

    def _set_status(self, text: str) -> None:
        # Caption detail preserves the state iconography: the UI renders
        # from self._assistant_state (issue #5), status strings are detail.
        if self._offline and not self._status_allowed_offline(text):
            # Stale worker idle callback (e.g. an in-flight "Transcribing…"
            # finishing after OFFLINE was pressed). Worker threads are never
            # joined, so their queued GLib.idle_add updates can land after
            # the stop — OFFLINE means actually offline, drop them.
            return
        self.shell.assistant.set_state(self._assistant_state, text)

    @staticmethod
    def _status_allowed_offline(text: str) -> bool:
        """Status texts that may paint while OFFLINE (explicit confirmations)."""
        return text in ("Offline", "Stopped.") or text.startswith("Asking with ")

    def set_assistant_state(self, state: AssistantState, detail: str | None = None) -> None:
        """Single entry point for assistant state (issue #5)."""
        self._assistant_state = state
        self.shell.assistant.set_state(state, detail)

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

    def _detect_hardware_async(self) -> None:
        def worker() -> None:
            available_gb, source = detect_available_model_memory_gb()
            # Show something immediately from the built-in or cached catalog,
            # so the suggestion never waits on the network.
            catalog = load_catalog()
            idle(
                self._apply_hardware_summary,
                available_gb,
                source,
                suggest_models(available_gb, catalog=catalog),
            )
            if not refresh_due():
                return
            try:
                refreshed = refresh_and_cache()
            except CatalogUnavailable:
                return  # Offline, or the registry is down; the cache still stands.
            if refreshed != catalog:
                idle(
                    self._apply_hardware_summary,
                    available_gb,
                    source,
                    suggest_models(available_gb, catalog=refreshed),
                )

        threading.Thread(target=worker, name="hardware-detect", daemon=True).start()

    def _apply_hardware_summary(self, available_gb: float, source: str, suggestions: list) -> bool:
        self._suggested_models = [model.name for model in suggestions]
        names = ", ".join(self._suggested_models)
        self._hardware_summary = f"Suggested for this machine (~{available_gb:.0f} GB {source}): {names}"
        self._has_gpu = source == "GPU VRAM"
        # Keep the gauge live for the app's lifetime once a GPU is detected,
        # not only while an Ollama query is in flight.
        if not self._closing:
            self._start_gpu_monitor()
        return False

    def _start_gpu_monitor(self) -> None:
        # Runs for the window's lifetime once hardware detection finds a
        # GPU; the guard below keeps repeated calls (from both detection
        # and ask_ai) from spawning a second poller.
        if not self._has_gpu or self._gpu_poll_stop is not None:
            return
        stop_event = threading.Event()
        self._gpu_poll_stop = stop_event

        def worker() -> None:
            # Sample immediately (off the UI thread) so the gauge doesn't sit
            # at 0% for a full interval before its first real reading.
            while True:
                usage = sample_gpu_usage()
                if usage is not None and not stop_event.is_set():
                    idle(self._apply_gpu_usage, usage)
                if stop_event.wait(GPU_POLL_INTERVAL_SECONDS):
                    break

        threading.Thread(target=worker, name="gpu-monitor", daemon=True).start()

    def _stop_gpu_monitor(self) -> None:
        if self._gpu_poll_stop is not None:
            self._gpu_poll_stop.set()
            self._gpu_poll_stop = None
        self.shell.set_gpu_text("")

    def _apply_gpu_usage(self, usage: GpuUsage) -> bool:
        self.shell.set_gpu_text(
            f"{usage.utilization_percent:.0f}%",
            f"{usage.utilization_percent:.0f}% utilization — "
            f"{usage.memory_used_gb:.1f} / {usage.memory_total_gb:.1f} GB VRAM",
        )
        return False

    def _apply_ollama_models(self, models: list[str]) -> bool:
        self.ollama_models = models
        if models and self.settings.ollama_model not in models:
            self.settings.ollama_model = models[0]
            self.config_store.save(self.settings)
        self._apply_model_combo(models)
        return False

    def _apply_model_combo(self, models: list[str]) -> None:
        self.shell.models.set_models(models, self.settings.ollama_model)

    @staticmethod
    def _scroll_to_end(view: Gtk.TextView) -> None:
        buffer = view.get_buffer()
        end_iter = buffer.get_end_iter()
        mark = buffer.create_mark(None, end_iter, False)
        view.scroll_to_mark(mark, 0.0, False, 0.0, 0.0)
        buffer.delete_mark(mark)

    def _start_ollama_install(self) -> None:
        if self._installing:
            self._toast("Ollama installation is already running.")
            return
        self._installing = True
        cancel_event = threading.Event()
        self._install_cancel = cancel_event

        dialog = Adw.Dialog(title="Installing Ollama", content_width=560, content_height=420)
        toolbar = Adw.ToolbarView()
        dialog.set_child(toolbar)
        toolbar.add_top_bar(Adw.HeaderBar(show_end_title_buttons=False))

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        toolbar.set_content(box)

        info = Gtk.Label(
            label="This downloads the official installer from ollama.com and runs it as "
            "root. You'll be asked for your password.",
            wrap=True,
            xalign=0,
        )
        box.append(info)

        log_view = Gtk.TextView(editable=False, cursor_visible=False, monospace=True)
        log_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_vexpand(True)
        scroller.set_child(log_view)
        box.append(scroller)

        cancel_button = Gtk.Button(label="Cancel", halign=Gtk.Align.END)
        box.append(cancel_button)

        def append_log(line: str) -> bool:
            buffer = log_view.get_buffer()
            buffer.insert(buffer.get_end_iter(), line + "\n")
            self._scroll_to_end(log_view)
            return False

        def on_cancel(*_args) -> None:
            cancel_event.set()
            cancel_button.set_sensitive(False)
            cancel_button.set_label("Stopping…")

        cancel_button.connect("clicked", on_cancel)
        dialog.connect("closed", lambda *_: cancel_event.set())
        dialog.present(self)

        def worker() -> None:
            try:
                exit_code = install_ollama(
                    on_output=lambda line: idle(append_log, line),
                    cancel_event=cancel_event,
                )
            except InstallerError as exc:
                idle(self._on_install_finished, dialog, False, str(exc))
                return
            if cancel_event.is_set():
                idle(self._on_install_finished, dialog, False, "Installation cancelled.")
            elif exit_code == 0:
                idle(self._on_install_finished, dialog, True, "Ollama installed successfully.")
            else:
                idle(self._on_install_finished, dialog, False, f"Installer exited with status {exit_code}.")

        threading.Thread(target=worker, name="ollama-install", daemon=True).start()

    def _on_install_finished(self, dialog: Adw.Dialog, success: bool, message: str) -> bool:
        self._installing = False
        dialog.close()
        self._toast(message)
        if success:
            self._refresh_ollama_models()
        return False

    def _show_model_manager(self) -> None:
        dialog = Adw.Dialog(title="Manage Ollama Models", content_width=560, content_height=520)
        toolbar = Adw.ToolbarView()
        dialog.set_child(toolbar)
        toolbar.add_top_bar(Adw.HeaderBar())

        page = Adw.PreferencesPage()
        scroller = Gtk.ScrolledWindow(vexpand=True)
        scroller.set_child(page)
        toolbar.set_content(scroller)

        installed_group = Adw.PreferencesGroup(title="Installed")
        page.add(installed_group)
        installed_rows: dict[str, Adw.ActionRow] = {}
        if self.ollama_models:
            for name in self.ollama_models:
                row = Adw.ActionRow(title=name, subtitle="Calculating size…")
                delete_button = Gtk.Button(icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER)
                delete_button.add_css_class("flat")
                delete_button.set_tooltip_text(f"Remove {name}")
                delete_button.connect(
                    "clicked", lambda _btn, model=name: self._confirm_delete_model(dialog, model)
                )
                row.add_suffix(delete_button)
                installed_rows[name] = row
                installed_group.add(row)
        else:
            installed_group.add(Adw.ActionRow(title="No models installed yet"))

        def apply_sizes(infos: list) -> bool:
            for info in infos:
                row = installed_rows.get(info.name)
                if row is not None:
                    row.set_subtitle(f"{info.size_bytes / (1024**3):.1f} GB")
            return False

        def fetch_sizes() -> None:
            try:
                infos = OllamaClient(self.settings.ollama_url).list_models_detailed()
            except OllamaError:
                return
            idle(apply_sizes, infos)

        if installed_rows:
            threading.Thread(target=fetch_sizes, name="ollama-sizes", daemon=True).start()

        pull_group = Adw.PreferencesGroup(
            title="Pull a model",
            description="Enter any Ollama model tag, or pick a suggestion for this machine.",
        )
        page.add(pull_group)

        pull_row = Adw.EntryRow(title="Model name")
        pull_button = Gtk.Button(label="Pull", valign=Gtk.Align.CENTER)
        pull_button.add_css_class("suggested-action")
        pull_row.add_suffix(pull_button)
        pull_group.add(pull_row)

        progress_bar = Gtk.ProgressBar(visible=False, show_text=True)
        pull_group.add(progress_bar)

        if self._suggested_models:
            suggestion_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            suggestion_box.set_margin_top(4)
            suggestion_box.set_margin_bottom(8)
            suggestion_box.set_margin_start(12)
            suggestion_box.set_margin_end(12)
            for name in self._suggested_models:
                chip = Gtk.Button(label=name)
                chip.connect("clicked", lambda _btn, model=name: pull_row.set_text(model))
                suggestion_box.append(chip)
            pull_group.add(suggestion_box)

        def set_pulling(active: bool) -> None:
            pull_button.set_sensitive(not active)
            pull_row.set_sensitive(not active)
            progress_bar.set_visible(active)
            if not active:
                progress_bar.set_fraction(0)
                progress_bar.set_text("")

        def on_progress(status: str, completed: int, total: int) -> bool:
            if total > 0:
                progress_bar.set_fraction(min(1.0, completed / total))
                progress_bar.set_text(f"{status} — {completed / (1024**2):.0f} / {total / (1024**2):.0f} MB")
            else:
                progress_bar.set_fraction(0.0)
                progress_bar.set_text(status)
            return False

        def on_pull_finished(success: bool, message: str) -> bool:
            set_pulling(False)
            self._toast(message)
            if success:
                dialog.close()
                self._show_model_manager()
            return False

        def start_pull(*_args) -> None:
            name = pull_row.get_text().strip()
            if not name:
                self._toast("Enter a model name first.")
                return
            set_pulling(True)
            cancel_event = threading.Event()
            client = OllamaClient(self.settings.ollama_url)

            def worker() -> None:
                try:
                    client.pull_model(
                        name,
                        cancel_event=cancel_event,
                        on_progress=lambda status, completed, total: idle(
                            on_progress, status, completed, total
                        ),
                    )
                    try:
                        models = client.list_models()
                    except OllamaError:
                        models = self.ollama_models
                    idle(self._apply_ollama_models, models)
                    idle(on_pull_finished, True, f"Pulled {name}.")
                except OllamaError as exc:
                    idle(on_pull_finished, False, str(exc))

            threading.Thread(target=worker, name="ollama-pull", daemon=True).start()

        pull_button.connect("clicked", start_pull)
        dialog.present(self)

    def _confirm_delete_model(self, parent_dialog: Adw.Dialog, model: str) -> None:
        confirm = Adw.AlertDialog(
            heading=f"Remove {model}?",
            body="This deletes the downloaded model from disk. You can pull it again later.",
        )
        confirm.add_response("cancel", "Cancel")
        confirm.add_response("delete", "Remove")
        confirm.set_response_appearance("delete", Adw.ResponseAppearance.DESTRUCTIVE)
        confirm.set_default_response("cancel")
        confirm.set_close_response("cancel")

        def on_response(_dialog, response: str) -> None:
            if response != "delete":
                return
            endpoint = self.settings.ollama_url

            def worker() -> None:
                try:
                    OllamaClient(endpoint).delete_model(model)
                    try:
                        models = OllamaClient(endpoint).list_models()
                    except OllamaError:
                        models = [name for name in self.ollama_models if name != model]
                    idle(self._apply_ollama_models, models)
                    idle(self._on_model_deleted, parent_dialog, True, f"Removed {model}.")
                except OllamaError as exc:
                    idle(self._on_model_deleted, parent_dialog, False, str(exc))

            threading.Thread(target=worker, name="ollama-delete", daemon=True).start()

        confirm.connect("response", on_response)
        confirm.present(parent_dialog)

    def _on_model_deleted(self, parent_dialog: Adw.Dialog, success: bool, message: str) -> bool:
        self._toast(message)
        if success:
            parent_dialog.close()
            self._show_model_manager()
        return False

    def _load_whisper(self, model_name: str) -> None:
        self.settings.whisper_model = model_name
        self.config_store.save(self.settings)
        self._set_status(f"Loading Whisper {model_name}…")
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

    def toggle_recording(self) -> None:
        if self.listening:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        if not self.whisper.ready:
            self._toast("Whisper is still loading.")
            return
        if not self.devices:
            self._toast("No microphone is available.")
            return
        if self.conversation_active:
            self.stop_conversation_mode()

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
            self._toast(str(exc))
            return

        self.listening = True
        self._latest_level = 0.0
        self._start_level_updates()
        self.set_assistant_state(AssistantState.LISTENING, "Listening…")

    def stop_recording(self) -> None:
        self.audio.stop()
        if self.dictation is not None:
            self.dictation.stop()
            self.dictation = None
        self.listening = False
        self._stop_level_updates()
        if not self.conversation_active and not self._offline:
            self.set_assistant_state(AssistantState.READY, "Ready")

    def _auto_stop_recording(self) -> bool:
        if self.listening:
            self.stop_recording()
        return False

    def _capture_error(self, error: str) -> bool:
        self.stop_recording()
        self._toast(f"Microphone error: {error}")
        return False

    def toggle_conversation(self) -> None:
        if self.conversation_active:
            self.stop_conversation_mode()
        else:
            self._activate_conversation()

    def start_conversation_mode(self) -> None:
        if not self.whisper.ready:
            self._toast("Whisper is still loading.")
            return
        if not self.devices:
            self._toast("No microphone is available.")
            return
        if self.listening:
            self.stop_recording()

        self.conversation = ConversationController(
            wake_whisper=self.wake_whisper if self.wake_whisper.ready else self.whisper,
            prompt_whisper=self.whisper,
            language=self.settings.language,
            wake_word=self.settings.wake_word,
            threshold=self.settings.voice_threshold,
            silence_ms=self.settings.silence_ms,
            max_segment_seconds=self.settings.max_segment_seconds,
            # _conversation_idle_status() still reports "Ready" here because
            # conversation_active flips True below; spell out the active text.
            idle_status=f"Awaiting orders — say “{self.settings.wake_word}” to ask something",
            on_woken=lambda: idle(self._on_conversation_woken),
            on_prompt=lambda text: idle(self._on_conversation_prompt, text),
            on_status=lambda text: idle(self._set_status, text),
            on_error=lambda text: idle(self._toast, text),
            on_exit=lambda kind: idle(self._on_conversation_exit, kind),
        )
        self.conversation.start()
        try:
            self.audio.start(
                self.settings.microphone_id,
                self.conversation.feed,
                self._queue_level,
                lambda error: idle(self._conversation_capture_error, error),
            )
        except Exception as exc:  # noqa: BLE001 - platform boundary
            self.conversation.stop()
            self.conversation = None
            self._toast(str(exc))
            return

        self.conversation_active = True
        self._latest_level = 0.0
        self._start_level_updates()
        self.shell.set_agent_state(running=True, offline=False)
        # Waiting for the wake word is READY, not LISTENING (issue #6):
        # LISTENING starts once the wake word is actually heard.
        self.set_assistant_state(
            AssistantState.READY,
            f"Say “{self.settings.wake_word}” to begin",
        )

    def stop_conversation_mode(self) -> None:
        self.audio.stop()
        if self.conversation is not None:
            self.conversation.stop()
            self.conversation.unmute()
            self.conversation = None
        self.conversation_active = False
        # A reply may still be playing: speech.stop() fires no callbacks, so
        # the mute/barge-in state is reset explicitly here.
        self.speech.stop()
        self._speaking_since = 0.0
        self._barge_in_streak = 0
        if self._pending_user_generation is not None and self._conversation_history:
            self._conversation_history.pop()
        self._pending_user_generation = None
        self.query_cancel.set()
        self._query_generation += 1
        self._stop_level_updates()
        self.shell.set_agent_state(running=False, offline=self._offline)
        if not self._offline:
            self.set_assistant_state(AssistantState.READY, "Ready")

    def _conversation_capture_error(self, error: str) -> bool:
        self.stop_conversation_mode()
        self._toast(f"Microphone error: {error}")
        return False

    def _on_conversation_woken(self) -> bool:
        if self._speaking_since > 0.0:
            # The wake word was heard over the assistant's own reply: stop it.
            # speech.stop() fires no callbacks, so reset the mute/barge-in
            # state explicitly (mirrors stop_conversation_mode).
            self.speech.stop()
            self._speaking_since = 0.0
            self._barge_in_streak = 0
            if self.conversation is not None:
                self.conversation.unmute()
            self._toast("Interrupted — go ahead.")
        else:
            self._toast(f"Heard “{self.settings.wake_word}” — listening…")
        self.set_assistant_state(AssistantState.LISTENING, "Listening for your request…")
        return False

    def _on_conversation_prompt(self, text: str) -> bool:
        if not text:
            return False
        if self._offline or not self.conversation_active:
            # Late worker callback after OFFLINE/stop: actually offline.
            return False
        self.ask_ai(text)
        return False

    def _on_conversation_exit(self, kind: str) -> bool:
        # The user ended the exchange by voice: cancel any in-flight request,
        # make the mic available again, and either drop back to listening for
        # the wake word ("cancel") or leave conversation mode ("goodbye").
        self.query_cancel.set()
        self._query_generation += 1
        if self._pending_user_generation is not None and self._conversation_history:
            self._conversation_history.pop()
        self._pending_user_generation = None
        if self.conversation is not None:
            self.conversation.unmute()
        self.speech.stop()
        self._speaking_since = 0.0
        self._barge_in_streak = 0
        if kind == "goodbye":
            self.stop_conversation_mode()
            self._set_status("Goodbye!")
        else:
            # Back to waiting for the wake word: READY, not LISTENING.
            self.set_assistant_state(
                AssistantState.READY, self._conversation_idle_status()
            )
        return False

    def _conversation_speak(self, text: str) -> None:
        if self.conversation is not None:
            self.conversation.mute()
        self._speaking_since = time.monotonic()
        self._barge_in_streak = 0
        self.set_assistant_state(AssistantState.SPEAKING, "Speaking…")
        self.speech.speak(
            text,
            self.settings.tts_rate,
            self.settings.tts_voice,
            on_started=lambda: idle(
                self.set_assistant_state, AssistantState.SPEAKING, "Speaking…"
            ),
            on_done=lambda: idle(self._on_conversation_speech_done),
            on_error=lambda error: idle(self._on_conversation_speech_error, error),
        )

    def _conversation_idle_status(self) -> str:
        return (
            f"Awaiting orders — say “{self.settings.wake_word}” to ask something"
            if self.conversation_active
            else "Ready"
        )

    def _on_conversation_speech_done(self) -> bool:
        if self.conversation is not None:
            self.conversation.unmute()
        self._speaking_since = 0.0
        self._barge_in_streak = 0
        self._set_status(self._conversation_idle_status())
        return False

    def _on_conversation_speech_error(self, error: str) -> bool:
        if self.conversation is not None:
            self.conversation.unmute()
        self._speaking_since = 0.0
        self._barge_in_streak = 0
        self._toast(error)
        self._set_status(self._conversation_idle_status())
        return False

    def _on_dictation_status(self, text: str) -> bool:
        # Keep the label stable while recording. Rapid label replacement caused
        # stale glyph fragments with GTK 4 on some NVIDIA/X11/Compiz desktops.
        if self.listening and text == "Transcribing…":
            return False
        self._set_status(text)
        return False

    def _queue_level(self, value: float) -> None:
        # GStreamer's callback runs outside the GTK main loop. Store only the
        # newest level instead of adding an unbounded series of idle callbacks.
        self._latest_level = max(0.0, min(4000.0, value))

    def _start_level_updates(self) -> None:
        if not self._level_source:
            self._level_source = GLib.timeout_add(50, self._flush_level)

    def _flush_level(self) -> bool:
        # Both dictation and conversation mode run the live level meter; in
        # conversation mode this also samples for barge-in.
        if not self.listening and not self.conversation_active:
            self._level_source = 0
            return False
        self.shell.assistant.set_audio_level(min(1.0, self._latest_level / 4000.0))
        self._maybe_barge_in()
        return True

    def _barge_in_target_level(self) -> float:
        # Speech over the assistant's own voice: well above the plain
        # voice-detection threshold so background noise never interrupts.
        return max(self.settings.voice_threshold * 1.4, self.settings.voice_threshold + 350)

    def _maybe_barge_in(self) -> None:
        """Interrupt a spoken reply when the user talks over it.

        The mic keeps reporting levels while muted, so sustained loud speech
        here means the user interrupted: stop the reply, unmute, and let the
        very next utterance become a prompt without the wake word.
        """
        if not self.conversation_active or self.conversation is None or not self.conversation.muted:
            self._barge_in_streak = 0
            return
        if self._speaking_since <= 0.0:
            return
        if time.monotonic() - self._speaking_since < BARGE_IN_GRACE_SECONDS:
            self._barge_in_streak = 0
            return
        if self._latest_level <= self._barge_in_target_level():
            self._barge_in_streak = 0
            return
        self._barge_in_streak += 1
        if self._barge_in_streak < BARGE_IN_STREAK:
            return
        self._barge_in_streak = 0
        self._speaking_since = 0.0
        self.speech.stop()
        self.conversation.unmute()
        self.conversation.arm_prompt()
        self._toast("Interrupted — go ahead.")
        self._set_status("Listening for your request…")

    def _stop_level_updates(self) -> None:
        if self._level_source:
            GLib.source_remove(self._level_source)
            self._level_source = 0
        self._latest_level = 0.0
        self.shell.assistant.set_audio_level(0.0)

    def _append_transcript(self, text: str) -> bool:
        if self._offline:
            # In-flight transcribe finished after OFFLINE: actually offline.
            return False
        end = self._transcript_buffer.get_end_iter()
        prefix = "" if self._transcript_buffer.get_char_count() == 0 else " "
        self._transcript_buffer.insert(end, prefix + text.strip())
        return False

    def _get_text(self, buffer: Gtk.TextBuffer) -> str:
        return buffer.get_text(
            buffer.get_start_iter(), buffer.get_end_iter(), True
        ).strip()

    def _set_text(self, buffer: Gtk.TextBuffer, text: str) -> None:
        buffer.set_text(text)

    def _stop_dictation_for_action(self) -> None:
        """Stop microphone capture before actions that consume or replace text.

        Skipped while conversation mode owns the shared ``self.audio``
        pipeline (dictation and conversation mode are mutually exclusive, so
        there is never a dictation session to stop in that case) — otherwise
        this would tear down the mic capture conversation mode itself needs,
        since ``ask_ai`` runs this on every automatic conversation turn.
        Otherwise it must run unconditionally: a dictation session can
        already be producing results while ``self.listening`` is not yet
        set. ``stop_recording`` is idempotent, so an idle call is harmless.
        """
        if not self.conversation_active:
            self.stop_recording()

    def _copy_buffer_text(self, buffer: Gtk.TextBuffer, *, empty_message: str, done_message: str) -> None:
        text = self._get_text(buffer)
        if not text:
            self._toast(empty_message)
            return
        display = Gdk.Display.get_default()
        if display is None:
            self._toast("The clipboard is unavailable.")
            return
        display.get_clipboard().set(text)
        self._toast(done_message)

    def copy_transcript(self) -> None:
        self._stop_dictation_for_action()
        self._copy_buffer_text(
            self._transcript_buffer,
            empty_message="There is no transcript to copy.",
            done_message="Transcript copied.",
        )

    def copy_response(self) -> None:
        self._copy_buffer_text(
            self._response_buffer,
            empty_message="There is no AI response to copy.",
            done_message="Response copied.",
        )

    def clear_all(self) -> None:
        self._stop_dictation_for_action()
        self.stop_current_work()
        # A fresh start includes forgetting the conversation thread: the model
        # shouldn't carry context from a cleared transcript into the next turn.
        self._conversation_history.clear()
        self._pending_user_generation = None
        self._set_text(self._transcript_buffer, "")
        self._set_text(self._response_buffer, "")
        self._set_status("Ready")

    def ask_ai(self, prompt: str | None = None) -> None:
        self._stop_dictation_for_action()
        if prompt is None:
            prompt = self._get_text(self._transcript_buffer)
        else:
            if self.conversation_active:
                # Keep the exchange visible while talking hands-free: each new
                # question adds a line instead of wiping the conversation.
                existing = self._get_text(self._transcript_buffer)
                self._set_text(self._transcript_buffer, f"{existing}\n{prompt}" if existing else prompt)
            else:
                self._set_text(self._transcript_buffer, prompt)
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

        # In conversation mode the question joins the running thread of turns;
        # a plain dictation/typed question stays a single-shot prompt.
        if self.conversation_active:
            if not self._conversation_history:
                self._conversation_history.append({"role": "system", "content": CONVERSATION_SYSTEM_PROMPT})
            self._conversation_history.append({"role": "user", "content": prompt})
            messages: list[dict[str, str]] | None = list(self._conversation_history)
            self._pending_user_generation = generation
        else:
            messages = None
            self._pending_user_generation = None

        model = self.settings.ollama_model
        endpoint = self.settings.ollama_url
        self._set_text(self._response_buffer, "")
        self.set_assistant_state(AssistantState.THINKING, f"Asking {model}…")
        self._start_gpu_monitor()

        # Batch streamed chunks so the main loop schedules at most one idle
        # callback per batch instead of one per token chunk.
        pending_chunks: list[str] = []

        def flush_chunks() -> None:
            if pending_chunks:
                batch = "".join(pending_chunks)
                pending_chunks.clear()
                idle(self._append_response, batch, generation, cancel_event)

        def on_chunk(chunk: str) -> None:
            pending_chunks.append(chunk)
            if len(pending_chunks) >= 8:
                flush_chunks()

        def worker() -> None:
            try:
                client = OllamaClient(endpoint)
                answer = client.generate_stream(
                    model=model,
                    prompt=prompt,
                    cancel_event=cancel_event,
                    on_chunk=on_chunk,
                    messages=messages,
                )
                flush_chunks()
                idle(self._on_query_finished, answer, generation, cancel_event)
            except OllamaError as exc:
                flush_chunks()
                idle(self._on_query_error, str(exc), generation, cancel_event)

        threading.Thread(target=worker, name=f"ollama-query-{generation}", daemon=True).start()

    def _query_is_current(self, generation: int, cancel_event: threading.Event) -> bool:
        return generation == self._query_generation and cancel_event is self.query_cancel

    def _append_response(self, batch: str, generation: int, cancel_event: threading.Event) -> bool:
        if not self._query_is_current(generation, cancel_event) or cancel_event.is_set():
            return False
        self._response_buffer.insert(self._response_buffer.get_end_iter(), batch)
        return False

    def _on_query_finished(
        self, answer: str, generation: int, cancel_event: threading.Event
    ) -> bool:
        if not self._query_is_current(generation, cancel_event):
            return False
        has_pending_turn = (
            self.conversation_active
            and self._conversation_history
            and self._pending_user_generation == generation
        )
        if cancel_event.is_set():
            if has_pending_turn:
                finalize_conversation_turn(self._conversation_history, "")
            self._pending_user_generation = None
            self._set_status("AI request stopped.")
            return False
        self._set_status("AI response complete.")
        # A thinking model streams its scratchpad in with the reply. Replace
        # what was streamed with just the answer, so the chain of thought is
        # neither left on screen nor read aloud in conversation mode.
        spoken = strip_reasoning(answer)
        if spoken != answer:
            self._response_buffer.set_text(spoken)
        if has_pending_turn:
            finalize_conversation_turn(self._conversation_history, spoken)
        self._pending_user_generation = None
        if spoken and self.conversation_active:
            self._conversation_speak(spoken)
        elif spoken and self.settings.auto_speak:
            self.speak_response()
        else:
            self.set_assistant_state(
                AssistantState.LISTENING if self.conversation_active else AssistantState.READY,
                "AI response complete.",
            )
        return False

    def _on_query_error(
        self, error: str, generation: int, cancel_event: threading.Event
    ) -> bool:
        if not self._query_is_current(generation, cancel_event) or cancel_event.is_set():
            return False
        if (
            self.conversation_active
            and self._conversation_history
            and self._pending_user_generation == generation
        ):
            # The turn produced no usable answer, so it must not sit in the
            # conversation history as an unanswered question with no reply.
            self._conversation_history.pop()
        self._pending_user_generation = None
        self._toast(error)
        self._set_status(self._conversation_idle_status() if self.conversation_active else "AI request failed.")
        return False

    def speak_response(self) -> None:
        text = self._get_text(self._response_buffer) or self._get_text(self._transcript_buffer)
        if not text:
            self._toast("There is no text to speak.")
            return
        self.set_assistant_state(AssistantState.SPEAKING, "Starting speech…")
        self.speech.speak(
            text,
            self.settings.tts_rate,
            self.settings.tts_voice,
            on_started=lambda: idle(
                self.set_assistant_state, AssistantState.SPEAKING, "Speaking…"
            ),
            on_done=lambda: idle(
                self.set_assistant_state,
                AssistantState.LISTENING
                if self.conversation_active
                else AssistantState.READY,
            ),
            on_error=lambda error: idle(self._speech_error, error),
        )

    def _speech_error(self, error: str) -> bool:
        self.set_assistant_state(
            AssistantState.LISTENING if self.conversation_active else AssistantState.READY,
            "Speech playback failed.",
        )
        self._toast(error)
        return False

    def stop_current_work(self) -> None:
        if self.listening:
            self.stop_recording()
        if self.conversation_active:
            self.stop_conversation_mode()
        if self._installing:
            self._install_cancel.set()
        self.query_cancel.set()
        self._query_generation += 1
        self.speech.stop()
        if not self.conversation_active and not self.listening:
            self.set_assistant_state(
                AssistantState.OFFLINE if self._offline else AssistantState.READY,
                "Stopped.",
            )

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
            dialog.set_content_width(820)
        else:
            dialog.set_size_request(820, -1)

        # One page per topic so settings don't share one long crammed column:
        # the dialog renders a sidebar with a row per page.
        appearance_page = Adw.PreferencesPage(title="Appearance", icon_name="color-select-symbolic")
        appearance_page.set_description("How the window looks on your desktop.")
        appearance_group = Adw.PreferencesGroup()
        appearance_page.add(appearance_group)
        appearance_row = Adw.ComboRow(
            title="Color scheme",
            subtitle="Follow the desktop or choose an explicit light or dark appearance",
        )
        appearance_row.set_model(Gtk.StringList.new(APPEARANCE_LABELS))
        appearance_row.set_selected(APPEARANCE_VALUES.index(self.settings.appearance))
        appearance_group.add(appearance_row)
        dialog.add(appearance_page)

        speech_page = Adw.PreferencesPage(title="Speech recognition", icon_name="microphone-sensitivity-high-symbolic")
        speech_page.set_description("How spoken input is transcribed.")
        speech_group = Adw.PreferencesGroup()
        speech_page.add(speech_group)

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
        dialog.add(speech_page)

        ai_page = Adw.PreferencesPage(title="Local AI", icon_name="system-run-symbolic")
        ai_page.set_description("The Ollama server that answers questions on this machine.")
        ai_group = Adw.PreferencesGroup()
        ai_page.add(ai_group)
        model_names = self.ollama_models or ["No models found"]
        ai_row = Adw.ComboRow(title="Ollama model", subtitle=self._hardware_summary)
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

        install_row = Adw.ActionRow(
            title="Install or update Ollama",
            subtitle="Downloads the latest installer from ollama.com and runs it with a password prompt",
        )
        install_button = Gtk.Button(label="Install", valign=Gtk.Align.CENTER)
        install_button.connect("clicked", lambda *_: self._start_ollama_install())
        install_row.add_suffix(install_button)
        ai_group.add(install_row)

        manage_row = Adw.ActionRow(title="Pull or remove models")
        manage_button = Gtk.Button(label="Manage models…", valign=Gtk.Align.CENTER)
        manage_button.connect("clicked", lambda *_: self._show_model_manager())
        manage_row.add_suffix(manage_button)
        ai_group.add(manage_row)
        dialog.add(ai_page)

        conversation_page = Adw.PreferencesPage(title="Conversation mode", icon_name="microphone-sensitivity-muted-symbolic")
        conversation_page.set_description("Hands-free exchanges: say the wake word, then your question.")
        conversation_group = Adw.PreferencesGroup(
            description="Say the wake word to start talking, then ask something — it's transcribed and sent to the Ollama model automatically.",
        )
        conversation_page.add(conversation_group)
        wake_word_row = Adw.EntryRow(title="Wake word")
        wake_word_row.set_text(self.settings.wake_word)
        conversation_group.add(wake_word_row)
        dialog.add(conversation_page)

        voice_page = Adw.PreferencesPage(title="Speech output", icon_name="audio-volume-high-symbolic")
        voice_page.set_description("The voice that reads AI responses aloud.")
        voice_group = Adw.PreferencesGroup()
        voice_page.add(voice_group)
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
        dialog.add(voice_page)

        dialog.connect(
            "closed",
            self._save_preferences,
            appearance_row,
            whisper_row,
            mic_row,
            ai_row,
            endpoint_row,
            auto_speak_row,
            wake_word_row,
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
        wake_word_row,
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
        self.settings.wake_word = wake_word_row.get_text().strip()
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
                        <child><object class="GtkShortcutsShortcut"><property name="title">Start or stop conversation mode</property><property name="accelerator">&lt;Control&gt;&lt;Shift&gt;r</property></object></child>
                        <child><object class="GtkShortcutsShortcut"><property name="title">Ask AI</property><property name="accelerator">&lt;Control&gt;Return</property></object></child>
                        <child><object class="GtkShortcutsShortcut"><property name="title">Copy transcript</property><property name="accelerator">&lt;Control&gt;&lt;Shift&gt;c</property></object></child>
                        <child><object class="GtkShortcutsShortcut"><property name="title">Copy AI response</property><property name="accelerator">&lt;Control&gt;&lt;Shift&gt;v</property></object></child>
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
        self._closing = True
        self.stop_current_work()
        self._stop_gpu_monitor()
        self.audio.stop()
        self.config_store.save(self.settings)
        return False
