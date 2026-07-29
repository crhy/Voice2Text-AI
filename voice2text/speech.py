from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import gi

try:
    import edge_tts
except ImportError:  # The offline fallback still works in source-only installs.
    edge_tts = None

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

Gst.init(None)


class SpeechService:
    """Stream natural speech immediately, with an offline eSpeak NG fallback."""

    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self.appsrc: Any | None = None
        self.temp_path: str | None = None
        self.cancel_event = threading.Event()
        self._on_started = None
        self._on_done = None
        self._on_error = None
        self._started_emitted = False

    def speak(
        self,
        text: str,
        rate: int,
        voice: str,
        *,
        on_started,
        on_done,
        on_error,
    ) -> None:
        self.stop()
        cancel_event = threading.Event()
        self.cancel_event = cancel_event
        self._on_started = on_started
        self._on_done = on_done
        self._on_error = on_error
        self._started_emitted = False

        if edge_tts is not None:
            try:
                self._start_natural_stream(text, rate, voice, cancel_event)
                return
            except Exception as exc:  # noqa: BLE001 - multimedia boundary
                natural_error = str(exc)
        else:
            natural_error = "Edge TTS is unavailable"

        self._start_offline_worker(text, rate, cancel_event, natural_error)

    def _start_natural_stream(
        self,
        text: str,
        rate: int,
        voice: str,
        cancel_event: threading.Event,
    ) -> None:
        pipeline = Gst.parse_launch(
            "appsrc name=source format=bytes block=true max-bytes=1048576 ! "
            "queue max-size-bytes=2097152 ! "
            "decodebin ! audioconvert ! audioresample ! autoaudiosink"
        )
        appsrc = pipeline.get_by_name("source")
        if appsrc is None:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("GStreamer appsrc is unavailable.")

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", self._on_eos, cancel_event)
        bus.connect("message::error", self._on_error_message, cancel_event)
        self.pipeline = pipeline
        self.appsrc = appsrc
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            self._finish_media()
            raise RuntimeError("The streaming speech pipeline could not start.")

        threading.Thread(
            target=self._natural_worker,
            args=(text, rate, voice, appsrc, cancel_event),
            name="natural-speech-stream",
            daemon=True,
        ).start()

    def _natural_worker(
        self,
        text: str,
        rate: int,
        voice: str,
        appsrc,
        cancel_event: threading.Event,
    ) -> None:
        received_audio = threading.Event()
        try:
            asyncio.run(
                self._push_natural_audio(
                    text,
                    rate,
                    voice,
                    appsrc,
                    cancel_event,
                    lambda: self._mark_audio_received(cancel_event, received_audio),
                )
            )
        except Exception as exc:  # noqa: BLE001 - network/service boundary
            if cancel_event.is_set() or cancel_event is not self.cancel_event:
                return
            if received_audio.is_set():
                GLib.idle_add(
                    self._emit_error_for,
                    f"Natural voice was interrupted: {exc}",
                    cancel_event,
                )
            else:
                GLib.idle_add(
                    self._begin_offline_fallback,
                    text,
                    rate,
                    cancel_event,
                    str(exc),
                )

    async def _push_natural_audio(
        self,
        text: str,
        rate: int,
        voice: str,
        appsrc,
        cancel_event: threading.Event,
        on_first_audio,
    ) -> None:
        if edge_tts is None:
            raise RuntimeError("Edge TTS is unavailable")
        percent = round(((rate - 180) / 120) * 50)
        percent = max(-50, min(50, percent))
        communicate = edge_tts.Communicate(
            text,
            voice or "en-US-AriaNeural",
            rate=f"{percent:+d}%",
        )
        received_audio = False
        async for message in communicate.stream():
            if cancel_event.is_set() or cancel_event is not self.cancel_event:
                return
            if message.get("type") != "audio":
                continue
            data = message.get("data", b"")
            if not data:
                continue
            if not received_audio:
                received_audio = True
                on_first_audio()
            buffer = Gst.Buffer.new_allocate(None, len(data), None)
            buffer.fill(0, data)
            flow = appsrc.emit("push-buffer", buffer)
            if flow != Gst.FlowReturn.OK:
                if cancel_event.is_set():
                    return
                raise RuntimeError(f"GStreamer rejected speech audio ({flow.value_nick}).")

        if not received_audio:
            raise RuntimeError("The speech service returned no audio.")
        if self._is_current(cancel_event) and appsrc is self.appsrc:
            appsrc.emit("end-of-stream")

    def _mark_audio_received(
        self,
        cancel_event: threading.Event,
        received_audio: threading.Event,
    ) -> None:
        received_audio.set()
        GLib.idle_add(self._emit_started_for, cancel_event)

    def _begin_offline_fallback(
        self,
        text: str,
        rate: int,
        cancel_event: threading.Event,
        natural_error: str,
    ) -> bool:
        if not self._is_current(cancel_event):
            return False
        self._finish_media()
        self._start_offline_worker(text, rate, cancel_event, natural_error)
        return False

    def _start_offline_worker(
        self,
        text: str,
        rate: int,
        cancel_event: threading.Event,
        natural_error: str,
    ) -> None:
        threading.Thread(
            target=self._synthesize_offline,
            args=(text, rate, cancel_event, natural_error),
            name="offline-speech-synthesis",
            daemon=True,
        ).start()

    def _synthesize_offline(
        self,
        text: str,
        rate: int,
        cancel_event: threading.Event,
        natural_error: str,
    ) -> None:
        try:
            result = subprocess.run(
                ["espeak-ng", "--stdout", "-s", str(rate), text],
                check=True,
                capture_output=True,
            )
            if cancel_event.is_set():
                return
            with tempfile.NamedTemporaryFile(
                prefix="voice2text-",
                suffix=".wav",
                delete=False,
            ) as handle:
                handle.write(result.stdout)
                path = handle.name
            GLib.idle_add(self._play_file, path, cancel_event)
        except FileNotFoundError:
            detail = f"Natural voice failed ({natural_error}); eSpeak NG is not installed."
            GLib.idle_add(self._emit_error_for, detail, cancel_event)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace")[:200]
            message = f"Natural voice failed ({natural_error}); offline speech failed: {detail}"
            GLib.idle_add(self._emit_error_for, message, cancel_event)

    def _play_file(self, path: str, cancel_event: threading.Event) -> bool:
        if not self._is_current(cancel_event):
            with contextlib.suppress(OSError):
                os.unlink(path)
            return False
        self.temp_path = path
        pipeline = Gst.ElementFactory.make("playbin")
        if pipeline is None:
            with contextlib.suppress(OSError):
                os.unlink(path)
            self._emit_error_for("GStreamer playback is unavailable.", cancel_event)
            return False
        pipeline.set_property("uri", Path(path).as_uri())
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", self._on_eos, cancel_event)
        bus.connect("message::error", self._on_error_message, cancel_event)
        self.pipeline = pipeline
        pipeline.set_state(Gst.State.PLAYING)
        self._emit_started_for(cancel_event)
        return False

    def _is_current(self, cancel_event: threading.Event) -> bool:
        return cancel_event is self.cancel_event and not cancel_event.is_set()

    def _emit_started_for(self, cancel_event: threading.Event) -> bool:
        if not self._is_current(cancel_event) or self._started_emitted:
            return False
        self._started_emitted = True
        if self._on_started is not None:
            self._on_started()
        return False

    def _on_eos(self, _bus, _message, cancel_event: threading.Event) -> None:
        if not self._is_current(cancel_event):
            return
        self._finish_media()
        if self._on_done is not None:
            self._on_done()

    def _on_error_message(self, _bus, message, cancel_event: threading.Event) -> None:
        if not self._is_current(cancel_event):
            return
        error, _debug = message.parse_error()
        self._emit_error_for(error.message, cancel_event)

    def _emit_error_for(self, message: str, cancel_event: threading.Event) -> bool:
        if not self._is_current(cancel_event):
            return False
        self._finish_media()
        if self._on_error is not None:
            self._on_error(message)
        return False

    def stop(self) -> None:
        self.cancel_event.set()
        self._finish_media()

    def _finish_media(self) -> None:
        appsrc, self.appsrc = self.appsrc, None
        if appsrc is not None:
            with contextlib.suppress(Exception):
                appsrc.emit("end-of-stream")
        pipeline, self.pipeline = self.pipeline, None
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        path, self.temp_path = self.temp_path, None
        if path:
            with contextlib.suppress(OSError):
                os.unlink(path)
