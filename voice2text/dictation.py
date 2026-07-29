from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

from .transcription import WhisperService


class DictationController:
    """Segments live PCM around speech pauses and transcribes off the UI thread."""

    def __init__(
        self,
        whisper: WhisperService,
        *,
        language: str,
        threshold: int,
        silence_ms: int,
        max_segment_seconds: float,
        on_text: Callable[[str], None],
        on_status: Callable[[str], None],
        on_auto_stop: Callable[[], None],
        on_error: Callable[[str], None],
    ) -> None:
        self.whisper = whisper
        self.language = language
        self.threshold = threshold
        self.silence_seconds = silence_ms / 1000.0
        self.max_segment_seconds = max_segment_seconds
        self.on_text = on_text
        self.on_status = on_status
        self.on_auto_stop = on_auto_stop
        self.on_error = on_error
        self.queue: queue.Queue[tuple[bytes, float] | None] = queue.Queue(maxsize=80)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="dictation-worker", daemon=True)
        self.thread.start()

    def feed(self, pcm: bytes, level: float) -> None:
        if self.stop_event.is_set():
            return
        try:
            self.queue.put_nowait((pcm, level))
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait((pcm, level))
            except queue.Full:
                pass

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass

    def _run(self) -> None:
        segment = bytearray()
        heard_voice = False
        last_voice = time.monotonic()
        last_any_voice = last_voice

        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.25)
            except queue.Empty:
                item = None
            now = time.monotonic()

            if item is None:
                if self.stop_event.is_set():
                    break
            else:
                pcm, level = item
                segment.extend(pcm)
                if level >= self.threshold:
                    heard_voice = True
                    last_voice = now
                    last_any_voice = now

            duration = len(segment) / (16000 * 2)
            pause_ready = heard_voice and duration >= 0.8 and now - last_voice >= self.silence_seconds
            max_ready = heard_voice and duration >= self.max_segment_seconds
            if pause_ready or max_ready:
                self._flush(segment)
                segment = bytearray()
                heard_voice = False
                last_voice = now

            if not heard_voice and duration > 2.0:
                segment = bytearray()

            if now - last_any_voice > 15.0:
                self.on_status("No speech detected; dictation stopped.")
                self.on_auto_stop()
                self.stop_event.set()
                break

        if heard_voice and segment:
            self._flush(segment)

    def _flush(self, segment: bytearray) -> None:
        if not segment:
            return
        self.on_status("Transcribing…")
        try:
            text = self.whisper.transcribe(bytes(segment), self.language)
            if text:
                self.on_text(text)
        except Exception as exc:  # noqa: BLE001 - worker boundary
            self.on_error(str(exc))
