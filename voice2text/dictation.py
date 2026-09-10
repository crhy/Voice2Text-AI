from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator

from .transcription import WhisperService


def segment_stream(
    input_queue: queue.Queue[tuple[bytes, float] | None],
    stop_event: threading.Event,
    *,
    threshold: int,
    silence_seconds: float,
    max_segment_seconds: float,
    idle_timeout_seconds: float | None = None,
    on_idle_timeout: Callable[[], None] | None = None,
) -> Iterator[bytes]:
    """Yield PCM segments split at speech pauses from a live audio queue.

    Pulls ``(pcm, level)`` pairs fed by an audio capture callback and yields
    a complete segment once a pause (or the max segment length) is reached.
    If ``idle_timeout_seconds`` elapses with no speech at all, ``on_idle_timeout``
    fires and the generator ends (without touching ``stop_event``, so callers
    decide what ending idle means for them).
    """
    segment = bytearray()
    heard_voice = False
    last_voice = time.monotonic()
    last_any_voice = last_voice

    while not stop_event.is_set():
        try:
            item = input_queue.get(timeout=0.25)
        except queue.Empty:
            item = None
        now = time.monotonic()

        if item is None:
            if stop_event.is_set():
                break
        else:
            pcm, level = item
            segment.extend(pcm)
            if level >= threshold:
                heard_voice = True
                last_voice = now
                last_any_voice = now

        duration = len(segment) / (16000 * 2)
        # Require a little recorded content before treating a pause as a
        # boundary; scale it with the silence setting so short pauses cut
        # sooner instead of accumulating toward the max-segment cutoff.
        min_content = min(0.8, max(0.3, silence_seconds / 2.0))
        pause_ready = heard_voice and duration >= min_content and now - last_voice >= silence_seconds
        max_ready = heard_voice and duration >= max_segment_seconds
        if pause_ready or max_ready:
            yield bytes(segment)
            segment = bytearray()
            heard_voice = False
            last_voice = now

        if not heard_voice and duration > 2.0:
            segment = bytearray()

        if idle_timeout_seconds is not None and now - last_any_voice > idle_timeout_seconds:
            if on_idle_timeout is not None:
                on_idle_timeout()
            return

    if heard_voice and segment:
        yield bytes(segment)


class DictationController:
    """Segments live PCM around speech pauses and transcribes off the UI thread."""

    IDLE_TIMEOUT_SECONDS = 15.0

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

    def _on_idle_timeout(self) -> None:
        self.on_status("No speech detected; dictation stopped.")
        self.on_auto_stop()

    def _run(self) -> None:
        for segment in segment_stream(
            self.queue,
            self.stop_event,
            threshold=self.threshold,
            silence_seconds=self.silence_seconds,
            max_segment_seconds=self.max_segment_seconds,
            idle_timeout_seconds=self.IDLE_TIMEOUT_SECONDS,
            on_idle_timeout=self._on_idle_timeout,
        ):
            self._flush(segment)

    def _flush(self, segment: bytes) -> None:
        if not segment:
            return
        self.on_status("Transcribing…")
        try:
            text = self.whisper.transcribe(segment, self.language)
            if text:
                self.on_text(text)
        except Exception as exc:  # noqa: BLE001 - worker boundary
            self.on_error(str(exc))
