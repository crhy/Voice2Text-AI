from __future__ import annotations

import queue
import threading
from collections.abc import Callable

from .dictation import segment_stream
from .transcription import WhisperService


def strip_wake_word(text: str, wake_word: str) -> str | None:
    """Return the text with the wake word removed, or ``None`` if absent.

    Matching is a case-insensitive substring search, so "Hey Computer, what
    time is it" and "Computer" both match a wake word of "computer". The
    returned remainder has leading punctuation left over from the wake word
    trimmed off.
    """
    wake = wake_word.casefold().strip()
    if not wake:
        return None
    lowered = text.casefold()
    index = lowered.find(wake)
    if index == -1:
        return None
    remainder = text[:index] + text[index + len(wake) :]
    return remainder.strip(" ,.!?—-\t\n")


class ConversationController:
    """Listens for a wake word, then captures the next utterance as a prompt.

    Reuses the same speech-pause segmentation as :class:`DictationController`
    (via :func:`segment_stream`) so no extra always-on model is required: the
    already-loaded Whisper model transcribes each detected utterance and the
    result is checked for the configured wake word. A dedicated low-latency
    wake-word engine could replace this later without changing the caller
    contract (``feed``/``start``/``stop`` plus the three callbacks).
    """

    PROMPT_TIMEOUT_SECONDS = 8.0

    def __init__(
        self,
        whisper: WhisperService,
        *,
        language: str,
        wake_word: str,
        threshold: int,
        silence_ms: int,
        max_segment_seconds: float,
        on_woken: Callable[[], None],
        on_prompt: Callable[[str], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
    ) -> None:
        self.whisper = whisper
        self.language = language
        self.wake_word = wake_word
        self.threshold = threshold
        self.silence_seconds = silence_ms / 1000.0
        self.max_segment_seconds = max_segment_seconds
        self.on_woken = on_woken
        self.on_prompt = on_prompt
        self.on_status = on_status
        self.on_error = on_error
        self.queue: queue.Queue[tuple[bytes, float] | None] = queue.Queue(maxsize=80)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._muted = threading.Event()

    def start(self) -> None:
        self.stop_event.clear()
        self._muted.clear()
        self.thread = threading.Thread(target=self._run, name="conversation-worker", daemon=True)
        self.thread.start()

    def feed(self, pcm: bytes, level: float) -> None:
        # Dropped while muted so the assistant's own spoken reply, played
        # through the speakers, is never picked back up as a new utterance.
        if self.stop_event.is_set() or self._muted.is_set():
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

    def mute(self) -> None:
        self._muted.set()

    def unmute(self) -> None:
        self._muted.clear()

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass

    def _next_segment(self, idle_timeout_seconds: float | None) -> bytes | None:
        for segment in segment_stream(
            self.queue,
            self.stop_event,
            threshold=self.threshold,
            silence_seconds=self.silence_seconds,
            max_segment_seconds=self.max_segment_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
        ):
            return segment
        return None

    def _transcribe(self, segment: bytes) -> str:
        try:
            return self.whisper.transcribe(segment, self.language)
        except Exception as exc:  # noqa: BLE001 - worker boundary
            self.on_error(str(exc))
            return ""

    def _run(self) -> None:
        waiting_for_prompt = False
        while not self.stop_event.is_set():
            idle_timeout = self.PROMPT_TIMEOUT_SECONDS if waiting_for_prompt else None
            segment = self._next_segment(idle_timeout)
            if segment is None:
                if waiting_for_prompt and not self.stop_event.is_set():
                    self.on_status(f"Didn't catch that — say “{self.wake_word}” again.")
                waiting_for_prompt = False
                continue

            text = self._transcribe(segment)
            if not text:
                continue

            if not waiting_for_prompt:
                remainder = strip_wake_word(text, self.wake_word)
                if remainder is None:
                    continue
                self.on_woken()
                if remainder:
                    self.on_prompt(remainder)
                else:
                    self.on_status("Listening for your request…")
                    waiting_for_prompt = True
            else:
                self.on_prompt(text)
                waiting_for_prompt = False
