from __future__ import annotations

import queue
import re
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


# Utterances that end a conversation turn, matched exactly (whitespace and
# punctuation normalized) so a normal question can never be mistaken for one.
# "cancel" abandons the current question and returns to listening for the
# wake word; "goodbye" ends conversation mode altogether.
_CANCEL_PHRASES = frozenset(
    {
        "cancel",
        "cancelled",
        "stop",
        "stop talking",
        "stop it",
        "never mind",
        "nevermind",
        "forget it",
    }
)
_GOODBYE_PHRASES = frozenset(
    {
        "goodbye",
        "bye",
        "goodbye for now",
        "see you",
        "see ya",
        "that's all",
        "thats all",
        "done",
    }
)


def detect_exit_phrase(text: str) -> str | None:
    """Return "cancel" or "goodbye" if the utterance ends the conversation.

    Comparison is exact after casefolding and stripping punctuation, so
    "Never mind." and "Goodbye!" match while "never mind that" does not.
    """
    normalized = " ".join(re.sub(r"[^a-z0-9'\s]", " ", text.casefold()).split())
    if normalized in _GOODBYE_PHRASES:
        return "goodbye"
    if normalized in _CANCEL_PHRASES:
        return "cancel"
    return None


class ConversationController:
    """Listens for a wake word, then captures the next utterance as a prompt.

    Reuses the same speech-pause segmentation as :class:`DictationController`
    (via :func:`segment_stream`), so no dedicated wake-word model is required
    — a small, always-loaded Whisper model transcribes each utterance heard
    while waiting, and the result is checked for the configured wake word.
    Once woken, the caller's real (possibly much larger) model transcribes
    the actual command, since accuracy matters there but not during idle
    listening. A dedicated low-latency wake-word engine could replace the
    wake phase later without changing the caller contract (``feed``/
    ``start``/``stop`` plus the four callbacks).
    """

    PROMPT_TIMEOUT_SECONDS = 8.0

    def __init__(
        self,
        *,
        wake_whisper: WhisperService,
        prompt_whisper: WhisperService,
        language: str,
        wake_word: str,
        threshold: int,
        silence_ms: int,
        max_segment_seconds: float,
        on_woken: Callable[[], None],
        on_prompt: Callable[[str], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
        on_exit: Callable[[str], None] | None = None,
    ) -> None:
        # The wake phase runs constantly in the background, so it uses a small,
        # dedicated model (see WAKE_WHISPER_MODEL in window.py) instead of
        # whatever (possibly much larger) model the user picked for real
        # transcription — that model is only needed once actually woken.
        self.wake_whisper = wake_whisper
        self.prompt_whisper = prompt_whisper
        self.language = language
        self.wake_word = wake_word
        self.threshold = threshold
        self.silence_seconds = silence_ms / 1000.0
        self.max_segment_seconds = max_segment_seconds
        self.on_woken = on_woken
        self.on_prompt = on_prompt
        self.on_status = on_status
        self.on_error = on_error
        self.on_exit = on_exit or (lambda _kind: None)
        self.queue: queue.Queue[tuple[bytes, float] | None] = queue.Queue(maxsize=80)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._muted = threading.Event()
        # An utterance currently heard becomes a prompt without requiring the
        # wake word first (set by the caller after a barge-in, so the user can
        # just keep talking after interrupting a spoken reply).
        self.waiting_for_prompt = False

    @property
    def muted(self) -> bool:
        return self._muted.is_set()

    def start(self) -> None:
        self.stop_event.clear()
        self._muted.clear()
        self.waiting_for_prompt = False
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

    def arm_prompt(self) -> None:
        """Skip the wake word: the next utterance becomes a prompt directly.

        Called by the UI after the user barge-in interrupts a spoken reply,
        so they can keep talking without repeating the wake word.
        """
        self.waiting_for_prompt = True

    def stop(self) -> None:
        self.stop_event.set()
        self.waiting_for_prompt = False
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

    def _transcribe(self, segment: bytes, whisper: WhisperService) -> str:
        try:
            self.on_status("Transcribing…")
            return whisper.transcribe(segment, self.language)
        except Exception as exc:  # noqa: BLE001 - worker boundary
            self.on_error(str(exc))
            return ""

    def _run(self) -> None:
        while not self.stop_event.is_set():
            idle_timeout = self.PROMPT_TIMEOUT_SECONDS if self.waiting_for_prompt else None
            segment = self._next_segment(idle_timeout)
            # Re-read after the segment: a barge-in can arm the prompt state
            # while the user is mid-utterance, and that utterance must be the
            # prompt, not a wake-word candidate.
            waiting_for_prompt = self.waiting_for_prompt
            if segment is None:
                if waiting_for_prompt and not self.stop_event.is_set():
                    self.on_status(f"Didn't catch that — say “{self.wake_word}” again.")
                self.waiting_for_prompt = False
                continue

            whisper = self.prompt_whisper if waiting_for_prompt else self.wake_whisper
            text = self._transcribe(segment, whisper)
            if not text:
                continue

            exit_kind = detect_exit_phrase(text)
            if exit_kind is not None:
                # The user is ending things: abandon any in-flight prompt and
                # either go back to waiting for the wake word (cancel) or
                # leave conversation mode entirely (goodbye — the caller stops
                # us when it hears that).
                self.waiting_for_prompt = False
                self.on_exit(exit_kind)
                if exit_kind == "goodbye":
                    break
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
                    self.waiting_for_prompt = True
            else:
                self.on_prompt(text)
                self.waiting_for_prompt = False
