from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

import pytest

from voxa.conversation import (
    ConversationController,
    detect_exit_phrase,
    finalize_conversation_turn,
    strip_wake_word,
)


def test_strip_wake_word_returns_none_when_absent() -> None:
    assert strip_wake_word("what time is it", "computer") is None


def test_strip_wake_word_returns_empty_remainder_for_bare_wake_word() -> None:
    assert strip_wake_word("Computer", "computer") == ""


def test_strip_wake_word_returns_trailing_command_in_same_utterance() -> None:
    assert (
        strip_wake_word("Computer, what's the weather today", "computer")
        == "what's the weather today"
    )


def test_strip_wake_word_is_case_insensitive_and_matches_multi_word_phrase() -> None:
    assert strip_wake_word("Hey Voice Assistant tell me a joke", "hey voice assistant") == "tell me a joke"


def test_strip_wake_word_with_empty_configured_word_never_matches() -> None:
    assert strip_wake_word("computer", "") is None


def test_strip_wake_word_strips_repeated_wake_words() -> None:
    # A duplicated wake word is a common whisper artifact; it must never
    # leak into the prompt as bare text.
    for utterance in ("Voxa voxa", "Voxa, Voxa!", "voxa. voxa"):
        assert strip_wake_word(utterance, "voxa") == "", utterance


def test_detect_exit_phrase_matches_cancel_phrases() -> None:
    for utterance in ("Never mind.", "CANCEL", "Stop talking", "stop it", "Forget it!", "nevermind"):
        assert detect_exit_phrase(utterance) == "cancel", utterance


def test_detect_exit_phrase_matches_goodbye_phrases() -> None:
    for utterance in ("Goodbye", "Bye", "Goodbye for now", "See you!", "That's all", "thats all", "Done"):
        assert detect_exit_phrase(utterance) == "goodbye", utterance


def test_detect_exit_phrase_ignores_plain_questions() -> None:
    for utterance in (
        "never mind that detail, what's the weather?",
        "good",
        "I'm done with my homework",
        "can you stop the weather updating?",
        "bye, and tell me a joke later",
    ):
        assert detect_exit_phrase(utterance) is None, utterance


def test_detect_exit_phrase_requires_exact_utterance() -> None:
    assert detect_exit_phrase("please cancel that") is None
    assert detect_exit_phrase("goodbye everyone") is None


def test_finalize_conversation_turn_preserves_user_and_adds_assistant() -> None:
    history = deque(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ]
    )

    assert finalize_conversation_turn(history, "Hi there.") is True
    assert list(history) == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there."},
    ]


def test_finalize_conversation_turn_discards_unanswered_user_message() -> None:
    history = deque(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ]
    )

    assert finalize_conversation_turn(history, "") is False
    assert list(history) == [{"role": "system", "content": "Be brief."}]


# ConversationController drives the same pause segmentation as dictation, so
# controller tests feed real chunks and let the 0.25 s queue.get timeout in
# segment_stream advance time (mirroring tests/test_dictation.py).
LOUD_CHUNK = (b"\x00\x10" * 1000, 1000.0)  # 0.0625 s of "speech", above threshold


class FakeWhisper:
    """A stand-in for WhisperService with canned transcripts and call counts."""

    def __init__(self, canned=(), ready: bool = True, delay: float = 0.0) -> None:
        self.canned = list(canned)
        self.calls = 0
        self.error: Exception | None = None
        self.ready = ready
        self.delay = delay

    def transcribe(self, pcm: bytes, language: str = "en") -> str:
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if self.canned:
            return self.canned.pop(0)
        return ""


class _Events:
    """Collects the conversation callbacks so tests can assert on them."""

    def __init__(self) -> None:
        self.woken = 0
        self.prompts: list[str] = []
        self.statuses: list[str] = []
        self.errors: list[str] = []
        self.exits: list[str] = []

    def on_woken(self) -> None:
        self.woken += 1

    def on_prompt(self, text: str) -> None:
        self.prompts.append(text)

    def on_status(self, text: str) -> None:
        self.statuses.append(text)

    def on_error(self, text: str) -> None:
        self.errors.append(text)

    def on_exit(self, kind: str) -> None:
        self.exits.append(kind)


def make_controller(
    wake: FakeWhisper,
    prompt: FakeWhisper,
    wake_word: str = "voxa",
    prompt_timeout: float | None = None,
    idle_status: str = "Awaiting orders…",
) -> tuple[ConversationController, _Events]:
    events = _Events()
    controller = ConversationController(
        wake_whisper=wake,
        prompt_whisper=prompt,
        language="en",
        wake_word=wake_word,
        threshold=450,
        silence_ms=100,
        max_segment_seconds=6.0,
        on_woken=events.on_woken,
        on_prompt=events.on_prompt,
        on_status=events.on_status,
        on_error=events.on_error,
        on_exit=events.on_exit,
        idle_status=idle_status,
    )
    if prompt_timeout is not None:
        controller.PROMPT_TIMEOUT_SECONDS = prompt_timeout
    return controller, events


@pytest.fixture()
def run_controller() -> Callable[ConversationController, ConversationController]:
    def start(controller: ConversationController) -> ConversationController:
        controller.start()
        started.append(controller)
        return controller

    started: list[ConversationController] = []
    yield start
    for controller in started:
        controller.stop()
        if controller.thread is not None:
            controller.thread.join(timeout=5)
        if controller._wake_thread is not None:
            controller._wake_thread.join(timeout=5)


def feed_speech(controller: ConversationController, chunks: int = 6) -> None:
    # Six loud chunks = 0.375 s, past the 0.3 s minimum content for one
    # complete segment; the next 0.25 s queue timeout closes it.
    for _ in range(chunks):
        controller.feed(*LOUD_CHUNK)


def wait_for(condition, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_wake_word_remainder_wakes_and_prompts_in_same_utterance(run_controller):
    wake = FakeWhisper(["Voxa, what's the weather today"])
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    feed_speech(controller)
    assert wait_for(lambda: len(events.prompts) == 1)
    assert events.woken == 1
    assert events.prompts == ["what's the weather today"]
    assert prompt.calls == 0  # the remainder never needs the real model


def test_bare_wake_word_then_prompt_on_next_utterance(run_controller):
    wake = FakeWhisper(["Voxa"])
    prompt = FakeWhisper(["what's the weather tomorrow"])
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    feed_speech(controller)
    assert wait_for(lambda: "Listening for your request…" in events.statuses)
    assert controller.waiting_for_prompt is True
    feed_speech(controller)
    assert wait_for(lambda: events.prompts == ["what's the weather tomorrow"])
    assert events.woken == 1
    assert wake.calls == 1
    assert prompt.calls == 1
    assert controller.waiting_for_prompt is False


def test_armed_prompt_bypasses_wake_word(run_controller):
    wake = FakeWhisper()
    prompt = FakeWhisper(["", "the weather tomorrow"])
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    controller.arm_prompt()
    feed_speech(controller)
    assert wait_for(lambda: prompt.calls == 1)
    assert events.prompts == []
    assert events.woken == 0
    assert controller.waiting_for_prompt is True  # empty transcript keeps waiting
    feed_speech(controller)
    assert wait_for(lambda: events.prompts == ["the weather tomorrow"])
    assert prompt.calls == 2
    assert wake.calls == 0
    assert controller.waiting_for_prompt is False


def test_muted_stream_is_never_treated_as_a_prompt(run_controller):
    # While muted the assistant's reply is playing through the speakers, so
    # nothing the mic picks up may become a prompt — only the wake word
    # matters, and this transcript doesn't contain it (echo guard).
    wake = FakeWhisper(["what's the weather"])
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    controller.mute()
    feed_speech(controller)
    assert wait_for(lambda: wake.calls >= 1)
    assert events.prompts == [] and events.woken == 0
    assert prompt.calls == 0
    assert controller.muted


def test_cancel_ends_turn_but_keeps_listening(run_controller):
    wake = FakeWhisper(["cancel", "Voxa"])
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    feed_speech(controller)
    assert wait_for(lambda: events.exits == ["cancel"])
    assert events.prompts == []
    assert controller.waiting_for_prompt is False
    assert controller.thread is not None and controller.thread.is_alive()
    feed_speech(controller)
    assert wait_for(lambda: events.woken == 1)


def test_exit_phrase_is_checked_even_in_prompt_phase(run_controller):
    wake = FakeWhisper()
    prompt = FakeWhisper(["stop talking"])
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    controller.arm_prompt()
    feed_speech(controller)
    assert wait_for(lambda: events.exits == ["cancel"])
    assert events.prompts == []
    assert controller.waiting_for_prompt is False
    assert controller.thread is not None and controller.thread.is_alive()


def test_goodbye_stops_the_worker(run_controller):
    wake = FakeWhisper(["Goodbye"])
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    feed_speech(controller)
    assert wait_for(lambda: events.exits == ["goodbye"])
    assert controller.thread is not None
    controller.thread.join(timeout=5)
    assert not controller.thread.is_alive()
    feed_speech(controller)
    time.sleep(0.5)
    assert events.prompts == [] and events.woken == 0


def test_prompt_idle_timeout_reports_and_resets(run_controller):
    wake = FakeWhisper(["Voxa"])
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt, prompt_timeout=1.0)
    run_controller(controller)
    feed_speech(controller)
    assert wait_for(lambda: "Listening for your request…" in events.statuses)
    assert wait_for(
        lambda: "Didn't catch that — say “voxa” again." in events.statuses,
        timeout=5.0,
    )
    assert controller.waiting_for_prompt is False


def test_transcription_error_is_reported_and_worker_survives(run_controller):
    wake = FakeWhisper()
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    wake.error = RuntimeError("model crash")
    feed_speech(controller)
    assert wait_for(lambda: events.errors == ["model crash"])
    assert events.woken == 0
    wake.error = None
    wake.canned.append("Voxa")
    feed_speech(controller)
    assert wait_for(lambda: events.woken == 1)
    assert controller.thread is not None and controller.thread.is_alive()


def test_wake_word_while_muted_wakes_and_unmutes(run_controller):
    # The bare wake word said over a reply wakes the controller and hands the
    # very next utterance to the prompt phase, unmuted.
    wake = FakeWhisper(["Voxa"])
    prompt = FakeWhisper(["what's the weather tomorrow"])
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    controller.mute()
    feed_speech(controller)
    assert wait_for(lambda: controller.waiting_for_prompt is True)
    assert wait_for(lambda: not controller.muted)
    assert events.woken == 1
    assert "Listening for your request…" in events.statuses
    feed_speech(controller)
    assert wait_for(lambda: events.prompts == ["what's the weather tomorrow"])
    assert events.woken == 1
    assert prompt.calls == 1
    assert controller.waiting_for_prompt is False


def test_wake_word_with_remainder_while_muted_prompts_directly(run_controller):
    wake = FakeWhisper(["Voxa, what's the time"])
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    controller.mute()
    feed_speech(controller)
    assert wait_for(lambda: events.prompts == ["what's the time"])
    assert wait_for(lambda: not controller.muted)
    assert events.woken == 1
    assert controller.waiting_for_prompt is False
    assert prompt.calls == 0  # the remainder never needs the real model


def test_repeated_wake_word_is_treated_as_bare_wake(run_controller):
    # Whisper artifacts like "voxa voxa" must wake but not become a prompt.
    wake = FakeWhisper(["Voxa voxa"])
    prompt = FakeWhisper(["the weather today"])
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    feed_speech(controller)
    assert wait_for(lambda: "Listening for your request…" in events.statuses)
    assert controller.waiting_for_prompt is True
    assert events.prompts == []
    assert wake.calls == 1 and prompt.calls == 0
    feed_speech(controller)
    assert wait_for(lambda: events.prompts == ["the weather today"])
    assert prompt.calls == 1
    assert controller.waiting_for_prompt is False


def test_repeated_wake_word_while_muted_is_a_bare_wake(run_controller):
    wake = FakeWhisper(["Voxa voxa"])
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    controller.mute()
    feed_speech(controller)
    assert wait_for(lambda: controller.waiting_for_prompt is True)
    assert wait_for(lambda: not controller.muted)
    assert events.prompts == []
    assert events.woken == 1
    assert prompt.calls == 0


def test_unmute_discards_stale_muted_audio(run_controller):
    # Audio still queued while muted (possibly the reply's own speaker echo)
    # is dropped on unmute, so it can never become a spurious wake.
    wake = FakeWhisper(canned=["hello there"], delay=1.0)
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    controller.mute()
    feed_speech(controller)
    assert wait_for(lambda: wake.calls == 1)  # busy transcribing the first segment
    controller.feed(*LOUD_CHUNK)  # arrives while the worker is blocked
    assert controller._wake_queue.qsize() >= 1
    controller.unmute()
    assert controller._wake_queue.qsize() == 0
    time.sleep(1.5)  # let the blocked transcribe finish and the loop settle
    assert events.woken == 0 and events.prompts == []
    assert prompt.calls == 0


def test_muted_stream_uses_fallback_model_when_wake_model_not_ready(run_controller):
    wake = FakeWhisper(ready=False)
    prompt = FakeWhisper(["Voxa"])
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    assert controller._muted_whisper is prompt
    controller.mute()
    feed_speech(controller)
    assert wait_for(lambda: events.woken == 1)
    assert prompt.calls == 1
    assert wake.calls == 0


def test_muted_stream_upgrades_to_wake_model_when_it_becomes_ready(run_controller):
    # The tiny model may finish loading after construction (the fallback was
    # used at build time); the muted worker must switch to it when ready.
    wake = FakeWhisper(ready=False)
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    assert controller._muted_whisper is prompt  # fallback while the tiny model loads
    controller.mute()
    wake.canned.append("Voxa")
    wake.ready = True  # the tiny model finished loading while muted
    feed_speech(controller)
    assert wait_for(lambda: events.woken == 1)
    assert wake.calls == 1
    assert prompt.calls == 0
    assert wait_for(lambda: not controller.muted)


def test_wake_model_error_while_muted_is_reported_and_worker_survives(run_controller):
    wake = FakeWhisper()
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    controller.mute()
    wake.error = RuntimeError("wake model crash")
    feed_speech(controller)
    assert wait_for(lambda: events.errors == ["wake model crash"])
    assert events.woken == 0
    wake.error = None
    wake.canned.append("Voxa")
    feed_speech(controller)
    assert wait_for(lambda: events.woken == 1)
    assert wait_for(lambda: not controller.muted)
    assert controller._wake_thread is not None and controller._wake_thread.is_alive()


def test_empty_transcript_resets_to_idle_status(run_controller):
    # Room noise transcribed to nothing must not leave "Transcribing…" stuck.
    wake = FakeWhisper([""])
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    feed_speech(controller)
    assert wait_for(lambda: events.statuses and events.statuses[-1] == "Awaiting orders…")
    assert events.prompts == [] and events.woken == 0
    assert controller.thread is not None and controller.thread.is_alive()


def test_utterance_without_wake_word_resets_to_idle_status(run_controller):
    # Heard speech with no wake word is nobody talking to us: back to idle.
    wake = FakeWhisper(["what time is it"])
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    feed_speech(controller)
    assert wait_for(lambda: "Awaiting orders…" in events.statuses)
    assert events.prompts == [] and events.woken == 0
    assert controller.thread is not None and controller.thread.is_alive()


def test_custom_idle_status_is_used(run_controller):
    wake = FakeWhisper([""])
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt, idle_status="Stand by…")
    run_controller(controller)
    feed_speech(controller)
    assert wait_for(lambda: events.statuses and events.statuses[-1] == "Stand by…")
