from __future__ import annotations

from voice2text.listening import (
    CAP_SLACK_SECONDS,
    CLARIFY_CONFIDENCE,
    MIN_PROMPT_WORDS,
    clarification_prompt,
    ends_request,
    is_follow_up,
    needs_clarification,
    resolve_follow_up,
)


def test_ends_request_accepts_a_segment_cut_by_a_pause() -> None:
    # 1.0 s of speech, a 0.1 s silence setting, a 5 s cap: the user stopped.
    assert ends_request(1.0, 0.1, 5.0) is True


def test_ends_request_rejects_a_segment_that_ran_to_the_length_cap() -> None:
    for segment_seconds in (5.0, 5.02, 5.4):
        assert ends_request(segment_seconds, 0.1, 5.0) is False, segment_seconds


def test_ends_request_treats_the_cap_boundary_with_a_small_slack() -> None:
    # The audio feed only checks the cap after whole chunks, so a cap-cut
    # segment can land a hair under the limit and must still count as cut off.
    assert ends_request(5.0 - CAP_SLACK_SECONDS / 2.0, 0.1, 5.0) is False
    # Comfortably inside the cap is a real pause.
    assert ends_request(5.0 - 0.5, 0.1, 5.0) is True


def test_ends_request_rejects_a_fragment_shorter_than_the_pause_floor() -> None:
    # The floor matches segment_stream's min_content: silence_seconds / 2 clamped
    # into 0.3..0.8, so a stray syllable never becomes a request.
    assert ends_request(0.2, 0.1, 5.0) is False
    assert ends_request(0.3, 0.1, 5.0) is True
    assert ends_request(0.6, 1.5, 5.0) is False
    assert ends_request(0.8, 1.5, 5.0) is True


def test_ends_request_ignores_a_non_positive_cap() -> None:
    assert ends_request(30.0, 0.1, 0.0) is True


def test_needs_clarification_flags_low_confidence() -> None:
    assert needs_clarification("what time is it", CLARIFY_CONFIDENCE - 0.01) is True
    assert needs_clarification("what time is it", CLARIFY_CONFIDENCE) is False
    assert needs_clarification("what time is it", 1.0) is False


def test_needs_clarification_flags_content_free_transcripts() -> None:
    assert needs_clarification("", 1.0) is True
    assert needs_clarification("   ", 0.9) is True
    assert needs_clarification("hmm", 1.0) is True
    assert needs_clarification("computer", 1.0) is True
    assert len("ok".split()) < MIN_PROMPT_WORDS


def test_needs_clarification_accepts_a_bare_follow_up_as_intentional() -> None:
    # Short on purpose, not a mis-hear: "why?" is resolved against the previous
    # request instead of being asked about.
    assert needs_clarification("why?", 1.0) is False
    assert needs_clarification("and again?", 1.0) is False


def test_needs_clarification_still_flags_a_low_confidence_follow_up() -> None:
    assert needs_clarification("why?", 0.2) is True


def test_is_follow_up_matches_bare_follow_ups() -> None:
    for text in (
        "why?",
        "Why",
        "and again?",
        "what about that",
        "how?",
        "more details",
        "say it again",
        "what else",
        "explain that",
    ):
        assert is_follow_up(text), text


def test_is_follow_up_rejects_requests_that_carry_their_own_topic() -> None:
    for text in (
        "why is the sky blue",
        "what's the weather today",
        "tell me a joke",
        "what time is it",
        "and then run the long tail of this sentence",
        "why does the moon move",
    ):
        assert not is_follow_up(text), text


def test_is_follow_up_rejects_empty_text() -> None:
    assert not is_follow_up("")
    assert not is_follow_up("?!")


def test_clarification_prompt_names_the_heard_text_and_the_wake_word() -> None:
    prompt = clarification_prompt("weather fro the mrning", "computer")
    assert "weather fro the mrning" in prompt
    assert "computer" in prompt
    assert prompt.endswith("clearly.")


def test_clarification_prompt_collapses_whitespace_in_the_heard_text() -> None:
    assert "weater  cold" not in clarification_prompt("weater   cold", "computer")
    assert "weater cold" in clarification_prompt("weater   cold", "computer")


def test_clarification_prompt_without_heard_text_still_gives_the_wake_word() -> None:
    prompt = clarification_prompt("   ", "voice assistant")
    assert "voice assistant" in prompt
    assert "  " not in prompt


def test_resolve_follow_up_folds_the_follow_up_into_the_previous_request() -> None:
    resolved = resolve_follow_up("what time is it", "why?")
    assert resolved.startswith("what time is it")
    assert "why?" in resolved
    assert resolved == 'what time is it — and the user then asked: "why?"'


def test_resolve_follow_up_without_a_previous_request_returns_the_follow_up() -> None:
    assert resolve_follow_up("", "why?") == "why?"
    assert resolve_follow_up("what time is it", "") == "what time is it"


def test_resolve_follow_up_collapses_whitespace() -> None:
    assert resolve_follow_up("  what   time  is it  ", "  why  ") == (
        'what time is it — and the user then asked: "why"'
    )
