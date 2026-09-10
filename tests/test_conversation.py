from __future__ import annotations

from voice2text.conversation import strip_wake_word


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
