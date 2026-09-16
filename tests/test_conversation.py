from __future__ import annotations

from voice2text.conversation import detect_exit_phrase, strip_wake_word


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
