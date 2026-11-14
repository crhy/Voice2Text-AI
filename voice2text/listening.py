"""Deciding when a captured utterance is a finished request.

Pure helpers — no I/O, no threads, no model calls — used by
:class:`voice2text.conversation.ConversationController` to tell three things
apart: the user paused and their request is ready to send, the user is still
talking and the length cap merely cut them off, and the user said something
too quiet or too short to be trusted as a request at all. Keeping the rules
here makes the conversational flows scriptable in tests without audio, a
Whisper model or timing sleeps.
"""

from __future__ import annotations

import re

# A segment counts as cap-cut once it reaches the configured maximum length.
# The tolerance absorbs the granularity of the audio feed: ``segment_stream``
# only checks the cap after whole chunks, so a segment can land a hair under
# (or over) the limit.
CAP_SLACK_SECONDS = 0.05

# Below this confidence a transcript is treated as a guess rather than as
# something the user actually said. Whisper's ``avg_logprob``/``no_speech_prob``
# are mapped into this 0..1 scale by
# :meth:`voice2text.transcription.WhisperService.transcribe_with_confidence`.
CLARIFY_CONFIDENCE = 0.45

# An utterance shorter than this is usually a mis-hear or a stray syllable, so
# it is asked about instead of being sent to the model as a question — unless
# it is a recognisable follow-up, which is deliberately short.
MIN_PROMPT_WORDS = 2

# Words that, on their own, only make sense as a continuation of the request
# the assistant already has in hand.
_FOLLOW_UP_WORDS = frozenset(
    {
        "why",
        "how",
        "where",
        "when",
        "who",
        "what",
        "which",
        "whichever",
        "again",
        "more",
        "else",
        "other",
        "others",
        "details",
        "detail",
        "longer",
        "shorter",
        "slower",
        "louder",
        "deeper",
        "simpler",
        "please",
        "say",
        "repeat",
        "rephrase",
        "explain",
        "elaborate",
        "continue",
    }
)

# Words that carry no topic of their own: articles, copulas, prepositions and
# vague stand-ins. A follow-up is made only of these plus at least one word
# above; a sentence containing a topic word ("why is the sky blue") is a
# request in its own right. Personal pronouns are deliberately *not* function
# words, so "how are you" stays a request rather than a follow-up.
_FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "any",
        "some",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "it",
        "its",
        "one",
        "ones",
        "thing",
        "things",
        "stuff",
        "way",
        "part",
        "point",
        "case",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "about",
        "of",
        "for",
        "to",
        "in",
        "on",
        "with",
        "from",
        "by",
        "and",
        "or",
        "not",
        "then",
        "so",
        "just",
        "really",
        "still",
        "much",
        "lot",
        "bit",
        "few",
        "own",
        "same",
        "instead",
        "back",
    }
)

# A follow-up is never a long sentence, whatever the words happen to be.
_MAX_FOLLOW_UP_WORDS = 4


def _words(text: str) -> list[str]:
    """Casefolded alphanumeric words of ``text``, punctuation collapsed away."""
    return re.sub(r"[^a-z0-9'\s]", " ", text.casefold()).split()


def ends_request(
    segment_seconds: float,
    silence_seconds: float,
    max_segment_seconds: float,
) -> bool:
    """Return whether a captured segment is a finished request.

    ``True`` means the user paused, so the text can be sent. ``False`` means
    the user is still talking: either the segment ran to ``max_segment_seconds``
    (the length cap cut a mid-sentence user off) or it is too short to be a
    request at all. The short-segment floor mirrors the ``min_content`` rule
    :func:`voice2text.dictation.segment_stream` uses to decide when a pause
    counts as a boundary.
    """
    if max_segment_seconds <= 0:
        # No cap configured: only a pause can end the segment.
        return True
    if segment_seconds >= max_segment_seconds - CAP_SLACK_SECONDS:
        return False
    min_content = min(0.8, max(0.3, silence_seconds / 2.0))
    if segment_seconds < min_content:
        return False
    return True


def needs_clarification(text: str, confidence: float) -> bool:
    """Return whether the utterance should be asked about instead of sent.

    Low transcription confidence (a guess) means the assistant should say what
    it heard and ask for a repeat rather than answer a question nobody asked.
    A very short utterance also gets that treatment, because a stray syllable
    transcribes plausibly but carries no intent — unless it is a recognisable
    follow-up, which is short on purpose.
    """
    if not text.strip():
        return True
    if confidence < CLARIFY_CONFIDENCE:
        return True
    if len(text.split()) < MIN_PROMPT_WORDS and not is_follow_up(text):
        return True
    return False


def clarification_prompt(text: str, wake_word: str) -> str:
    """Return the user-facing wording for a clarification.

    Names what was heard (so the user can see the mis-hear) and how to call
    the assistant again; it stays a question and is never presented as an
    answer or as a command the user gave.
    """
    heard = " ".join(text.split())
    if not heard:
        return f"Didn't catch a request — say “{wake_word}” and ask again."
    return (
        f"Heard “{heard}” but that isn't a clear request — "
        f"say “{wake_word}” and ask again, or repeat it more clearly."
    )


def is_follow_up(text: str) -> bool:
    """Return whether the utterance only makes sense after an earlier request.

    Bare follow-ups such as "why?", "and again?" and "what about that" match;
    a sentence that merely contains a follow-up word ("why is the sky blue")
    does not, because it carries its own topic.
    """
    words = _words(text)
    if not words or len(words) > _MAX_FOLLOW_UP_WORDS:
        return False
    if not any(word in _FOLLOW_UP_WORDS for word in words):
        return False
    return all(word in _FOLLOW_UP_WORDS or word in _FUNCTION_WORDS for word in words)


def resolve_follow_up(previous_prompt: str, text: str) -> str:
    """Return the follow-up as a self-contained request.

    The previous prompt is what the assistant was already asked, so the follow-
    up is appended to it as an explicit instruction rather than sent alone:
    "why?" against "what time is it" becomes a question about the time. Returns
    the bare follow-up when there is nothing to resolve it against.
    """
    previous = " ".join(previous_prompt.split()).strip()
    follow_up = " ".join(text.split()).strip()
    if not previous:
        return follow_up
    if not follow_up:
        return previous
    return f'{previous} — and the user then asked: "{follow_up}"'
