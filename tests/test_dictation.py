from __future__ import annotations

import queue
import threading

from voice2text.dictation import segment_stream

LOUD_CHUNK = (b"\x00\x10" * 1000, 1000.0)  # 2000 bytes, above any test threshold
QUIET_CHUNK = (b"\x00\x00" * 1000, 0.0)


def test_segment_stream_yields_one_segment_after_a_pause() -> None:
    q: queue.Queue = queue.Queue()
    for _ in range(10):
        q.put(LOUD_CHUNK)

    segments = list(
        segment_stream(
            q,
            threading.Event(),
            threshold=500,
            silence_seconds=0.1,
            max_segment_seconds=5.0,
            idle_timeout_seconds=0.3,
        )
    )

    assert len(segments) == 1
    assert len(segments[0]) == 10 * len(LOUD_CHUNK[0])


def test_segment_stream_ignores_audio_below_threshold() -> None:
    q: queue.Queue = queue.Queue()
    for _ in range(5):
        q.put(QUIET_CHUNK)

    segments = list(
        segment_stream(
            q,
            threading.Event(),
            threshold=500,
            silence_seconds=0.1,
            max_segment_seconds=5.0,
            idle_timeout_seconds=0.3,
        )
    )

    assert segments == []


def test_segment_stream_stops_immediately_when_stop_event_is_set() -> None:
    q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    stop_event.set()

    segments = list(
        segment_stream(
            q,
            stop_event,
            threshold=500,
            silence_seconds=0.1,
            max_segment_seconds=5.0,
        )
    )

    assert segments == []


def test_segment_stream_calls_on_idle_timeout() -> None:
    q: queue.Queue = queue.Queue()
    called = threading.Event()

    segments = list(
        segment_stream(
            q,
            threading.Event(),
            threshold=500,
            silence_seconds=0.1,
            max_segment_seconds=5.0,
            idle_timeout_seconds=0.2,
            on_idle_timeout=called.set,
        )
    )

    assert segments == []
    assert called.is_set()
