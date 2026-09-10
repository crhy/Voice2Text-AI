from __future__ import annotations

import io
import threading
from unittest.mock import patch

import pytest

from voice2text.ollama import OllamaClient, OllamaError, strip_reasoning


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_list_models_sorts_names() -> None:
    response = FakeResponse(b'{"models":[{"name":"zeta"},{"name":"alpha"}]}')
    with patch("urllib.request.urlopen", return_value=response):
        assert OllamaClient().list_models() == ["alpha", "zeta"]


def test_list_models_detailed_returns_sizes_sorted_by_name() -> None:
    response = FakeResponse(
        b'{"models":[{"name":"zeta","size":2000},{"name":"alpha","size":1000}]}'
    )
    with patch("urllib.request.urlopen", return_value=response):
        infos = OllamaClient().list_models_detailed()
    assert [(info.name, info.size_bytes) for info in infos] == [("alpha", 1000), ("zeta", 2000)]


def test_pull_model_reports_progress_and_stops_on_success() -> None:
    response = FakeResponse(
        b'{"status":"pulling manifest"}\n'
        b'{"status":"downloading","completed":50,"total":100}\n'
        b'{"status":"success"}\n'
    )
    events: list[tuple[str, int, int]] = []
    with patch("urllib.request.urlopen", return_value=response):
        OllamaClient().pull_model(
            "qwen2.5:0.5b",
            cancel_event=threading.Event(),
            on_progress=lambda status, completed, total: events.append((status, completed, total)),
        )
    assert events == [
        ("pulling manifest", 0, 0),
        ("downloading", 50, 100),
        ("success", 0, 0),
    ]


def test_pull_model_raises_on_stream_error() -> None:
    response = FakeResponse(b'{"error":"model not found"}\n')
    with patch("urllib.request.urlopen", return_value=response), pytest.raises(OllamaError):
        OllamaClient().pull_model(
            "does-not-exist",
            cancel_event=threading.Event(),
            on_progress=lambda *_args: None,
        )


def test_pull_model_rejects_blank_name() -> None:
    with pytest.raises(OllamaError):
        OllamaClient().pull_model("  ", cancel_event=threading.Event(), on_progress=lambda *_args: None)


def test_delete_model_sends_request() -> None:
    response = FakeResponse(b"")
    with patch("urllib.request.urlopen", return_value=response) as mocked:
        OllamaClient().delete_model("qwen2.5:0.5b")
    request = mocked.call_args[0][0]
    assert request.get_method() == "DELETE"
    assert request.full_url.endswith("/api/delete")


def test_streaming_response_calls_chunk_callback() -> None:
    response = FakeResponse(
        b'{"response":"Hello","done":false}\n'
        b'{"response":" world","done":false}\n'
        b'{"done":true}\n'
    )
    chunks: list[str] = []
    with patch("urllib.request.urlopen", return_value=response):
        answer = OllamaClient().generate_stream(
            model="test",
            prompt="hello",
            cancel_event=threading.Event(),
            on_chunk=chunks.append,
        )
    assert answer == "Hello world"
    assert chunks == ["Hello", " world"]


def test_strip_reasoning_removes_a_matched_think_block():
    assert strip_reasoning("<think>weighing it up</think>\n\nGNOME.") == "GNOME."


def test_strip_reasoning_handles_an_unopened_closing_tag():
    # What Ollama actually served for a Qwen3 GGUF: the opening tag was already
    # consumed by the chat template, leaving the scratchpad bare.
    reply = 'We need answer user. Simple. Need final concise.\n</think>\n\nGNOME, KDE Plasma, and XFCE.'
    assert strip_reasoning(reply) == "GNOME, KDE Plasma, and XFCE."


def test_strip_reasoning_leaves_an_ordinary_reply_alone():
    assert strip_reasoning("GNOME, KDE Plasma, and XFCE.") == "GNOME, KDE Plasma, and XFCE."


def test_strip_reasoning_keeps_the_last_answer_when_several_blocks_appear():
    assert strip_reasoning("<think>a</think>mid</think>final") == "final"
