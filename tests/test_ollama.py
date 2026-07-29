from __future__ import annotations

import io
import threading
from unittest.mock import patch

from voice2text.ollama import OllamaClient


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_list_models_sorts_names() -> None:
    response = FakeResponse(b'{"models":[{"name":"zeta"},{"name":"alpha"}]}')
    with patch("urllib.request.urlopen", return_value=response):
        assert OllamaClient().list_models() == ["alpha", "zeta"]


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
