from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass


class OllamaError(RuntimeError):
    pass


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Drop a reasoning model's scratchpad from its reply.

    Qwen3, DeepSeek-R1 and other thinking models wrap their working in
    <think>...</think>. Ollama serves some GGUF builds with the opening tag
    already consumed by the chat template, leaving a bare closing tag, so
    anything ahead of the last </think> is treated as reasoning rather than
    insisting on a matched pair. Text with no closing tag is left alone.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1]
    return cleaned.strip()


@dataclass(slots=True, frozen=True)
class ModelInfo:
    name: str
    size_bytes: int


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _list_models_payload(self) -> list[dict]:
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Could not connect to Ollama: {exc}") from exc
        models = payload.get("models", []) if isinstance(payload, dict) else []
        return [item for item in models if isinstance(item, dict)]

    def list_models(self) -> list[str]:
        names = [item.get("name", "") for item in self._list_models_payload()]
        return sorted(name for name in names if name)

    def list_models_detailed(self) -> list[ModelInfo]:
        infos = [
            ModelInfo(name=item["name"], size_bytes=int(item.get("size", 0) or 0))
            for item in self._list_models_payload()
            if item.get("name")
        ]
        return sorted(infos, key=lambda info: info.name)

    def generate_stream(
        self,
        *,
        model: str,
        prompt: str,
        cancel_event: threading.Event,
        on_chunk: Callable[[str], None],
        num_predict: int = 768,
    ) -> str:
        if not model:
            raise OllamaError("No Ollama model is selected.")
        if not prompt.strip():
            raise OllamaError("There is no text to send.")

        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt.strip(),
                "stream": True,
                "options": {"num_predict": num_predict},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        chunks: list[str] = []
        stream_error: str | None = None
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                for raw_line in response:
                    if cancel_event.is_set():
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    # Newer Ollama servers (0.1.33+) may serve streaming responses
                    # as Server-Sent Events: skip "event:" lines and strip the
                    # "data:" prefix before parsing the JSON payload.
                    if line.startswith("event:"):
                        continue
                    if line.startswith("data:"):
                        line = line.removeprefix("data:").strip()
                        if not line:
                            continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("error"):
                        # Record but do not raise mid-iteration: let the loop
                        # finish so any already-delivered chunks stay in the
                        # UI, and raise after the connection is closed.
                        stream_error = str(event["error"])
                        break
                    text = event.get("response", "")
                    if text:
                        chunks.append(text)
                        on_chunk(text)
                    if event.get("done"):
                        break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

        if stream_error is not None:
            raise OllamaError(stream_error)

        return "".join(chunks).strip()

    def pull_model(
        self,
        model: str,
        *,
        cancel_event: threading.Event,
        on_progress: Callable[[str, int, int], None],
    ) -> None:
        """Pull ``model``, reporting ``(status, completed_bytes, total_bytes)`` as it downloads."""
        name = model.strip()
        if not name:
            raise OllamaError("No model name was given.")

        payload = json.dumps({"name": name, "stream": True}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/pull",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                for raw_line in response:
                    if cancel_event.is_set():
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("error"):
                        raise OllamaError(str(event["error"]))
                    status = str(event.get("status", ""))
                    total = int(event.get("total", 0) or 0)
                    completed = int(event.get("completed", 0) or 0)
                    on_progress(status, completed, total)
                    if status == "success":
                        break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise OllamaError(f"Pulling {name} failed: {exc}") from exc

    def delete_model(self, model: str) -> None:
        name = model.strip()
        if not name:
            raise OllamaError("No model name was given.")

        payload = json.dumps({"name": name}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/delete",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="DELETE",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OllamaError(f"Could not delete {name}: {exc}") from exc
