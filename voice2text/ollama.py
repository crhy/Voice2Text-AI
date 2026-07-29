from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def list_models(self) -> list[str]:
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Could not connect to Ollama: {exc}") from exc

        models = payload.get("models", []) if isinstance(payload, dict) else []
        names = [item.get("name", "") for item in models if isinstance(item, dict)]
        return sorted(name for name in names if name)

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
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
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

        return "".join(chunks).strip()
