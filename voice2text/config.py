from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


def _clean_dirs(values: list[str]) -> list[str]:
    """Trim the allowlist, drop blank entries and keep it in order.

    A blank entry is never a wildcard: an empty allowlist means nothing may
    be touched, so entries that resolve to nothing are removed rather than
    left to mean "everything".
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    return cleaned


@dataclass(slots=True)
class Settings:
    microphone_id: str = ""
    microphone_name: str = ""
    whisper_model: str = "base"
    ollama_model: str = ""
    ollama_url: str = "http://127.0.0.1:11434"
    language: str = "en"
    tts_rate: int = 180
    tts_voice: str = "en-US-AriaNeural"
    appearance: str = "system"
    auto_speak: bool = True
    silence_ms: int = 900
    voice_threshold: int = 450
    max_segment_seconds: float = 6.0
    wake_word: str = "computer"
    # OpenCode continuation is opt-in per directory: the feature stays off
    # until the user turns it on *and* names a project inside an allowed
    # directory, and auto-run needs a second switch on top of that.
    opencode_enabled: bool = False
    opencode_auto_run: bool = False
    opencode_project: str = ""
    opencode_allowed_dirs: list[str] = field(default_factory=list)
    opencode_model: str = ""

    def normalized(self) -> Settings:
        self.tts_rate = max(80, min(350, int(self.tts_rate)))
        if self.appearance not in {"system", "light", "dark"}:
            self.appearance = "system"
        self.tts_voice = (self.tts_voice or "en-US-AriaNeural").strip()
        self.silence_ms = max(300, min(4000, int(self.silence_ms)))
        self.voice_threshold = max(50, min(5000, int(self.voice_threshold)))
        self.max_segment_seconds = max(2.0, min(20.0, float(self.max_segment_seconds)))
        self.ollama_url = self.ollama_url.rstrip("/") or "http://127.0.0.1:11434"
        self.language = (self.language or "en").strip()[:16]
        self.wake_word = (self.wake_word or "computer").strip()[:32] or "computer"
        self.opencode_enabled = bool(self.opencode_enabled)
        self.opencode_auto_run = bool(self.opencode_auto_run)
        self.opencode_project = self.opencode_project.strip()
        # Keep the model as the CLI's own "provider/model" text; anything the
        # user typed is passed through unchanged so a new provider naming
        # scheme does not get rejected by this app.
        self.opencode_model = self.opencode_model.strip()[:80]
        self.opencode_allowed_dirs = _clean_dirs(self.opencode_allowed_dirs)
        return self


class ConfigStore:
    """Small JSON settings store using the XDG configuration directory."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
            path = base / "voice2text-ai" / "config.json"
        self.path = path
        self.legacy_path = Path.home() / ".voice_config.json"

    def load(self) -> Settings:
        self._migrate_legacy_file()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return Settings()
        if not isinstance(payload, dict):
            return Settings()

        accepted = {field.name for field in fields(Settings)}
        clean: dict[str, Any] = {key: value for key, value in payload.items() if key in accepted}

        # Compatibility with the pre-0.4 configuration keys.
        if "selected_model" in payload and "ollama_model" not in clean:
            clean["ollama_model"] = payload["selected_model"]
        if "microphone_name" in payload:
            clean["microphone_name"] = payload["microphone_name"]

        try:
            return Settings(**clean).normalized()
        except (TypeError, ValueError):
            return Settings()

    def save(self, settings: Settings) -> None:
        settings.normalized()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(asdict(settings), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.path)

    def _migrate_legacy_file(self) -> None:
        if self.path.exists() or not self.legacy_path.exists():
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.legacy_path, self.path)
        except OSError:
            # Migration is best-effort; load() will fall back to defaults.
            return
