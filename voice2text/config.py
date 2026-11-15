from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


# The continuation feature used to drive the OpenCode CLI; it drives pi now. These
# are the settings keys of that earlier era, read once and carried over so a
# user's existing choices are not lost.
PI_KEY_MIGRATION = {
    "opencode_enabled": "pi_enabled",
    "opencode_auto_run": "pi_auto_run",
    "opencode_project": "pi_project",
    "opencode_allowed_dirs": "pi_allowed_dirs",
    "opencode_model": "pi_model",
}


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
    # pi continuation is opt-in per directory: the feature stays off until the
    # user turns it on *and* names a project inside an allowed directory, and
    # auto-run needs a second switch on top of that.
    pi_enabled: bool = False
    pi_auto_run: bool = False
    pi_project: str = ""
    pi_allowed_dirs: list[str] = field(default_factory=list)
    pi_provider: str = ""
    pi_model: str = ""

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
        self.pi_enabled = bool(self.pi_enabled)
        self.pi_auto_run = bool(self.pi_auto_run)
        self.pi_project = self.pi_project.strip()
        # pi picks its model with ``--provider <name> --model <id>``; both stay
        # free text here, so a provider naming scheme this app does not know
        # about is passed through instead of rejected.
        self.pi_provider = self.pi_provider.strip()[:40]
        self.pi_model = self.pi_model.strip()[:80]
        self.pi_allowed_dirs = _clean_dirs(self.pi_allowed_dirs)
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
        for legacy_key, new_key in PI_KEY_MIGRATION.items():
            if new_key not in clean and legacy_key in payload:
                clean[new_key] = payload[legacy_key]

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
