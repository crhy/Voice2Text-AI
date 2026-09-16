from __future__ import annotations

import json
from pathlib import Path

from voxa.config import ConfigStore, Settings


def test_defaults_when_config_is_missing(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    assert store.load() == Settings()


def test_round_trip_and_normalization(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    settings = Settings(tts_rate=999, silence_ms=10, ollama_url="http://localhost:11434/")
    store.save(settings)

    loaded = store.load()
    assert loaded.tts_rate == 350
    assert loaded.silence_ms == 300
    assert loaded.ollama_url == "http://localhost:11434"


def test_legacy_keys_are_migrated_in_memory(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"selected_model": "qwen3:8b", "whisper_model": "small"}), encoding="utf-8")
    loaded = ConfigStore(path).load()
    assert loaded.ollama_model == "qwen3:8b"
    assert loaded.whisper_model == "small"


def test_appearance_and_voice_are_normalized(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    settings = Settings(appearance="sepia", tts_voice="")
    store.save(settings)

    loaded = store.load()
    assert loaded.appearance == "system"
    assert loaded.tts_voice == "en-US-AriaNeural"


def test_wake_word_falls_back_to_default_when_blank(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.save(Settings(wake_word="   "))

    assert store.load().wake_word == "voxa"


def test_legacy_config_dir_is_migrated(tmp_path: Path) -> None:
    old_dir = tmp_path / "config" / "voice2text-ai"
    old_dir.mkdir(parents=True)
    (old_dir / "config.json").write_text(json.dumps({"tts_rate": 200}), encoding="utf-8")

    store = ConfigStore(tmp_path / "config" / "voxa" / "config.json")
    loaded = store.load()

    assert loaded.tts_rate == 200
    assert (tmp_path / "config" / "voxa" / "config.json").is_file()
