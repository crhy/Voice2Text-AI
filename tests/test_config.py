from __future__ import annotations

import json
from pathlib import Path

from voice2text.config import ConfigStore, Settings


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

    assert store.load().wake_word == "computer"


def test_pi_settings_default_to_off(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    loaded = store.load()

    assert loaded.pi_enabled is False
    assert loaded.pi_auto_run is False
    assert loaded.pi_project == ""
    assert loaded.pi_allowed_dirs == []
    assert loaded.pi_provider == ""
    assert loaded.pi_model == ""


def test_pi_settings_round_trip_and_normalise(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.save(
        Settings(
            pi_enabled=True,
            pi_auto_run=True,
            pi_project="  /home/user/project  ",
            pi_allowed_dirs=[" /home/user ", "", " /home/user ", "/opt/projects"],
            pi_provider=" ollama ",
            pi_model=" anthropic/claude-4.5 ",
        )
    )
    loaded = store.load()

    assert loaded.pi_enabled is True
    assert loaded.pi_auto_run is True
    assert loaded.pi_project == "/home/user/project"
    assert loaded.pi_allowed_dirs == ["/home/user", "/opt/projects"]
    assert loaded.pi_provider == "ollama"
    assert loaded.pi_model == "anthropic/claude-4.5"


def test_continuation_settings_migrate_from_the_earlier_names(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "opencode_enabled": True,
                "opencode_auto_run": True,
                "opencode_project": "/home/user/project",
                "opencode_allowed_dirs": ["/home/user"],
                "opencode_model": "ollama/qwen3:8b",
            }
        ),
        encoding="utf-8",
    )
    loaded = ConfigStore(path).load()

    assert loaded.pi_enabled is True
    assert loaded.pi_auto_run is True
    assert loaded.pi_project == "/home/user/project"
    assert loaded.pi_allowed_dirs == ["/home/user"]
    assert loaded.pi_model == "ollama/qwen3:8b"


def test_new_continuation_keys_win_over_the_old_ones(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "opencode_enabled": True,
                "opencode_project": "/old",
                "pi_enabled": False,
                "pi_project": "/new",
            }
        ),
        encoding="utf-8",
    )
    loaded = ConfigStore(path).load()

    assert loaded.pi_enabled is False
    assert loaded.pi_project == "/new"
