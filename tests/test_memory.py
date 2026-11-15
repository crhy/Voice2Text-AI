"""Tests for voice2text.memory (issue #27, first milestone)."""

from __future__ import annotations

import json
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from voice2text.backup import REDACTED_MARKER
from voice2text.config import ConfigStore, Settings
from voice2text.memory import (
    EVAL_CASES,
    MemoryStore,
    MemoryStoreError,
    default_memory_path,
    evaluate,
    main,
)


def _seed(store: MemoryStore) -> None:
    store.record("what film did I watch last night", "The film I named was Dune.")
    store.record("which restaurant do I like", "Try the restaurant Le Cresset.")
    store.record("what time should I wake up", "Wake at 6:30 on weekdays.")
    store.record("where are my holiday photos", "The photos are in the shoebox by the bed.")
    store.record("explain Einstein to a child", "Einstein was born in Ulm.")


@pytest.fixture
def memory_file(tmp_path: Path) -> Path:
    return tmp_path / "memory.jsonl"


def test_default_settings_record_nothing(memory_file: Path) -> None:
    assert Settings().memory_enabled is False

    store = MemoryStore(memory_file, enabled=Settings().memory_enabled)
    assert store.record("what did I say", "nothing at all") is None
    assert not memory_file.exists()
    assert store.load() == []
    assert store.context_for("what did I say") == []


def test_retention_days_are_clamped(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.save(Settings(memory_enabled=True, memory_retention_days=99999))

    loaded = store.load()
    assert loaded.memory_enabled is True
    assert loaded.memory_retention_days == 3650

    store.save(Settings(memory_retention_days=0))
    assert store.load().memory_retention_days == 1


def test_default_path_follows_xdg_data_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert default_memory_path() == tmp_path / "voice2text-ai" / "memory.jsonl"


def test_record_round_trips_through_inspect(memory_file: Path) -> None:
    store = MemoryStore(memory_file, enabled=True)
    written = store.record(
        "what film did I watch last night",
        "The film I named was Dune.",
        provenance={"channel": "conversation", "model": "qwen3:8b"},
    )
    assert written is not None

    hits = store.inspect("film", limit=5)
    assert len(hits) == 1
    assert hits[0]["prompt"] == "what film did I watch last night"
    assert hits[0]["reply"] == "The film I named was Dune."
    assert hits[0]["kind"] == "turn"
    assert hits[0]["provenance"]["model"] == "qwen3:8b"
    assert hits[0]["digest"]

    # The line on disk is JSONL and reloads as the same record.
    lines = memory_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["prompt"] == "what film did I watch last night"


def test_secret_looking_values_come_back_redacted(memory_file: Path) -> None:
    store = MemoryStore(memory_file, enabled=True)
    token = "deadbeefcafebabe1234567890abcdef"
    written = store.record(
        f"my password is {token} and here is sk-proj-Zk9xYq7Lm2Nd",
        "Noted.",
        provenance={"password": "hunter2", "channel": "conversation"},
    )
    assert written is not None
    assert REDACTED_MARKER in written["prompt"]
    assert token not in written["prompt"]
    assert "sk-proj-Zk9xYq7Lm2Nd" not in written["prompt"]
    assert written["provenance"]["password"] == REDACTED_MARKER
    assert written["redacted"] == ["prompt", "provenance.password"]

    hit = store.inspect("password", limit=5)[0]
    assert hit["prompt"] == written["prompt"]
    assert token not in hit["prompt"]

    # Ordinary prose is left alone.
    plain = store.record("where do I keep my holiday photos", "In the shoebox by the bed.")
    assert plain is not None
    assert plain["redacted"] == []
    assert plain["prompt"] == "where do I keep my holiday photos"


def test_record_never_raises_when_the_disk_fails(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")

    store = MemoryStore(blocked / "memory.jsonl", enabled=True)
    assert store.record("hello", "there") is None
    assert store.load() == []

    # A value that cannot be serialised must not escape record() either.
    good = MemoryStore(tmp_path / "memory.jsonl", enabled=True)
    assert good.record("hello", "there", provenance={"clock": object()}) is None


def test_export_writes_json_that_reloads(memory_file: Path, tmp_path: Path) -> None:
    store = MemoryStore(memory_file, enabled=True)
    _seed(store)
    # A record written before the redaction policy existed: export must scrub
    # on the way out as well as on the way in.
    memory_file.write_text(
        memory_file.read_text(encoding="utf-8")
        + json.dumps(
            {
                "when": datetime.now(UTC).isoformat(timespec="seconds"),
                "kind": "turn",
                "prompt": "my password is deadbeefcafebabe1234567890abcdef",
                "reply": "Noted.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    export_path = tmp_path / "out" / "memory-export.json"
    summary = store.export(export_path)

    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["count"] == 6 == summary["count"]
    assert len(payload["records"]) == 6
    assert payload["format"] == 1
    assert summary["redacted_keys"] == ["prompt"]
    assert "deadbeefcafebabe1234567890abcdef" not in export_path.read_text(encoding="utf-8")
    assert REDACTED_MARKER in export_path.read_text(encoding="utf-8")


def test_delete_removes_exactly_what_the_query_names(memory_file: Path) -> None:
    store = MemoryStore(memory_file, enabled=True)
    _seed(store)

    assert store.delete("restaurant") == 1
    assert store.count() == 4
    assert store.inspect("restaurant", limit=5) == []

    assert store.delete("photos") == 1
    assert store.delete("no such thing at all") == 0

    with pytest.raises(MemoryStoreError):
        store.delete("   ")

    assert store.delete_all() == 3
    assert store.count() == 0
    assert not memory_file.exists()


def test_prune_forgets_only_turns_past_the_retention_window(memory_file: Path) -> None:
    store = MemoryStore(memory_file, enabled=True)
    old = (datetime.now(UTC) - timedelta(days=45)).isoformat(timespec="seconds")
    fresh = json.dumps(
        {"when": datetime.now(UTC).isoformat(timespec="seconds"), "prompt": "a fresh question", "reply": "fresh"}
    )
    memory_file.write_text(
        json.dumps({"when": old, "kind": "turn", "prompt": "an old question", "reply": "an old answer"}) + "\n"
        + fresh + "\n",
        encoding="utf-8",
    )

    assert store.prune(days=30) == 1
    remaining = store.load()
    assert len(remaining) == 1
    assert remaining[0]["prompt"] == "a fresh question"
    assert store.prune(days=365) == 0


def test_context_for_is_empty_when_disabled_and_fills_when_enabled(memory_file: Path) -> None:
    disabled = MemoryStore(memory_file, enabled=False)
    assert disabled.context_for("film") == []

    enabled = MemoryStore(memory_file, enabled=True)
    enabled.record("what film did I watch last night", "The film I named was Dune.")
    messages = enabled.context_for("what film did I watch")
    assert messages
    assert messages[0]["role"] == "system"
    assert "Dune" in messages[0]["content"]
    assert enabled.context_for("no matching question at all here") == []


def test_evaluate_reports_expected_passes_and_failures(memory_file: Path) -> None:
    store = MemoryStore(memory_file, enabled=True)
    _seed(store)

    cases = [
        ("what film did I say I watched", "film"),
        ("which restaurant did I recommend", "restaurant"),
        ("what time do I wake up", "wake"),
        ("where do I keep my holiday photos", "photos"),
        ("who did I ask the model to explain", "Einstein"),
        ("what book did I say I borrowed", "book"),
    ]
    report = evaluate(store, cases)
    assert report["cases"] == 6
    assert report["passed"] == 5
    assert report["failed"] == 1
    assert report["failures"] == ["what book did I say I borrowed"]

    empty = MemoryStore(memory_file.with_name("empty.jsonl"), enabled=True)
    assert evaluate(empty, EVAL_CASES)["passed"] == 0


def test_cli_drives_each_subcommand(memory_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Consent: recording without --enabled writes nothing at all.
    assert main(["--memory-file", str(memory_file), "record", "what film did I watch", "--reply", "Dune"]) == 1
    assert "Nothing recorded" in capsys.readouterr().out

    assert main(
        [
            "--memory-file",
            str(memory_file),
            "record",
            "what film did I watch",
            "--reply",
            "Dune",
            "--enabled",
        ]
    ) == 0
    assert (
        main(["--memory-file", str(memory_file), "record", "who is Einstein", "--reply", "A physicist.", "--enabled"])
        == 0
    )
    assert "Recorded (turn)" in capsys.readouterr().out

    assert main(["--memory-file", str(memory_file), "inspect", "film"]) == 0
    out = capsys.readouterr().out
    assert "what film did I watch" in out
    assert "Dune" in out
    assert "Einstein" not in out

    export_path = tmp_path / "export.json"
    assert main(["--memory-file", str(memory_file), "export", "--output", str(export_path)]) == 0
    assert "Exported 2 remembered turn(s)" in capsys.readouterr().out
    assert json.loads(export_path.read_text(encoding="utf-8"))["count"] == 2

    assert main(["--memory-file", str(memory_file), "prune", "--days", "30"]) == 0
    assert "Removed 0 turn(s)" in capsys.readouterr().out

    assert main(["--memory-file", str(memory_file), "eval"]) == 0
    assert "fixture case(s)" in capsys.readouterr().out

    assert main(["--memory-file", str(memory_file), "delete", "einstein"]) == 0
    assert "Deleted 1 remembered turn(s)" in capsys.readouterr().out
    assert MemoryStore(memory_file, enabled=True).count() == 1

    assert main(["--memory-file", str(memory_file), "delete", "--all"]) == 0
    assert MemoryStore(memory_file, enabled=True).count() == 0

    # A delete with nothing to name is refused rather than wiping the store.
    _seed(MemoryStore(memory_file, enabled=True))
    assert main(["--memory-file", str(memory_file), "delete"]) == 1
    assert MemoryStore(memory_file, enabled=True).count() == 5


def test_cli_records_through_the_default_location(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert main(["record", "where are my photos", "--reply", "In the shoebox.", "--enabled"]) == 0
    store = MemoryStore(enabled=True)
    assert store.path == tmp_path / "voice2text-ai" / "memory.jsonl"
    assert store.count() == 1
    assert "In the shoebox." in store.context_for("photos")[0]["content"]


def test_the_app_records_a_turn_only_when_the_user_consented(tmp_path: Path) -> None:
    # voice2text.window imports GTK at import time and the CI test container
    # installs no GTK, so this wiring check skips there instead of failing.
    pytest.importorskip("gi")
    from voice2text.window import MainWindow

    path = tmp_path / "memory.jsonl"
    window = MainWindow.__new__(MainWindow)  # the widget tree is not needed here
    window.settings = Settings()
    window._conversation_history = deque(
        [
            {"role": "system", "content": "You are a hands-free voice assistant."},
            {"role": "user", "content": "what film did I watch"},
        ],
        maxlen=24,
    )

    window.memory = MemoryStore(path, enabled=window.settings.memory_enabled)
    MainWindow._remember_turn(window, "Dune.")
    assert not path.exists()

    window.settings.memory_enabled = True
    window.memory = MemoryStore(path, enabled=window.settings.memory_enabled)
    MainWindow._remember_turn(window, "Dune.")
    records = window.memory.load()
    assert len(records) == 1
    assert records[0]["kind"] == "conversation"
    assert records[0]["prompt"] == "what film did I watch"
    assert records[0]["provenance"]["channel"] == "conversation"
    assert "Dune." in window.memory.context_for("what film did I watch")[0]["content"]


def test_load_skips_a_half_written_line(memory_file: Path) -> None:
    memory_file.write_text(
        '{"prompt": "whole record", "reply": "kept"}\n'
        "not json at all\n"
        + json.dumps({"v": "one", "prompt": "odd version", "reply": "still readable"}) + "\n",
        encoding="utf-8",
    )
    store = MemoryStore(memory_file, enabled=True)
    records: list[dict[str, Any]] = store.load()
    assert len(records) == 2
    assert "whole record" in [record["prompt"] for record in records]
    assert records[-1]["v"] == 1
