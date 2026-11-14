from __future__ import annotations

import json
from pathlib import Path

import pytest

from voice2text import actions
from voice2text.actions import (
    ActionConflict,
    ActionError,
    ActionSpec,
    ActionStore,
    Match,
    Param,
    argv_for,
    match_action,
    normalize_phrase,
    plan,
    starter_actions,
)


def test_normalize_phrase_drops_case_punctuation_and_whitespace() -> None:
    assert normalize_phrase("  Play   Bossnova!!  ") == "play bossnova"
    assert normalize_phrase("Open <Bookmark>, please") == "open <bookmark> please"
    assert normalize_phrase("don't — do this") == "don t do this"


def test_matching_is_case_and_punctuation_insensitive() -> None:
    actions_list = starter_actions()
    for text in ("play bossnova", "Play Bossnova!", "  PLAY   bossnova  "):
        match = match_action(text, actions_list)
        assert match is not None
        assert match.action.id == "play-bossnova"
        assert match.exact is True


def test_placeholder_captures_one_spoken_word() -> None:
    match = match_action("Open GitHub.", starter_actions())
    assert match is not None
    assert match.action.id == "open-bookmark"
    assert match.params == {"bookmark": "github"}
    assert match.exact is False


def test_placeholder_captures_the_rest_of_a_search_phrase() -> None:
    match = match_action("search for how far is the moon", starter_actions())
    assert match is not None
    assert match.params == {"text": "how far is the moon"}


def test_unrelated_speech_matches_nothing() -> None:
    assert match_action("hello there", starter_actions()) is None
    # A sentence that merely mentions a trigger must not fire it.
    assert match_action("ask me about opening github", starter_actions()) is None
    assert match_action("", starter_actions()) is None


def _similar_pair(*, enabled: bool = True) -> list[ActionSpec]:
    return [
        ActionSpec(id="open-site", phrase="open <site>", adapter="open-url", enabled=enabled),
        ActionSpec(id="open-page", phrase="open <page>", adapter="open-url", enabled=enabled),
    ]


def test_equal_specificity_conflicts_instead_of_guessing() -> None:
    with pytest.raises(ActionConflict):
        match_action("open docs", _similar_pair())


def test_disabled_actions_never_match() -> None:
    assert match_action("open docs", _similar_pair(enabled=False)) is None


def test_longest_matching_trigger_wins() -> None:
    broad = ActionSpec(id="play-thing", phrase="play <thing>", adapter="play-url")
    narrow = ActionSpec(id="play-today", phrase="play <thing> today", adapter="play-url")
    match = match_action("play bossnova today", [broad, narrow])
    assert match is not None and match.action.id == "play-today"
    # The broader trigger still answers on its own, and no conflict is raised.
    assert match_action("play bossnova", [broad, narrow]).action.id == "play-thing"


def test_starter_set_answers_the_issue_phrases() -> None:
    bossnova = match_action("play bossnova", starter_actions())
    assert bossnova is not None
    assert argv_for(bossnova.action, bossnova.params) == [
        "mpv",
        "--no-terminal",
        "https://somafm.com/m3u/bossa.m3u",
    ]

    bookmark = match_action("open github", starter_actions())
    assert bookmark is not None
    assert argv_for(bookmark.action, bookmark.params) == [
        "xdg-open",
        "https://github.com/crhy/Voice2Text-AI",
    ]


def test_argv_is_a_list_and_never_a_shell_string() -> None:
    match = match_action("search for rm -rf /; echo hi", starter_actions())
    assert match is not None
    argv = argv_for(match.action, match.params)
    assert isinstance(argv, list)
    assert all(isinstance(part, str) for part in argv)
    assert argv[0] == "xdg-open"
    # The dangerous words are percent-encoded inside one query argument.
    assert len(argv) == 2
    assert ";" not in argv[1] and " " not in argv[1]


def test_unknown_bookmark_is_refused_rather_than_opened() -> None:
    match = match_action("open not-a-bookmark", starter_actions())
    assert match is not None
    with pytest.raises(ActionError, match="unknown bookmark"):
        argv_for(match.action, match.params)


def test_the_module_itself_never_touches_a_subprocess() -> None:
    assert "subprocess" not in vars(actions)


def test_plan_is_a_reversible_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoShell:
        def __getattr__(self, name: str):
            raise AssertionError(f"plan() touched {name}")

    monkeypatch.setattr(actions, "subprocess", _NoShell(), raising=False)
    action = starter_actions()[0]
    argv = plan(action, "play bossnova")
    assert argv[0] == "mpv"
    assert "somafm.com" in argv[-1]


def test_plan_rejects_a_live_process_handle() -> None:
    class _LiveProcess:
        def poll(self) -> None:
            return None

    with pytest.raises(ActionError, match="never executes"):
        plan(starter_actions()[0], "play bossnova", process=_LiveProcess())


def test_plan_refuses_text_that_does_not_trigger_the_action() -> None:
    with pytest.raises(ActionError, match="does not trigger"):
        plan(starter_actions()[0], "play something else")


def test_privileged_actions_need_explicit_permission() -> None:
    destructive = ActionSpec(
        id="trash-file",
        phrase="trash <path>",
        params=(Param("path", "text"),),
        adapter="delete-path",
        confirm=True,
    )
    with pytest.raises(ActionError, match="privileged adapter"):
        argv_for(destructive, {"path": "notes/old.md"})

    argv = argv_for(destructive, {"path": "notes/old.md"}, allow_privileged=True)
    assert argv == ["gio", "trash", "--", "notes/old.md"]

    # Even when permitted, the path stays inside the user's own files.
    with pytest.raises(ActionError, match="user's own files"):
        argv_for(destructive, {"path": "/etc/passwd"}, allow_privileged=True)


def test_privileged_messaging_needs_explicit_permission_too() -> None:
    sender = ActionSpec(
        id="send-note",
        phrase="send <text>",
        params=(Param("text", "text"), Param("url", "url", "https://example.com/notify")),
        adapter="send-text",
        confirm=True,
    )
    with pytest.raises(ActionError, match="privileged adapter"):
        argv_for(sender, {"text": "meeting at 3"})

    argv = argv_for(sender, {"text": "meeting at 3"}, allow_privileged=True)
    assert argv == ["xdg-open", "https://example.com/notify?text=meeting%20at%203"]
    # The spoken words stay inside one query argument.
    assert len(argv) == 2


def test_privileged_action_without_confirmation_is_invalid() -> None:
    unconfirmed = ActionSpec(id="send-note", phrase="send <text>", adapter="send-text", confirm=False)
    with pytest.raises(ActionError, match="must require confirmation"):
        argv_for(unconfirmed, {"text": "hi", "url": "https://example.com"}, allow_privileged=True)


def test_declared_but_unimplemented_adapter_stays_inert() -> None:
    spend = ActionSpec(
        id="buy-thing",
        phrase="buy <text>",
        params=(Param("text", "text"),),
        adapter="spend-credit",
        confirm=True,
    )
    with pytest.raises(ActionError, match="no executable adapter"):
        argv_for(spend, {"text": "a coffee"}, allow_privileged=True)


def test_store_round_trip_keeps_schema_and_policy(tmp_path: Path) -> None:
    path = tmp_path / "actions.json"
    store = ActionStore(path)
    store.create(starter_actions()[1])
    store.disable("open-bookmark")

    reloaded = ActionStore(path)
    assert reloaded.load() == store.actions
    assert reloaded.load()[0].enabled is False
    assert json.loads(path.read_text())["schema"] == actions.SCHEMA_VERSION
    assert not path.with_suffix(".tmp").exists()


def test_missing_file_loads_empty(tmp_path: Path) -> None:
    store = ActionStore(tmp_path / "nested" / "actions.json")
    assert store.load() == []


def test_export_import_round_trip(tmp_path: Path) -> None:
    source = ActionStore(tmp_path / "a.json")
    source.actions = starter_actions()
    payload = json.loads(source.export_json())

    target = ActionStore(tmp_path / "b.json")
    target.import_json(payload)
    assert target.actions == starter_actions()

    # An id clash is refused until the caller asks to replace.
    with pytest.raises(ActionError, match="already defined"):
        target.import_json(payload)
    target.import_json(payload, replace=True)
    assert target.actions == starter_actions()


def test_store_rejects_bad_payloads(tmp_path: Path) -> None:
    path = tmp_path / "actions.json"

    def write(payload: dict) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    write({"schema": 99, "actions": []})
    with pytest.raises(ActionError, match="schema 99 is not supported"):
        ActionStore(path).load()

    write({"schema": 1, "actions": [{"id": "a", "phrase": "do it", "adapter": "teleport"}]})
    with pytest.raises(ActionError, match="unknown adapter"):
        ActionStore(path).load()

    write(
        {
            "schema": 1,
            "actions": [
                {"id": "a", "phrase": "do it", "adapter": "open-url"},
                {"id": "a", "phrase": "do that", "adapter": "open-url"},
            ],
        }
    )
    with pytest.raises(ActionError, match="duplicate action id"):
        ActionStore(path).load()

    for phrase in ("", "   ", "!!!"):
        write({"schema": 1, "actions": [{"id": "a", "phrase": phrase, "adapter": "open-url"}]})
        with pytest.raises(ActionError, match="empty trigger phrase"):
            ActionStore(path).load()

    write(
        {
            "schema": 1,
            "actions": [
                {
                    "id": "a",
                    "phrase": "open it",
                    "adapter": "open-url",
                    "params": [{"name": "url", "type": "url", "value": "file:///etc/passwd"}],
                }
            ],
        }
    )
    with pytest.raises(ActionError, match="http or https"):
        ActionStore(path).load()

    write(
        {
            "schema": 1,
            "actions": [
                {
                    "id": "a",
                    "phrase": "send <text>",
                    "adapter": "send-text",
                    "confirm": False,
                    "params": [{"name": "text", "type": "text"}],
                }
            ],
        }
    )
    with pytest.raises(ActionError, match="must set confirm to true"):
        ActionStore(path).load()


def test_edit_changes_wording_and_enable_only_flips_enabled(tmp_path: Path) -> None:
    store = ActionStore(tmp_path / "actions.json")
    store.create(starter_actions()[1])

    assert store.edit("open-bookmark", phrase="open the <bookmark>") == [
        ActionSpec(
            id="open-bookmark",
            phrase="open the <bookmark>",
            params=(Param("bookmark", "bookmark"),),
            adapter="open-url",
        )
    ]
    with pytest.raises(ActionError, match=r"enable\(\) or disable\(\)"):
        store.edit("open-bookmark", enabled=False)

    assert store.disable("open-bookmark")[0].enabled is False
    assert store.enable("open-bookmark")[0].enabled is True
    assert store.actions == ActionStore(store.path).load()


def test_edit_refuses_an_unusable_rewrite(tmp_path: Path) -> None:
    store = ActionStore(tmp_path / "actions.json")
    store.create(starter_actions()[1])
    with pytest.raises(ActionError, match="unknown adapter"):
        store.edit("open-bookmark", adapter="warp")
    assert store.actions == [starter_actions()[1]]


def test_match_carries_the_utterance_for_the_status_line() -> None:
    match = match_action("Play Bossnova", starter_actions())
    assert isinstance(match, Match)
    assert match.text == "Play Bossnova"
