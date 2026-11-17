from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import voice2text.actions as actions
from voice2text.actions import Action, available_apps
from voice2text.config import ConfigStore, Settings


def test_registry_ids_are_unique_and_known() -> None:
    ids = [action.action_id for action in actions.ACTIONS]
    assert len(ids) == len(set(ids))
    assert actions.ACTION_IDS == set(ids)
    assert {action.app_id for action in actions.ACTIONS} == set(actions.APP_IDS)


def test_every_action_declares_a_permission_and_readable_text() -> None:
    for action in actions.ACTIONS:
        assert isinstance(action, Action)
        assert action.title and action.detail
        assert action.permission.startswith("--")
        assert action.app_id in actions.APP_NAMES
        assert isinstance(action.destructive, bool)


def test_registry_only_offers_read_only_or_export_actions() -> None:
    # No action may claim to edit a project in place: the only write flags are
    # binds on a folder an export may create files in.
    for action in actions.ACTIONS:
        assert "--dev=" not in action.permission
        assert "--device" not in action.permission
        assert "--allow=dev-" not in action.permission


def test_available_apps_reads_fake_flatpak_list_output(tmp_path: Path) -> None:
    output = "org.gimp.GIMP\norg.blender.Blender.Codecs\ncom.example.Other\n"
    found = available_apps(flatpak_output=output, desktop_dirs=(tmp_path,))
    assert [app.app_id for app in found] == ["org.gimp.GIMP"]
    assert found[0].name == "GIMP"
    assert found[0].source == "flatpak"


def test_available_apps_falls_back_to_fake_applications_directory(tmp_path: Path) -> None:
    (tmp_path / "org.freecad.FreeCAD.desktop").write_text("[Desktop Entry]\n", encoding="utf-8")
    # A sibling launcher of the same suite is not evidence of the app itself.
    (tmp_path / "org.libreoffice.LibreOffice.base.desktop").write_text("[Desktop Entry]\n", encoding="utf-8")

    found = available_apps(flatpak_output="", desktop_dirs=(tmp_path,))
    assert [app.app_id for app in found] == ["org.freecad.FreeCAD"]
    assert found[0].source == "desktop"


def test_available_apps_merges_flatpak_and_desktop_sources(tmp_path: Path) -> None:
    (tmp_path / "org.blender.Blender.desktop").write_text("[Desktop Entry]\n", encoding="utf-8")
    found = available_apps(flatpak_output="org.gimp.GIMP\n", desktop_dirs=(tmp_path,))
    assert [(app.app_id, app.source) for app in found] == [
        ("org.gimp.GIMP", "flatpak"),
        ("org.blender.Blender", "desktop"),
    ]


def test_available_apps_only_asks_flatpak_for_metadata(tmp_path: Path) -> None:
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="org.gimp.GIMP\n", stderr="")
    with patch("subprocess.run", return_value=fake) as run:
        found = available_apps(desktop_dirs=(tmp_path,))
    assert [app.app_id for app in found] == ["org.gimp.GIMP"]
    # Exactly one probe, and it is the metadata listing: no app is ever started.
    assert len(run.call_args_list) == 1
    assert run.call_args.args[0] == ["flatpak", "list", "--columns=application"]


def test_available_apps_survives_a_missing_flatpak(tmp_path: Path) -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert available_apps(desktop_dirs=(tmp_path,)) == []


def test_nothing_is_granted_by_default() -> None:
    settings = Settings()
    assert settings.allowed_actions == []
    assert actions.allowed(settings) == []
    assert actions.granted(settings, "gimp.open-project") is None
    assert actions.granted_text(settings) == "none granted"


def test_unknown_ids_are_never_accepted() -> None:
    settings = Settings(allowed_actions=["gimp.open-project", "made.up.action"])
    assert actions.granted(settings, "made.up.action") is None
    assert actions.granted(settings, "GIMP.OPEN-PROJECT") is None
    assert actions.grant(settings, "made.up.action") is False
    assert settings.allowed_actions == ["gimp.open-project", "made.up.action"]
    # Even a hand-written id that is not in the registry is dropped on save.
    assert "made.up.action" not in settings.normalized().allowed_actions


def test_grant_and_revoke_round_trip_through_the_config_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    store = ConfigStore()
    assert store.path == tmp_path / "voice2text-ai" / "config.json"

    settings = store.load()
    assert actions.grant(settings, "libreoffice.overwrite-document") is True
    store.save(settings)

    reloaded = store.load()
    assert reloaded.allowed_actions == ["libreoffice.overwrite-document"]
    granted = actions.granted(reloaded, "libreoffice.overwrite-document")
    assert granted is not None and granted.destructive is True

    assert actions.revoke(reloaded, "libreoffice.overwrite-document") is True
    store.save(reloaded)
    assert store.load().allowed_actions == []
    assert actions.granted(store.load(), "libreoffice.overwrite-document") is None


def test_normalized_drops_unknown_ids_and_sorts_the_rest() -> None:
    settings = Settings(
        allowed_actions=[
            "libreoffice.overwrite-document",
            "not.in.the.registry",
            " gimp.open-project ",
            "gimp.open-project",
        ]
    )
    assert settings.normalized().allowed_actions == [
        "gimp.open-project",
        "libreoffice.overwrite-document",
    ]


def test_normalized_tolerates_a_hand_edited_value() -> None:
    assert Settings(allowed_actions="gimp.open-project").normalized().allowed_actions == []
    assert Settings(allowed_actions=[None, 7]).normalized().allowed_actions == []


def test_an_old_config_file_without_the_key_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"whisper_model": "small", "wake_word": "hey"}', encoding="utf-8")
    loaded = ConfigStore(path).load()
    assert loaded.whisper_model == "small"
    assert loaded.allowed_actions == []


def test_every_destructive_action_needs_an_explicit_grant() -> None:
    destructive = [action for action in actions.ACTIONS if action.destructive]
    assert destructive, "the registry must exercise the confirm path"

    for action in destructive:
        assert actions.granted(Settings(), action.action_id) is None
        assert actions.granted(Settings(allowed_actions=[action.action_id]), action.action_id) is action
        # A destructive action replaces something, so it can never be covered
        # by a read-only permission.
        assert "--ro-bind" not in action.permission
        assert "--bind=" in action.permission


def test_allowed_lists_only_granted_actions_in_registry_order() -> None:
    settings = Settings(allowed_actions=["freecad.open-project", "gimp.export-caption"])
    assert [action.action_id for action in actions.allowed(settings)] == [
        "freecad.open-project",
        "gimp.export-caption",
    ]


def test_actions_for_groups_the_registry_by_app() -> None:
    for app_id in actions.APP_IDS:
        assert actions.actions_for(app_id)
        assert {action.app_id for action in actions.actions_for(app_id)} == {app_id}
    assert actions.actions_for("com.example.Other") == []


def test_describe_shows_the_permission_and_flags_destructive_actions() -> None:
    read_only = actions.ACTIONS[0]
    assert read_only.permission in actions.describe(read_only)
    assert "destructive" not in actions.describe(read_only)

    destructive = next(action for action in actions.ACTIONS if action.destructive)
    assert "destructive" in actions.describe(destructive)


def test_availability_text_reports_nothing_detected(tmp_path: Path) -> None:
    assert (
        actions.availability_text(available_apps(flatpak_output="", desktop_dirs=(tmp_path,)))
        == "none detected"
    )


def test_the_window_surfaces_apps_and_confirms_destructive_actions() -> None:
    # The UI is not importable in a headless test, so this checks the source the
    # way test_release_metadata checks the manifest: the status strip gains the
    # availability and grant state, and a destructive action only becomes granted
    # through the same confirm shape _confirm_delete_model uses.
    source = (Path(__file__).resolve().parents[1] / "voice2text/window.py").read_text(encoding="utf-8")

    assert "available_apps(" in source
    assert "granted_text(self.settings)" in source
    assert "def _confirm_action" in source
    assert "Adw.ResponseAppearance.DESTRUCTIVE" in source
    # No new dialog: the two Adw.Dialog instances are the existing installer and
    # model manager, and the confirm reuses Adw.AlertDialog like the model delete.
    assert source.count("Adw.Dialog(") == 2
    assert source.count("Adw.AlertDialog(") == 2
