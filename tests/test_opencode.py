from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from voice2text import opencode


SESSION_ID = "ses_01ABC"


def make_fake_binary(
    tmp_path: Path,
    *,
    sleep_seconds: float = 0.0,
    session_id: str = SESSION_ID,
    exit_code: int = 0,
) -> tuple[Path, Path, Path]:
    """Create an executable fake ``opencode`` that records how it was invoked.

    It writes argv and cwd to ``invocation.json``, prints JSON events the way
    ``--format json`` does, and can pause so a stop event has something to
    interrupt.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary = bin_dir / "opencode"
    record = tmp_path / "invocation.json"

    script = "\n".join(
        [
            f"#!{sys.executable}",
            "import json, sys, time",
            "from pathlib import Path",
            f"Path({str(record)!r}).write_text(",
            "    json.dumps({'argv': sys.argv[1:], 'cwd': str(Path.cwd())}), encoding='utf-8')",
            f"time.sleep({sleep_seconds})",
            f"print(json.dumps({{'type': 'session', 'sessionID': {session_id!r}}}))",
            "print(json.dumps({'type': 'text', 'text': 'noted'}))",
            f"sys.exit({exit_code})",
        ]
    )
    binary.write_text(script + "\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary, record, bin_dir


def make_repo(tmp_path: Path, *, branch: str = "feature", dirty: bool = False) -> Path:
    """A tiny git repository, built with real git so the status parsing is tested."""
    repo = tmp_path / "project"
    repo.mkdir()
    git = ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "-C", str(repo)]

    def run(args: list[str]) -> None:
        subprocess.run([*git, *args], check=True, capture_output=True)

    run(["init", "-q"])
    (repo / "work.txt").write_text("one\n", encoding="utf-8")
    run(["add", "."])
    run(["commit", "-q", "-m", "first commit"])
    run(["checkout", "-q", "-b", branch])
    if dirty:
        (repo / "work.txt").write_text("one\ntwo\n", encoding="utf-8")
        (repo / "extra.txt").write_text("new\n", encoding="utf-8")
    return repo


def on_path(monkeypatch, bin_dir: Path) -> None:
    """Put the fake binary where ``find_binary()`` looks for it."""
    monkeypatch.setenv("PATH", str(bin_dir))


# --- binary discovery -------------------------------------------------------


def test_find_binary_returns_none_when_nothing_is_installed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert opencode.find_binary() is None


def test_find_binary_falls_back_to_the_opencode_install_directory(tmp_path: Path, monkeypatch) -> None:
    installed = tmp_path / ".opencode" / "bin" / "opencode"
    installed.parent.mkdir(parents=True)
    installed.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    installed.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert opencode.find_binary() == str(installed)


# --- the allowlist ----------------------------------------------------------


def test_project_allowed_denies_everything_when_the_allowlist_is_empty(tmp_path: Path) -> None:
    assert opencode.project_allowed(tmp_path, []) is False
    assert opencode.project_allowed(tmp_path / "sub", ()) is False


def test_project_allowed_accepts_the_entry_itself_and_paths_inside_it(tmp_path: Path) -> None:
    assert opencode.project_allowed(tmp_path, [str(tmp_path)]) is True
    assert opencode.project_allowed(tmp_path / "sub" / "deep", [str(tmp_path)]) is True


def test_project_allowed_rejects_parent_traversal_and_sibling_prefixes(tmp_path: Path) -> None:
    assert opencode.project_allowed(tmp_path / "..", [str(tmp_path)]) is False
    assert opencode.project_allowed(tmp_path / "sub" / ".." / "..", [str(tmp_path)]) is False
    assert opencode.project_allowed(Path(str(tmp_path) + "x"), [str(tmp_path)]) is False


def test_project_allowed_rejects_a_symlink_that_leaves_the_tree(tmp_path: Path) -> None:
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = inside / "escape"
    link.symlink_to(outside)
    assert opencode.project_allowed(link, [str(inside)]) is False
    assert opencode.project_allowed(link / "file.txt", [str(inside)]) is False


# --- read-only status -------------------------------------------------------


def test_read_only_status_reports_branch_commits_and_dirty_files(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, dirty=True)
    status = opencode.read_only_status(repo, allowed_dirs=[str(repo)])

    assert status.is_git_repo is True
    assert status.branch == "feature"
    assert status.dirty is True
    assert {name.split()[-1] for name in status.changed_files} == {"work.txt", "extra.txt"}
    assert status.recent_commits and "first commit" in status.recent_commits[0]
    assert "work.txt" in status.diffstat


def test_read_only_status_reports_a_clean_tree(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    status = opencode.read_only_status(repo, allowed_dirs=[str(repo)])
    assert status.dirty is False
    assert status.changed_files == ()
    assert status.diffstat == ""


def test_read_only_status_refuses_a_project_outside_the_allowlist(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    status = opencode.read_only_status(repo, allowed_dirs=[str(tmp_path / "elsewhere")])
    assert status.is_git_repo is False
    assert "allowed" in status.error


def test_read_only_status_reports_a_non_repository(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    status = opencode.read_only_status(plain, allowed_dirs=[str(plain)])
    assert status.is_git_repo is False
    assert "git" in status.error


def test_suggestions_stay_read_only_and_never_offer_auto(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, dirty=True)
    status = opencode.read_only_status(repo, allowed_dirs=[str(repo)])
    lines = opencode.suggested_next_actions(status, "add a pause button\nsecond line")

    assert any("uncommitted" in line for line in lines)
    assert any("add a pause button" in line for line in lines)
    assert any("approval" in line for line in lines)
    assert all("--auto" not in line for line in lines)


def test_suggestions_explain_why_there_is_nothing_to_suggest() -> None:
    status = opencode.ProjectStatus(project="/nowhere", error="this is not a git repository")
    assert opencode.suggested_next_actions(status) == [
        "Nothing to suggest yet: this is not a git repository."
    ]


# --- argv shape -------------------------------------------------------------


def test_continuation_argv_is_minimal_and_carries_no_auto() -> None:
    argv = opencode.continuation_argv("next step", binary="/opt/opencode")
    assert argv == ["/opt/opencode", "run", "next step", "--format", "json"]
    assert not any(str(part).startswith("--auto") for part in argv)


def test_continuation_argv_adds_only_the_flags_that_have_values() -> None:
    argv = opencode.continuation_argv(
        "next step",
        project="/home/user/project",
        model="ollama/qwen3:8b",
        session=SESSION_ID,
        binary="/opt/opencode",
        capture_session=False,
    )
    assert argv == [
        "/opt/opencode",
        "run",
        "next step",
        "--dir",
        "/home/user/project",
        "-m",
        "ollama/qwen3:8b",
        "-s",
        SESSION_ID,
    ]
    assert "--format" not in argv


def test_session_id_round_trips_into_the_next_run() -> None:
    stdout = json.dumps({"type": "session", "sessionID": SESSION_ID})
    captured = opencode.parse_session_id(stdout)
    assert captured == SESSION_ID
    argv = opencode.continuation_argv("continue", session=captured, binary="/opt/opencode")
    assert argv[-4:] == ["-s", SESSION_ID, "--format", "json"]


def test_parse_session_id_tolerates_other_event_shapes() -> None:
    assert opencode.parse_session_id("not json at all") == ""
    assert opencode.parse_session_id('{"session": {"id": "ses_2"}}') == "ses_2"


# --- running a continuation ------------------------------------------------


def test_run_continuation_runs_inside_the_allowed_project(tmp_path: Path, monkeypatch) -> None:
    binary, record, bin_dir = make_fake_binary(tmp_path)
    on_path(monkeypatch, bin_dir)
    project = tmp_path / "project"
    project.mkdir()

    argv = opencode.continuation_argv(
        "next step", project=str(project), model="ollama/qwen3:8b", binary=str(binary)
    )
    result = opencode.run_continuation(argv, cwd=project, allowed_dirs=[str(project)], timeout=30.0)

    assert result.returncode == 0
    assert result.stopped is False and result.timed_out is False
    assert result.session_id == SESSION_ID

    invocation = json.loads(record.read_text(encoding="utf-8"))
    assert Path(invocation["cwd"]) == project
    assert invocation["argv"] == [
        "run",
        "next step",
        "--dir",
        str(project),
        "-m",
        "ollama/qwen3:8b",
        "--format",
        "json",
    ]
    assert "--auto" not in invocation["argv"]


def test_run_continuation_refuses_before_any_subprocess_starts(tmp_path: Path, monkeypatch) -> None:
    binary, record, bin_dir = make_fake_binary(tmp_path)
    on_path(monkeypatch, bin_dir)
    project = tmp_path / "project"
    project.mkdir()
    argv = opencode.continuation_argv("next step", project=str(project), binary=str(binary))

    with pytest.raises(opencode.OpenCodeError):
        opencode.run_continuation(argv, cwd=project, allowed_dirs=[], timeout=30.0)
    assert not record.exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(opencode.OpenCodeError):
        opencode.run_continuation(argv, cwd=outside, allowed_dirs=[str(project)], timeout=30.0)
    assert not record.exists()


def test_run_continuation_refuses_a_missing_binary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    project = tmp_path / "project"
    project.mkdir()

    with pytest.raises(opencode.OpenCodeError) as raised:
        opencode.run_continuation(
            opencode.continuation_argv("next step"),
            cwd=project,
            allowed_dirs=[str(project)],
            timeout=5.0,
        )
    assert "not installed" in str(raised.value)


def test_run_continuation_refuses_an_argv_that_carries_auto(tmp_path: Path, monkeypatch) -> None:
    binary, record, bin_dir = make_fake_binary(tmp_path)
    on_path(monkeypatch, bin_dir)
    project = tmp_path / "project"
    project.mkdir()

    argv = opencode.continuation_argv("next step", binary=str(binary)) + ["--auto"]
    with pytest.raises(opencode.OpenCodeError) as raised:
        opencode.run_continuation(argv, cwd=project, allowed_dirs=[str(project)], timeout=5.0)
    assert "--auto" in str(raised.value)
    assert not record.exists()


def test_stop_event_terminates_a_running_continuation(tmp_path: Path, monkeypatch) -> None:
    binary, record, bin_dir = make_fake_binary(tmp_path, sleep_seconds=5.0)
    on_path(monkeypatch, bin_dir)
    project = tmp_path / "project"
    project.mkdir()

    stop_event = threading.Event()
    threading.Timer(0.6, stop_event.set).start()
    argv = opencode.continuation_argv("next step", binary=str(binary))
    result = opencode.run_continuation(
        argv, cwd=project, allowed_dirs=[str(project)], stop_event=stop_event, timeout=30.0
    )

    assert result.stopped is True
    assert result.returncode != 0
    assert record.exists()  # the child really did start, then was killed


def test_timeout_bounds_a_run_that_never_finishes(tmp_path: Path, monkeypatch) -> None:
    binary, _record, bin_dir = make_fake_binary(tmp_path, sleep_seconds=10.0)
    on_path(monkeypatch, bin_dir)
    project = tmp_path / "project"
    project.mkdir()

    argv = opencode.continuation_argv("next step", binary=str(binary))
    result = opencode.run_continuation(argv, cwd=project, allowed_dirs=[str(project)], timeout=0.5)
    assert result.timed_out is True


def test_run_continuation_reports_a_failing_run(tmp_path: Path, monkeypatch) -> None:
    binary, _record, bin_dir = make_fake_binary(tmp_path, exit_code=3)
    on_path(monkeypatch, bin_dir)
    project = tmp_path / "project"
    project.mkdir()

    argv = opencode.continuation_argv("next step", binary=str(binary))
    result = opencode.run_continuation(
        argv, cwd=project, allowed_dirs=[str(project)], timeout=30.0
    )
    assert result.returncode == 3
    assert opencode.parse_reply_text(result.stdout) == "noted"


# --- plan, checkpoints, audit log -------------------------------------------


def test_plan_and_checkpoints_survive_a_reload(tmp_path: Path) -> None:
    path = tmp_path / "opencode-plan.json"
    store = opencode.PlanStore(path)
    store.save(
        project="/home/user/project",
        steps=["run the tests", "open an issue"],
        session_id=SESSION_ID,
    )
    store.checkpoint(
        project="/home/user/project", step="run the tests", session_id=SESSION_ID, status="finished"
    )

    reloaded = opencode.PlanStore(path).load()
    assert reloaded["steps"] == ["run the tests", "open an issue"]
    assert reloaded["session_id"] == SESSION_ID
    assert len(reloaded["checkpoints"]) == 1
    checkpoint = reloaded["checkpoints"][0]
    assert checkpoint["step"] == "run the tests"
    assert checkpoint["session_id"] == SESSION_ID
    assert checkpoint["when"]


def test_plan_checkpoints_append_without_losing_the_plan(tmp_path: Path) -> None:
    path = tmp_path / "opencode-plan.json"
    store = opencode.PlanStore(path)
    store.save(project="/p", steps=["a", "b"])
    store.checkpoint(project="/p", step="a", status="finished")
    store.checkpoint(project="/p", step="b", status="stopped")

    reloaded = opencode.PlanStore(path).load()
    assert [item["step"] for item in reloaded["checkpoints"]] == ["a", "b"]
    assert reloaded["steps"] == ["a", "b"]


def test_plan_store_tolerates_a_missing_or_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "opencode-plan.json"
    empty = {"project": "", "steps": [], "checkpoints": [], "session_id": ""}
    assert opencode.PlanStore(path).load() == empty
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert opencode.PlanStore(path).load() == empty


def test_audit_log_records_the_run_without_the_message_body(tmp_path: Path) -> None:
    path = tmp_path / "opencode-audit.log"
    log = opencode.AuditLog(path)
    argv = opencode.continuation_argv(
        "a secret sentence about the user",
        binary="/opt/opencode",
        project="/home/user/project",
    )
    log.record("run", argv=argv, detail="exit=0")
    log.record("stopped", detail="user pressed pause")

    lines = opencode.AuditLog(path).read()
    assert len(lines) == 2
    assert "<message:" in lines[0]
    assert "secret sentence" not in path.read_text(encoding="utf-8")
    assert "/opt/opencode run" in lines[0]
    assert lines[1].endswith("stopped user pressed pause")


def test_audit_log_appends_across_reloads(tmp_path: Path) -> None:
    path = tmp_path / "opencode-audit.log"
    opencode.AuditLog(path).record("run", detail="first")
    opencode.AuditLog(path).record("run", detail="second")
    assert [line.split(" ", 1)[1] for line in opencode.AuditLog(path).read()] == [
        "run first",
        "run second",
    ]
