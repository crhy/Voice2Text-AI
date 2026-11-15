from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from voice2text import pi_agent


SESSION_ID = "11111111-2222-4444-8888-555555555555"


def pi_events(
    session_id: str = SESSION_ID, reply: str = "noted", turns: tuple[str, ...] | None = None
) -> list[dict]:
    """The JSONL records a real ``pi -p --mode json`` run writes to stdout.

    ``turns`` lets a run produce several assistant messages, so the tests can
    check that the *final* one is what the app shows.
    """
    replies = list(turns) if turns is not None else [reply]
    events: list[dict] = [
        {
            "type": "session",
            "version": 3,
            "id": session_id,
            "timestamp": "2024-12-03T14:00:00.000Z",
            "cwd": "/not-the-run-directory",
        },
        {"type": "agent_start"},
        {"type": "turn_start"},
        {"type": "message_start", "message": {"role": "user", "content": "next step"}},
        {"type": "message_end", "message": {"role": "user", "content": "next step"}},
    ]
    for index, text in enumerate(replies):
        events.extend(
            [
                {"type": "message_start", "message": {"role": "assistant", "content": []}},
                {
                    "type": "message_update",
                    "usage": {"input": 10, "output": 1},
                    "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": text},
                },
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "private deliberation"},
                            {"type": "text", "text": text},
                        ],
                    },
                },
                {"type": "turn_end", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}},
            ]
        )
    # A vocabulary this app has never seen must not break the parse.
    events.append({"type": "brand_new_event_from_a_later_version", "whatever": 1})
    events.append({"type": "agent_end", "messages": [], "willRetry": False})
    events.append({"type": "agent_settled", "aborted": False})
    return events


def make_fake_binary(
    tmp_path: Path,
    *,
    sleep_seconds: float = 0.0,
    session_id: str = SESSION_ID,
    exit_code: int = 0,
    events: list[dict] | None = None,
) -> tuple[Path, Path, Path]:
    """Create an executable fake ``pi`` that records how it was invoked.

    It writes argv and cwd to ``invocation.json``, prints JSON events the way
    ``--mode json`` does, and can pause so a stop event has something to
    interrupt. No test ever runs the real pi binary.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary = bin_dir / "pi"
    record = tmp_path / "invocation.json"

    script = [
        f"#!{sys.executable}",
        "import json, sys, time",
        "from pathlib import Path",
        f"Path({str(record)!r}).write_text(",
        "    json.dumps({'argv': sys.argv[1:], 'cwd': str(Path.cwd())}), encoding='utf-8')",
        f"time.sleep({sleep_seconds})",
    ]
    for event in (events if events is not None else pi_events(session_id=session_id)):
        script.append(f"print(json.dumps({event!r}))")
    script.append(f"sys.exit({exit_code})")

    binary.write_text("\n".join(script) + "\n", encoding="utf-8")
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
    assert pi_agent.find_binary() is None


def test_find_binary_falls_back_to_the_user_local_bin(tmp_path: Path, monkeypatch) -> None:
    installed = tmp_path / ".local" / "bin" / "pi"
    installed.parent.mkdir(parents=True)
    installed.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    installed.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert pi_agent.find_binary() == str(installed)


def test_find_binary_falls_back_to_the_pi_directory(tmp_path: Path, monkeypatch) -> None:
    installed = tmp_path / ".pi" / "bin" / "pi"
    installed.parent.mkdir(parents=True)
    installed.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    installed.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert pi_agent.find_binary() == str(installed)


# --- the allowlist ----------------------------------------------------------


def test_project_allowed_denies_everything_when_the_allowlist_is_empty(tmp_path: Path) -> None:
    assert pi_agent.project_allowed(tmp_path, []) is False
    assert pi_agent.project_allowed(tmp_path / "sub", ()) is False


def test_project_allowed_accepts_the_entry_itself_and_paths_inside_it(tmp_path: Path) -> None:
    assert pi_agent.project_allowed(tmp_path, [str(tmp_path)]) is True
    assert pi_agent.project_allowed(tmp_path / "sub" / "deep", [str(tmp_path)]) is True


def test_project_allowed_rejects_parent_traversal_and_sibling_prefixes(tmp_path: Path) -> None:
    assert pi_agent.project_allowed(tmp_path / "..", [str(tmp_path)]) is False
    assert pi_agent.project_allowed(tmp_path / "sub" / ".." / "..", [str(tmp_path)]) is False
    assert pi_agent.project_allowed(Path(str(tmp_path) + "x"), [str(tmp_path)]) is False


def test_project_allowed_rejects_a_symlink_that_leaves_the_tree(tmp_path: Path) -> None:
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = inside / "escape"
    link.symlink_to(outside)
    assert pi_agent.project_allowed(link, [str(inside)]) is False
    assert pi_agent.project_allowed(link / "file.txt", [str(inside)]) is False


# --- read-only status -------------------------------------------------------


def test_read_only_status_reports_branch_commits_and_dirty_files(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, dirty=True)
    status = pi_agent.read_only_status(repo, allowed_dirs=[str(repo)])

    assert status.is_git_repo is True
    assert status.branch == "feature"
    assert status.dirty is True
    assert {name.split()[-1] for name in status.changed_files} == {"work.txt", "extra.txt"}
    assert status.recent_commits and "first commit" in status.recent_commits[0]
    assert "work.txt" in status.diffstat


def test_read_only_status_reports_a_clean_tree(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    status = pi_agent.read_only_status(repo, allowed_dirs=[str(repo)])
    assert status.dirty is False
    assert status.changed_files == ()
    assert status.diffstat == ""


def test_read_only_status_refuses_a_project_outside_the_allowlist(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    status = pi_agent.read_only_status(repo, allowed_dirs=[str(tmp_path / "elsewhere")])
    assert status.is_git_repo is False
    assert "allowed" in status.error


def test_read_only_status_reports_a_non_repository(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    status = pi_agent.read_only_status(plain, allowed_dirs=[str(plain)])
    assert status.is_git_repo is False
    assert "git" in status.error


def test_suggestions_stay_read_only_and_never_offer_approve(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, dirty=True)
    status = pi_agent.read_only_status(repo, allowed_dirs=[str(repo)])
    lines = pi_agent.suggested_next_actions(status, "add a pause button\nsecond line")

    assert any("uncommitted" in line for line in lines)
    assert any("add a pause button" in line for line in lines)
    assert any("allow" in line for line in lines)
    assert all("--approve" not in line for line in lines)


def test_suggestions_explain_why_there_is_nothing_to_suggest() -> None:
    status = pi_agent.ProjectStatus(project="/nowhere", error="this is not a git repository")
    assert pi_agent.suggested_next_actions(status) == [
        "Nothing to suggest yet: this is not a git repository."
    ]


# --- argv shape -------------------------------------------------------------


def test_continuation_argv_is_the_documented_pi_shape() -> None:
    argv = pi_agent.continuation_argv("next step", binary="/usr/local/bin/pi")
    assert argv == [
        "/usr/local/bin/pi",
        "-p",
        "--no-session",
        "--offline",
        "--mode",
        "json",
        "--no-approve",
        "--",
        "next step",
    ]
    assert not any(str(part).startswith("--approve") for part in argv)
    assert "--approve" not in argv and "-a" not in argv


def test_continuation_argv_adds_only_the_flags_that_have_values() -> None:
    argv = pi_agent.continuation_argv(
        "next step",
        provider="ollama",
        model="qwen3:8b",
        binary="/usr/local/bin/pi",
    )
    assert argv == [
        "/usr/local/bin/pi",
        "-p",
        "--no-session",
        "--offline",
        "--mode",
        "json",
        "--no-approve",
        "--provider",
        "ollama",
        "--model",
        "qwen3:8b",
        "--",
        "next step",
    ]
    # pi has no --dir: the run's working directory places it in the project.
    assert "--dir" not in argv


def test_continuation_argv_keeps_a_transcript_that_starts_with_a_dash_as_text() -> None:
    argv = pi_agent.continuation_argv("--not-a-flag", binary="/usr/local/bin/pi")
    assert argv[-2:] == ["--", "--not-a-flag"]


def test_continuation_argv_resumes_a_session_instead_of_staying_ephemeral() -> None:
    argv = pi_agent.continuation_argv(
        "continue", session=SESSION_ID, binary="/usr/local/bin/pi"
    )
    assert "--session" in argv and argv[argv.index("--session") + 1] == SESSION_ID
    assert "--no-session" not in argv


def test_continuation_argv_refuses_a_session_id_that_is_not_a_plain_id() -> None:
    for smuggled in ("../../etc/passwd", "--approve", "/home/user/session.jsonl"):
        with pytest.raises(pi_agent.PiAgentError):
            pi_agent.continuation_argv("continue", session=smuggled, binary="/usr/local/bin/pi")


# --- parsing the event stream -----------------------------------------------


def test_session_id_round_trips_into_the_next_run() -> None:
    stdout = "\n".join(json.dumps(event) for event in pi_events())
    captured = pi_agent.parse_session_id(stdout)
    assert captured == SESSION_ID
    argv = pi_agent.continuation_argv("continue", session=captured, binary="/usr/local/bin/pi")
    assert argv[argv.index("--session") + 1] == SESSION_ID


def test_parse_session_id_tolerates_other_event_shapes() -> None:
    assert pi_agent.parse_session_id("not json at all") == ""
    assert pi_agent.parse_session_id('{"type":"message_end","session_id":"ses_2"}') == "ses_2"
    assert pi_agent.parse_session_id('{"session": {"id": "ses_3"}}') == "ses_3"


def test_parse_reply_text_returns_the_final_assistant_text() -> None:
    stdout = "\n".join(json.dumps(event) for event in pi_events(turns=("working", "done")))
    assert pi_agent.parse_reply_text(stdout) == "done"


def test_parse_reply_text_ignores_thinking_and_unknown_events() -> None:
    stdout = "\n".join(json.dumps(event) for event in pi_events(reply="noted"))
    text = pi_agent.parse_reply_text(stdout)
    assert text == "noted"
    assert "private deliberation" not in text


def test_parse_reply_text_falls_back_to_streaming_deltas() -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "agent_start"}),
            json.dumps(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": "half a "},
                }
            ),
            json.dumps(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": "sentence"},
                }
            ),
        ]
    )
    assert pi_agent.parse_reply_text(stdout) == "half a sentence"


def test_parse_reply_text_falls_back_to_the_raw_output() -> None:
    assert pi_agent.parse_reply_text("pi said this in plain text") == "pi said this in plain text"


# --- running a continuation ------------------------------------------------


def test_run_continuation_runs_inside_the_allowed_project(tmp_path: Path, monkeypatch) -> None:
    binary, record, bin_dir = make_fake_binary(tmp_path)
    on_path(monkeypatch, bin_dir)
    project = tmp_path / "project"
    project.mkdir()

    argv = pi_agent.continuation_argv(
        "next step", provider="ollama", model="qwen3:8b", binary=str(binary)
    )
    result = pi_agent.run_continuation(
        argv, cwd=project, allowed_dirs=[str(project)], timeout=30.0
    )

    assert result.returncode == 0
    assert result.stopped is False and result.timed_out is False
    assert result.session_id == SESSION_ID
    assert pi_agent.parse_reply_text(result.stdout) == "noted"

    invocation = json.loads(record.read_text(encoding="utf-8"))
    assert Path(invocation["cwd"]) == project
    assert invocation["argv"] == [
        "-p",
        "--no-session",
        "--offline",
        "--mode",
        "json",
        "--no-approve",
        "--provider",
        "ollama",
        "--model",
        "qwen3:8b",
        "--",
        "next step",
    ]
    assert "--approve" not in invocation["argv"] and "-a" not in invocation["argv"]


def test_run_continuation_refuses_before_any_subprocess_starts(tmp_path: Path, monkeypatch) -> None:
    binary, record, bin_dir = make_fake_binary(tmp_path)
    on_path(monkeypatch, bin_dir)
    project = tmp_path / "project"
    project.mkdir()
    argv = pi_agent.continuation_argv("next step", binary=str(binary))

    with pytest.raises(pi_agent.PiAgentError):
        pi_agent.run_continuation(argv, cwd=project, allowed_dirs=[], timeout=30.0)
    assert not record.exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(pi_agent.PiAgentError):
        pi_agent.run_continuation(argv, cwd=outside, allowed_dirs=[str(project)], timeout=30.0)
    assert not record.exists()


def test_run_continuation_refuses_a_missing_binary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    project = tmp_path / "project"
    project.mkdir()

    with pytest.raises(pi_agent.PiAgentError) as raised:
        pi_agent.run_continuation(
            pi_agent.continuation_argv("next step"),
            cwd=project,
            allowed_dirs=[str(project)],
            timeout=5.0,
        )
    assert "not installed" in str(raised.value)


def test_run_continuation_refuses_an_argv_that_trusts_project_files(tmp_path: Path, monkeypatch) -> None:
    binary, record, bin_dir = make_fake_binary(tmp_path)
    on_path(monkeypatch, bin_dir)
    project = tmp_path / "project"
    project.mkdir()

    for flag in ("--approve", "-a", "--approve=yes"):
        argv = pi_agent.continuation_argv("next step", binary=str(binary))
        argv.insert(argv.index("--"), flag)  # smuggled into the flag position
        with pytest.raises(pi_agent.PiAgentError) as raised:
            pi_agent.run_continuation(argv, cwd=project, allowed_dirs=[str(project)], timeout=5.0)
        assert "project-local files" in str(raised.value)
    assert not record.exists()


def test_a_transcript_that_looks_like_a_flag_stays_text(tmp_path: Path, monkeypatch) -> None:
    binary, record, bin_dir = make_fake_binary(tmp_path)
    on_path(monkeypatch, bin_dir)
    project = tmp_path / "project"
    project.mkdir()

    argv = pi_agent.continuation_argv("--approve", binary=str(binary))
    result = pi_agent.run_continuation(
        argv, cwd=project, allowed_dirs=[str(project)], timeout=30.0
    )

    assert result.returncode == 0
    invocation = json.loads(record.read_text(encoding="utf-8"))
    assert invocation["argv"][-2:] == ["--", "--approve"]
    assert invocation["argv"].count("--no-approve") == 1


def test_stop_event_terminates_a_running_continuation(tmp_path: Path, monkeypatch) -> None:
    binary, record, bin_dir = make_fake_binary(tmp_path, sleep_seconds=5.0)
    on_path(monkeypatch, bin_dir)
    project = tmp_path / "project"
    project.mkdir()

    stop_event = threading.Event()
    threading.Timer(0.6, stop_event.set).start()
    argv = pi_agent.continuation_argv("next step", binary=str(binary))
    result = pi_agent.run_continuation(
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

    argv = pi_agent.continuation_argv("next step", binary=str(binary))
    result = pi_agent.run_continuation(argv, cwd=project, allowed_dirs=[str(project)], timeout=0.5)
    assert result.timed_out is True


def test_run_continuation_reports_a_failing_run(tmp_path: Path, monkeypatch) -> None:
    binary, _record, bin_dir = make_fake_binary(tmp_path, exit_code=3)
    on_path(monkeypatch, bin_dir)
    project = tmp_path / "project"
    project.mkdir()

    argv = pi_agent.continuation_argv("next step", binary=str(binary))
    result = pi_agent.run_continuation(
        argv, cwd=project, allowed_dirs=[str(project)], timeout=30.0
    )
    assert result.returncode == 3
    assert pi_agent.parse_reply_text(result.stdout) == "noted"


# --- plan, checkpoints, audit log -------------------------------------------


def test_plan_and_checkpoints_survive_a_reload(tmp_path: Path) -> None:
    path = tmp_path / "pi-plan.json"
    store = pi_agent.PlanStore(path)
    store.save(
        project="/home/user/project",
        steps=["run the tests", "open an issue"],
        session_id=SESSION_ID,
    )
    store.checkpoint(
        project="/home/user/project", step="run the tests", session_id=SESSION_ID, status="finished"
    )

    reloaded = pi_agent.PlanStore(path).load()
    assert reloaded["steps"] == ["run the tests", "open an issue"]
    assert reloaded["session_id"] == SESSION_ID
    assert len(reloaded["checkpoints"]) == 1
    checkpoint = reloaded["checkpoints"][0]
    assert checkpoint["step"] == "run the tests"
    assert checkpoint["session_id"] == SESSION_ID
    assert checkpoint["when"]


def test_plan_checkpoints_append_without_losing_the_plan(tmp_path: Path) -> None:
    path = tmp_path / "pi-plan.json"
    store = pi_agent.PlanStore(path)
    store.save(project="/p", steps=["a", "b"])
    store.checkpoint(project="/p", step="a", status="finished")
    store.checkpoint(project="/p", step="b", status="stopped")

    reloaded = pi_agent.PlanStore(path).load()
    assert [item["step"] for item in reloaded["checkpoints"]] == ["a", "b"]
    assert reloaded["steps"] == ["a", "b"]


def test_plan_store_tolerates_a_missing_or_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "pi-plan.json"
    empty = {"project": "", "steps": [], "checkpoints": [], "session_id": ""}
    assert pi_agent.PlanStore(path).load() == empty
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert pi_agent.PlanStore(path).load() == empty


def test_plan_store_migrates_the_plan_from_the_earlier_feature_slice(tmp_path: Path) -> None:
    old = tmp_path / "opencode-plan.json"
    old.write_text(
        json.dumps({"version": 1, "project": "/home/user/project", "steps": ["a", "b"]}),
        encoding="utf-8",
    )
    path = tmp_path / "pi-plan.json"

    reloaded = pi_agent.PlanStore(path).load()
    assert reloaded["project"] == "/home/user/project"
    assert reloaded["steps"] == ["a", "b"]
    assert old.exists()  # the old file is left alone

    pi_agent.PlanStore(path).checkpoint(project="/home/user/project", step="a", status="finished")
    assert path.is_file()
    assert json.loads(old.read_text(encoding="utf-8"))["steps"] == ["a", "b"]


def test_audit_log_records_the_run_without_the_message_body(tmp_path: Path) -> None:
    path = tmp_path / "pi-audit.log"
    log = pi_agent.AuditLog(path)
    argv = pi_agent.continuation_argv(
        "a secret sentence about the user",
        binary="/usr/local/bin/pi",
        provider="ollama",
        model="qwen3:8b",
    )
    log.record("run", argv=argv, detail="exit=0")
    log.record("stopped", detail="user pressed pause")

    lines = pi_agent.AuditLog(path).read()
    assert len(lines) == 2
    assert "<message:" in lines[0]
    assert "secret sentence" not in path.read_text(encoding="utf-8")
    assert "/usr/local/bin/pi -p --no-session --offline --mode json" in lines[0]
    assert lines[1].endswith("stopped user pressed pause")


def test_audit_log_appends_across_reloads(tmp_path: Path) -> None:
    path = tmp_path / "pi-audit.log"
    pi_agent.AuditLog(path).record("run", detail="first")
    pi_agent.AuditLog(path).record("run", detail="second")
    assert [line.split(" ", 1)[1] for line in pi_agent.AuditLog(path).read()] == [
        "run first",
        "run second",
    ]


def test_audit_log_keeps_the_history_from_the_earlier_feature_slice(tmp_path: Path) -> None:
    old = tmp_path / "opencode-audit.log"
    old.write_text("2026-01-01T00:00:00+00:00 run argv=/usr/bin/true\n", encoding="utf-8")
    path = tmp_path / "pi-audit.log"

    pi_agent.AuditLog(path).record("run", detail="after the switch")

    lines = pi_agent.AuditLog(path).read()
    assert len(lines) == 2
    assert lines[0].endswith("run argv=/usr/bin/true")
    assert lines[1].endswith("run after the switch")
