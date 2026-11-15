# SPDX-License-Identifier: MIT
"""Supervised hand-off to the OpenCode CLI, kept inside an explicit allowlist.

OpenCode is a general-purpose coding agent: pointed at a project it can edit
files, run tests and reach the network. "Let the project keep moving without
the user" therefore has to be built as a set of boundaries rather than as one
big feature, and this module is the first slice of it:

* **Read-only first.** ``read_only_status`` only ever runs git inspection
  commands, and ``suggested_next_actions`` turns that into plain text. Nothing
  here writes to a project.
* **An explicit allowlist.** Every path that reaches a subprocess is resolved
  (symlinks and ``..`` collapsed) and must sit inside one of the user's
  ``opencode_allowed_dirs``. An empty allowlist means nothing may run at all.
* **No silent permission growth.** ``--auto`` is the CLI's own
  "auto-approve permissions that are not explicitly denied (dangerous!)"
  switch. It is never emitted, and ``run_continuation`` refuses an argv that
  contains it, so a caller cannot smuggle it in.
* **Pause / kill / audit / checkpoint.** A run honours a stop event, is
  time-bounded, appends one line to an append-only audit log, and records a
  checkpoint (plus the session id, so a later run can continue the same
  session with ``-s``).

Still deliberately absent, per the issue's ordering: supervised edits/tests
run on the user's behalf, resumable unattended work in a sandbox, and
recovery-after-crash proof. See docs/OPENCODE.md.

No GTK here: everything is pure or ``subprocess`` so it can be tested offline.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

AUTO_FLAG = "--auto"
DEFAULT_TIMEOUT_SECONDS = 300.0
PLAN_FILENAME = "opencode-plan.json"
AUDIT_FILENAME = "opencode-audit.log"
# The CLI's own help fixes the contract this code emits:
# ``opencode run [message..]``, ``--dir <path>``, ``-m provider/model``,
# ``-s <session id>``, ``--format default|json``.
RUN_SUBCOMMAND = "run"


class OpenCodeError(RuntimeError):
    """Raised when a continuation cannot be started, or failed while running."""


@dataclass(slots=True, frozen=True)
class ProjectStatus:
    """What a project looks like right now, from read-only git inspection."""

    project: str = ""
    is_git_repo: bool = False
    branch: str = ""
    dirty: bool = False
    changed_files: tuple[str, ...] = ()
    recent_commits: tuple[str, ...] = ()
    diffstat: str = ""
    error: str = ""


@dataclass(slots=True, frozen=True)
class RunResult:
    """One OpenCode run: what it printed, how it ended, and its session id."""

    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    stopped: bool = False
    timed_out: bool = False
    session_id: str = ""


def find_binary() -> str | None:
    """Locate the OpenCode CLI, PATH first then the installer's own directory."""
    found = shutil.which("opencode")
    if found:
        return found
    candidate = Path.home() / ".opencode" / "bin" / "opencode"
    try:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    except OSError:
        return None
    return None


def _resolved(value: str | Path) -> Path | None:
    """Absolute, symlink-free view of a path, or None if it cannot be resolved.

    Resolving is what makes the allowlist meaningful: ``..`` segments and
    symlinks that point out of the permitted tree are collapsed to the real
    target, so a path that escapes is compared as the escape it actually is.
    """
    text = str(value).strip()
    if not text:
        return None
    try:
        return Path(text).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def project_allowed(path: str | Path, allowed_dirs: Sequence[str]) -> bool:
    """True when ``path`` is inside one of ``allowed_dirs``.

    An empty allowlist denies everything — the feature is opt-in per directory,
    and a missing setting must never mean "everything is fair game".
    """
    if not allowed_dirs:
        return False
    target = _resolved(path)
    if target is None:
        return False
    for entry in allowed_dirs:
        root = _resolved(entry)
        if root is None:
            continue
        # is_relative_to compares path parts, so /tmp/ab does not match /tmp/abc.
        if target == root or target.is_relative_to(root):
            return True
    return False


def _git(cwd: Path, *args: str, timeout: float = 10.0) -> tuple[int, str]:
    """Run one read-only git command and return ``(returncode, stdout)``.

    Negative codes stand for "git could not run at all", which callers report
    instead of guessing.
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return -1, "git is not installed"
    except subprocess.TimeoutExpired:
        return -2, "git timed out"
    except OSError as exc:
        return -3, str(exc)
    return completed.returncode, (completed.stdout or "").strip()


def read_only_status(
    project: str | Path, *, allowed_dirs: Sequence[str] | None = None
) -> ProjectStatus:
    """Inspect a project with git only: branch, dirty files, recent commits, diffstat.

    Pass ``allowed_dirs`` to have the allowlist enforced here as well as at the
    call site; the inspection commands themselves never modify the checkout.
    """
    target = _resolved(project)
    if target is None or not target.is_dir():
        return ProjectStatus(project=str(project), error="the project directory does not exist")
    if allowed_dirs is not None and not project_allowed(target, allowed_dirs):
        return ProjectStatus(
            project=str(target),
            error="the project is not inside an allowed directory",
        )

    inside, _ = _git(target, "rev-parse", "--is-inside-work-tree")
    if inside != 0:
        return ProjectStatus(project=str(target), error="this is not a git repository")

    branch_code, branch = _git(target, "rev-parse", "--abbrev-ref", "HEAD")
    if branch_code != 0 or not branch:
        # A repository with no commits yet has no branch to name.
        branch = ""

    status_code, raw_status = _git(target, "status", "--porcelain")
    changed = tuple(line for line in raw_status.splitlines() if line.strip()) if status_code == 0 else ()

    log_code, log = _git(target, "--no-pager", "log", "-n", "5", "--oneline")
    commits = tuple(log.splitlines()) if log_code == 0 else ()

    diff_code, diffstat = _git(target, "--no-pager", "diff", "--stat")
    if diff_code != 0:
        diffstat = ""

    return ProjectStatus(
        project=str(target),
        is_git_repo=True,
        branch=branch,
        dirty=bool(changed),
        changed_files=changed,
        recent_commits=commits,
        diffstat=diffstat,
    )


def _first_line(text: str, limit: int = 160) -> str:
    """One trimmed line of user text, used for suggestions and the audit log."""
    for line in text.splitlines():
        if line.strip():
            line = line.strip()
            return line[:limit] + ("…" if len(line) > limit else "")
    return ""


def suggested_next_actions(status: ProjectStatus, transcript: str = "") -> list[str]:
    """Plain-text suggestions for the next step — read-only, never commands.

    The transcript's first line is treated as the user's stated intent, so the
    advice is about *their* task rather than a generic checklist.
    """
    if status.error:
        return [f"Nothing to suggest yet: {status.error}."]

    lines: list[str] = []
    branch = status.branch or "the current branch"
    if status.dirty:
        lines.append(
            f"{len(status.changed_files)} file(s) have uncommitted changes on {branch}: "
            "finish or set those aside yourself before asking for a continuation, "
            "so the run starts from a checkpoint you agree with."
        )
    else:
        lines.append(f"{branch} is clean — a good checkpoint to continue from.")

    if status.recent_commits:
        lines.append(f"Last commit: {status.recent_commits[0]}")
    if status.diffstat:
        lines.append(f"Unstaged diff: {_first_line(status.diffstat)}")

    intent = _first_line(transcript)
    if intent:
        lines.append(f"Next action from your transcript: {intent}")
    else:
        lines.append(
            "Next action: say or type the change you want, then press Continue "
            "to run it with your approval."
        )

    lines.append(
        "Edits, tests and anything that leaves this machine need your approval: "
        "this app never auto-approves permissions."
    )
    return lines


def continuation_argv(
    message: str,
    *,
    project: str = "",
    model: str = "",
    session: str = "",
    binary: str | None = None,
    capture_session: bool = True,
) -> list[str]:
    """Build the argv for one supervised continuation run.

    Only flags the CLI documents are used, and only when a value is set, so
    the command stays minimal and reviewable. ``--format json`` is added when
    the session id has to be captured for a later ``-s`` continuation; ``--auto``
    is never added.
    """
    resolved_binary = binary or find_binary() or "opencode"
    argv = [resolved_binary, RUN_SUBCOMMAND, message]
    if project:
        argv += ["--dir", project]
    if model:
        argv += ["-m", model]
    if session:
        argv += ["-s", session]
    if capture_session:
        argv += ["--format", "json"]
    return argv


def _events(stdout: str) -> list[dict]:
    """Parse the JSON event lines ``--format json`` produces."""
    events: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def parse_session_id(stdout: str) -> str:
    """Recover the session id of a run so a later run can continue it with ``-s``.

    The CLI's JSON event shape is not stable across versions, so any session
    identifier found in the events is accepted rather than one exact key.
    """
    for event in _events(stdout):
        for key in ("sessionID", "session_id", "sessionId"):
            value = event.get(key)
            if isinstance(value, str) and value:
                return value
        session = event.get("session")
        if isinstance(session, dict):
            for key in ("id", "sessionID"):
                value = session.get(key)
                if isinstance(value, str) and value:
                    return value
    return ""


def parse_reply_text(stdout: str) -> str:
    """Best-effort plain text of a run's reply, for the AI response pane.

    Falls back to the raw output when no text event is recognised, so the
    user always sees what actually happened rather than a blank pane.
    """
    parts: list[str] = []
    for event in _events(stdout):
        for key in ("text", "content", "response"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    text = "\n".join(parts).strip()
    return text if text else stdout.strip()


def _contains_auto(argv: Sequence[str]) -> bool:
    return any(
        str(arg) == AUTO_FLAG or str(arg).startswith(f"{AUTO_FLAG}=") for arg in argv
    )


def run_continuation(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    stop_event: threading.Event | None = None,
    allowed_dirs: Sequence[str] = (),
) -> RunResult:
    """Run one continuation inside the allowlist, killable and time-bounded.

    Every guard runs *before* the subprocess starts: no OpenCode binary means a
    clear failure instead of a mysterious one, an argv carrying ``--auto`` is
    refused, and ``cwd`` must be an allowed project directory. The kill switch
    is polled while the child runs, so pausing really terminates it.
    """
    if find_binary() is None:
        raise OpenCodeError(
            "OpenCode is not installed: 'opencode' is not on PATH and "
            "~/.opencode/bin/opencode is missing."
        )
    if _contains_auto(argv):
        raise OpenCodeError(f"Refusing to run: {AUTO_FLAG} would auto-approve permissions.")
    if not project_allowed(cwd, allowed_dirs):
        raise OpenCodeError(
            f"Refusing to run in {cwd}: it is not inside an allowed project directory."
        )

    command = [str(part) for part in argv]
    if _contains_auto(command):
        raise OpenCodeError(f"Refusing to run: {AUTO_FLAG} would auto-approve permissions.")

    kill = stop_event or threading.Event()
    try:
        process = subprocess.Popen(
            command,
            cwd=str(_resolved(cwd)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, ValueError) as exc:
        raise OpenCodeError(f"Could not start OpenCode: {exc}") from exc

    stopped = False
    timed_out = False
    deadline = timeout
    while True:
        if kill.is_set():
            stopped = True
            break
        try:
            process.wait(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            pass
        deadline -= 0.1
        if deadline <= 0:
            timed_out = True
            break

    if stopped or timed_out:
        try:
            process.terminate()
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            try:
                process.kill()
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass
        except OSError:
            pass

    try:
        stdout, stderr = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
    except OSError as exc:
        raise OpenCodeError(f"Could not read OpenCode output: {exc}") from exc

    returncode = process.returncode if process.returncode is not None else -1
    return RunResult(
        stdout=stdout or "",
        stderr=stderr or "",
        returncode=returncode,
        stopped=stopped,
        timed_out=timed_out,
        session_id=parse_session_id(stdout or ""),
    )


class PlanStore:
    """The task plan and its checkpoints, as one JSON file under the XDG dir.

    Keeping the plan on disk is what makes the feature resumable: the app can
    be closed mid-plan and reopened later with the same steps, the same
    allowlist and the last session id.
    """

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
            path = base / "voice2text-ai" / PLAN_FILENAME
        self.path = path

    def load(self) -> dict:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            payload = None
        if not isinstance(payload, dict):
            payload = {}
        plan = payload.get("steps")
        checkpoints = payload.get("checkpoints")
        return {
            "project": str(payload.get("project", "")),
            "steps": [str(step) for step in (plan if isinstance(plan, list) else [])],
            "checkpoints": [item for item in checkpoints if isinstance(item, dict)]
            if isinstance(checkpoints, list)
            else [],
            "session_id": str(payload.get("session_id", "")),
        }

    def save(
        self,
        *,
        project: str,
        steps: Sequence[str],
        session_id: str = "",
        checkpoints: Sequence[dict] | None = None,
    ) -> None:
        existing = self.load()
        payload = {
            "version": 1,
            "project": str(project),
            "steps": [str(step).strip() for step in steps if str(step).strip()],
            "session_id": str(session_id or existing.get("session_id", "")),
            "checkpoints": list(checkpoints)
            if checkpoints is not None
            else existing.get("checkpoints", []),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temp_path.replace(self.path)

    def checkpoint(
        self,
        *,
        project: str,
        step: str,
        session_id: str = "",
        status: str = "finished",
        when: str | None = None,
    ) -> dict:
        """Append one checkpoint for a finished step and persist it."""
        entry = {
            "when": when or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "project": str(project),
            "step": str(step).strip()[:200],
            "status": str(status)[:40],
            "session_id": str(session_id),
        }
        record = self.load()
        checkpoints = record.get("checkpoints", [])
        checkpoints.append(entry)
        # A plan can run for a long time; only the newest checkpoints matter.
        checkpoints = checkpoints[-50:]
        self.save(
            project=project or record.get("project", ""),
            steps=record.get("steps", []),
            session_id=session_id or record.get("session_id", ""),
            checkpoints=checkpoints,
        )
        return entry


class AuditLog:
    """Append-only record of what was actually run.

    Only the command shape is recorded, never message bodies: a transcript can
    hold anything the user said, and the log is meant to be readable later
    without turning it into a copy of the user's private speech.
    """

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
            path = base / "voice2text-ai" / AUDIT_FILENAME
        self.path = path

    def record(self, event: str, *, argv: Sequence[str] = (), detail: str = "") -> str:
        """Append one line and return it."""
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {event}"
        if argv:
            line += f" argv={_redact(argv)}"
        if detail:
            line += f" {detail}"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line[:500].rstrip() + "\n")
        return line

    def read(self, limit: int = 50) -> list[str]:
        """Newest lines, for the status view."""
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, OSError):
            return []
        return lines[-limit:]


def _redact(argv: Sequence[str]) -> str:
    """Quote argv but replace the message body with its length.

    The message is the user's own words (often the transcript); the audit log
    only needs to show that a message of roughly that size was sent. Flags and
    paths stay readable, because those are exactly what needs auditing.
    """
    message_index = -1
    for index, arg in enumerate(argv):
        if str(arg) == RUN_SUBCOMMAND:
            message_index = index + 1
            break
    parts: list[str] = []
    for index, arg in enumerate(argv):
        if index == message_index:
            parts.append(f"<message:{len(str(arg))} chars>")
        else:
            parts.append(shlex.quote(str(arg)))
    return " ".join(parts)


