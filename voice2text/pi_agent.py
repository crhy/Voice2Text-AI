# SPDX-License-Identifier: MIT
"""Supervised hand-off to the pi coding agent, kept inside an explicit allowlist.

pi is a general-purpose coding agent: pointed at a project it can read, edit and
execute with the permissions of the account that started it, and it does not
ask before every tool call. "Let the project keep moving without the user"
therefore has to be built as a set of boundaries rather than as one big feature,
and this module is the first slice of it:

* **Read-only first.** ``read_only_status`` only ever runs git inspection
  commands, and ``suggested_next_actions`` turns that into plain text. Nothing
  here writes to a project.
* **An explicit allowlist.** Every path that reaches a subprocess is resolved
  (symlinks and ``..`` collapsed) and must sit inside one of the user's
  ``pi_allowed_dirs``. An empty allowlist means nothing may run at all.
* **No silent permission growth.** pi's ``--approve`` ("trust project-local
  files for this run") is never emitted, ``run_continuation`` refuses an argv
  that contains it, and ``--no-approve`` is emitted instead so a project folder
  cannot load extensions or settings behind the user's back — not even if the
  user's global ``defaultProjectTrust`` says "always".
* **Pause / kill / audit / checkpoint.** A run honours a stop event, is
  time-bounded, appends one line to an append-only audit log, and records a
  checkpoint (plus the session id, so a later run can resume the same pi
  session with ``--session``).

This module replaces the OpenCode experiment of the same feature (settings
keys and plan/audit files from that era are migrated once, see
``_migrate_legacy``); pi is the harness this app targets.

Still deliberately absent, per the issue's ordering: supervised edits/tests
run on the user's behalf, resumable unattended work in a sandbox, and
recovery-after-crash proof. See docs/PI_AGENT.md.

No GTK here: everything is pure or ``subprocess`` so it can be tested offline.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# The command shape this app emits, fixed by ``pi --help``:
# ``pi -p --no-session --offline --mode json "<prompt>"`` with the model
# chosen by ``--provider <name> --model <id>``. Resuming a session swaps
# ``--no-session`` for ``--session <id>``.
BASE_ARGS = ("-p", "--no-session", "--offline", "--mode", "json")
END_OF_OPTIONS = "--"
# pi's own "trust project-local files for this run" switch and its short form.
# Emitting it would let a project folder load extensions and settings without
# the user deciding, so it is refused; ``--no-approve`` is emitted instead.
APPROVE_FLAGS = ("--approve", "-a")
NO_APPROVE_FLAG = "--no-approve"
DEFAULT_TIMEOUT_SECONDS = 300.0
PLAN_FILENAME = "pi-plan.json"
AUDIT_FILENAME = "pi-audit.log"
# Names used by the earlier OpenCode-based slice of this same feature. Read
# once and carried over so existing settings and history are not lost.
LEGACY_PLAN_FILENAME = "opencode-plan.json"
LEGACY_AUDIT_FILENAME = "opencode-audit.log"
# pi accepts a session file path or a (partial) UUID for --session. Only a
# plain id that cannot look like an option is ever emitted here: no path
# separators, no leading dash.
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class PiAgentError(RuntimeError):
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
    """One pi run: what it printed, how it ended, and its session id."""

    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    stopped: bool = False
    timed_out: bool = False
    session_id: str = ""


def find_binary() -> str | None:
    """Locate the pi CLI, PATH first then the locations pipx/user installs use."""
    found = shutil.which("pi")
    if found:
        return found
    for candidate in (
        Path.home() / ".local" / "bin" / "pi",
        Path.home() / ".pi" / "bin" / "pi",
    ):
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        except OSError:
            continue
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
            "Next action: say or type the change you want, then press pi "
            "to run it with your approval."
        )

    lines.append(
        "pi does not ask before each tool call, so this app only ever runs it "
        "inside the directories you allow and never trusts project-local files: "
        "review what it did before it leaves your machine."
    )
    return lines


def continuation_argv(
    message: str,
    *,
    provider: str = "",
    model: str = "",
    session: str = "",
    binary: str | None = None,
) -> list[str]:
    """Build the argv for one supervised continuation run.

    Only flags pi documents are used, and only when a value is set, so the
    command stays minimal and reviewable. The run is ephemeral (``--no-session``)
    unless a session id is handed back explicitly, in which case pi resumes that
    session with ``--session`` instead. ``--no-approve`` is always emitted and
    ``--approve`` never is. pi has no ``--dir``: the working directory set by
    ``run_continuation`` is what places the run inside the allowed project.
    """
    resolved_binary = binary or find_binary() or "pi"
    if session:
        if not _SESSION_ID_RE.fullmatch(session):
            raise PiAgentError(
                "Refusing to resume: the session id is not a plain id, so it "
                "could name a path or an option."
            )
        argv = [resolved_binary, "-p", "--offline", "--mode", "json", "--session", session]
    else:
        argv = [resolved_binary, *BASE_ARGS]
    argv += [NO_APPROVE_FLAG]
    if provider:
        argv += ["--provider", provider]
    if model:
        argv += ["--model", model]
    # ``--`` keeps a transcript that starts with a dash from being read as a flag.
    return [*argv, END_OF_OPTIONS, message]


def _events(stdout: str) -> list[dict]:
    """Parse the JSONL records pi writes to stdout.

    One JSON object per LF-terminated line. Anything that is not such an
    object is ignored: the event vocabulary grows, and an unknown event must
    never break a run that finished.
    """
    events: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip().rstrip("\r")
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
    """Recover the session id of a run so a later run can resume it with ``--session``.

    The first JSON-mode record is pi's session header (``{"type":"session",
    "id":...}``), but the event shape is not guaranteed across versions, so any
    session identifier found in the events is accepted rather than one exact key.
    """
    for event in _events(stdout):
        if event.get("type") == "session":
            value = event.get("id")
            if isinstance(value, str) and value:
                return value
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


def _message_text(message: dict) -> str:
    """Plain text of one message record: a string, or text blocks only.

    Thinking/reasoning blocks are skipped on purpose — the response pane shows
    what pi concluded, not its internal deliberation.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def parse_reply_text(stdout: str) -> str:
    """Best-effort plain text of a run's reply, for the AI response pane.

    The authoritative value is the text of the *final* assistant message
    (``message_end``, or ``turn_end`` for versions that only emit that).
    Streaming deltas and ``agent_end`` are accepted as fallbacks, and the raw
    output is returned when nothing is recognised, so the user always sees
    what actually happened rather than a blank pane.
    """
    events = _events(stdout)

    final = ""
    for event in events:
        if event.get("type") not in ("message_end", "turn_end"):
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        text = _message_text(message).strip()
        if text:
            final = text
    if final:
        return final

    # A run killed mid-stream still has its deltas.
    deltas: list[str] = []
    for event in events:
        if event.get("type") != "message_update":
            continue
        nested = event.get("assistantMessageEvent")
        if isinstance(nested, dict) and nested.get("type") == "text_delta":
            delta = nested.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
    text = "".join(deltas).strip()
    if text:
        return text

    for event in events:
        if event.get("type") != "agent_end":
            continue
        messages = event.get("messages")
        if not isinstance(messages, list):
            continue
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                text = _message_text(message).strip()
                if text:
                    return text

    for event in events:
        for key in ("text", "content", "response"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return stdout.strip()


def _contains_approve(argv: Sequence[str]) -> str:
    """Return the offending flag when an argv asks pi to trust project files.

    Only the arguments before ``--`` are flags; anything after it is the
    message, where such a word is inert text rather than a switch.
    """
    for arg in argv:
        text = str(arg)
        if text == END_OF_OPTIONS:
            break
        if text in APPROVE_FLAGS or text.startswith("--approve="):
            return text
    return ""


def run_continuation(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    stop_event: threading.Event | None = None,
    allowed_dirs: Sequence[str] = (),
) -> RunResult:
    """Run one continuation inside the allowlist, killable and time-bounded.

    Every guard runs *before* the subprocess starts: no pi binary means a
    clear failure instead of a mysterious one, an argv carrying ``--approve``
    is refused, and ``cwd`` must be an allowed project directory. The kill
    switch is polled while the child runs, so pausing really terminates it.
    """
    if find_binary() is None:
        raise PiAgentError(
            "pi is not installed: 'pi' is not on PATH and neither "
            "~/.local/bin/pi nor ~/.pi/bin/pi is present."
        )
    flag = _contains_approve(argv)
    if flag:
        raise PiAgentError(
            f"Refusing to run: {flag} would trust project-local files without asking."
        )
    if not project_allowed(cwd, allowed_dirs):
        raise PiAgentError(
            f"Refusing to run in {cwd}: it is not inside an allowed project directory."
        )

    command = [str(part) for part in argv]
    smuggled = _contains_approve(command)
    if smuggled:
        raise PiAgentError(
            f"Refusing to run: {smuggled} would trust project-local files without asking."
        )

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
        raise PiAgentError(f"Could not start pi: {exc}") from exc

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
        raise PiAgentError(f"Could not read pi output: {exc}") from exc

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
        self.legacy_path = path.parent / LEGACY_PLAN_FILENAME

    def _migrate_legacy(self) -> None:
        """Carry the plan written by the OpenCode-era slice over to the new name.

        One-time and non-destructive: the old file is left in place, and the
        new file wins as soon as it exists.
        """
        if self.path.exists() or not self.legacy_path.exists():
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(self.legacy_path.read_bytes())
        except OSError:
            # Migration is best-effort; load() falls back to defaults.
            return

    def load(self) -> dict:
        self._migrate_legacy()
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
        self.legacy_path = path.parent / LEGACY_AUDIT_FILENAME

    def _migrate_legacy(self) -> None:
        """Keep the history written under the OpenCode-era name readable.

        Copied once into the new file, which is what every later append uses;
        the old file is never modified or removed.
        """
        if self.path.exists() or not self.legacy_path.exists():
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(self.legacy_path.read_bytes())
        except OSError:
            return

    def record(self, event: str, *, argv: Sequence[str] = (), detail: str = "") -> str:
        """Append one line and return it."""
        self._migrate_legacy()
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
        self._migrate_legacy()
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, OSError):
            return []
        return lines[-limit:]


def _redact(argv: Sequence[str]) -> str:
    """Quote argv but replace the message body with its length.

    The message is the user's own words (often the transcript); the audit log
    only needs to show that a message of roughly that size was sent. Flags and
    paths stay readable, because those are exactly what needs auditing. The
    message is whatever follows ``--``, which is where this app puts it.
    """
    parts: list[str] = []
    redacting = False
    for arg in argv:
        text = str(arg)
        if text == END_OF_OPTIONS:
            redacting = True
            parts.append(text)
            continue
        if redacting:
            parts.append(f"<message:{len(text)} chars>")
        else:
            parts.append(shlex.quote(text))
    return " ".join(parts)
