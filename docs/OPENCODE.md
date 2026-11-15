# OpenCode continuation (supervised, allowlisted)

Voice2Text AI can hand **one step at a time** of a project to the
[OpenCode](https://opencode.ai/) CLI: it runs `opencode run` in a directory you
have explicitly allowed, records what it ran, and lets you pause it at any
moment. This is the first slice of "let projects keep moving forward": read-only
status plus suggested next actions, and one supervised run. It is **off by
default**.

Nothing here is a replacement for reviewing a coding agent's work. The point of
this feature is that the boundaries are visible and enforced by the app, not
left to the CLI's defaults.

## Boundaries

| Boundary | What the app does |
| -------- | ---- |
| **Permitted directories** | `opencode_allowed_dirs` is an explicit list. Every path that reaches a subprocess is resolved first (symlinks and `..` collapsed), then must sit inside one entry. An empty list denies everything — it is never a wildcard. A sibling like `/tmp/ab` does not match an allowlist entry `/tmp/a`, and a symlink pointing out of the tree is rejected because the *real* target is what gets compared. |
| **Permitted tools** | The status view runs git's own inspection commands only (`rev-parse`, `status --porcelain`, `log --oneline`, `diff --stat`). A continuation runs exactly one command shape: `opencode run [message] [--dir] [-m] [-s] [--format json]`. |
| **Approval for destructive/external actions** | A continuation never starts on its own unless you switch on *both* `opencode_enabled` and `opencode_auto_run`, and even then it only follows a transcript that finished by itself. OpenCode's own permission prompts still apply, because the app never answers them: `--auto` ("auto-approve permissions that are not explicitly denied") is never emitted, and `run_continuation` refuses an argv that contains it, so it cannot be smuggled in by a caller. |
| **Persisted plan and checkpoints** | `$XDG_CONFIG_HOME/voice2text-ai/opencode-plan.json` holds the plan steps, the project, the last session id and the newest 50 checkpoints. Each finished run appends one checkpoint, so a later run can continue the same OpenCode session with `-s`. |
| **Audit log** | `$XDG_CONFIG_HOME/voice2text-ai/opencode-audit.log` is append-only: one line per status check, refusal, and run, with the argv quoted. The message body is replaced by its length (`<message:42 chars>`) — the log records what was *allowed*, not what you said. |
| **Stop on uncertainty** | A run is time-bounded (10 minutes by default). A project that is not inside the allowlist, a missing OpenCode binary, or an argv carrying `--auto` all fail *before* any subprocess starts, with the reason in the response pane. |
| **Pause / kill** | The **Stop** button (and closing the window) sets the run's stop event: the child process is terminated, the status line reads **Paused.**, and the checkpoint records `stopped`. The OpenCode button is greyed out while the feature is off, so there is nothing to press by accident. |
| **Rollback** | The app never edits your project, so it has nothing to undo. Rollback is git's: the status view shows the branch, the uncommitted files and the recent commits, and the checkpoint log shows which step ran when. Commit or stash before asking for a continuation and the checkpoint you want to return to is one `git reset` away. |

## Settings

In `~/.config/voice2text-ai/config.json`, edited from **Preferences → OpenCode**:

| Setting | Default | Meaning |
| ------- | ------- | ------- |
| `opencode_enabled` | `false` | Master switch. Off means no continuation can start; the read-only status view still inspects only an allowed project. |
| `opencode_auto_run` | `false` | Additionally lets a finished transcript trigger the continuation. |
| `opencode_project` | `""` | The project to continue. |
| `opencode_allowed_dirs` | `[]` | Directories OpenCode may work in (separated by `:` in the dialog). |
| `opencode_model` | `""` | `provider/model`, passed to `-m`; blank uses OpenCode's own default. |

## In the app

- **OpenCode status** (menu): read-only status of the project — branch,
  uncommitted files, recent commits, diffstat — plus plain-text *suggested
  next actions*. The suggestions are advice, never commands.
- **OpenCode** button / **OpenCode continue** (menu): one supervised run of the
  current transcript inside the allowed project. The reply lands in the AI
  response pane; the run is audited and checkpointed.
- **Stop**: pauses the run (see above).

OpenCode itself is installed and configured by you — this app does not install
it, does not manage its credentials, and does not touch its configuration.
`v2t-backup` still archives `~/.config/opencode/opencode.json` and friends
(secrets redacted), as before.

## What is deliberately not enabled yet

The issue's ordering is being followed, so these stay open:

- **Supervised edits/tests on your behalf** — approving an individual edit or
  test run, and showing a diff before it is applied.
- **Resumable unattended work in a sandbox** — running for many steps with no
  press from you, inside a container or a dedicated worktree.
- **Recovery tests** — proving a plan resumes correctly after a crash, a
  killed run, or a corrupt plan file.
- **Live proof that permissions cannot expand** — the code refuses `--auto`
  and the tests assert it, but the guarantee that OpenCode itself cannot
  escalate a permission on a real install has not been demonstrated.

Until those land, treat a continuation as "ask the agent, then review what it
did", not as hands-free automation.
