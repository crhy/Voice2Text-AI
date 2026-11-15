# pi continuation (supervised, allowlisted)

Voice2Text AI can hand **one step at a time** of a project to the
[pi](https://github.com/earendil-works/pi) coding agent: it runs pi in a directory
you have explicitly allowed, records what it ran, and lets you pause it at any
moment. This is the first slice of "let projects keep moving forward": read-only
status plus suggested next actions, and one supervised run. It is **off by
default**.

This replaced an earlier slice that drove the OpenCode CLI. pi is the harness this
app targets (same pass rate in rhY's A/B, roughly 30% less time and 35% fewer
tokens). Settings keys, the plan file and the audit log written under the old
names are read once and carried over, so nothing you had configured is lost.

Nothing here is a replacement for reviewing a coding agent's work. The point of
this feature is that the boundaries are visible and enforced by the app, not
left to the CLI's defaults — and pi, unlike an interactive assistant, does not
ask before each tool call.

## Boundaries

| Boundary | What the app does |
| -------- | ---- |
| **Permitted directories** | `pi_allowed_dirs` is an explicit list. Every path that reaches a subprocess is resolved first (symlinks and `..` collapsed), then must sit inside one entry. An empty list denies everything — it is never a wildcard. A sibling like `/tmp/ab` does not match an allowlist entry `/tmp/a`, and a symlink pointing out of the tree is rejected because the *real* target is what gets compared. |
| **Permitted tools** | The status view runs git's own inspection commands only (`rev-parse`, `status --porcelain`, `log --oneline`, `diff --stat`). A continuation runs exactly one command shape: `pi -p --no-session --offline --mode json --no-approve [--provider <name>] [--model <id>] -- <message>`. pi has no `--dir`: the working directory of the child process is the allowed project, which is also where pi looks for its project configuration. |
| **Approval for destructive/external actions** | A continuation never starts on its own unless you switch on *both* `pi_enabled` and `pi_auto_run`, and even then it only follows a transcript that finished by itself. pi's `--approve` ("trust project-local files for this run") is never emitted, `--no-approve` is emitted instead, and `run_continuation` refuses an argv that contains `--approve` or `-a`, so a project folder cannot load extensions or settings behind your back — not even if your global `defaultProjectTrust` says "always" — and the flag cannot be smuggled in by a caller. |
| **Persisted plan and checkpoints** | `$XDG_CONFIG_HOME/voice2text-ai/pi-plan.json` holds the plan steps, the project, the last session id and the newest 50 checkpoints. Each finished run appends one checkpoint. A plan written under the old name is copied over the first time it is read. |
| **Audit log** | `$XDG_CONFIG_HOME/voice2text-ai/pi-audit.log` is append-only: one line per status check, refusal, and run, with the argv quoted. The message body is replaced by its length (`<message:42 chars>`) — the log records what was *allowed*, not what you said. History kept under the old name is carried into this file once. |
| **Stop on uncertainty** | A run is time-bounded (10 minutes by default). A project that is not inside the allowlist, a missing pi binary, an argv carrying `--approve`, and a session id that is not a plain id all fail *before* any subprocess starts, with the reason in the response pane. |
| **Pause / kill** | The **Stop** button (and closing the window) sets the run's stop event: the child process is terminated, the status line reads **Paused.**, and the checkpoint records `stopped`. The pi button is greyed out while the feature is off, so there is nothing to press by accident. |
| **Rollback** | The app never edits your project, so it has nothing to undo. Rollback is git's: the status view shows the branch, the uncommitted files and the recent commits, and the checkpoint log shows which step ran when. Commit or stash before asking for a continuation and the checkpoint you want to return to is one `git reset` away. |

## Settings

In `~/.config/voice2text-ai/config.json`, edited from **Preferences → pi continuation**:

| Setting | Default | Meaning |
| ------- | ------- | ------- |
| `pi_enabled` | `false` | Master switch. Off means no continuation can start; the read-only status view still inspects only an allowed project. |
| `pi_auto_run` | `false` | Additionally lets a finished transcript trigger the continuation. |
| `pi_project` | `""` | The project to continue. |
| `pi_allowed_dirs` | `[]` | Directories pi may work in (separated by `:` in the dialog). |
| `pi_provider` | `""` | Provider name, passed to `--provider`; blank lets pi resolve the model itself. |
| `pi_model` | `""` | Model as pi names it (a `provider/id` pattern works on its own), passed to `--model`; blank uses pi's default. |

The keys of the earlier slice (`opencode_enabled`, `opencode_auto_run`,
`opencode_project`, `opencode_allowed_dirs`, `opencode_model`) are read once and
copied onto the names above when the new key is absent. They are never written
again, and a value you set under the new name always wins.

## In the app

- **pi status** (menu): read-only status of the project — branch,
  uncommitted files, recent commits, diffstat — plus plain-text *suggested
  next actions*. The suggestions are advice, never commands.
- **pi** button / **pi continue** (menu): one supervised run of the current
  transcript inside the allowed project. The reply lands in the AI response
  pane; the run is audited and checkpointed.
- **Stop**: pauses the run (see above).

The reply shown in the response pane is the **final assistant text** of pi's
JSON event stream: pi writes one JSON record per line, unknown event types are
ignored, thinking blocks are not displayed, and the raw output is shown if no
assistant message can be recognised — so a version change in pi cannot blank
the pane.

Runs are **ephemeral** (`--no-session`): pi's own context is not carried from
one step to the next, which is why the plan file, not pi's session log, is the
thing that remembers your steps. `continuation_argv` can resume a session
(`--session <id>`, replacing `--no-session`) for a later slice; the id is
validated so it cannot name a path or an option.

pi itself is installed and configured by you — this app does not install it,
does not manage its credentials, and does not touch `~/.pi`. `v2t-backup` still
archives OpenCode configuration as it always did; archiving pi's own state is a
separate, open item.

## What is deliberately not enabled yet

The issue's ordering is being followed, so these stay open:

- **Supervised edits/tests on your behalf** — approving an individual edit or
  test run, and showing a diff before it is applied.
- **Resumable unattended work in a sandbox** — running for many steps with no
  press from you, inside a container or a dedicated worktree.
- **Recovery tests** — proving a plan resumes correctly after a crash, a
  killed run, or a corrupt plan file.
- **Live proof that permissions cannot expand** — the code refuses `--approve`
  and the tests assert it, but pi runs its tools with the permissions of your
  account, and `--offline` only disables startup network operations. The
  real boundary against a wrong or hostile generated action is an operating-
  system or container boundary, which this app does not create.

Until those land, treat a continuation as "ask the agent, then review what it
did", not as hands-free automation.
