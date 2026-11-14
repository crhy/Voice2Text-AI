"""Programmable actions the hands-free assistant can act on.

An action is a trigger phrase ("play bossnova", "open <bookmark>") plus a name
from a **closed** adapter set that turns the captured parameters into an argv
list. Nothing in this module runs a command: ``plan()`` and ``argv_for()`` are
the reversible test mode, and the caller that decides to execute does so with
``subprocess.Popen(argv)`` — never through a shell, so a word heard from the
microphone can never be reinterpreted as a command line.

Two closed adapter sets exist. ``ADAPTERS`` are the safe ones the assistant may
run hands-free. ``PRIVILEGED_ADAPTERS`` (messaging, deleting, spending) are
recognised by the schema so they can be imported, reviewed and disabled, but
``argv_for`` refuses them unless the caller passes ``allow_privileged=True``
explicitly, and a privileged action without ``confirm=True`` is rejected at
load time: it can never reach the user without a confirmation dialog in front
of it.

A trigger has to account for the whole utterance, and a trailing placeholder
captures the rest of it. That is what makes matching deterministic, and it is
why an action cannot fire on a sentence that merely mentions it: "ask me about
opening github" triggers nothing.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# Safe adapters: open a page, play a stream, search for a phrase.
ADAPTERS = ("open-url", "play-url", "search-text")
# Privileged adapters: recognised by the schema, never run without an opt-in.
PRIVILEGED_ADAPTERS = ("send-text", "delete-path", "spend-credit")
KNOWN_ADAPTERS = ADAPTERS + PRIVILEGED_ADAPTERS

PARAM_TYPES = ("url", "bookmark", "text")

# A placeholder captures one word, so this cap only guards fixed values.
MAX_PARAM_CHARS = 200

# Named bookmarks shipped with the app. The issue's "Bossanovaville" nickname
# is Soma FM's bossa-nova stream, served as an https playlist mpv can follow.
BOOKMARKS: dict[str, str] = {
    "bossnova": "https://somafm.com/m3u/bossa.m3u",
    "bossanovaville": "https://somafm.com/m3u/bossa.m3u",
    "ollama": "https://ollama.com/library",
    "github": "https://github.com/crhy/Voice2Text-AI",
}

_OPENER = "xdg-open"
_PLAYER = "mpv"
# Destructive actions go to the desktop trash, which is reversible: this module
# never builds a command that erases anything permanently.
_TRASH = ("gio", "trash", "--")


class ActionError(ValueError):
    """An action, phrase, parameter or stored payload is not usable."""


class ActionConflict(ActionError):
    """Two enabled triggers matched equally well; guessing is never correct."""


class _NeverExecuted:
    """Sentinel handed to plan(): it reports having run nothing, ever."""

    returncode = 0

    def poll(self) -> int:
        return self.returncode


_NEVER_EXECUTED = _NeverExecuted()


@dataclass(frozen=True, slots=True)
class Param:
    """A typed parameter of a trigger phrase.

    ``value`` is set for a fixed parameter baked into the action (the Bossanovaville
    stream URL, say); a ``<name>`` placeholder in the phrase captures the value
    from what the user said instead.
    """

    name: str
    type: str = "text"
    value: str = ""


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """One programmable action: trigger phrase, parameters, adapter, policy."""

    id: str
    phrase: str
    params: tuple[Param, ...] = ()
    adapter: str = "open-url"
    confirm: bool = False
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class Match:
    """The result of matching an utterance against an action."""

    action: ActionSpec
    text: str
    params: dict[str, str]
    exact: bool = False


class _Placeholder(str):
    """Marks a phrase token that captures one spoken word."""

    __slots__ = ()


_PLACEHOLDER = re.compile(r"<\s*([a-z0-9_]+)\s*>")
_PUNCT = re.compile(r"[^\w\s]+")
_SPACES = re.compile(r"\s+")


def normalize_phrase(text: str) -> str:
    """Casefold, drop punctuation, collapse whitespace.

    ``<name>`` placeholders survive as single tokens; every other punctuation
    character becomes a space.
    """

    text = text.casefold()
    parts: list[str] = []
    pos = 0
    for match in _PLACEHOLDER.finditer(text):
        head = text[pos:match.start()]
        parts.append(_PUNCT.sub(" ", head))
        parts.append(f" <{match.group(1)}> ")
        pos = match.end()
    parts.append(_PUNCT.sub(" ", text[pos:]))
    return _SPACES.sub(" ", "".join(parts)).strip()


def _phrase_tokens(phrase: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for token in normalize_phrase(phrase).split():
        if len(token) > 2 and token.startswith("<") and token.endswith(">"):
            tokens.append(_Placeholder(token[1:-1]))
        else:
            tokens.append(token)
    return tuple(tokens)


def _capture(action: ActionSpec, text: str) -> dict[str, str] | None:
    """Match the action's phrase against text, returning captured parameters."""

    tokens = _phrase_tokens(action.phrase)
    words = normalize_phrase(text).split()
    if not tokens or not words:
        return None

    params: dict[str, str] = {}
    index = 0
    for position, token in enumerate(tokens):
        if isinstance(token, _Placeholder):
            if position == len(tokens) - 1:
                # A trailing placeholder takes the rest of the utterance, so
                # "search for <text>" can capture a whole question.
                if index >= len(words):
                    return None
                params[str(token)] = " ".join(words[index:])
                index = len(words)
            else:
                if index >= len(words):
                    return None
                params[str(token)] = words[index]
                index += 1
        else:
            if index >= len(words) or words[index] != token:
                return None
            index += 1
    if index != len(words):
        # Words the phrase does not account for: this action does not match.
        return None
    return params


def _specificity(action: ActionSpec) -> tuple[int, int]:
    """How specific a trigger is: literal words first, then total length."""

    tokens = _phrase_tokens(action.phrase)
    literals = sum(1 for token in tokens if not isinstance(token, _Placeholder))
    return (literals, len(tokens))


def match_action(text: str, actions: Iterable[ActionSpec]) -> Match | None:
    """Return the single best enabled action for text, or None.

    An exact normalized phrase wins outright; otherwise the longest (most
    specific) matching trigger wins. Two triggers of equal specificity that both
    match raise ActionConflict instead of picking one.
    """

    candidates = [action for action in actions if action.enabled]
    if not candidates:
        return None

    normalized = normalize_phrase(text)
    if not normalized:
        return None

    exact = [action for action in candidates if normalize_phrase(action.phrase) == normalized]
    if len(exact) > 1:
        raise ActionConflict(
            f"actions {', '.join(sorted(action.id for action in exact))} share the trigger {normalized!r}"
        )
    if exact:
        action = exact[0]
        return Match(action, text, _capture(action, text) or {}, exact=True)

    scored = []
    for action in candidates:
        captured = _capture(action, text)
        if captured is not None:
            scored.append((_specificity(action), action, captured))
    if not scored:
        return None

    best = max(scored, key=lambda entry: entry[0])
    tied = [entry for entry in scored if entry[0] == best[0]]
    if len(tied) > 1:
        raise ActionConflict(
            "actions "
            + ", ".join(sorted(entry[1].id for entry in tied))
            + f" match {normalized!r} with equal specificity"
        )
    return Match(best[1], text, best[2])


def _clean(value: str, name: str) -> str:
    value = _SPACES.sub(" ", value).strip()
    if not value or len(value) > MAX_PARAM_CHARS or any(ord(char) < 32 for char in value):
        raise ActionError(f"parameter {name} must be 1-{MAX_PARAM_CHARS} printable characters")
    return value


def _http_url(url: str, name: str) -> str:
    if not url.startswith(("http://", "https://")):
        raise ActionError(f"{name} must be an http or https URL, got {url!r}")
    return url


def _quote(text: str) -> str:
    # Escapes everything a URL query cannot carry literally, so a spoken search
    # phrase cannot smuggle extra query parameters into the URL.
    return re.sub(r"[^A-Za-z0-9._~-]", lambda match: f"%{ord(match.group(0)):02X}", text)


def _resolve(action: ActionSpec, params: dict[str, str]) -> dict[str, str]:
    """Apply each declared parameter's type to the value it carries."""

    resolved: dict[str, str] = {}
    for param in action.params:
        raw = params.get(param.name, param.value)
        if not raw:
            raise ActionError(f"action {action.id} is missing parameter {param.name}")
        value = _clean(raw, param.name)
        if param.type == "url":
            value = _http_url(value, f"action {action.id} parameter {param.name}")
        elif param.type == "bookmark":
            if value not in BOOKMARKS:
                raise ActionError(f"action {action.id} names unknown bookmark {value!r}")
            value = _http_url(BOOKMARKS[value], f"bookmark {value}")
        resolved[param.name] = value
    return resolved


def _adapter_input(action: ActionSpec, resolved: dict[str, str], required: str) -> str:
    """Pick the value an adapter consumes, by name or by being the only one."""

    value = resolved.get(required)
    if value is None and len(resolved) == 1:
        value = next(iter(resolved.values()))
    if not value:
        raise ActionError(f"adapter {action.adapter} needs a {required} parameter")
    return value


def argv_for(
    action: ActionSpec,
    params: dict[str, str],
    *,
    allow_privileged: bool = False,
) -> list[str]:
    """Build the argv list for an action: a list, never a shell string, never run."""

    if action.adapter in PRIVILEGED_ADAPTERS:
        if not allow_privileged:
            raise ActionError(
                f"action {action.id} uses privileged adapter {action.adapter}; it runs only "
                "when the caller explicitly permits privileged actions"
            )
        if not action.confirm:
            raise ActionError(f"privileged action {action.id} must require confirmation")
        resolved = _resolve(action, params)
        if action.adapter == "delete-path":
            target = _clean(resolved.get("path", ""), "path")
            if target.startswith(("/", "~")) or ".." in target.split("/"):
                raise ActionError(f"path {target!r} must stay inside the user's own files")
            return [*_TRASH, target]
        if action.adapter == "send-text":
            base = _http_url(resolved.get("url", ""), f"action {action.id} parameter url")
            body = _clean(resolved.get("text", ""), "text")
            return [_OPENER, f"{base}?text={_quote(body)}"]
        raise ActionError(f"privileged adapter {action.adapter} has no executable adapter here")

    resolved = _resolve(action, params)
    if action.adapter == "open-url":
        return [_OPENER, _http_url(_adapter_input(action, resolved, "url"), "url")]
    if action.adapter == "play-url":
        return [_PLAYER, "--no-terminal", _http_url(_adapter_input(action, resolved, "url"), "url")]
    if action.adapter == "search-text":
        return [_OPENER, f"https://duckduckgo.com/?q={_quote(_adapter_input(action, resolved, 'text'))}"]
    raise ActionError(f"unknown adapter {action.adapter!r}")


def plan(
    action: ActionSpec,
    text: str,
    *,
    process: Any = _NEVER_EXECUTED,
    allow_privileged: bool = False,
) -> list[str]:
    """Reversible test mode: the argv that *would* run, with nothing executed.

    ``process`` defaults to a sentinel that reports having started nothing; a
    live process handle is rejected, so a dry run stays observable and inert.
    """

    captured = _capture(action, text)
    if captured is None:
        raise ActionError(f"{text!r} does not trigger {action.id}")
    argv = argv_for(action, captured, allow_privileged=allow_privileged)
    if process.poll() is None:
        raise ActionError("plan() never executes: it reports argv, it does not run it")
    return argv


def starter_actions() -> list[ActionSpec]:
    """The small safe starter set shipped with the app."""

    return [
        ActionSpec(
            id="play-bossnova",
            phrase="play bossnova",
            params=(Param("url", "url", BOOKMARKS["bossnova"]),),
            adapter="play-url",
        ),
        ActionSpec(
            id="open-bookmark",
            phrase="open <bookmark>",
            params=(Param("bookmark", "bookmark"),),
            adapter="open-url",
        ),
        ActionSpec(
            id="search-text",
            phrase="search for <text>",
            params=(Param("text", "text"),),
            adapter="search-text",
        ),
    ]


def _entry_of(action: ActionSpec) -> dict[str, Any]:
    return {
        "id": action.id,
        "phrase": action.phrase,
        "params": [
            {"name": param.name, "type": param.type, "value": param.value}
            for param in action.params
        ],
        "adapter": action.adapter,
        "confirm": action.confirm,
        "enabled": action.enabled,
    }


class ActionStore:
    """Versioned action registry in ``$XDG_CONFIG_HOME/voice2text-ai/actions.json``."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
            path = base / "voice2text-ai" / "actions.json"
        self.path = path
        self.actions: list[ActionSpec] = []

    def load(self) -> list[ActionSpec]:
        self.actions = self.parse(self._read_file())
        return self.actions

    def _read_file(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def parse(self, payload: dict[str, Any]) -> list[ActionSpec]:
        """Validate a payload (from disk or from an export) into action specs."""

        schema = payload.get("schema", SCHEMA_VERSION)
        if schema != SCHEMA_VERSION:
            raise ActionError(f"actions schema {schema!r} is not supported")

        raw_actions = payload.get("actions", [])
        if not isinstance(raw_actions, list):
            raise ActionError("actions must be a list")

        actions = [self._action_from_entry(entry) for entry in raw_actions]
        self._check_ids(actions)
        return actions

    def import_json(self, payload: Any, *, replace: bool = False) -> list[ActionSpec]:
        """Merge an exported payload into the registry, refusing id clashes."""

        if isinstance(payload, str):
            payload = json.loads(payload)
        incoming = self.parse(payload if isinstance(payload, dict) else {})
        if replace:
            self.actions = incoming
        else:
            existing = {action.id for action in self.actions}
            clash = sorted(action.id for action in incoming if action.id in existing)
            if clash:
                raise ActionError(
                    f"actions already defined: {', '.join(clash)}; pass replace=True to overwrite"
                )
            self.actions = [*self.actions, *incoming]
        self.validate(self.actions)
        return self.actions

    def export_json(self, actions: list[ActionSpec] | None = None) -> str:
        actions = list(actions) if actions is not None else self.actions
        self.validate(actions)
        payload = {"schema": SCHEMA_VERSION, "actions": [_entry_of(action) for action in actions]}
        return json.dumps(payload, indent=2, sort_keys=True)

    def create(self, action: ActionSpec) -> list[ActionSpec]:
        if any(existing.id == action.id for existing in self.actions):
            raise ActionError(f"action id {action.id} already exists")
        self.validate([*self.actions, action])
        self.actions = [*self.actions, action]
        self.save(self.actions)
        return self.actions

    def edit(self, action_id: str, **changes: Any) -> list[ActionSpec]:
        """Change an action's wording or policy — never whether it is live."""

        if "enabled" in changes:
            raise ActionError("use enable() or disable() to change whether an action is live")
        allowed = {"phrase", "params", "adapter", "confirm"}
        unknown = set(changes) - allowed
        if unknown:
            raise ActionError(f"cannot change: {', '.join(sorted(unknown))}")

        updated: list[ActionSpec] = []
        for action in self.actions:
            if action.id != action_id:
                updated.append(action)
                continue
            fields = {
                "id": action.id,
                "phrase": changes.get("phrase", action.phrase),
                "params": changes.get("params", action.params),
                "adapter": changes.get("adapter", action.adapter),
                "confirm": changes.get("confirm", action.confirm),
                "enabled": action.enabled,
            }
            updated.append(ActionSpec(**fields))
        if not any(action.id == action_id for action in updated):
            raise ActionError(f"no action {action_id}")
        self.validate(updated)
        self.actions = updated
        self.save(self.actions)
        return self.actions

    def disable(self, action_id: str) -> list[ActionSpec]:
        return self._set_enabled(action_id, False)

    def enable(self, action_id: str) -> list[ActionSpec]:
        return self._set_enabled(action_id, True)

    def _set_enabled(self, action_id: str, enabled: bool) -> list[ActionSpec]:
        if not any(action.id == action_id for action in self.actions):
            raise ActionError(f"no action {action_id}")
        self.actions = [
            dc_replace(action, enabled=enabled) if action.id == action_id else action
            for action in self.actions
        ]
        self.save(self.actions)
        return self.actions

    def save(self, actions: list[ActionSpec] | None = None) -> None:
        actions = list(actions) if actions is not None else self.actions
        self.validate(actions)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(self.export_json(actions) + "\n", encoding="utf-8")
        temp_path.replace(self.path)
        self.actions = actions

    def validate(self, actions: list[ActionSpec]) -> None:
        self._check_ids(actions)
        for action in actions:
            if not action.id or not normalize_phrase(action.phrase):
                raise ActionError("every action needs an id and a non-empty trigger phrase")
            if not _phrase_tokens(action.phrase):
                raise ActionError(f"action {action.id} has a trigger phrase with no words")
            if action.adapter not in KNOWN_ADAPTERS:
                raise ActionError(f"action {action.id} uses unknown adapter {action.adapter!r}")
            if action.adapter in PRIVILEGED_ADAPTERS and not action.confirm:
                raise ActionError(f"privileged action {action.id} must require confirmation")
            placeholders = {
                str(token)
                for token in _phrase_tokens(action.phrase)
                if isinstance(token, _Placeholder)
            }
            names: set[str] = set()
            for param in action.params:
                if param.name in names:
                    raise ActionError(f"action {action.id} repeats parameter {param.name}")
                names.add(param.name)
                if param.type not in PARAM_TYPES:
                    raise ActionError(f"action {action.id} parameter {param.name} has bad type")
                if param.name not in placeholders and not param.value:
                    raise ActionError(
                        f"action {action.id} parameter {param.name} is neither a placeholder in "
                        "the phrase nor a fixed value"
                    )
                if param.value:
                    value = _clean(param.value, param.name)
                    if param.type == "url":
                        _http_url(value, f"action {action.id} parameter {param.name}")
                    elif param.type == "bookmark" and value not in BOOKMARKS:
                        raise ActionError(f"action {action.id} references unknown bookmark {value!r}")

    def _check_ids(self, actions: list[ActionSpec]) -> None:
        seen: set[str] = set()
        for action in actions:
            if action.id in seen:
                raise ActionError(f"duplicate action id {action.id}")
            seen.add(action.id)

    def _action_from_entry(self, entry: Any) -> ActionSpec:
        if not isinstance(entry, dict):
            raise ActionError("each action must be an object")
        accepted = {"id", "phrase", "params", "adapter", "confirm", "enabled"}
        unknown = set(entry) - accepted
        if unknown:
            raise ActionError(f"unknown action keys: {', '.join(sorted(unknown))}")

        action_id = str(entry.get("id", "")).strip()
        phrase = str(entry.get("phrase", ""))
        if not action_id:
            raise ActionError("every action needs an id")
        if not normalize_phrase(phrase):
            raise ActionError(f"action {action_id} has an empty trigger phrase")

        adapter = str(entry.get("adapter", ""))
        if adapter not in KNOWN_ADAPTERS:
            raise ActionError(f"action {action_id} uses unknown adapter {adapter!r}")

        params: list[Param] = []
        for raw_param in entry.get("params", []):
            if not isinstance(raw_param, dict):
                raise ActionError(f"action {action_id} has a malformed parameter")
            name = str(raw_param.get("name", "")).strip()
            kind = str(raw_param.get("type", "text"))
            value = str(raw_param.get("value", ""))
            if not name:
                raise ActionError(f"action {action_id} has a parameter with no name")
            if kind not in PARAM_TYPES:
                raise ActionError(f"action {action_id} parameter {name} has bad type")
            if value:
                value = _clean(value, name)
                if kind == "url":
                    _http_url(value, f"action {action_id} parameter {name}")
                elif kind == "bookmark" and value not in BOOKMARKS:
                    raise ActionError(f"action {action_id} references unknown bookmark {value!r}")
            params.append(Param(name, kind, value))

        confirm = bool(entry.get("confirm", False))
        if adapter in PRIVILEGED_ADAPTERS and not confirm:
            raise ActionError(f"privileged action {action_id} must set confirm to true")

        return ActionSpec(
            id=action_id,
            phrase=phrase,
            params=tuple(params),
            adapter=adapter,
            confirm=confirm,
            enabled=bool(entry.get("enabled", True)),
        )
