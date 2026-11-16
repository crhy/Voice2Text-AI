"""Which platform the application is actually running on, and what is supported there.

Nothing else in the tree knows which distribution it is on: capture falls back to
GStreamer's ``autoaudiosrc`` and an unsupported host simply sees no microphones.
These probes name the host honestly (path arguments so tests can point them at
fixtures, never an estimate) and classify it against the Flatpak baseline that
ships in ``packaging/flatpak``. See ``docs/PLATFORM-SUPPORT.md`` for the matrix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .installer import is_flatpak

# Tried in order; the first readable file wins. Inside the Flatpak this is the
# runtime image's own release file, so ``os_release_search_paths()`` puts the
# host's copy (bound by Flatpak at /run/host/os-release) first when sandboxed.
OS_RELEASE_PATHS: tuple[Path, ...] = (Path("/etc/os-release"), Path("/usr/lib/os-release"))
FLATPAK_HOST_RELEASE = Path("/run/host/os-release")

# The Debian-family distributions the Flatpak baseline is verified on. A host
# qualifies through its ID or through ID_LIKE, so Debian-derived systems land
# on the same baseline; anything else that looks like a Linux desktop is
# reported as partial, never as supported. See docs/PLATFORM-SUPPORT.md.
TESTED_DISTRIBUTIONS: frozenset[str] = frozenset({"linuxmint", "ubuntu", "debian", "devuan", "spaced"})

SUPPORTED = "supported"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"

# Client tools each session backend answers through. ``pactl`` is counted for
# both: PipeWire ships it and PulseAudio installs it, so it proves a session
# exists without naming which one.
_PIPEWIRE_TOOLS: tuple[str, ...] = ("pwls",)
_PULSEAUDIO_TOOLS: tuple[str, ...] = ("pulseaudio", "pactl")

_NOT_YET = "Windows, macOS and Android are not supported yet"


@dataclass(slots=True, frozen=True)
class Distribution:
    id: str = ""
    id_like: str = ""
    version_id: str = ""
    pretty_name: str = ""

    @property
    def label(self) -> str:
        """How the distribution is named to the user, or an honest blank."""
        if self.pretty_name:
            return self.pretty_name
        if self.id:
            return f"{self.id} {self.version_id}".strip()
        return ""

    @property
    def family(self) -> frozenset[str]:
        """Every distribution id this host claims, including the families it is like."""
        ids = {part.casefold() for part in f"{self.id} {self.id_like}".split() if part}
        return frozenset(ids)

    @property
    def identified(self) -> bool:
        return bool(self.id)


@dataclass(slots=True, frozen=True)
class AudioBackends:
    pipewire: bool = False
    pulseaudio: bool = False
    alsa: bool = False

    @property
    def names(self) -> tuple[str, ...]:
        found: list[str] = []
        if self.pipewire:
            found.append("PipeWire")
        if self.pulseaudio:
            found.append("PulseAudio")
        if self.alsa:
            found.append("ALSA")
        return tuple(found)

    @property
    def any_backend(self) -> bool:
        return bool(self.names)

    @property
    def session_backend(self) -> bool:
        """True when a PipeWire or PulseAudio session is reachable."""
        return self.pipewire or self.pulseaudio


@dataclass(slots=True, frozen=True)
class SupportReport:
    level: str
    reason: str


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _parse_os_release(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        values[key.strip().upper()] = _unquote(raw)
    return values


def os_release_search_paths(flatpak: bool | None = None) -> tuple[Path, ...]:
    """os-release locations to try, host-first when running inside the Flatpak."""
    sandboxed = is_flatpak() if flatpak is None else flatpak
    if sandboxed:
        return (FLATPAK_HOST_RELEASE, *OS_RELEASE_PATHS)
    return OS_RELEASE_PATHS


def read_os_release(paths: tuple[Path, ...] = OS_RELEASE_PATHS) -> Distribution:
    """The distribution this machine reports about itself, or an empty one."""
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        values = _parse_os_release(text)
        if not values:
            continue
        return Distribution(
            id=values.get("ID", ""),
            id_like=values.get("ID_LIKE", ""),
            version_id=values.get("VERSION_ID", ""),
            pretty_name=values.get("PRETTY_NAME", ""),
        )
    return Distribution()


def _on_path(name: str, search_path: str) -> bool:
    for directory in search_path.split(os.pathsep):
        if not directory:
            continue
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return True
    return False


def audio_backends(
    path: str | None = None,
    runtime_dir: Path | None = None,
    dev_sound: Path = Path("/dev/snd"),
) -> AudioBackends:
    """Which audio backends this machine makes reachable.

    Deliberately imports no ``gi``: ``voice2text.audio`` loads GStreamer lazily,
    so a probe must stay importable on a host that has no GStreamer at all.
    """
    search_path = path if path is not None else os.environ.get("PATH", "")
    if runtime_dir is None:
        configured = os.environ.get("XDG_RUNTIME_DIR", "")
        runtime_dir = Path(configured) if configured else None

    pipewire = any(_on_path(tool, search_path) for tool in _PIPEWIRE_TOOLS)
    pulseaudio = any(_on_path(tool, search_path) for tool in _PULSEAUDIO_TOOLS)
    if not pulseaudio and runtime_dir is not None:
        # The per-user PulseAudio socket directory, which Flatpak's
        # --socket=pulseaudio also provides inside the sandbox.
        pulseaudio = (runtime_dir / "pulse").exists()

    return AudioBackends(pipewire=pipewire, pulseaudio=pulseaudio, alsa=dev_sound.is_dir())


def classify(distribution: Distribution, backends: AudioBackends) -> SupportReport:
    """Grade the host against the Flatpak baseline without guessing at anything."""
    host = distribution.label
    found = ", ".join(backends.names)

    if not distribution.identified:
        return SupportReport(
            UNSUPPORTED,
            "this machine reported no distribution ID in os-release, so Voice2Text AI "
            "cannot name it; the Flatpak baseline covers Linux desktops only.",
        )
    if not backends.any_backend:
        return SupportReport(
            UNSUPPORTED,
            f"{host} exposes no audio backend at all: no PipeWire, no PulseAudio and no "
            "ALSA devices were found, so there is nothing to capture your voice from.",
        )
    if not distribution.family & TESTED_DISTRIBUTIONS:
        return SupportReport(
            PARTIAL,
            f"{host} is not in the tested matrix; it found {found}, and would also have to "
            "provide Wayland or X11 and a reachable Ollama before it could be called supported.",
        )
    if not backends.session_backend:
        return SupportReport(
            PARTIAL,
            f"{host} is in the tested matrix but exposes only {found}; the Flatpak baseline "
            "expects PipeWire or PulseAudio.",
        )
    return SupportReport(
        SUPPORTED,
        f"{host} found {found}, which matches the Flatpak baseline of Wayland or X11, "
        "PipeWire or PulseAudio, and a reachable Ollama.",
    )


def support_sentence(report: SupportReport) -> str:
    """One plain sentence about this machine: what it is, and what is not supported."""
    labels = {SUPPORTED: "Supported here", PARTIAL: "Partially supported here", UNSUPPORTED: "Not supported here"}
    label = labels.get(report.level, "Support unknown here")
    return f"{label}: {report.reason} {_NOT_YET}."
