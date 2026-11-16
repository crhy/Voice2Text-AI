# Platform support

Voice2Text AI ships as a **Linux Flatpak** (`io.github.crhy.voice2textai.yml`, `packaging/flatpak/`).
That Flatpak baseline is the only configuration this project claims to support. Everything below
Windows/macOS/Android is tracked as future work, not as a promise: the application now names the
platform it is actually running on and reports its own support level instead of implying parity.

## How the application reports its own support

`voice2text/platform_support.py` probes the machine with injectable paths (the same idiom as
`voice2text/hardware.py`) and never estimates:

- **`read_os_release()`** — reads `/etc/os-release`, then `/usr/lib/os-release`. Inside the Flatpak
  the host's copy, bound by Flatpak at `/run/host/os-release`, is tried first, so the app names the
  distribution you are on rather than the GNOME runtime image.
- **`audio_backends()`** — PipeWire when `pwls` is reachable; PulseAudio (or a PulseAudio-compatible
  socket) when `pulseaudio`/`pactl` is reachable or `$XDG_RUNTIME_DIR/pulse` exists; plain ALSA when
  `/dev/snd` is present. It imports no `gi`, so a host without PyGObject still gets an answer.
- **`classify()`** — grades the pair as `supported`, `partial` or `unsupported`.
- **`support_sentence()`** — one plain line, shown as the subtitle of the **Ollama model** row in
  Preferences, next to the hardware summary.
- **`voice2text/audio.py`** — when no microphone is selected and no backend at all is reachable,
  capture now fails with a message naming the detected distribution and what was missing, instead of
  GStreamer's `autoaudiosrc` quietly producing an empty microphone list.

A host is `supported` only when its distribution is in the matrix below (by `ID` or by `ID_LIKE`)
**and** a PipeWire or PulseAudio session is reachable. Debian-family hosts with ALSA only, and Linux
desktops outside the matrix, are `partial`. A machine that names no distribution in `os-release`, or
that exposes no audio backend at all, is `unsupported`.

## Supported Linux distribution matrix

The Flatpak baseline needs all of: a **Wayland or X11** session, **PipeWire or PulseAudio** (a
PulseAudio-compatible socket counts), a **reachable Ollama** server with at least one model pulled,
and Flatpak 1.10+ on the host.

| Distribution | Status | Notes |
| ------------ | ------ | ----- |
| Linux Mint 13 (and 12) | supported | The reference desktop for the shipped Flatpak; PipeWire by default. |
| Ubuntu 22.04 / 24.04 | supported | PipeWire or PulseAudio out of the box. |
| Debian 12 | supported | PulseAudio session per user; install `flatpak` yourself. |
| Devuan 5 | supported | No desktop service manager starts PulseAudio for you — the session must already be running. |
| Spaced Linux 11.x | supported | Debian-family; published through the Spaced Linux `spaced-github` repository. |
| Fedora / RHEL / openSUSE | not in the matrix — reported `partial` | Not verified; see what an unlisted desktop must provide below. |
| Arch and AUR-based | not in the matrix — reported `partial` | Not verified; GStreamer/PipeWire packaging differs. |
| Alpine (musl) | not in the matrix — reported `partial` | The pinned Python wheels in `python3-requirements-flatpak.json` are built for glibc. |
| NixOS | not in the matrix — reported `partial` | The Flatpak needs a host Flatpak install and a user PulseAudio/PipeWire session. |
| Windows, macOS, Android | not supported | See the feature matrix below. |

## What an unlisted Linux desktop must provide

To move from `partial` to `supported`, a Linux desktop outside the matrix has to supply all of:

1. A **Wayland or X11** session the sandbox can reach (`--socket=wayland`, `--socket=fallback-x11`).
2. A **PipeWire or PulseAudio** session reachable through `--socket=pulseaudio`; ALSA-only hosts stay
   `partial`, because `autoaudiosrc` has no session to fall back to and monitor/loopback nodes are
   refused.
3. An **Ollama** server answering at the configured address, with at least one model pulled. The in-app
   installer runs the official script through `flatpak-spawn --host pkexec`, which needs a PolicyKit
   agent on the host.
4. **Flatpak 1.10+** installed on the host (or a native package for that distribution).
5. A real **microphone** node, not a `Monitor of …` loopback node.

Running from source instead of the Flatpak additionally needs GTK 4, libadwaita, GStreamer 1.0 with
the Python bindings (`python3-gi`), and Python 3.11+.

## Feature/support matrix

| Capability | Linux (Flatpak baseline) | Windows | macOS | Android |
| ---------- | ------ | ------- | ----- | ------- |
| Audio capture | supported (PipeWire/PulseAudio; ALSA only = partial) | not supported | not supported | not supported |
| Transcription (Faster Whisper, ctranslate2) | supported | not supported | not supported | not supported |
| Ollama transport (HTTP to the configured address) | supported | not supported | not supported | not supported |
| TTS (Edge TTS, eSpeak NG offline fallback) | supported | not supported | not supported | not supported |
| Settings (`$XDG_CONFIG_HOME/voice2text-ai/config.json`) | supported | not supported | not supported | not supported |
| UI (GTK 4 + libadwaita, theme colours) | supported | not supported | not supported | not supported |
| Backup / restore (`v2t-backup`) | supported | not supported | not supported | not supported |

Nothing in this table is claimed for Windows, macOS or Android. The rows are the seams the triage of
crhy/Voice2Text-AI#10 asks for — audio capture, transcription, Ollama transport, TTS, settings and UI
each need their own platform adapter before any of them can move to `supported` elsewhere.

### What Windows would need

- A capture adapter over DirectShow/WASAPI or the system's own loopback-free device list, since there is
  no PipeWire/PulseAudio session and no `/dev/snd`.
- A packaging prototype (MSI or an appx bundle) with the pinned wheels rebuilt for Windows, plus a smoke
  test that runs headless in CI.
- `%APPDATA%`/registry-free settings: an explicit per-user settings path instead of `$XDG_CONFIG_HOME`.
- A TTS path for the offline fallback (eSpeak NG is not packaged there) and a documented Ollama endpoint.

### What macOS would need

- A capture adapter over CoreAudio device enumeration instead of GStreamer's ALSA/PulseAudio backends.
- A packaging prototype (`.pkg`/`.dmg` or a Homebrew formula) with rebuilt wheels and a CI smoke test.
- A settings location outside `$XDG_CONFIG_HOME` (or a documented default under `~/.config`).
- A signed, sandboxable UI story for GTK 4/libadwaita, which is not a native macOS stack.

### What Android would need

Tracked in crhy/Voice2Text-AI#2: its lifecycle and local-model constraints differ enough that it is a
separate delivery, not a Linux build. It would need a touch-first UI rather than GTK 4/libadwaita,
capture through the platform's own audio API, per-app storage instead of XDG paths, a model-size policy
for the device's memory, and a restart-safe session model.

## Smoke tests

`tests/test_platform_support.py` covers the probe and the honest reporting on Linux: `os-release`
fixtures (Devuan, Debian, Ubuntu, Linux Mint, a host that names no `ID`), a fake `PATH` for backend
detection, the three classification levels, the sentence never claiming parity, and `voice2text/audio.py`
raising the platform-named error when no backend exists while keeping the `autoaudiosrc` path when one
does. Per-platform smoke tests for Windows, macOS and Android do not exist yet and are part of the
cross-platform work in crhy/Voice2Text-AI#10, not of this baseline.
