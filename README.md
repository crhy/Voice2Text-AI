# Voice2Text AI

Fast native Linux dictation with local AI. Record from your microphone, have
speech transcribed on-device with Faster Whisper, prompt a local Ollama model,
and hear the answer spoken back with a natural neural voice and an offline
eSpeak NG fallback.

Built with GTK 4 and libadwaita, powered by GStreamer, and distributed as a
Flatpak. Works on Wayland and X11 with PipeWire or PulseAudio.

![Voice2Text AI 0.4 main window](docs/screenshots/v0.4.0-main.png)

## Features

- **On-device dictation** — microphone audio is transcribed locally by Faster
  Whisper (nothing leaves your machine).
- **Local AI prompts** — send the transcript to any Ollama model you have
  pulled, with streaming responses and a live GPU utilization gauge while
  it's thinking (NVIDIA and AMD).
- **Natural speech output** — Edge TTS voices stream immediately; if the
  network voice is unavailable, eSpeak NG speaks offline automatically.
- **Adaptive segmentation** — speech is split at pauses, so dictation flows
  naturally while transcribing in the background.
- **Conversation mode** — say a wake word (default "computer") and it starts
  listening, transcribes your request, asks the selected Ollama model, and
  speaks the reply back — hands-free from wake word to answer. The model
  remembers the earlier turns of the exchange, you can talk over a spoken
  reply to interrupt it, and "cancel" or "goodbye" ends the conversation by
  voice.
- **Hardware-aware model suggestions** — Preferences shows Ollama models
  sized to fit your detected GPU VRAM (or system RAM if there's no GPU).
- **One-click Ollama install and model management** — install or update
  Ollama with a native password prompt, and pull or remove models from
  Preferences, all without a terminal.
- **Coordinated appearance** — follows the system light/dark setting, with an
  explicit override in Preferences.

## Install (Flatpak)

The easiest way is the Flatpak from the [releases](https://github.com/crhy/Voice2Text-AI/releases):

```bash
flatpak install --user Voice2Text-AI.flatpak
flatpak run io.github.crhy.voice2textai
```

For local AI, you need [Ollama](https://ollama.com/) with at least one model
pulled. Preferences → Local AI has an **Install** button that downloads the
official installer and runs it with a native password prompt (via `pkexec`),
and a **Manage models…** button that pulls or removes models — including
suggestions sized to fit your machine's detected GPU VRAM (or RAM if there's
no GPU) — without a terminal.

To do the same by hand instead:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:0.5b
```

`qwen2.5:0.5b` is the smallest useful model at well under 1 GB. Depending on
your hardware you may want a larger one — Preferences shows what fits.

The Whisper model downloads on first launch (the `base` model is the default;
smaller models use less memory and start faster).

## Run from source

Requires Python 3.11+, GTK 4, libadwaita, and GStreamer with the Python
bindings. Install the Python dependencies and launch:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python voice2text_ai.py
```

`voice2text_ai.py` is a thin launcher over the `voice2text` package; the
PyPI entry point is `voice2text-ai`.

## Keyboard shortcuts

| Shortcut            | Action                        |
| ------------------- | ----------------------------- |
| `Ctrl+R`            | Start or stop dictation       |
| `Ctrl+Shift+R`      | Start or stop conversation mode |
| `Ctrl+Enter`        | Ask AI                        |
| `Ctrl+Shift+C`      | Copy transcript               |
| `Ctrl+Shift+V`      | Copy AI response               |
| `Ctrl+L`            | Clear                         |
| `Ctrl+,`            | Preferences                   |
| `Ctrl+Q`            | Quit                          |

## Building the Flatpak

The GitHub Actions workflow generates pinned Python dependencies, builds the
Flatpak bundle, and attaches it to a draft release for every `v*` tag. To build
locally, generate the pinned module and run flatpak-builder:

```bash
git clone https://github.com/flatpak/flatpak-builder-tools.git /tmp/flatpak-builder-tools
python3 /tmp/flatpak-builder-tools/pip/flatpak-pip-generator \
  --requirements-file=packaging/flatpak/requirements.txt \
  --runtime org.gnome.Sdk//50 \
  --prefer-wheels=ctranslate2,onnxruntime,tokenizers,av,numpy,pyyaml,protobuf \
  --wheel-arches=x86_64 \
  --output=python3-requirements-flatpak
flatpak-builder --user --install --force-clean build-dir io.github.crhy.voice2textai.yml
```

## Configuration

Settings are stored in `$XDG_CONFIG_HOME/voice2text-ai/config.json` and edited
from the Preferences dialog.

Backups and restores are covered by the `v2t-backup` CLI, which archives
settings, Ollama model manifests and optional model weights into a
GPG/AES-256 encrypted archive; see [docs/BACKUP.md](docs/BACKUP.md).

## Acknowledgements

- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) for on-device transcription
- [Ollama](https://ollama.com/) for local language models
- [Edge TTS](https://github.com/rany2/edge-tts) for natural voices
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng) for offline speech
- GTK 4, libadwaita, and GStreamer

## License

MIT — see [LICENSE](LICENSE).
