# Voice2Text AI

Fast native Linux dictation with local AI. Record from your microphone, have
speech transcribed on-device with Faster Whisper, prompt a local Ollama model,
and hear the answer spoken back with a natural neural voice and an offline
eSpeak NG fallback.

Built with GTK 4 and libadwaita, powered by GStreamer, and distributed as a
Flatpak. Works on Wayland and X11 with PipeWire or PulseAudio.

## Features

- **On-device dictation** — microphone audio is transcribed locally by Faster
  Whisper (nothing leaves your machine).
- **Local AI prompts** — send the transcript to any Ollama model you have
  pulled, with streaming responses.
- **Natural speech output** — Edge TTS voices stream immediately; if the
  network voice is unavailable, eSpeak NG speaks offline automatically.
- **Adaptive segmentation** — speech is split at pauses, so dictation flows
  naturally while transcribing in the background.
- **Coordinated appearance** — follows the system light/dark setting, with an
  explicit override in Preferences.

## Install (Flatpak)

The easiest way is the Flatpak from the [releases](https://github.com/crhy/Voice2Text-AI/releases):

```bash
flatpak install --user Voice2Text-AI.flatpak
flatpak run io.github.crhy.voice2textai
```

For local AI, install [Ollama](https://ollama.com/) and pull a model, for example:

```bash
ollama pull llama3.1:8b
```

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
| `Ctrl+Enter`        | Ask AI                        |
| `Ctrl+Shift+C`      | Copy transcript               |
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
  --runtime org.gnome.Sdk//48 \
  --prefer-wheels=ctranslate2,onnxruntime,tokenizers,av,numpy,pyyaml,protobuf \
  --wheel-arches=x86_64 \
  --output=python3-requirements-flatpak
flatpak-builder --user --install --force-clean build-dir io.github.crhy.voice2textai.yml
```

## Configuration

Settings are stored in `$XDG_CONFIG_HOME/voice2text-ai/config.json` and edited
from the Preferences dialog.

## Acknowledgements

- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) for on-device transcription
- [Ollama](https://ollama.com/) for local language models
- [Edge TTS](https://github.com/rany2/edge-tts) for natural voices
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng) for offline speech
- GTK 4, libadwaita, and GStreamer

## License

MIT — see [LICENSE](LICENSE).
