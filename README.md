# Voice2Text-AI

A Linux desktop app for voice dictation, local AI prompts, and spoken responses.

Voice2Text-AI records from your microphone, transcribes speech with Faster Whisper, sends text to a local Ollama model, and can read responses aloud with text-to-speech.

![Voice2Text-AI main window](docs/screenshots/mainwindow.png)

## Features

- Voice dictation
- Faster Whisper transcription
- Ollama model selection
- AI response panel
- Text-to-speech playback
- Offline eSpeak NG fallback
- PySide6/Qt interface
- Flatpak packaging for Linux

## Screenshots

![Typical query](docs/screenshots/typicalquery.png)

![AI response](docs/screenshots/AIresponse.png)

## Install

Download the latest `.flatpak` bundle from GitHub Releases.

    flatpak install --user ./Voice2Text-AI-v0.3.2-x86_64.flatpak
    flatpak run io.github.crhy.voice2textai

## Ollama

AI responses require Ollama running locally.

    ollama serve
    ollama pull qwen3:8b

Restart Voice2Text-AI after starting Ollama.

## Run from source

    git clone https://github.com/crhy/Voice2Text-AI.git
    cd Voice2Text-AI

    python3.13 -m venv .venv313
    source .venv313/bin/activate

    python -m pip install -r requirements.txt
    python qt_app.py

## Build the Flatpak

    flatpak-builder \
      --disable-rofiles-fuse \
      --force-clean \
      --user \
      --install-deps-from=flathub \
      --install \
      build-dir \
      io.github.crhy.voice2textai.yml

## License

See `LICENSE`.
