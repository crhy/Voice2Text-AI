# Voice2Text AI

<img src="logo.png" alt="Voice2Text AI Logo" width="200"/>

## Application Screenshot

<img src="Screenshot at 2025-11-03 01-22-46.png" alt="Voice2Text AI Application Screenshot" width="600"/>

## Downloads

- [Windows](https://github.com/crhy/Voice2Text-AI/releases/download/v0.2/Voice2Text.exe)
- [macOS](https://github.com/crhy/Voice2Text-AI/releases/download/v0.2/Voice2Text)
- Linux: Flatpak coming soon to [Flathub](https://flathub.org/apps/com.voice2text.app)

A cross-platform Python application that transcribes voice input using GPU-accelerated Whisper, sends text to local Ollama AI models for intelligent responses, and reads the output aloud with text-to-speech.

## Features

- 🎙️ Real-time voice transcription with OpenAI Whisper (GPU-accelerated)
- 🤖 AI chat integration with local Ollama models
- 🔊 Text-to-speech output
- 🎨 Modern dark gradient UI
- 📦 Cross-platform executables (Windows/macOS/Linux)
- 🖥️ Flatpak available on Flathub

## Project Structure

```
OpenCode/
├── Voice2Text/
│   ├── public/
│   │   ├── electron.js
│   │   ├── favicon.ico
│   │   └── index.html
│   ├── src/
│   │   ├── Components/
│   │   │   ├── Details/
│   │   │   │   ├── Details.css
│   │   │   │   └── Details.jsx
│   │   │   ├── Editor/
│   │   │   │   ├── assets/
│   │   │   │   │   └── star.png
│   │   │   │   ├── Editor.css
│   │   │   │   └── Editor.jsx
│   │   │   ├── Navbar/
│   │   │   │   ├── Navbar.css
│   │   │   │   └── Navbar.jsx
│   │   │   └── Options/
│   │   │       ├── assets/
│   │   │       │   ├── headset.png
│   │   │       │   ├── headset_on.png
│   │   │       │   ├── star.png
│   │   │       │   ├── trash.png
│   │   │       │   └── volume.png
│   │   │       ├── Options.css
│   │   │       └── Options.jsx
│   │   ├── App.css
│   │   ├── App.js
│   │   ├── index.css
│   │   └── index.js
│   ├── .gitattributes
│   ├── .gitignore
│   ├── README.md
│   ├── com.voice2text.app.desktop
│   ├── com.voice2text.app.metainfo.xml
│   ├── com.voice2text.app.yml
│   ├── electron.js
│   ├── package-lock.json
│   ├── package.json
│   ├── requirements.txt
│   ├── run_voice_app.sh
│   ├── test_whisper.py
│   ├── voice_app.py
│   └── voice_config.json
├── Voice2Text-Windows/
│   └── Voice2Text.exe
├── Voice2Text-macOS/
│   └── Voice2Text
├── flathub-repo/
│   └── com.voice2text.app/
│       └── com.voice2text.app.yml
├── voice_env/
│   └── (Python virtual environment files)
├── README.md
├── VOICE_README.md
├── buildozer.spec
├── config.json
├── main.py
├── opencode.json
├── requirements.txt
├── test_voice.py
├── test_whisper.py
├── voice_app.desktop
├── voice_app.py
├── voice_app.spec
├── voice_app_kivy.py
├── voice_config.json
└── voice_to_opencode.py
```

## Requirements

- Python 3.8+
- Ollama running locally (http://localhost:11434)
- Microphone access
- For GPU acceleration: CUDA-compatible GPU (optional)

## Installation

### Option 1: Download Executable (Windows/macOS)
Download the latest release for your platform from [GitHub Releases](https://github.com/crhy/Voice2Text-AI/releases):
- **Windows**: `Voice2Text.exe`
- **macOS**: `Voice2Text`

### Option 2: Linux Installer (Recommended for Linux)
For Linux users who don't have Ollama installed, download and run the installer script to set up Ollama and the model automatically:
```bash
wget https://github.com/crhy/Voice2Text-AI/releases/download/v0.2/install.sh
chmod +x install.sh
./install.sh
```
This installs Ollama and the required model. For the app, install the flatpak from Flathub once available:
```bash
flatpak install flathub com.voice2text.app
```

### Option 3: From Source
1. Clone the repository:
    ```bash
    git clone https://github.com/crhy/Voice2Text-AI.git
    cd Voice2Text-AI
    ```

2. Create a virtual environment:
    ```bash
    python3 -m venv venv
    ```

3. Activate the virtual environment:
    ```bash
    source venv/bin/activate
    ```

4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

5. Ensure Ollama is installed and running with models available.

## Usage

### Running the App
- **Executable**: Double-click the downloaded executable
- **From Source**: 
  ```bash
  source venv/bin/activate
  python voice_app.py
  ```

### Interface Guide
1. **Model Selection**: Choose your preferred Ollama model from the dropdown
2. **Voice Input**: Click "Start Recording" to begin voice transcription
3. **Speak**: Talk clearly into your microphone
4. **Stop**: Click "Stop Recording" when finished
5. **Edit Text**: Review and edit the transcribed text in the input area
6. **Send to AI**: Click "Send to AI" to get Ollama's response
7. **Listen**: The response will be displayed and spoken aloud

The app features a modern dark UI with real-time status indicators and remembers your settings between sessions.

## Notes

- Speech recognition uses OpenAI Whisper (runs locally, internet required for initial model download)
- Text-to-speech uses Google TTS (requires internet)
- GPU acceleration available for faster Whisper transcription
- Settings are automatically saved
- Works offline once models are downloaded

## Troubleshooting

- **Microphone Issues**: Run `python test_mic.py` to test audio input
- **Ollama Connection**: Ensure Ollama is running at http://localhost:11434
- **GPU Support**: Check CUDA installation with `nvidia-smi`
- **Permissions**: Grant microphone access to the application
- **Model Download**: First run may take time to download Whisper models

## Contributing

Contributions welcome! Please submit issues and pull requests on GitHub.