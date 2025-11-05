# Voice2Text AI

<img src="logo.png" alt="Voice2Text AI Logo" width="200"/>

A cross-platform Python application that transcribes voice input using GPU-accelerated Whisper, sends text to local Ollama AI models for intelligent responses, and reads the output aloud with text-to-speech.

## Features

- 🎙️ Real-time voice transcription with OpenAI Whisper (GPU-accelerated)
- 🤖 AI chat integration with local Ollama models
- 🔊 Text-to-speech output with pause/resume
- 🎨 Modern dark gradient UI
- 📦 Cross-platform executables (Windows/macOS/Linux)
- 🖥️ Flatpak support
- ⚡ Optimized performance with silence detection
- 🔄 Automatic retry logic for network requests



## Requirements

- Python 3.8+
- Ollama running locally (http://localhost:11434)
- Microphone access
- CUDA-compatible GPU (optional, for faster Whisper processing)

## Installation

### Pre-built Executables
Download the latest release from [GitHub Releases](https://github.com/crhy/Voice2Text-AI/releases):
- **Windows**: `Voice2Text.exe`
- **macOS**: `Voice2Text`

### Linux (Flatpak)
```bash
flatpak install flathub com.voice2text.app
```

### From Source
```bash
git clone https://github.com/crhy/Voice2Text-AI.git
cd Voice2Text-AI
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Development
```bash
# Run tests
python test_voice.py
python test_mic.py
python test_whisper.py

# Build executable
pyinstaller voice_app.spec

# Build Flatpak
flatpak-builder build com.voice2text.app.yml --force-clean
```

### Setup Ollama
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.2  # or any preferred model
```

## Usage

### Running the App
- **Executable**: Double-click the downloaded file
- **From Source**: `python voice_app.py`
- **Flatpak**: `flatpak run com.voice2text.app`

### Quick Start
1. Select your Ollama model from the dropdown
2. Click "🎙️ Start Dictation"
3. Speak clearly into your microphone
4. Click "⏹️ Stop Dictation" when finished
5. Click "🤖 Query AI" to get AI responses
6. Responses are displayed and spoken aloud

The app automatically saves your settings and provides real-time status updates.

## Notes

- Whisper models run locally (internet required for initial download)
- Google TTS requires internet for speech synthesis
- GPU acceleration speeds up transcription significantly
- Settings persist between sessions
- App works offline after initial model downloads

## Troubleshooting

- **Microphone**: Run `python test_mic.py`
- **Ollama**: Ensure it's running at `http://localhost:11434`
- **GPU**: Check with `nvidia-smi` (optional)
- **Audio**: Grant microphone permissions
- **Models**: First run downloads Whisper models (~2GB)

## Project Structure

```
Voice2Text-AI/
├── voice_app.py          # Main GUI application
├── main.py              # Alternative Tkinter version
├── voice_app_kivy.py    # Kivy mobile version
├── voice_to_opencode.py  # CLI version
├── requirements.txt      # Python dependencies
├── test_*.py            # Test scripts
├── *.spec               # PyInstaller configs
├── *.desktop            # Linux desktop files
├── com.voice2text.app.*  # Flatpak manifests
├── config.json          # App configuration
├── voice_config.json    # Voice app settings
├── logo.png             # Application logo
├── install.sh           # Linux installer
└── README.md
```

## Contributing

Contributions welcome! Please submit issues and pull requests on [GitHub](https://github.com/crhy/Voice2Text-AI).

## License

This project is open source. See individual files for license information.