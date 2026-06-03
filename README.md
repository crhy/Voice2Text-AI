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

- Python 3.8+ (maybe, I think the standalone app runs without it)
- Ollama running locally (http://localhost:11434) (only for AI interaction)
- Microphone access (The whole point is dictation)
- CUDA-compatible GPU (optional, for faster Whisper processing.  Highly recommended.)

## Installation

### Pre-built Executable v0.3.1
Coming Soon

### Pre-built Executables Version 0.2.0
Download the latest release from [GitHub Releases](https://github.com/crhy/Voice2Text-AI/releases):
- **Windows**: `Voice2Text.exe` Untested
- **macOS**: `Voice2Text` Untested
- **Linux**: 'Voice2Text' https://drive.google.com/file/d/1MmF6Vr_3nz1yket2SdnHCI_Od5OpIzQd/view?usp=sharing

### Linux (Flatpak) v0.3.1

### From Source (This might actually work)
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

# Build Flatpak (This has never worked.  Yet)
flatpak-builder --force-clean build com.voice2text.app.yml
flatpak-builder --user --install build com.voice2text.app.yml

# Export Flatpak for distribution
flatpak build-bundle build voice2text.flatpak com.voice2text.app
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
- **Executable**: Right Click, Select "Allow Executing File as Program", Double-click the downloaded file (wait, it's slow to load.  Once it's up it's crazy fast)
- **From Source**: `python voice_app.py`
- **Flatpak**: `flatpak run com.voice2text.app` (Good luck?)

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
- Edge TTS provides local speech synthesis
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

## Distribution

### Building Releases
```bash
# Create GitHub release with all platform binaries
# Upload these files to GitHub Releases:
# - Voice2Text.exe (Windows)
# - Voice2Text (macOS)
# - voice2text.flatpak (Linux)
```

### Flathub Submission
To submit to Flathub for official distribution:
```bash
# Fork the Flathub repository
# Add your manifest to: https://github.com/flathub/flathub
# Submit pull request with com.voice2text.app.yml
```

## Contributing

Contributions welcome! Please submit issues and pull requests on [GitHub](https://github.com/crhy/Voice2Text-AI).

## License

This project is open source. See individual files for license information.
