# Voice2Text AI

<img src="logo.png" alt="Voice2Text AI Logo" width="200"/>

A cross-platform Python application that transcribes voice input using GPU-accelerated Whisper, sends text to local Ollama AI models for intelligent responses, and reads the output aloud with text-to-speech.

## Features

- 🎙️ Real-time voice transcription with OpenAI Whisper (GPU-accelerated)
- 🤖 AI chat integration with local Ollama models
- 🔊 Text-to-speech output
- 🎨 Modern dark gradient UI
- 📦 Cross-platform executables (Windows/macOS/Linux)
- 🖥️ Flatpak available on Flathub

## Requirements

- Python 3.8+
- Ollama running locally (http://localhost:11434)
- Microphone access
- For GPU acceleration: CUDA-compatible GPU (optional)

## Installation

### Option 1: Download Executable (Recommended)
Download the latest release for your platform from [GitHub Releases](https://github.com/crhy/Voice2Text-AI/releases):
- **Windows**: `Voice2Text.exe`
- **macOS**: `Voice2Text.app`
- **Linux**: `Voice2Text` (or install via Flatpak below)

### Option 2: Flatpak (Linux)
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