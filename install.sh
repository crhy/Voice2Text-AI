#!/bin/bash

# Voice2Text AI Linux Installer
# This script installs Ollama, pulls the required model, installs the flatpak, and tests Ollama.

set -e

echo "Voice2Text AI Linux Installer"
echo "============================="

# Install flatpak if not installed
if ! command -v flatpak &> /dev/null; then
    echo "Installing flatpak..."
    sudo apt-get update
    sudo apt-get install -y flatpak
    flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
fi

# Download and install latest Ollama
echo "Downloading latest Ollama..."
OLLAMA_URL=$(curl -s https://api.github.com/repos/ollama/ollama/releases/latest | grep "browser_download_url.*linux-amd64" | cut -d '"' -f 4)
if [ -z "$OLLAMA_URL" ]; then
    echo "Failed to find Ollama download URL"
    exit 1
fi
wget -O ollama-linux-amd64.tgz "$OLLAMA_URL"
tar -xzf ollama-linux-amd64.tgz
sudo mv ollama /usr/local/bin/
rm ollama-linux-amd64.tgz

# Start Ollama service (assuming systemd)
echo "Starting Ollama service..."
sudo useradd -r -s /bin/false ollama || true
sudo mkdir -p /etc/systemd/system
cat <<EOF | sudo tee /etc/systemd/system/ollama.service
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
sleep 10

# Pull the model
echo "Pulling qwen2.5:0.5b model..."
ollama pull qwen2.5:0.5b

# Install the flatpak
echo "Installing Voice2Text AI flatpak..."
flatpak install -y flathub com.voice2text.app

# Test Ollama
echo "Testing Ollama..."
if ollama list | grep -q "qwen2.5:0.5b"; then
    echo "Ollama test passed: Model qwen2.5:0.5b is installed."
else
    echo "Ollama test failed: Model not found."
    exit 1
fi

echo "Installation complete! Run 'flatpak run com.voice2text.app' to start the app."