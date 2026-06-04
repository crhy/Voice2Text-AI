#!/usr/bin/env python3
from faster_whisper import WhisperModel
import torch

print("Testing Faster Whisper...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Using device: {device}, compute_type: {compute_type}")
    model = WhisperModel("small", device=device, compute_type=compute_type)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")