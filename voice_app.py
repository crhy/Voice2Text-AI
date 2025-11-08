#!/usr/bin/env python3
"""
Voice 2 Text GUI Application

A standalone GUI app for voice recognition that integrates with AI.
Features a simple interface with start/stop buttons and automatic clipboard copying.
"""

import os

# Disable Jack audio to prevent errors
os.environ['JACK_NO_START_SERVER'] = '1'
os.environ['SDL_AUDIODRIVER'] = 'alsa'
os.environ['ALSA_NO_JACK'] = '1'

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pyperclip
import threading
import time
import json
import pyaudio
import numpy as np
import tempfile
import wave
from faster_whisper import WhisperModel
from scipy.signal import resample
import torch
import requests
from gtts import gTTS
import pygame
import asyncio
import edge_tts
from PIL import Image, ImageTk
import datetime
import queue
from tqdm import tqdm

class GuiTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app = getattr(GuiTqdm, 'app', None)

    def update(self, n=1):
        super().update(n)
        if self.app and self.total and self.total > 0:
            progress = min(100, self.n / self.total * 100)
            self.app.queue.put(('progress', progress))


class VoiceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice 2 Text")
        self.root.geometry("900x800")
        self.root.configure(bg='black')
        self.root.resizable(True, True)

        # Create canvas with gradient background
        self.canvas = tk.Canvas(self.root, width=900, height=800, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        # Bind resize event
        self.canvas.bind('<Configure>', self.on_canvas_resize)

        # Create initial gradient
        self.create_gradient(900, 800)

        # Config
        self.config_file = os.path.expanduser('~/.voice_config.json')
        print(f"Config file: {self.config_file}")
        self.config = self.load_config()

        # TTS settings
        self.tts_rate = self.config.get('tts_rate', 180)

        # Audio devices
        self.audio = pyaudio.PyAudio()
        self.microphones = self.get_microphones()
        self.selected_mic_index = 0
        self.selected_mic_name = self.config.get('microphone_name', '')

        # Create vars
        self.whisper_var = tk.StringVar()
        self.mic_var = tk.StringVar()
        self.model_var = tk.StringVar()

        # Set mic from saved name or default
        if self.microphones:
            if self.selected_mic_name and self.selected_mic_name in self.microphones:
                self.mic_var.set(self.selected_mic_name)
                self.selected_mic_index = self.microphones.index(self.selected_mic_name)
            else:
                self.mic_var.set(self.microphones[0])
                self.selected_mic_index = 0
                self.selected_mic_name = self.microphones[0]
        else:
            self.mic_var.set("No microphone detected")
            self.selected_mic_name = ''

        # Whisper models
        self.whisper_models = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
        self.selected_whisper_model = self.config.get('whisper_model', 'tiny')
        # Model sizes in MB and estimated download times in minutes (assuming 1 MB/s)
        self.model_info = {
            "tiny": {"size": 39, "eta": 1},
            "base": {"size": 74, "eta": 1},
            "small": {"size": 244, "eta": 4},
            "medium": {"size": 769, "eta": 13},
            "large-v2": {"size": 1550, "eta": 26},
            "large-v3": {"size": 1550, "eta": 26}
        }

        # Speech recognition with Faster Whisper
        threading.Thread(target=self.load_whisper_model, daemon=True).start()

        # Text-to-speech engine
        pygame.mixer.init()
        self.tts_playing = False

        # Queue for thread-safe GUI updates
        self.queue = queue.Queue()

        # Performance and memory optimization
        self.audio_frames = []  # Limit memory usage
        self.max_audio_frames = 1000  # Prevent excessive memory usage

        # Ollama models
        self.ollama_models = self.get_ollama_models()
        self.selected_model = self.config.get('selected_model', "llama3.2" if "llama3.2" in self.ollama_models else (self.ollama_models[0] if self.ollama_models else "llama3.2"))
        # Ensure selected model is valid
        if self.ollama_models and self.selected_model not in self.ollama_models:
            self.selected_model = self.ollama_models[0]
        self.is_listening = False
        self.current_text = ""
        self.audio_stream = None

        # Create GUI
        self.create_gui()

        self.process_queue()

        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_time(self):
        current_time = datetime.datetime.now().strftime("%I:%M:%S %p")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)

    def process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == "update_status":
                    self.update_status(msg[1], msg[2])
                elif msg[0] == "update_transcript":
                    self.text_area.insert(tk.END, f"{msg[1]}\n")
                    self.text_area.see(tk.END)
                    self.root.update_idletasks()
                elif msg[0] == "show_error":
                    messagebox.showerror("Error", msg[1])
                elif msg[0] == "clear_ai":
                    self.ai_text_area.delete(1.0, tk.END)
                elif msg[0] == "insert_ai":
                    self.ai_text_area.insert(tk.END, msg[1])

                elif msg[0] == "stop_dictation":
                    self.stop_dictation()
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)

    def get_ollama_models(self):
        """Get available Ollama models with improved error handling."""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except requests.exceptions.Timeout:
            self.update_status("Ollama connection timeout - check if Ollama is running", "orange")
            return []
        except requests.exceptions.ConnectionError:
            self.update_status("Cannot connect to Ollama - start with 'ollama serve'", "red")
            return []
        except requests.exceptions.RequestException as e:
            self.update_status(f"Ollama request error: {str(e)[:50]}", "red")
            return []
        except (KeyError, ValueError) as e:
            self.update_status(f"Invalid Ollama response: {str(e)[:50]}", "red")
            return []

    def load_config(self):
        config = {}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                config = {}
        print(f"Loaded config: {config}")
        return config

    def save_config(self):
        config = {
            'microphone_name': self.selected_mic_name,
            'selected_model': self.selected_model,
            'whisper_model': self.selected_whisper_model,
            'tts_rate': self.tts_rate
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
            print(f"Saved config: {config}")
        except Exception as e:
            print(f"Error saving config: {e}")

    def on_close(self):
        self.save_config()
        self.audio.terminate()
        self.root.destroy()

    def create_gradient(self, width, height):
        color1 = (0, 0, 0)
        color2 = (0, 0, 51)
        img = Image.new('RGB', (width, height), color1)
        for y in range(height):
            r = int(color1[0] + (color2[0] - color1[0]) * y / height)
            g = int(color1[1] + (color2[1] - color1[1]) * y / height)
            b = int(color1[2] + (color2[2] - color1[2]) * y / height)
            for x in range(width):
                img.putpixel((x, y), (r, g, b))
        self.bg_photo = ImageTk.PhotoImage(img)
        self.canvas.delete("gradient")
        self.canvas.create_image(0, 0, anchor='nw', image=self.bg_photo, tags="gradient")

    def on_canvas_resize(self, event):
        width = event.width
        height = event.height
        self.create_gradient(width, height)

    def get_microphones(self):
        microphones = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            max_input = info.get('maxInputChannels', 0)
            if isinstance(max_input, (int, float)) and max_input > 0:
                microphones.append(f"{info.get('name')} (Index: {i})")
        return microphones

    def get_mic_device_index(self, mic_string):
        import re
        match = re.search(r'Index: (\d+)', mic_string)
        return int(match.group(1)) if match else 0

    def load_whisper_model(self):
        """Load Whisper model with improved error handling and performance."""
        info = self.model_info.get(self.selected_whisper_model, {"size": "unknown", "eta": "unknown"})
        size = info["size"]
        eta = info["eta"]
        self.update_status(f"Loading Whisper model: {self.selected_whisper_model} ({size} MB) - estimated download time: {eta} min", "#ffaa00")
        if isinstance(eta, int) and eta > 0:
            self.root.after(0, lambda: self.progress_bar.config(mode='determinate', maximum=100, value=0))
            # Start progress update
            total_time = eta * 60  # seconds
            step = 100 / total_time
            def update_progress():
                current = self.progress_bar['value']
                if current < 100:
                    self.progress_bar['value'] = min(100, current + step)
                    self.root.after(1000, update_progress)
            self.root.after(1000, update_progress)
        else:
            self.root.after(0, lambda: self.progress_bar.config(mode='indeterminate'))
            self.root.after(0, lambda: self.progress_bar.start())

        # Determine optimal device and compute type
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "int8"

        try:
            self.model = WhisperModel(
                self.selected_whisper_model,
                device=device,
                compute_type=compute_type,
                cpu_threads=1 if device == "cuda" else 4
            )
            self.update_status("Whisper model loaded successfully!", "#00aa00")
            self.root.after(0, lambda: self.progress_bar.config(value=100))
            self.root.after(0, lambda: self.loaded_label.config(text=f"Loaded: {self.selected_whisper_model}"))

        except Exception as e:
            loaded = False
            if device == "cuda":
                self.update_status("CUDA failed, falling back to CPU...", "#ffaa00")
                try:
                    device = "cpu"
                    compute_type = "int8"
                    self.model = WhisperModel(
                        self.selected_whisper_model,
                        device=device,
                        compute_type=compute_type,
                        cpu_threads=4
                    )
                    self.update_status("Whisper model loaded on CPU!", "#00aa00")
                    self.root.after(0, lambda: self.progress_bar.config(value=100))
                    self.root.after(0, lambda: self.loaded_label.config(text=f"Loaded: {self.selected_whisper_model}"))
                    loaded = True
                except Exception as e2:
                    error_msg = f"Failed to load Whisper model on CPU: {str(e2)[:100]}"
                    self.update_status(error_msg, "red")
                    self.root.after(0, lambda: self.progress_bar.config(value=0))
                    self.model = None
                    self.root.after(0, lambda: self.loaded_label.config(text="Loaded: Failed"))
            if not loaded:
                error_msg = f"Failed to load Whisper model: {str(e)[:100]}"
                self.update_status(error_msg, "red")
                self.root.after(0, lambda: self.progress_bar.stop())
                self.model = None
                self.root.after(0, lambda: self.loaded_label.config(text="Loaded: Failed"))
                error_msg = f"Failed to load Whisper model: {str(e)[:100]}"
                self.update_status(error_msg, "red")
                self.root.after(0, lambda: self.progress_bar.stop())
                self.model = None
                self.root.after(0, lambda: self.loaded_label.config(text="Loaded: Failed"))
                error_msg = f"Failed to load Whisper model: {str(e)[:100]}"
                self.update_status(error_msg, "red")
                self.root.after(0, lambda: self.progress_bar.stop())
                self.model = None
                self.root.after(0, lambda: self.loaded_label.config(text="Loaded: Failed"))

        except Exception as e:
            error_msg = f"Unexpected error loading Whisper model: {str(e)[:100]}"
            self.update_status(error_msg, "red")
            self.root.after(0, lambda: self.progress_bar.stop())
            self.model = None
            self.root.after(0, lambda: self.loaded_label.config(text="Loaded: Failed"))

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream with memory management"""
        if self.is_listening:
            self.audio_frames.append(in_data)
            # Prevent excessive memory usage by limiting frame buffer
            if len(self.audio_frames) > self.max_audio_frames:
                # Remove oldest frames to maintain buffer size
                remove_count = len(self.audio_frames) - self.max_audio_frames
                self.audio_frames = self.audio_frames[remove_count:]
        return (in_data, pyaudio.paContinue)

    def create_gui(self):
        # Style
        style = ttk.Style()
        style.configure('TFrame', background='#000022')
        style.configure('TButton', font=('Helvetica', 12), padding=10, background='#000022', foreground='white')
        style.configure('TLabel', font=('Helvetica', 10), background='#000000', foreground='white')
        style.configure('TCombobox', font=('Helvetica', 10), fieldbackground='white', foreground='black', selectbackground='#000055', selectforeground='white')
        style.configure('TCombobox.Listbox', background='#000022', foreground='white', selectbackground='#000055', selectforeground='white')
        style.configure('Vertical.TScrollbar', background='#000022', troughcolor='#000022', arrowcolor='white', bordercolor='#000022')
        style.configure('TProgressbar', background='#00aa00', troughcolor='#000033', bordercolor='#000033')

        # Title
        title_label = tk.Label(self.root, text="Voice 2 Text", font=('Arial Black', 26, 'bold'), bg='black', fg='white')
        self.canvas.create_window(450, 50, window=title_label)

        # Version
        version_label = tk.Label(self.root, text="v0.02", font=('Helvetica', 8), bg='#000000', fg='white')
        self.canvas.create_window(850, 20, window=version_label)

        # Time
        self.time_label = tk.Label(self.root, text="", font=('Helvetica', 10), bg='#000000', fg='white')
        self.canvas.create_window(850, 40, window=self.time_label)
        self.update_time()

        # Whisper model selection
        whisper_frame = ttk.Frame(self.root, style='TFrame')
        self.canvas.create_window(450, 610, window=whisper_frame)

        tk.Label(whisper_frame, text="Whisper Model:", bg='#000022', fg='white', font=('Arial', 12, 'bold')).pack(side='left')
        self.whisper_combo = ttk.Combobox(whisper_frame, textvariable=self.whisper_var, values=self.whisper_models, state='readonly', width=40)
        self.whisper_combo.pack(side='left', padx=(10, 0))
        self.whisper_var.set(self.selected_whisper_model)
        self.whisper_combo.bind('<<ComboboxSelected>>', self.on_whisper_change)

        # Loaded model indicator
        self.loaded_label = tk.Label(whisper_frame, text="Loaded: None", bg='#000010', fg='white', font=('Helvetica', 8))
        self.loaded_label.pack(side='left', padx=(10, 0))

        # Microphone selection
        mic_frame = ttk.Frame(self.root, style='TFrame')
        self.canvas.create_window(450, 650, window=mic_frame)

        tk.Label(mic_frame, text="Microphone:", bg='#000022', fg='white', font=('Arial', 12, 'bold')).pack(side='left')
        mic_values = self.microphones if self.microphones else ["No microphone detected"]
        self.mic_combo = ttk.Combobox(mic_frame, textvariable=self.mic_var, values=mic_values, state='readonly', width=40)
        self.mic_combo.pack(side='left', padx=(10, 0))
        self.mic_combo.bind('<<ComboboxSelected>>', self.on_mic_change_combo)

        # AI Model selection
        model_frame = ttk.Frame(self.root, style='TFrame')
        self.canvas.create_window(450, 690, window=model_frame)

        tk.Label(model_frame, text="AI Model:", bg='#000022', fg='white', font=('Arial', 12, 'bold')).pack(side='left')
        model_values = self.ollama_models if self.ollama_models else ["Ollama not running"]
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=model_values, state='readonly', width=40)
        self.model_combo.pack(side='left', padx=(10, 0))
        if self.ollama_models:
            self.model_var.set(self.selected_model)
        else:
            self.model_var.set("Ollama not running")
        self.model_var.trace('w', self.on_model_change)

        # Status label with loading indicator
        self.status_label = tk.Label(self.root, text="Ready", font=('Helvetica', 12, 'bold'), bg='#000033', fg='yellow')
        self.canvas.create_window(450, 110, window=self.status_label)

        # Progress bar for model download
        self.progress_bar = ttk.Progressbar(self.root, orient='horizontal', mode='indeterminate', length=400)
        self.canvas.create_window(450, 130, window=self.progress_bar)

        # Text area
        text_frame = ttk.Frame(self.root)
        self.canvas.create_window(250, 290, window=text_frame)

        tk.Label(text_frame, text="Transcribed Text:", bg='black', fg='white', font=('Helvetica', 10, 'bold')).pack(fill='x')
        self.text_area = scrolledtext.ScrolledText(text_frame, height=15, width=35, wrap=tk.WORD,
                                                    bg='black', fg='white', insertbackground='white',
                                                    font=('Consolas', 10), borderwidth=0, relief='flat')
        self.text_area.pack(fill='x', expand=False)

        # AI Response area
        ai_frame = ttk.Frame(self.root)
        self.canvas.create_window(650, 290, window=ai_frame)

        tk.Label(ai_frame, text="AI Response:", bg='black', fg='white', font=('Helvetica', 10, 'bold')).pack(fill='x')
        self.ai_text_area = scrolledtext.ScrolledText(ai_frame, height=15, width=35, wrap=tk.WORD,
                                                       bg='black', fg='white', insertbackground='white',
                                                       font=('Consolas', 10), borderwidth=0, relief='flat')
        self.ai_text_area.pack(fill='x', expand=False)

        # TTS Controls
        tts_frame = ttk.Frame(self.root, style='TFrame')
        self.canvas.create_window(450, 570, window=tts_frame)

        tk.Label(tts_frame, text="TTS Speed:", bg='#000022', fg='white', font=('Arial', 12, 'bold')).pack(side='left')
        self.tts_scale = tk.Scale(tts_frame, from_=100, to=300, orient='horizontal', bg='#000022', fg='white', troughcolor='#000055', highlightbackground='#000022')
        self.tts_scale.set(self.tts_rate)
        self.tts_scale.pack(side='left', padx=(10, 0))
        self.tts_scale.bind('<ButtonRelease-1>', self.on_tts_rate_change)

        # Buttons
        button_frame = ttk.Frame(self.root)
        self.canvas.create_window(450, 500, window=button_frame)

        self.dictation_button = ttk.Button(button_frame, text="🎙️ Start Dictation",
                                           command=self.toggle_dictation)
        self.dictation_button.pack(side='left', padx=5)

        self.copy_button = ttk.Button(button_frame, text="📋 Copy Text",
                                      command=self.copy_text)
        self.copy_button.pack(side='left', padx=5)

        self.send_ai_button = ttk.Button(button_frame, text="🤖 Query AI",
                                         command=self.send_to_ai)
        self.send_ai_button.pack(side='left', padx=5)

        self.stop_tts_button = ttk.Button(button_frame, text="Stop Speech",
                                          command=self.stop_tts)
        self.stop_tts_button.pack(side='left', padx=5)

        self.clear_button = ttk.Button(button_frame, text="🗑️ Clear",
                                       command=self.clear_text)
        self.clear_button.pack(side='left', padx=5)



    def on_mic_change_combo(self, event=None):
        value = self.mic_var.get()
        if value in self.microphones:
            self.selected_mic_index = self.microphones.index(value)
            self.selected_mic_name = value
            self.save_config()
            self.update_status(f"Selected: {value.split(' (')[0]}")

    def on_model_change(self, *args):
        self.selected_model = self.model_var.get()
        self.save_config()
        self.update_status(f"AI Model: {self.selected_model}")

    def on_whisper_change(self, event=None):
        old_model = self.selected_whisper_model
        self.selected_whisper_model = self.whisper_var.get()
        if self.selected_whisper_model != old_model:
            self.save_config()
            self.model = None  # Free old model
            self.loaded_label.config(text="Loaded: None")
            self.update_status(f"Loading Whisper model: {self.selected_whisper_model}...", "#ffaa00")
            threading.Thread(target=self.load_whisper_model, daemon=True).start()

    def on_tts_rate_change(self, event=None):
        self.tts_rate = int(self.tts_scale.get())
        self.save_config()

    def update_status(self, message, color='gray', progress_text=""):
        """Update status with optional progress indicator."""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=message, fg=color)
            self.root.update_idletasks()

    def toggle_dictation(self):
        if self.is_listening:
            self.stop_dictation()
        else:
            self.start_dictation()

    def start_dictation(self):
        if not self.microphones:
            messagebox.showerror("Error", "No microphones found!")
            return

        self.is_listening = True
        self.current_text = ""
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, "🎙️ Listening... Speak now!\n\n")

        self.dictation_button.config(text="⏹️ Stop Dictation")
        self.update_status("🎙️ Listening...", "#00aa00")

        # Start listening in background thread
        threading.Thread(target=self.listen_loop, daemon=True).start()

    def stop_dictation(self):
        self.is_listening = False
        self.dictation_button.config(text="🎙️ Start Dictation")

        if self.current_text.strip():
            pyperclip.copy(self.current_text.strip())
            self.update_status("📋 Text copied to clipboard!", "#0066cc")
        else:
            self.update_status("Ready", "black")

        # Small delay to ensure audio stream is fully closed
        time.sleep(0.1)

    def copy_text(self):
        if self.is_listening:
            self.stop_dictation()
        text = self.text_area.get(1.0, tk.END).strip()
        prompt = "🎙️ Listening... Speak now!\n\n"
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        if text:
            pyperclip.copy(text)
            self.update_status("📋 Text copied to clipboard!", "#0066cc")
        else:
            self.update_status("No text to copy", "black")

    def send_to_ai(self):
        if self.is_listening:
            self.stop_dictation()
        text = self.text_area.get(1.0, tk.END).strip()
        if text:
            self.ai_text_area.delete(1.0, tk.END)
            self.update_status("🤖 Sending to AI...", "#ffaa00")
            threading.Thread(target=self.query_ollama_and_speak, args=(text,), daemon=True).start()
        else:
            self.update_status("No text to send to AI", "black")

    def query_ollama_and_speak(self, user_text):
        """Query Ollama with retry logic and improved error handling."""
        if not user_text or not user_text.strip():
            self.update_status("No text to send to AI", "orange")
            return

        # Check if Ollama is running
        if not self.ollama_models:
            self.update_status("Ollama not running - start with 'ollama serve'", "red")
            return

        # Sanitize input
        user_text = user_text.strip()
        if len(user_text) > 10000:  # Limit input size
            user_text = user_text[:10000] + "..."
            self.update_status("Input truncated to 10,000 characters", "orange")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.update_status(f"🤖 Querying AI... (attempt {attempt + 1}/{max_retries})", "#ffaa00")

                # Query Ollama with reasonable timeout
                response = requests.post('http://localhost:11434/api/generate',
                                        json={
                                            "model": self.selected_model,
                                            "prompt": user_text,
                                            "stream": False
                                        },
                                        timeout=120)  # Increased timeout for slower models

                response.raise_for_status()
                ai_response = response.json().get('response', '').strip()

                if ai_response:
                    # Display AI response
                    self.queue.put(("clear_ai",))
                    self.queue.put(("insert_ai", ai_response))

                    # Speak the response with TTS
                    self.update_status("🎵 Generating speech...", "#00aa00")
                    self.speak_with_tts(ai_response)
                    self.update_status("🤖 AI responded successfully!", "#00aa00")
                    return  # Success, exit function
                    self.update_status("AI gave empty response", "orange")
                    return

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    self.update_status(f"AI timeout, retrying... ({attempt + 1}/{max_retries})", "orange")
                    time.sleep(2)  # Wait before retry
                    continue
                    self.update_status("AI timeout - model may be slow or overloaded", "red")

            except requests.exceptions.ConnectionError:
                self.update_status("Cannot connect to Ollama - check if running", "red")
                break  # Don't retry connection errors

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else "unknown"
                self.update_status(f"Ollama HTTP error {status_code}: {str(e)[:50]}", "red")
                break  # Don't retry HTTP errors

            except (KeyError, ValueError) as e:
                self.update_status(f"Invalid response from Ollama: {str(e)[:50]}", "red")
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    self.update_status(f"AI error, retrying... ({attempt + 1}/{max_retries})", "orange")
                    time.sleep(1)
                    continue
                    self.update_status(f"AI error: {str(e)[:50]}", "red")

    def speak_with_tts(self, text):
        """Speak text with edge-tts or fallback to gTTS."""
        if not text or not text.strip():
            self.update_status("No text to speak", "orange")
            return

        # Limit text length for TTS
        text = text.strip()
        # Remove hashtags and asterisks for cleaner speech
        text = text.replace('#', '').replace('*', '')
        if len(text) > 5000:
            text = text[:5000] + "..."
            self.update_status("Speech truncated to 5000 characters", "orange")

        try:
            self.update_status("🔊 Generating speech...", "#00aa00")
            self.tts_playing = True

            # Calculate rate for edge-tts: map 100-300 to -50% to +50%
            rate_percent = ((self.tts_rate - 180) / 120) * 50  # 180 is neutral
            rate_str = f"{rate_percent:+.0f}%"

            async def generate_speech():
                try:
                    voice = "en-US-AriaNeural"
                    communicate = edge_tts.Communicate(text, voice, rate=rate_str)
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', mode='w+b')
                    temp_file.close()
                    await communicate.save(temp_file.name)
                    return temp_file.name
                except Exception as e:
                    raise e

            temp_file_name = asyncio.run(generate_speech())
            pygame.mixer.music.load(temp_file_name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and self.tts_playing:
                pygame.time.wait(100)
            pygame.mixer.music.stop()
            os.unlink(temp_file_name)
            self.update_status("Speech completed", "#00aa00")

        except Exception as e:
            # Fallback to gTTS
            try:
                self.update_status("Edge TTS failed, using gTTS...", "orange")
                tts = gTTS(text=text, lang='en', slow=False, tld='co.uk')
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', mode='w+b')
                temp_file.close()
                tts.save(temp_file.name)
                pygame.mixer.music.load(temp_file.name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and self.tts_playing:
                    pygame.time.wait(100)
                pygame.mixer.music.stop()
                os.unlink(temp_file.name)
                self.update_status("Speech completed (gTTS)", "#00aa00")
            except Exception as e2:
                error_msg = f"TTS error: {str(e)[:30]}, gTTS: {str(e2)[:30]}"
                self.update_status(error_msg, "red")
        finally:
            self.tts_playing = False
            self.queue.put(("update_status", "Ready", "white"))

    def stop_tts(self):
        if self.is_listening:
            self.stop_dictation()
        self.tts_playing = False
        pygame.mixer.music.stop()
        self.update_status("TTS stopped", "orange")

    def clear_text(self):
        if self.is_listening:
            self.stop_dictation()
        self.text_area.delete(1.0, tk.END)
        self.ai_text_area.delete(1.0, tk.END)
        self.current_text = ""
        self.update_status("Ready", "black")

    def listen_loop(self):
        try:
            device_index = self.get_mic_device_index(self.microphones[self.selected_mic_index])

            # Start audio stream - try different sample rates
            self.audio_frames = []
            sample_rates = [48000, 44100, 32000, 22050, 16000, 8000]  # Try common rates

            for rate in sample_rates:
                try:
                    self.audio_stream = self.audio.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=1024,
                        stream_callback=self.audio_callback
                    )
                    self.sample_rate = rate
                    break
                except Exception as e:
                    print(f"Failed to open stream at {rate} Hz: {e}")
                    continue
                self.root.after(0, lambda: self.update_status("No audio device available - check microphone setup", "red"))
                return

            self.audio_stream.start_stream()
            self.queue.put(("update_status", "🎙️ Listening... (real-time)", "#00aa00"))

            processed_frames = 0
            chunk_duration = 3  # Increased to 3 seconds for better accuracy and less CPU usage
            silence_threshold = 500  # RMS threshold for silence detection
            consecutive_silent_chunks = 0
            max_silent_chunks = 5  # Stop after 15 seconds of silence

            # Real-time transcription loop
            while self.is_listening:
                time.sleep(chunk_duration)

                # Check if we have new frames to process
                if len(self.audio_frames) > processed_frames:
                    chunk_frames = self.audio_frames[processed_frames:]
                    processed_frames = len(self.audio_frames)

                    # Quick silence detection to avoid unnecessary processing
                    if len(chunk_frames) > 0:
                        try:
                            audio_data = np.frombuffer(b''.join(chunk_frames), dtype=np.int16)
                            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))

                            if rms < silence_threshold:
                                consecutive_silent_chunks += 1
                                if consecutive_silent_chunks >= max_silent_chunks:
                                    self.queue.put(("update_status", "Silence detected, stopping...", "#ffaa00"))
                                    self.stop_dictation()
                                    break
                                consecutive_silent_chunks = 0
                        except:
                            pass  # Continue processing if RMS calculation fails

                    # Process this chunk
                    self.queue.put(("update_status", "🔍 Recognizing...", "#ffaa00"))
                    try:
                        # Convert chunk to numpy and resample (optimized)
                        audio_data = np.frombuffer(b''.join(chunk_frames), dtype=np.int16)

                        # Only resample if necessary and cache the resampled data
                        if self.sample_rate != 16000:
                            num_samples = len(audio_data)
                            target_samples = int(num_samples * 16000 / self.sample_rate)
                            audio_data = resample(audio_data, target_samples).astype(np.int16)

                        # Save chunk to temp WAV
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                            temp_filename = temp_file.name

                            with wave.open(temp_filename, 'wb') as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(16000)
                                wf.writeframes(audio_data.tobytes())

                        # Transcribe chunk with error checking
                        if self.model is None:
                            raise RuntimeError("Whisper model not loaded")

                        segments, info = self.model.transcribe(temp_filename, language="en")
                        text = " ".join(segment.text for segment in segments).strip()

                        os.unlink(temp_filename)

                        if text:
                            self.current_text += text + " "
                            self.queue.put(("update_transcript", text))
                            self.queue.put(("update_status", "🎙️ Listening... (real-time)", "#00aa00"))
                            self.queue.put(("update_status", "🎙️ Listening... (real-time)", "#00aa00"))

                    except Exception as e:
                        self.queue.put(("update_transcript", f"[Error: {e}]"))
                        self.queue.put(("update_status", "🎙️ Listening... (real-time)", "#00aa00"))

            # Stop recording
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None

            # Process any remaining frames
            if len(self.audio_frames) > processed_frames:
                remaining_frames = self.audio_frames[processed_frames:]
                self.queue.put(("update_status", "🔍 Finalizing...", "#ffaa00"))
                try:
                    audio_data = np.frombuffer(b''.join(remaining_frames), dtype=np.int16)
                    if self.sample_rate != 16000:
                        num_samples = len(audio_data)
                        target_samples = int(num_samples * 16000 / self.sample_rate)
                        audio_data = resample(audio_data, target_samples).astype(np.int16)

                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_filename = temp_file.name

                        with wave.open(temp_filename, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(16000)
                            wf.writeframes(audio_data.tobytes())

                    if self.model is None:
                        raise RuntimeError("Whisper model not loaded")

                    segments, info = self.model.transcribe(temp_filename, language="en")
                    text = " ".join(segment.text for segment in segments).strip()

                    os.unlink(temp_filename)

                    if text:
                        self.current_text += text + " "
                        self.queue.put(("update_transcript", text))

                except Exception as e:
                    self.root.after(0, self.update_transcript, f"[Error: {e}]")

            self.root.after(0, lambda: self.update_status("Ready", "black"))

        except Exception as e:
            self.queue.put(("show_error", f"Recognition error: {e}"))
            self.queue.put(("stop_dictation",))

    def update_transcript(self, text):
        self.text_area.insert(tk.END, f"{text}\n")
        self.text_area.see(tk.END)
        self.root.update_idletasks()

def main():
    try:
        root = tk.Tk()
        app = VoiceApp(root)
        root.mainloop()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install required packages:")
        print("pip install SpeechRecognition pyperclip pyaudio pocketsphinx")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()