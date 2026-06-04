#!/usr/bin/env python3
"""
Voice 2 Text Qt Application

Qt/PySide6 port of the Voice2Text-AI desktop app.
"""

import os
import sys
import contextlib
import math
import json
import re
import time
import queue
import tempfile
import subprocess
import shutil
import asyncio
import threading
import datetime

if sys.platform.startswith("linux"):
    os.environ.setdefault("JACK_NO_START_SERVER", "1")
    os.environ.setdefault("ALSA_NO_JACK", "1")
    if os.environ.get("FLATPAK_ID"):
        os.environ.setdefault("SDL_AUDIODRIVER", "pulseaudio")

import numpy as np
import pyaudio
import pyperclip
import requests
import pygame
import edge_tts
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

from PySide6.QtCore import Qt, QObject, Signal, QTimer
from PySide6.QtGui import QFont, QPainter, QLinearGradient, QColor, QBrush
from PySide6.QtWidgets import (
    QApplication, QComboBox, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

APP_VERSION = "v0.3.1"


class AppSignals(QObject):
    status = Signal(str, str)
    transcript = Signal(str)
    ai_clear = Signal()
    ai_append = Signal(str)
    progress_mode = Signal(bool)
    progress_value = Signal(int)
    loaded_label = Signal(str)
    stop_dictation = Signal()
    error = Signal(str)
    button_text = Signal(str)


class GradientWidget(QWidget):
    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor(0, 0, 0))
        gradient.setColorAt(1.0, QColor(0, 0, 51))
        painter.fillRect(self.rect(), QBrush(gradient))


class VoiceAppQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.shutdown_event = threading.Event()
        self.model_lock = threading.Lock()
        self.audio_queue = queue.Queue(maxsize=256)
        self.model = None
        self.sample_rate = 16000
        self.audio = None
        self.audio_stream = None
        self.is_listening = False
        self.current_text = ""
        self.tts_playing = False
        self.tts_available = False

        self.config_file = os.path.expanduser("~/.voice_config.json")
        print(f"Config file: {self.config_file}")
        self.config = self.load_config()
        self.tts_rate = int(self.config.get("tts_rate", 180))

        self.signals = AppSignals()

        self.setWindowTitle("Voice 2 Text")
        self.resize(1000, 940)
        self.setMinimumSize(900, 860)

        self.whisper_models = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
        self.selected_whisper_model = self.config.get("whisper_model", "tiny")
        if self.selected_whisper_model not in self.whisper_models:
            self.selected_whisper_model = "tiny"
        self.model_info = {
            "tiny": {"size": 39, "eta": 1},
            "base": {"size": 74, "eta": 1},
            "small": {"size": 244, "eta": 4},
            "medium": {"size": 769, "eta": 13},
            "large-v2": {"size": 1550, "eta": 26},
            "large-v3": {"size": 1550, "eta": 26},
        }

        try:
            self.audio = pyaudio.PyAudio()
            self.microphones = self.get_microphones()
        except Exception as e:
            print(f"Audio initialization failed: {e}")
            self.microphones = []

        self.selected_mic_index = 0
        self.selected_mic_name = self.config.get("microphone_name", "")
        self.ollama_models = self.get_ollama_models()
        self.selected_model = self.config.get(
            "selected_model",
            "llama3.2" if "llama3.2" in self.ollama_models else (self.ollama_models[0] if self.ollama_models else "llama3.2"),
        )
        if self.ollama_models and self.selected_model not in self.ollama_models:
            self.selected_model = self.ollama_models[0]

        try:
            pygame.mixer.init()
            self.tts_available = True
        except Exception as e:
            print(f"TTS audio output initialization failed: {e}")

        self.build_ui()
        self.connect_signals()
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_time)
        self.clock_timer.start(1000)
        self.update_time()
        threading.Thread(target=self.load_whisper_model, daemon=True).start()

    def connect_signals(self):
        self.signals.status.connect(self.set_status)
        self.signals.transcript.connect(self.update_transcript)
        self.signals.ai_clear.connect(self.ai_text_area.clear)
        self.signals.ai_append.connect(self.ai_text_area_append)
        self.signals.progress_mode.connect(self.set_progress_busy)
        self.signals.progress_value.connect(self.set_progress_value)
        self.signals.loaded_label.connect(self.loaded_label_set_text)
        self.signals.stop_dictation.connect(self.stop_dictation)
        self.signals.error.connect(self.show_error)
        self.signals.button_text.connect(self.set_dictation_button_text)

    def build_ui(self):
        self.root_widget = GradientWidget()
        self.setCentralWidget(self.root_widget)
        self.root_widget.setStyleSheet("""
            QWidget { color: white; font-family: "Noto Sans", "DejaVu Sans", "Liberation Sans", sans-serif; font-size: 12pt; }
            QLabel { color: white; background: transparent; }
            QPushButton { background-color: #000022; color: white; border: 1px solid #666699; padding: 10px 14px; font-size: 12pt; }
            QPushButton:hover { background-color: #000044; }
            QPushButton:pressed { background-color: #000066; }
            QPlainTextEdit { background-color: black; color: white; border: 1px solid #dddddd; selection-background-color: #003366; font-family: "Noto Sans", "DejaVu Sans", "Liberation Sans", sans-serif; font-size: 12pt; }
            QComboBox { background-color: #dddddd; color: black; border: 1px solid #888888; padding: 2px 4px; font-size: 10pt; }
            QComboBox QAbstractItemView { background-color: white; color: black; selection-background-color: #003366; selection-color: white; }
            QSlider::groove:horizontal { background: #000055; height: 16px; border: 1px solid #444488; }
            QSlider::handle:horizontal { background: #888888; width: 14px; border: 1px solid #cccccc; margin: -2px 0; }
            QProgressBar { border: 1px solid #66cc66; background: #000033; height: 18px; text-align: center; }
            QProgressBar::chunk { background-color: #00bb00; }
        """)

        outer = QVBoxLayout(self.root_widget)
        outer.setContentsMargins(60, 25, 60, 35)
        outer.setSpacing(14)

        top_row = QHBoxLayout()
        top_row.addStretch()

        header_right = QVBoxLayout()
        header_right.setSpacing(2)

        self.version_label = QLabel(APP_VERSION)
        self.version_label.setAlignment(Qt.AlignRight)
        self.version_label.setFont(QFont("Noto Sans", 10))
        header_right.addWidget(self.version_label)

        self.time_label = QLabel("")
        self.time_label.setAlignment(Qt.AlignRight)
        self.time_label.setFont(QFont("Noto Sans", 10))
        header_right.addWidget(self.time_label)

        top_row.addLayout(header_right)
        outer.addLayout(top_row)

        self.title_label = QLabel("Voice 2 Text")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setMinimumHeight(95)
        self.title_label.setFont(QFont("Noto Sans", 46, QFont.Bold))
        self.title_label.setStyleSheet(
            "font-size: 46pt; "
            "font-weight: 800; "
            "color: white; "
            "background: transparent;"
        )
        outer.addWidget(self.title_label)


        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Noto Sans", 13, QFont.Bold))
        self.status_label.setStyleSheet("background-color: #000033; color: yellow; padding: 4px;")
        outer.addWidget(self.status_label, alignment=Qt.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(400)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        outer.addWidget(self.progress_bar, alignment=Qt.AlignCenter)

        text_grid = QGridLayout()
        text_grid.setHorizontalSpacing(55)
        left_label = QLabel("Transcribed Text:")
        left_label.setAlignment(Qt.AlignCenter)
        left_label.setFont(QFont("Noto Sans", 11, QFont.Bold))
        right_label = QLabel("AI Response:")
        right_label.setAlignment(Qt.AlignCenter)
        right_label.setFont(QFont("Noto Sans", 11, QFont.Bold))
        self.text_area = QPlainTextEdit()
        self.text_area.setFixedHeight(285)
        self.text_area.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.ai_text_area = QPlainTextEdit()
        self.ai_text_area.setFixedHeight(285)
        self.ai_text_area.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        text_grid.addWidget(left_label, 0, 0)
        text_grid.addWidget(right_label, 0, 1)
        text_grid.addWidget(self.text_area, 1, 0)
        text_grid.addWidget(self.ai_text_area, 1, 1)
        outer.addLayout(text_grid)
        outer.addSpacing(38)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        self.dictation_button = QPushButton("Start Dictation")
        self.dictation_button.clicked.connect(self.toggle_dictation)
        self.copy_button = QPushButton("Copy Text")
        self.copy_button.clicked.connect(self.copy_text)
        self.send_ai_button = QPushButton("Query AI")
        self.send_ai_button.clicked.connect(self.send_to_ai)
        self.stop_tts_button = QPushButton("Stop Speech")
        self.stop_tts_button.clicked.connect(self.stop_tts)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_text)
        # Direct shiny button styling
        shiny_button_style = """
        QPushButton {
            color: white;
            background: qlineargradient(
                x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 #2d3d9f,
                stop: 0.18 #1b2875,
                stop: 0.52 #0b1249,
                stop: 1 #030622
            );
            border-top: 2px solid #8796dd;
            border-left: 2px solid #5f6fc2;
            border-right: 2px solid #05082a;
            border-bottom: 2px solid #020315;
            border-radius: 8px;
            padding: 10px 16px;
            font-size: 12pt;
            font-weight: 700;
        }
        QPushButton:hover {
            background: qlineargradient(
                x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 #3b50c9,
                stop: 0.20 #263597,
                stop: 0.55 #10195c,
                stop: 1 #05082a
            );
            border-top: 2px solid #b8c4ff;
            border-left: 2px solid #7f90e6;
            border-right: 2px solid #10195c;
            border-bottom: 2px solid #070a2f;
        }
        QPushButton:pressed {
            background: qlineargradient(
                x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 #020315,
                stop: 0.48 #070d34,
                stop: 1 #1b2875
            );
            padding-top: 12px;
            padding-bottom: 8px;
            border-top: 2px solid #020315;
            border-left: 2px solid #020315;
            border-right: 2px solid #5f6fc2;
            border-bottom: 2px solid #8796dd;
        }
        """
        for button in (
            self.dictation_button,
            self.copy_button,
            self.send_ai_button,
            self.stop_tts_button,
            self.clear_button,
        ):
            button.setMinimumHeight(50)
            button.setCursor(Qt.PointingHandCursor)
            button.setStyleSheet(shiny_button_style)

        for button in (self.dictation_button, self.copy_button, self.send_ai_button, self.stop_tts_button, self.clear_button):
            button_row.addWidget(button)
        outer.addLayout(button_row)

        tts_row = QHBoxLayout()
        tts_row.addStretch()
        tts_label = QLabel("TTS Speed:")
        tts_label.setFont(QFont("Noto Sans", 12, QFont.Bold))
        tts_row.addWidget(tts_label)
        self.tts_value_label = QLabel(str(self.tts_rate))
        self.tts_value_label.setFixedWidth(45)
        self.tts_value_label.setAlignment(Qt.AlignCenter)
        self.tts_slider = QSlider(Qt.Horizontal)
        self.tts_slider.setRange(100, 300)
        self.tts_slider.setValue(self.tts_rate)
        self.tts_slider.setFixedWidth(130)
        self.tts_slider.valueChanged.connect(self.on_tts_rate_change)
        tts_row.addWidget(self.tts_value_label)
        tts_row.addWidget(self.tts_slider)
        tts_row.addStretch()
        outer.addLayout(tts_row)

        controls = QGridLayout()
        controls.setHorizontalSpacing(10)
        controls.setVerticalSpacing(12)
        self.whisper_combo = QComboBox()
        self.whisper_combo.addItems(self.whisper_models)
        self.whisper_combo.setCurrentText(self.selected_whisper_model)
        self.whisper_combo.setFixedWidth(420)
        self.whisper_combo.currentTextChanged.connect(self.on_whisper_change)
        self.loaded_label = QLabel("Loaded: None")
        self.loaded_label.setFont(QFont("Noto Sans", 9))
        self.mic_combo = QComboBox()
        mic_values = self.microphones if self.microphones else ["No microphone detected"]
        self.mic_combo.addItems(mic_values)
        if self.microphones:
            if self.selected_mic_name in self.microphones:
                self.selected_mic_index = self.microphones.index(self.selected_mic_name)
            else:
                self.selected_mic_index = 0
                self.selected_mic_name = self.microphones[0]
            self.mic_combo.setCurrentIndex(self.selected_mic_index)
        self.mic_combo.setFixedWidth(420)
        self.mic_combo.currentIndexChanged.connect(self.on_mic_change_combo)
        self.model_combo = QComboBox()
        model_values = self.ollama_models if self.ollama_models else ["Ollama not running"]
        self.model_combo.addItems(model_values)
        if self.ollama_models:
            self.model_combo.setCurrentText(self.selected_model)
        self.model_combo.setFixedWidth(420)
        self.model_combo.currentTextChanged.connect(self.on_model_change)
        labels = [QLabel("Whisper Model:"), QLabel("Microphone:"), QLabel("AI Model:")]
        for label in labels:
            label.setFont(QFont("Noto Sans", 12, QFont.Bold))
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        controls.addWidget(labels[0], 0, 0)
        controls.addWidget(self.whisper_combo, 0, 1)
        controls.addWidget(self.loaded_label, 0, 2)
        controls.addWidget(labels[1], 1, 0)
        controls.addWidget(self.mic_combo, 1, 1, 1, 2)
        controls.addWidget(labels[2], 2, 0)
        controls.addWidget(self.model_combo, 2, 1, 1, 2)
        controls.setColumnStretch(1, 1)
        outer.addLayout(controls)
        outer.addStretch()

    def update_time(self):
        self.time_label.setText(datetime.datetime.now().strftime("%I:%M:%S %p"))

    def set_status(self, message, color="gray"):
        qcolor = {"red": "#ff4444", "orange": "#ffaa00", "black": "#ffffff", "gray": "#dddddd"}.get(color, color)
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"background-color: #000033; color: {qcolor}; padding: 4px;")

    def update_status(self, message, color="gray"):
        self.signals.status.emit(message, color)

    def update_transcript(self, text):
        self.text_area.appendPlainText(text)
        self.text_area.verticalScrollBar().setValue(self.text_area.verticalScrollBar().maximum())

    def ai_text_area_append(self, text):
        self.ai_text_area.appendPlainText(text)
        self.ai_text_area.verticalScrollBar().setValue(self.ai_text_area.verticalScrollBar().maximum())

    def set_progress_busy(self, busy):
        self.progress_bar.setRange(0, 0 if busy else 100)

    def set_progress_value(self, value):
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(value)

    def loaded_label_set_text(self, text):
        self.loaded_label.setText(text)

    def set_dictation_button_text(self, text):
        self.dictation_button.setText(text)

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                print(f"Loaded config: {config}")
                return config
            except Exception as e:
                print(f"Error loading config: {e}")
        print("Loaded config: {}")
        return {}

    def save_config(self):
        config = {
            "microphone_name": self.selected_mic_name,
            "selected_model": self.selected_model,
            "whisper_model": self.selected_whisper_model,
            "tts_rate": self.tts_rate,
        }
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f)
            print(f"Saved config: {config}")
        except Exception as e:
            print(f"Error saving config: {e}")

    def get_ollama_models(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=1.5)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except requests.exceptions.Timeout:
            self.update_status("Ollama connection timeout - check if Ollama is running", "orange")
        except requests.exceptions.ConnectionError:
            self.update_status("Cannot connect to Ollama - start with 'ollama serve'", "red")
        except requests.exceptions.RequestException as e:
            self.update_status(f"Ollama request error: {str(e)[:50]}", "red")
        except (KeyError, ValueError) as e:
            self.update_status(f"Invalid Ollama response: {str(e)[:50]}", "red")
        return []

    def get_microphones(self):
        microphones = []
        if not self.audio:
            return microphones
        try:
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                max_input = info.get("maxInputChannels", 0)
                if isinstance(max_input, (int, float)) and max_input > 0:
                    microphones.append(f"{info.get('name')} (Index: {i})")
        except Exception as e:
            print(f"Could not enumerate microphones: {e}")
        return microphones

    def get_mic_device_index(self, mic_string):
        match = re.search(r"Index: (\d+)", mic_string)
        return int(match.group(1)) if match else 0

    def on_mic_change_combo(self, index):
        if 0 <= index < len(self.microphones):
            self.selected_mic_index = index
            self.selected_mic_name = self.microphones[index]
            self.save_config()
            self.update_status(f"Selected: {self.selected_mic_name.split(' (')[0]}")

    def on_model_change(self, value):
        if value and value != "Ollama not running":
            self.selected_model = value
            self.save_config()
            self.update_status(f"AI Model: {self.selected_model}")

    def on_whisper_change(self, value):
        if value and value != self.selected_whisper_model:
            self.selected_whisper_model = value
            self.save_config()
            with self.model_lock:
                self.model = None
            self.loaded_label.setText("Loaded: None")
            self.update_status(f"Loading Whisper model: {self.selected_whisper_model}...", "#ffaa00")
            threading.Thread(target=self.load_whisper_model, daemon=True).start()

    def on_tts_rate_change(self, value):
        self.tts_rate = int(value)
        self.tts_value_label.setText(str(self.tts_rate))
        self.save_config()

    def load_whisper_model(self):
        model_name = self.selected_whisper_model
        info = self.model_info.get(model_name, {"size": "unknown", "eta": "unknown"})
        self.update_status(f"Loading Whisper model: {model_name} ({info['size']} MB) - estimated download time: {info['eta']} min", "#ffaa00")
        self.signals.progress_mode.emit(True)
        last_error = None
        try:
            model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=max(1, min(4, (os.cpu_count() or 4))))
            with self.model_lock:
                if model_name != self.selected_whisper_model:
                    return
                self.model = model
            self.signals.progress_mode.emit(False)
            self.signals.progress_value.emit(100)
            self.signals.loaded_label.emit(f"Loaded: {model_name}")
            self.update_status("Whisper model loaded on CPU!", "#00aa00")
        except Exception as e:
            last_error = e
            with self.model_lock:
                self.model = None
            self.signals.progress_mode.emit(False)
            self.signals.progress_value.emit(0)
            self.signals.loaded_label.emit("Loaded: Failed")
            self.update_status(f"Failed to load Whisper model: {str(last_error)[:120]}", "red")

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_listening:
            try:
                self.audio_queue.put_nowait(in_data)
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self.audio_queue.get_nowait()
                with contextlib.suppress(queue.Full):
                    self.audio_queue.put_nowait(in_data)
        return (in_data, pyaudio.paContinue)

    def toggle_dictation(self):
        self.stop_dictation() if self.is_listening else self.start_dictation()

    def start_dictation(self):
        if not self.microphones:
            QMessageBox.critical(self, "Error", "No microphones found!")
            return
        with self.model_lock:
            model_ready = self.model is not None
        if not model_ready:
            self.update_status("Whisper model is still loading; try again shortly", "orange")
            return
        self.is_listening = True
        self.current_text = ""
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.text_area.clear()
        self.text_area.appendPlainText("Listening... Speak now!\n")
        self.signals.button_text.emit("Stop Dictation")
        self.update_status("Listening...", "#00aa00")
        threading.Thread(target=self.listen_loop, daemon=True).start()

    def stop_dictation(self):
        self.is_listening = False
        self.signals.button_text.emit("Start Dictation")
        if self.current_text.strip():
            self.update_status("Dictation stopped. Ready to query AI or copy text.", "#0066cc")
        else:
            self.update_status("Ready", "black")

    def copy_text(self):
        if self.is_listening:
            self.stop_dictation()
        text = self.text_area.toPlainText().strip()
        prompt = "Listening... Speak now!"
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        if text:
            try:
                pyperclip.copy(text)
                self.update_status("Text copied to clipboard!", "#0066cc")
            except Exception as e:
                self.update_status(f"Clipboard unavailable: {str(e)[:60]}", "orange")
        else:
            self.update_status("No text to copy", "black")

    def send_to_ai(self):
        if self.is_listening:
            self.stop_dictation()
        text = self.text_area.toPlainText().strip()
        prompt = "Listening... Speak now!"
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        if text:
            self.ai_text_area.clear()
            self.update_status("Sending to AI...", "#ffaa00")
            threading.Thread(target=self.query_ollama_and_speak, args=(text,), daemon=True).start()
        else:
            self.update_status("No text to send to AI", "black")

    def query_ollama_and_speak(self, user_text):
        if not user_text or not user_text.strip():
            self.update_status("No text to send to AI", "orange")
            return
        if not self.ollama_models:
            self.ollama_models = self.get_ollama_models()
            if not self.ollama_models:
                self.update_status("Ollama not running - start with 'ollama serve'", "red")
                return
        user_text = user_text.strip()
        if len(user_text) > 10000:
            user_text = user_text[:10000] + "..."
            self.update_status("Input truncated to 10,000 characters", "orange")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.update_status(f"Querying AI... (attempt {attempt + 1}/{max_retries})", "#ffaa00")
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": self.selected_model, "prompt": user_text, "stream": False, "options": {"num_predict": 512}},
                    timeout=(5, 120),
                )
                response.raise_for_status()
                ai_response = response.json().get("response", "").strip()
                if not ai_response:
                    self.update_status("AI gave empty response", "orange")
                    return
                self.signals.ai_clear.emit()
                self.signals.ai_append.emit(ai_response)
                self.update_status("Generating speech...", "#00aa00")
                self.speak_with_tts(ai_response)
                self.update_status("AI responded successfully!", "#00aa00")
                return
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    self.update_status(f"AI timeout, retrying... ({attempt + 1}/{max_retries})", "orange")
                    time.sleep(2)
                    continue
                self.update_status("AI timeout - model may be slow or overloaded", "red")
            except requests.exceptions.ConnectionError:
                self.update_status("Cannot connect to Ollama - check if running", "red")
                break
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else "unknown"
                self.update_status(f"Ollama HTTP error {status_code}: {str(e)[:50]}", "red")
                break
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
        if not text.strip():
            self.update_status("No text to speak", "orange")
            return
        if not self.tts_available:
            self.update_status("TTS not available", "orange")
            return
        temp_files = []
        self.tts_playing = True
        try:
            try:
                self.update_status("Generating speech...", "#0066cc")
                rate_percent = ((self.tts_rate - 180) / 120) * 50
                rate_percent = max(-50, min(50, rate_percent))
                rate_str = f"{rate_percent:+.0f}%"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", mode="w+b")
                temp_file_name = temp_file.name
                temp_file.close()
                temp_files.append(temp_file_name)

                async def save_edge_tts():
                    communicate = edge_tts.Communicate(text, "en-US-AriaNeural", rate=rate_str)
                    await communicate.save(temp_file_name)

                asyncio.run(save_edge_tts())
                pygame.mixer.music.load(temp_file_name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and self.tts_playing:
                    pygame.time.wait(100)
                pygame.mixer.music.stop()
                if self.tts_playing:
                    self.update_status("Speech completed", "#00aa00")
                return
            except Exception as edge_error:
                try:
                    if not shutil.which("espeak-ng"):
                        raise FileNotFoundError("espeak-ng is not installed")
                    self.update_status("Edge TTS failed, using offline eSpeak NG...", "orange")
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="w+b")
                    temp_file_name = temp_file.name
                    temp_file.close()
                    temp_files.append(temp_file_name)
                    subprocess.run(["espeak-ng", "-s", str(int(self.tts_rate)), "-w", temp_file_name, text], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    pygame.mixer.music.load(temp_file_name)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy() and self.tts_playing:
                        pygame.time.wait(100)
                    pygame.mixer.music.stop()
                    if self.tts_playing:
                        self.update_status("Speech completed (eSpeak NG)", "#00aa00")
                except Exception as espeak_error:
                    self.update_status(f"TTS error: Edge: {str(edge_error)[:30]}, eSpeak NG: {str(espeak_error)[:30]}", "orange")
        finally:
            self.tts_playing = False
            with contextlib.suppress(Exception):
                pygame.mixer.music.unload()
            for temp_file_name in temp_files:
                with contextlib.suppress(Exception):
                    os.unlink(temp_file_name)

    def stop_tts(self):
        if self.is_listening:
            self.stop_dictation()
        self.tts_playing = False
        if self.tts_available:
            with contextlib.suppress(Exception):
                pygame.mixer.music.stop()
        self.update_status("TTS stopped", "orange")

    def clear_text(self):
        if self.is_listening:
            self.stop_dictation()
        self.text_area.clear()
        self.ai_text_area.clear()
        self.current_text = ""
        self.update_status("Ready", "black")

    def transcribe_audio(self, audio_data, sample_rate):
        if audio_data.size == 0:
            return ""
        if sample_rate != 16000:
            divisor = math.gcd(sample_rate, 16000)
            audio_data = resample_poly(audio_data, 16000 // divisor, sample_rate // divisor).astype(np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0
        with self.model_lock:
            model = self.model
        if model is None:
            raise RuntimeError("Whisper model not loaded")
        segments, _info = model.transcribe(audio_float, language="en", beam_size=1, vad_filter=True, condition_on_previous_text=False)
        return " ".join(segment.text for segment in segments).strip()

    def listen_loop(self):
        try:
            if not self.audio:
                self.update_status("No audio input available", "red")
                self.signals.stop_dictation.emit()
                return
            device_index = self.get_mic_device_index(self.microphones[self.selected_mic_index])
            sample_rates = [48000, 44100, 16000, 22050, 8000]
            self.audio_stream = None
            for rate in sample_rates:
                try:
                    self.audio_stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, input_device_index=device_index, frames_per_buffer=1024, stream_callback=self.audio_callback)
                    self.sample_rate = rate
                    break
                except Exception as e:
                    print(f"Failed to open stream at {rate} Hz: {e}")
            if self.audio_stream is None:
                self.update_status("No audio device available - check microphone setup", "red")
                self.signals.stop_dictation.emit()
                return
            self.audio_stream.start_stream()
            self.update_status("Listening... (real-time)", "#00aa00")
            chunk_duration = 3.0
            silence_threshold = 500
            consecutive_silent_chunks = 0
            max_silent_chunks = 5
            pending_frames = []
            next_process = time.monotonic() + chunk_duration
            while self.is_listening and not self.shutdown_event.is_set():
                timeout = max(0.05, next_process - time.monotonic())
                try:
                    pending_frames.append(self.audio_queue.get(timeout=timeout))
                except queue.Empty:
                    pass
                if time.monotonic() < next_process:
                    continue
                next_process = time.monotonic() + chunk_duration
                if not pending_frames:
                    continue
                chunk = b"".join(pending_frames)
                pending_frames.clear()
                audio_data = np.frombuffer(chunk, dtype=np.int16)
                if audio_data.size == 0:
                    continue
                try:
                    rms = float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))
                    if rms < silence_threshold:
                        consecutive_silent_chunks += 1
                        if consecutive_silent_chunks >= max_silent_chunks:
                            self.update_status("Silence detected, stopping...", "#ffaa00")
                            self.is_listening = False
                            break
                        continue
                    consecutive_silent_chunks = 0
                except Exception:
                    pass
                self.update_status("Recognizing...", "#ffaa00")
                try:
                    text = self.transcribe_audio(audio_data, self.sample_rate)
                    if text:
                        self.current_text += text + " "
                        self.signals.transcript.emit(text)
                    self.update_status("Listening... (real-time)", "#00aa00")
                except Exception as e:
                    self.signals.transcript.emit(f"[Error: {e}]")
                    self.update_status("Listening... (real-time)", "#00aa00")
            if pending_frames:
                self.update_status("Finalizing...", "#ffaa00")
                try:
                    audio_data = np.frombuffer(b"".join(pending_frames), dtype=np.int16)
                    text = self.transcribe_audio(audio_data, self.sample_rate)
                    if text:
                        self.current_text += text + " "
                        self.signals.transcript.emit(text)
                except Exception as e:
                    self.signals.transcript.emit(f"[Error: {e}]")
            if self.audio_stream:
                with contextlib.suppress(Exception):
                    self.audio_stream.stop_stream()
                with contextlib.suppress(Exception):
                    self.audio_stream.close()
                self.audio_stream = None
            self.update_status("Ready", "black")
            self.signals.stop_dictation.emit()
        except Exception as e:
            self.signals.error.emit(f"Recognition error: {e}")
            self.signals.stop_dictation.emit()

    def closeEvent(self, event):
        self.shutdown_event.set()
        self.is_listening = False
        self.tts_playing = False
        self.save_config()
        if self.audio_stream:
            with contextlib.suppress(Exception):
                self.audio_stream.stop_stream()
            with contextlib.suppress(Exception):
                self.audio_stream.close()
            self.audio_stream = None
        if self.audio:
            with contextlib.suppress(Exception):
                self.audio.terminate()
        if self.tts_available:
            with contextlib.suppress(Exception):
                pygame.mixer.music.stop()
            with contextlib.suppress(Exception):
                pygame.mixer.quit()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Voice2Text-AI")
    app.setApplicationDisplayName("Voice 2 Text")
    app.setFont(QFont("Noto Sans", 11))
    window = VoiceAppQt()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
