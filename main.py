#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk
import tkinter.scrolledtext as scrolledtext
import speech_recognition as sr
import requests
import threading
import json
import pyaudio
import re
import os
import time
from gtts import gTTS
import pygame
import audioop
from kokoro_tts import KokoroTTS, KOKORO_VOICES

class DictationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dictation to Ollama")
        self.root.geometry("700x600")
        self.root.configure(bg='#333333')

        # Config
        self.config_file = 'config.json'
        self.config = self.load_config()

        # Audio devices
        self.input_devices = self.get_audio_devices('input')
        self.output_devices = self.get_audio_devices('output')
        self.selected_input = tk.StringVar(value=self.config.get('input_device', ''))
        self.selected_output = tk.StringVar(value=self.config.get('output_device', ''))

        # Ollama models
        self.models = self.get_ollama_models()
        self.selected_model = tk.StringVar(value=self.config.get('model', ''))

        # Status and GPU
        self.status_var = tk.StringVar(value="Ready")
        self.gpu_var = tk.StringVar(value="GPU: Checking...")

        # Recognition method
        self.recognizer_method = tk.StringVar(value=self.config.get('recognizer', 'Google'))

        # TTS engine
        self.tts_engine = tk.StringVar(value=self.config.get('tts_engine', 'gTTS'))

        # Kokoro voices
        self.kokoro_voices = KOKORO_VOICES
        self.selected_voice = tk.StringVar(value=self.config.get('voice', 'af_heart'))



        # GUI elements
        self.create_widgets()

        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.5  # Allow longer pauses
        self.is_listening = False

        # TTS
        self.current_channel = None
        self.tts_paused = False
        pygame.mixer.init()

        # VU
        self.monitor_stream = None
        self.pyaudio_instance = pyaudio.PyAudio()

        # Bind save on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)



    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_config(self):
        config = {
            'input_device': self.selected_input.get(),
            'output_device': self.selected_output.get(),
            'model': self.selected_model.get(),
            'recognizer': self.recognizer_method.get(),
            'tts_engine': self.tts_engine.get(),
            'voice': self.selected_voice.get()
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
        except:
            pass

    def on_close(self):
        self.save_config()
        # if self.monitor_stream:
        #     self.monitor_stream.close()
        self.pyaudio_instance.terminate()
        self.root.destroy()

    def update_vu(self):
        while self.is_listening and self.monitor_stream:
            try:
                data = self.monitor_stream.read(1024, exception_on_overflow=False)
                rms = audioop.rms(data, 2)
                # level = min(100, rms / 50)  # calibrate
                # self.root.after(0, self.draw_vu, level)
                time.sleep(0.05)
            except:
                break
        # self.root.after(0, self.draw_vu, 0)



    def create_widgets(self):
        # Styles
        style = ttk.Style()
        style.configure('Large.TButton', font=('Helvetica', 12), padding=10, background='#333333')
        style.configure('Desc.TLabel', font=('Helvetica', 10, 'bold'), foreground='blue')

        # Configure grid
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(4, weight=1)

        # Audio devices on left
        tk.Label(self.root, text="Input Device (Microphone):", bg='#333333', fg='white', font=('Helvetica', 10, 'bold'), relief='flat').grid(row=0, column=0, pady=5, padx=10, sticky='w')
        self.input_combo = ttk.Combobox(self.root, textvariable=self.selected_input, values=self.input_devices)
        self.input_combo.grid(row=0, column=1, pady=5, padx=10, sticky='ew')
        if self.selected_input.get() not in self.input_devices and self.input_devices:
            self.selected_input.set(self.input_devices[0])

        tk.Label(self.root, text="Output Device (Speakers):", bg='#333333', fg='white', font=('Helvetica', 10, 'bold'), relief='flat').grid(row=1, column=0, pady=5, padx=10, sticky='w')
        self.output_combo = ttk.Combobox(self.root, textvariable=self.selected_output, values=self.output_devices)
        self.output_combo.grid(row=1, column=1, pady=5, padx=10, sticky='ew')
        if self.selected_output.get() not in self.output_devices and self.output_devices:
            self.selected_output.set(self.output_devices[0])

        # Recognizer
        tk.Label(self.root, text="Speech Recognizer:", bg='#333333', fg='white', font=('Helvetica', 10, 'bold'), relief='flat').grid(row=2, column=0, pady=5, padx=10, sticky='w')
        self.recognizer_combo = ttk.Combobox(self.root, textvariable=self.recognizer_method, values=['Google', 'Sphinx'])
        self.recognizer_combo.grid(row=2, column=1, pady=5, padx=10, sticky='ew')

        # TTS Engine
        tk.Label(self.root, text="TTS Engine:", bg='#333333', fg='white', font=('Helvetica', 10, 'bold'), relief='flat').grid(row=3, column=0, pady=5, padx=10, sticky='w')
        self.tts_combo = ttk.Combobox(self.root, textvariable=self.tts_engine, values=['gTTS', 'kokoro'])
        self.tts_combo.grid(row=3, column=1, pady=5, padx=10, sticky='ew')

        # Kokoro Voice
        tk.Label(self.root, text="Kokoro Voice:", bg='#333333', fg='white', font=('Helvetica', 10, 'bold'), relief='flat').grid(row=3, column=2, pady=5, padx=10, sticky='w')
        self.voice_combo = ttk.Combobox(self.root, textvariable=self.selected_voice, values=self.kokoro_voices)
        self.voice_combo.grid(row=3, column=3, pady=5, padx=10, sticky='ew')

        # Model on right
        tk.Label(self.root, text="Ollama Model:", bg='#333333', fg='white', font=('Helvetica', 10, 'bold'), relief='flat').grid(row=0, column=2, pady=5, padx=10, sticky='w')
        self.model_combo = ttk.Combobox(self.root, textvariable=self.selected_model, values=self.models)
        self.model_combo.grid(row=0, column=3, pady=5, padx=10, sticky='ew')
        if self.selected_model.get() not in self.models and self.models:
            self.selected_model.set(self.models[0])

        # Text area
        self.text_area = scrolledtext.ScrolledText(self.root, height=15, width=60, bg='#333333', fg='white', insertbackground='white')
        self.text_area.grid(row=4, column=0, columnspan=4, pady=10, padx=10, sticky='nsew')

        # Status
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bg='#333333', fg='white')
        self.status_label.grid(row=6, column=0, columnspan=4, pady=5)

        # Buttons
        button_frame = tk.Frame(self.root, bg='#333333')
        button_frame.grid(row=7, column=0, columnspan=4, pady=10)

        # VU disabled
        # vu_frame = tk.Frame(button_frame, bg='#333333', relief='sunken', bd=2)
        # vu_frame.pack(side=tk.LEFT, padx=10)
        # self.vu_canvas = tk.Canvas(vu_frame, width=50, height=200, bg='black', highlightthickness=0)
        # self.vu_canvas.pack()

        self.start_button = tk.Button(button_frame, text="Start Dictation", command=self.start_dictation, font=('Helvetica', 12), padx=10, pady=5, bg='lightgray', fg='black')
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(button_frame, text="Stop Dictation", command=self.stop_dictation, state=tk.DISABLED, font=('Helvetica', 12), padx=10, pady=5, bg='lightgray', fg='black')
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.send_button = tk.Button(button_frame, text="Send to Ollama", command=self.send_current_text, font=('Helvetica', 12), padx=10, pady=5, bg='lightgray', fg='black')
        self.send_button.pack(side=tk.LEFT, padx=5)

        self.pause_tts_button = tk.Button(button_frame, text="Pause Speech", command=self.pause_tts, font=('Helvetica', 12), padx=10, pady=5, bg='lightgray', fg='black')
        self.pause_tts_button.pack(side=tk.LEFT, padx=5)



    def get_audio_devices(self, type):
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if type == 'input' and int(info.get('maxInputChannels', 0)) > 0:
                devices.append(f"{info.get('name')} (Index: {i})")
            elif type == 'output' and int(info.get('maxOutputChannels', 0)) > 0:
                devices.append(f"{info.get('name')} (Index: {i})")
        p.terminate()
        return devices

    def get_device_index(self, device_str):
        match = re.search(r'Index: (\d+)', device_str)
        return int(match.group(1)) if match else None

    def get_ollama_models(self):
        """Get available Ollama models with improved error handling."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except requests.exceptions.Timeout:
            self.status_var.set("Ollama connection timeout")
            return []
        except requests.exceptions.ConnectionError:
            self.status_var.set("Cannot connect to Ollama - start with 'ollama serve'")
            return []
        except (KeyError, ValueError):
            self.status_var.set("Invalid response from Ollama")
            return []
        except Exception as e:
            self.status_var.set(f"Ollama error: {str(e)[:50]}")
            return []

    def start_dictation(self):
        self.is_listening = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Listening")
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, "Listening...\n")

        # Start VU monitoring
        # Disabled due to PulseAudio conflict
        # device_index = self.get_device_index(self.selected_input.get())
        # if device_index is not None:
        #     self.monitor_stream = self.pyaudio_instance.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, input_device_index=device_index, frames_per_buffer=1024)
        #     threading.Thread(target=self.update_vu).start()

        threading.Thread(target=self.listen_and_process).start()

    def stop_dictation(self):
        self.is_listening = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        # if self.monitor_stream:
        #     self.monitor_stream.close()
        #     self.monitor_stream = None
        self.status_var.set("Ready")

    def listen_and_process(self):
        device_index = self.get_device_index(self.selected_input.get())
        if device_index is None:
            self.text_area.insert(tk.END, "Invalid input device selected\n")
            return
        try:
            with sr.Microphone(device_index=device_index) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                while self.is_listening:
                    try:
                        audio = self.recognizer.listen(source, timeout=10)
                        if self.recognizer_method.get() == 'Google':
                            text = self.recognizer.recognize_google(audio)
                        else:
                            text = self.recognizer.recognize_sphinx(audio)
                        self.text_area.insert(tk.END, f"You said: {text}\n")
                        threading.Thread(target=self.send_to_ollama, args=(text,)).start()
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        self.text_area.insert(tk.END, "Could not understand audio\n")
                    except sr.RequestError as e:
                        self.text_area.insert(tk.END, f"Error: {e}\n")
        except Exception as e:
            self.text_area.insert(tk.END, f"Error opening microphone: {e}\n")

    def send_current_text(self):
        text = self.text_area.get(1.0, tk.END).strip()
        if not text:
            self.text_area.insert(tk.END, "No text to send\n")
            return
        self.status_var.set("Sending")
        self.send_to_ollama(text)

    def pause_tts(self):
        if self.current_channel:
            if self.tts_paused:
                self.current_channel.unpause()
                self.tts_paused = False
                self.pause_tts_button.config(text="Pause TTS")
            else:
                self.current_channel.pause()
                self.tts_paused = True
                self.pause_tts_button.config(text="Resume TTS")

    def send_to_ollama(self, text):
        """Send text to Ollama with retry logic and better error handling."""
        model = self.selected_model.get()
        if not model:
            self.text_area.insert(tk.END, "No model selected\n")
            return

        if not text or not text.strip():
            self.text_area.insert(tk.END, "No text to send\n")
            return

        # Sanitize input
        text = text.strip()
        if len(text) > 10000:
            text = text[:10000] + "..."
            self.text_area.insert(tk.END, "Input truncated to 10,000 characters\n")

        self.text_area.insert(tk.END, f"Sending to {model}...\n")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.status_var.set(f"Processing (attempt {attempt + 1}/{max_retries})")
                payload = {
                    "model": model,
                    "prompt": text,
                    "stream": False
                }

                response = requests.post("http://localhost:11434/api/generate",
                                       json=payload, timeout=120)

                response.raise_for_status()
                data = response.json()
                reply = data.get('response', '').strip()

                if reply:
                    self.text_area.insert(tk.END, f"Ollama: {reply}\n")
                    self.speak_response(reply)
                else:
                    self.text_area.insert(tk.END, "Ollama returned empty response\n")

                self.status_var.set("Ready")
                return  # Success

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    self.text_area.insert(tk.END, f"Timeout, retrying... ({attempt + 1}/{max_retries})\n")
                    time.sleep(2)
                    continue
                else:
                    self.text_area.insert(tk.END, "Ollama timeout - model may be slow\n")

            except requests.exceptions.ConnectionError:
                self.text_area.insert(tk.END, "Cannot connect to Ollama - check if running\n")
                break

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else "unknown"
                self.text_area.insert(tk.END, f"Ollama HTTP error {status_code}: {str(e)[:50]}\n")
                break

            except (KeyError, ValueError) as e:
                self.text_area.insert(tk.END, f"Invalid response from Ollama: {str(e)[:50]}\n")
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    self.text_area.insert(tk.END, f"Error, retrying... ({attempt + 1}/{max_retries})\n")
                    time.sleep(1)
                    continue
                else:
                    self.text_area.insert(tk.END, f"Error connecting to Ollama: {str(e)[:50]}\n")

        self.status_var.set("Ready")

    def speak_response(self, text):
        # TODO: Use selected output device
        self.status_var.set("Speaking")
        # Preprocess text
        text = text.replace('*', 'star')
        threading.Thread(target=self._speak_response, args=(text,)).start()

    def _speak_response(self, text):
        engine = self.tts_engine.get()
        try:
            temp_file = None
            
            if engine == 'gTTS':
                tts = gTTS(text)
                temp_file = 'temp_tts.mp3'
                tts.save(temp_file)                
                
            elif engine == 'kokoro':
                # Initialize Kokoro TTS
                kokoro = KokoroTTS(lang_code=self.selected_voice.get()[0], speed=1.0)  # American English
                temp_file = 'temp_tts_kokoro.wav'
                kokoro.synthesize(text, voice=self.selected_voice.get(), output_path=temp_file)
            
            if temp_file:
                sound = pygame.mixer.Sound(temp_file)
                self.current_channel = sound.play()
                self.tts_paused = False
                self.pause_tts_button.config(text="Pause TTS")
                while self.current_channel and self.current_channel.get_busy():
                    time.sleep(0.1)
                os.remove(temp_file)
                
        except Exception as e:
            self.text_area.insert(tk.END, f"TTS error ({engine}): {e}\n")
        finally:
            self.current_channel = None
            self.status_var.set("Ready")

if __name__ == "__main__":
    root = tk.Tk()
    app = DictationApp(root)
    root.mainloop()