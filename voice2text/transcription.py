from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import os
import threading
from collections.abc import Callable
from typing import Any

import numpy as np


class TranscriptionError(RuntimeError):
    pass


def _cuda_compute_usable() -> bool:
    """True only when the CUDA *compute* stack (cuBLAS + cuDNN) can be loaded.

    The NVIDIA driver may be present (e.g. GDK/GL runtime) while the compute
    libraries are missing, which lets CTranslate2 report a CUDA device yet fail
    at inference time. Probe the compute libraries directly with dlopen because
    LD_LIBRARY_PATH-based lookups are not reflected in ldconfig's cache.
    """
    if os.getenv("VOICE2TEXT_FORCE_CUDA") == "1":
        return True
    for name in ("libcublas.so.12", "libcudart.so.12"):
        try:
            ctypes.CDLL(name)
        except OSError:
            return False
    return True


class WhisperService:
    """Thread-safe, lazily loaded Faster Whisper service."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._model_name = ""
        self._load_generation = 0

    @property
    def model_name(self) -> str:
        with self._lock:
            return self._model_name

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._model is not None

    def load_async(
        self,
        model_name: str,
        on_ready: Callable[[str, str], None],
        on_error: Callable[[str], None],
    ) -> None:
        with self._lock:
            self._load_generation += 1
            generation = self._load_generation
            self._model = None
            self._model_name = ""

        thread = threading.Thread(
            target=self._load,
            args=(model_name, generation, on_ready, on_error),
            name=f"whisper-load-{model_name}",
            daemon=True,
        )
        thread.start()

    def _load(
        self,
        model_name: str,
        generation: int,
        on_ready: Callable[[str, str], None],
        on_error: Callable[[str], None],
    ) -> None:
        try:
            from faster_whisper import WhisperModel

            available = 0
            with contextlib.suppress(OSError):
                available = len(os.sched_getaffinity(0))
            cpu_threads = max(1, min(8, available or os.cpu_count() or 4))
            requested = os.environ.get("VOICE2TEXT_DEVICE", "auto")
            attempts = []
            if requested == "auto":
                if _cuda_compute_usable():
                    attempts.append(("cuda", "default"))
                attempts.append(("cpu", "int8"))
            else:
                attempts = [(requested, "default"), ("cpu", "int8")]
            last_error: Exception | None = None
            model = None
            backend = ""
            for device, compute_type in attempts:
                try:
                    candidate = WhisperModel(
                        model_name,
                        device=device,
                        compute_type=compute_type,
                        cpu_threads=cpu_threads,
                    )
                    if device.startswith("cuda"):
                        self._probe_inference(candidate)
                    model = candidate
                    backend = f"{device}/{compute_type}"
                    break
                except Exception as exc:  # noqa: BLE001 - backend fallback is intentional
                    last_error = exc
            if model is None:
                raise TranscriptionError(str(last_error or "Unable to load Whisper model"))

            with self._lock:
                if generation != self._load_generation:
                    return
                self._model = model
                self._model_name = model_name
            on_ready(model_name, backend)
        except Exception as exc:  # noqa: BLE001 - worker boundary
            with self._lock:
                if generation != self._load_generation:
                    return
            on_error(str(exc))

    def _probe_inference(self, model: Any) -> None:
        """Run a tiny inference to confirm the device actually works.

        CTranslate2 can report a CUDA device during load and only fail once a
        compute kernel runs (e.g. missing cuBLAS), so verify before committing.
        """
        audio = np.zeros(1600, dtype=np.float32)
        for _segment in model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        ):
            pass

    def transcribe(self, pcm_s16le: bytes, language: str = "en") -> str:
        if not pcm_s16le:
            return ""
        with self._lock:
            model = self._model
        if model is None:
            raise TranscriptionError("The Whisper model has not finished loading.")

        audio = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0
        segments, _info = model.transcribe(
            audio,
            language=language or None,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 350},
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
