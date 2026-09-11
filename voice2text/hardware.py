from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class ModelSuggestion:
    name: str
    approx_gb: float
    description: str
    # Curated entries are the deliberately chosen ladder in MODEL_CATALOG;
    # catalog.discover_catalog() marks anything it finds in the library as
    # False so a discovered model cannot quietly displace a chosen one.
    curated: bool = True


# Approximate default-quantization weight size for each tag, in GB, smallest
# first. Used only to rank suggestions, not as an exact figure, and kept at or
# just above the real size so the headroom rule below stays conservative.
# catalog.refresh_catalog() re-measures these against Ollama's registry.
MODEL_CATALOG: tuple[ModelSuggestion, ...] = (
    ModelSuggestion("qwen2.5:0.5b", 0.4, "Fastest, runs on almost anything"),
    ModelSuggestion("gemma3:1b", 0.8, "Very small, still writes coherently"),
    ModelSuggestion("qwen2.5:1.5b", 1.0, "Very fast, good for low-memory devices"),
    ModelSuggestion("llama3.2:3b", 2.0, "Good balance for laptops without a GPU"),
    ModelSuggestion("qwen3.5:2b", 2.6, "Newer generation, small enough for a laptop"),
    ModelSuggestion("qwen3.5:4b", 3.2, "Newer generation, good on modest hardware"),
    ModelSuggestion("qwen2.5:7b", 4.7, "Strong general-purpose model"),
    ModelSuggestion("llama3.1:8b", 4.9, "Strong general-purpose model"),
    ModelSuggestion("qwen3.5:9b", 6.2, "Newer generation, strong for its size"),
    ModelSuggestion("qwen2.5:14b", 9.0, "Noticeably smarter, wants a mid-range GPU"),
    ModelSuggestion("qwen3.8:27b", 15.7, "Latest generation, wants a 24GB GPU"),
    ModelSuggestion("qwen2.5:32b", 20.0, "High quality, wants a 24GB+ GPU"),
    ModelSuggestion("qwen3.5:35b", 22.3, "Newer generation at the top end, wants 30GB+"),
    ModelSuggestion("llama3.1:70b", 40.0, "Top quality, wants multiple GPUs or a lot of unified memory"),
)

# Extra headroom beyond raw model weights for KV cache, activations, and the
# rest of the OS/desktop, so a suggestion isn't a model that merely fits on
# disk but chokes the moment inference starts.
# How far below the largest model that fits a curated one may be and still
# be preferred to it.
CURATED_PREFERENCE_BAND = 0.10

_HEADROOM_FACTOR = 1.3
_HEADROOM_FLOOR_GB = 1.0


def _fits(available_gb: float, model: ModelSuggestion) -> bool:
    return available_gb >= model.approx_gb * _HEADROOM_FACTOR + _HEADROOM_FLOOR_GB


def suggest_models(
    available_gb: float,
    *,
    limit: int = 3,
    catalog: tuple[ModelSuggestion, ...] | None = None,
) -> list[ModelSuggestion]:
    """Return up to ``limit`` catalog models that fit in ``available_gb``, best first."""
    entries = catalog or MODEL_CATALOG
    fitting = [model for model in entries if _fits(available_gb, model)]
    if not fitting:
        return [entries[0]]
    ranked = sorted(fitting, key=lambda model: model.approx_gb, reverse=True)
    best = ranked[0]
    if not best.curated:
        # Size stands in for quality, which is fine for a handful of chosen
        # models and poor once discovery fills the list with near-identical
        # ones: a tenth of a gigabyte should not decide that an automatically
        # found model beats a deliberately chosen one. A curated model within
        # a short reach of the largest that fits takes the top slot instead.
        reach = best.approx_gb * (1 - CURATED_PREFERENCE_BAND)
        preferred = next(
            (model for model in ranked if model.curated and model.approx_gb >= reach), None
        )
        if preferred is not None:
            ranked.remove(preferred)
            ranked.insert(0, preferred)
    return ranked[:limit]


def detect_gpu_vram_gb(sysfs_base: Path = Path("/sys/class/drm")) -> float | None:
    """Best-effort total VRAM in GB for the most capable GPU, or None if undetectable."""
    for probe in (_nvidia_vram_gb, _rocm_vram_gb):
        vram = probe()
        if vram is not None:
            return vram
    return _sysfs_amdgpu_vram_gb(sysfs_base)


def _nvidia_vram_gb() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not values:
        return None
    return max(values) / 1024.0


def _rocm_vram_gb() -> float | None:
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    totals = [int(value) for value in re.findall(r'"VRAM Total Memory \(B\)":\s*"?(\d+)"?', result.stdout)]
    if not totals:
        return None
    return max(totals) / (1024.0**3)


def _sysfs_amdgpu_vram_gb(base: Path = Path("/sys/class/drm")) -> float | None:
    totals: list[int] = []
    for path in base.glob("card*/device/mem_info_vram_total"):
        try:
            totals.append(int(path.read_text().strip()))
        except (OSError, ValueError):
            continue
    if not totals:
        return None
    return max(totals) / (1024.0**3)


def detect_system_ram_gb(meminfo_path: Path = Path("/proc/meminfo")) -> float | None:
    try:
        with meminfo_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    kib = int(line.split()[1])
                    return kib / (1024.0**2)
    except (OSError, ValueError, IndexError):
        return None
    return None


@dataclass(slots=True, frozen=True)
class GpuUsage:
    utilization_percent: float
    memory_used_gb: float
    memory_total_gb: float


def sample_gpu_usage() -> GpuUsage | None:
    """One-shot GPU utilization + VRAM sample, or None if no GPU tool is available.

    Cheap enough to poll every second or so from a background thread — each
    call is a single subprocess invocation, not a persistent connection.
    """
    for probe in (_nvidia_gpu_usage, _rocm_gpu_usage):
        usage = probe()
        if usage is not None:
            return usage
    return None


def _nvidia_gpu_usage() -> GpuUsage | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = next((line for line in result.stdout.splitlines() if line.strip()), "")
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        return None
    try:
        util, used_mib, total_mib = (float(part) for part in parts)
    except ValueError:
        return None
    return GpuUsage(utilization_percent=util, memory_used_gb=used_mib / 1024.0, memory_total_gb=total_mib / 1024.0)


def _rocm_gpu_usage() -> GpuUsage | None:
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    card = next(iter(payload.values()), None) if isinstance(payload, dict) else None
    if not isinstance(card, dict):
        return None
    try:
        util = float(card["GPU use (%)"])
        used_bytes = float(card["VRAM Total Used Memory (B)"])
        total_bytes = float(card["VRAM Total Memory (B)"])
    except (KeyError, TypeError, ValueError):
        return None
    return GpuUsage(
        utilization_percent=util,
        memory_used_gb=used_bytes / (1024.0**3),
        memory_total_gb=total_bytes / (1024.0**3),
    )


def detect_available_model_memory_gb() -> tuple[float, str]:
    """The best figure available to size a model against, and where it came from."""
    vram = detect_gpu_vram_gb()
    if vram is not None:
        return vram, "GPU VRAM"
    ram = detect_system_ram_gb()
    if ram is not None:
        return ram, "system RAM — no GPU detected"
    return 4.0, "an undetected machine (conservative default)"
