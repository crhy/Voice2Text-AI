"""Speed and GPU-offload reporting for one Ollama answer.

Two questions a user can act on, answered only from what the machine actually
reported: how fast the model wrote, and whether the GPU did any of the work.
Nothing here estimates a number the server or the GPU tools did not give, so a
silent server stays "no timing", not a plausible-looking speed.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from .hardware import GpuUsage, sample_gpu_usage
from .ollama import GenerationStats, OllamaClient

# nvidia-smi and rocm-smi both read a busy GPU well above this; anything at or
# under it means the card sat idle while the model was answering.
_BUSY_UTILIZATION_PERCENT = 1.0

_NO_TIMING = "this server reported no timing for that answer"
_NO_GPU = "no GPU was reported on this computer"
_NO_GPU_ACTIVITY = "no GPU activity was seen while the model answered, so it ran on the CPU"


@dataclass(slots=True, frozen=True)
class BenchmarkResult:
    """One measured answer: the server's speed numbers plus the GPU's own readings."""

    tokens_per_second: float | None
    first_token_ms: float | None
    peak_utilization_percent: float
    memory_used_gb: float
    verdict: str


def tokens_per_second(stats: GenerationStats | None) -> float | None:
    """Generation rate from the server's own token count and duration."""
    if stats is None or stats.eval_count <= 0 or stats.eval_duration_ns <= 0:
        return None
    return stats.eval_count * 1e9 / stats.eval_duration_ns


def time_to_first_token_ms(stats: GenerationStats | None) -> float | None:
    """How long the model spent reading the prompt before it started writing."""
    if stats is None or stats.prompt_eval_duration_ns <= 0:
        return None
    return stats.prompt_eval_duration_ns / 1e6


def speed_sentence(stats: GenerationStats | None) -> str:
    """The speed numbers in one honest sentence, for the status strip or a log."""
    if stats is None:
        return _NO_TIMING
    rate = tokens_per_second(stats)
    if rate is None:
        return _NO_TIMING
    sentence = f"about {rate:.1f} tokens per second"
    first_token_ms = time_to_first_token_ms(stats)
    if first_token_ms is not None:
        sentence += f", first token after {first_token_ms:.0f} ms"
    return sentence


def offload_verdict(samples: list[GpuUsage]) -> str:
    """Did the GPU take the work, in words a user can act on?

    ``samples`` are the GPU readings taken while the answer was streaming; an
    empty list means the sampler found no GPU tool at all on this machine.
    """
    if not samples:
        return _NO_GPU
    peak = max(sample.utilization_percent for sample in samples)
    if peak <= _BUSY_UTILIZATION_PERCENT:
        return _NO_GPU_ACTIVITY
    busiest = max(samples, key=lambda sample: sample.utilization_percent)
    return (
        f"the GPU was busy while the model answered — peak {peak:.0f}% utilization, "
        f"{busiest.memory_used_gb:.1f} of {busiest.memory_total_gb:.1f} GB VRAM in use"
    )


def benchmark(
    client: OllamaClient,
    model: str,
    prompt: str,
    *,
    sampler: Callable[[], GpuUsage | None] = sample_gpu_usage,
    poll_seconds: float = 1.0,
    cancel_event: threading.Event | None = None,
    max_tokens: int = 64,
) -> BenchmarkResult:
    """Stream one short answer and sample the GPU while it streams.

    The answer runs on its own thread so this thread can poll ``sampler`` at
    ``poll_seconds`` intervals for exactly the window the model was working —
    the reading that answers "is it really using the GPU, or quietly falling
    back to the CPU?"
    """
    cancel = cancel_event if cancel_event is not None else threading.Event()
    stats: list[GenerationStats] = []
    samples: list[GpuUsage] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            client.generate_stream(
                model=model,
                prompt=prompt,
                cancel_event=cancel,
                on_chunk=lambda _chunk: None,
                num_predict=max_tokens,
                on_stats=stats.append,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised on this thread
            failure.append(exc)

    thread = threading.Thread(target=run, name="ollama-benchmark", daemon=True)
    thread.start()
    while True:
        # Always take at least one reading, so an answer that streams faster
        # than the poll interval still gets a GPU verdict instead of an empty
        # sample list that would read as "no GPU on this computer".
        usage = sampler()
        if usage is not None:
            samples.append(usage)
        if cancel.is_set() or not thread.is_alive():
            break
        thread.join(poll_seconds)
    thread.join()

    if failure:
        raise failure[0]

    final = stats[0] if stats else None
    peak = max((sample.utilization_percent for sample in samples), default=0.0)
    memory = max((sample.memory_used_gb for sample in samples), default=0.0)
    return BenchmarkResult(
        tokens_per_second=tokens_per_second(final),
        first_token_ms=time_to_first_token_ms(final),
        peak_utilization_percent=peak,
        memory_used_gb=memory,
        verdict=offload_verdict(samples),
    )
