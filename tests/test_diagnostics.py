from __future__ import annotations

import io
import json
import threading
from unittest.mock import patch

import pytest

from voice2text.diagnostics import (
    BenchmarkResult,
    benchmark,
    offload_verdict,
    speed_sentence,
    time_to_first_token_ms,
    tokens_per_second,
)
from voice2text.hardware import GpuUsage
from voice2text.ollama import GenerationStats, OllamaClient, OllamaError


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class GatedResponse:
    """Fake NDJSON body that holds the stream open until the poller reports in.

    Without the gate a fake transport finishes before the poller gets a second
    reading, and the sample list would say nothing about the answer's lifetime.
    """

    def __init__(self, lines: list[bytes], gate: threading.Event) -> None:
        self._lines = iter(lines)
        self._gate = gate

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        for index, line in enumerate(self._lines):
            if index == 1:
                self._gate.wait(5.0)
            yield line


def _stats(
    eval_count: int = 128,
    eval_duration_ns: int = 6_400_000_000,
    prompt_eval_duration_ns: int = 412_000_000,
) -> GenerationStats:
    return GenerationStats(
        eval_count=eval_count,
        eval_duration_ns=eval_duration_ns,
        prompt_eval_count=7,
        prompt_eval_duration_ns=prompt_eval_duration_ns,
        total_duration_ns=7_100_000_000,
    )


def test_tokens_per_second_divides_the_servers_own_token_count_by_its_duration() -> None:
    # 128 tokens over 6.4 s.
    assert tokens_per_second(_stats()) == pytest.approx(20.0)


def test_time_to_first_token_ms_reads_the_prompt_duration() -> None:
    assert time_to_first_token_ms(_stats()) == pytest.approx(412.0)


def test_speed_sentence_words_the_measured_numbers() -> None:
    assert speed_sentence(_stats()) == "about 20.0 tokens per second, first token after 412 ms"


def test_speed_sentence_stays_honest_when_nothing_was_measured() -> None:
    honest = "this server reported no timing for that answer"
    assert speed_sentence(None) == honest
    # A done event that carried a zero duration must not become a division,
    # an infinity, or an invented speed.
    zero = _stats(eval_duration_ns=0, prompt_eval_duration_ns=0)
    assert speed_sentence(zero) == honest
    assert tokens_per_second(zero) is None
    assert time_to_first_token_ms(zero) is None
    for sentence in (speed_sentence(None), speed_sentence(zero)):
        assert not any(char.isdigit() for char in sentence)


def test_speed_sentence_keeps_talking_when_only_the_generation_rate_arrives() -> None:
    sentence = speed_sentence(_stats(prompt_eval_duration_ns=0))
    assert sentence == "about 20.0 tokens per second"


def test_offload_verdict_says_the_gpu_was_busy() -> None:
    samples = [
        GpuUsage(utilization_percent=0.0, memory_used_gb=0.0, memory_total_gb=8.0),
        GpuUsage(utilization_percent=97.0, memory_used_gb=6.5, memory_total_gb=8.0),
    ]
    verdict = offload_verdict(samples)
    assert "GPU was busy" in verdict
    assert "97% utilization" in verdict
    assert "6.5 of 8.0 GB VRAM" in verdict


def test_offload_verdict_blames_the_cpu_when_the_card_stayed_idle() -> None:
    samples = [
        GpuUsage(utilization_percent=0.0, memory_used_gb=0.0, memory_total_gb=8.0),
        GpuUsage(utilization_percent=0.0, memory_used_gb=0.0, memory_total_gb=8.0),
    ]
    assert (
        offload_verdict(samples)
        == "no GPU activity was seen while the model answered, so it ran on the CPU"
    )


def test_offload_verdict_says_no_gpu_was_reported_for_an_empty_sample_list() -> None:
    assert offload_verdict([]) == "no GPU was reported on this computer"


def test_benchmark_streams_an_answer_while_sampling_the_gpu() -> None:
    gate = threading.Event()
    response = GatedResponse(
        [
            b'{"response":"Hi","done":false}\n',
            b'{"response":" there","done":false}\n',
            b'{"done":true,"eval_count":128,"eval_duration":6400000000,'
            b'"prompt_eval_count":7,"prompt_eval_duration":412000000,"total_duration":7100000000}\n',
        ],
        gate,
    )
    readings = iter(
        [
            GpuUsage(utilization_percent=0.0, memory_used_gb=0.0, memory_total_gb=8.0),
            GpuUsage(utilization_percent=97.0, memory_used_gb=6.5, memory_total_gb=8.0),
        ]
    )

    def sampler() -> GpuUsage | None:
        usage = next(readings, None)
        if usage is None:
            # Past the scripted readings: keep reporting the busy card, and
            # let the stream resume so the test cannot hang.
            gate.set()
            return GpuUsage(utilization_percent=97.0, memory_used_gb=6.5, memory_total_gb=8.0)
        if usage.utilization_percent > 0:
            gate.set()
        return usage

    with patch("urllib.request.urlopen", return_value=response) as mocked:
        result = benchmark(
            OllamaClient(),
            "qwen2.5:0.5b",
            "say hi",
            sampler=sampler,
            poll_seconds=0.01,
        )

    assert isinstance(result, BenchmarkResult)
    assert result.tokens_per_second == pytest.approx(20.0)
    assert result.first_token_ms == pytest.approx(412.0)
    assert result.peak_utilization_percent == 97.0
    assert result.memory_used_gb == 6.5
    assert "GPU was busy" in result.verdict
    body = json.loads(mocked.call_args[0][0].data)
    assert body["model"] == "qwen2.5:0.5b"
    assert body["options"] == {"num_predict": 64}


def test_benchmark_reports_what_it_could_not_measure() -> None:
    # A server that sends no timing block, on a machine with no GPU tool:
    # every number stays None and the verdict says so plainly.
    response = FakeResponse(b'{"response":"Hi","done":false}\n{"done":true}\n')
    with patch("urllib.request.urlopen", return_value=response):
        result = benchmark(
            OllamaClient(),
            "qwen2.5:0.5b",
            "say hi",
            sampler=lambda: None,
            poll_seconds=0.01,
        )
    assert result.tokens_per_second is None
    assert result.first_token_ms is None
    assert result.peak_utilization_percent == 0.0
    assert result.verdict == "no GPU was reported on this computer"
    assert speed_sentence(None) == "this server reported no timing for that answer"


def test_benchmark_propagates_a_server_error_instead_of_reporting_zero_speed() -> None:
    response = FakeResponse(b'{"error":"model not found"}\n')
    with patch("urllib.request.urlopen", return_value=response), pytest.raises(OllamaError):
        benchmark(
            OllamaClient(),
            "does-not-exist",
            "say hi",
            sampler=lambda: None,
            poll_seconds=0.01,
        )
