from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import voice2text.hardware as hardware
from voice2text.hardware import (
    _nvidia_smi_command,
    context_tokens,
    hardware_limit,
    detect_gpu_vram_gb,
    detect_system_ram_gb,
    sample_gpu_usage,
    suggest_models,
)


def test_suggest_models_picks_largest_that_fits() -> None:
    # 6GB clears qwen3.5:4b at 3.2 * 1.3 + 1 = 5.2 but not qwen2.5:7b at 7.1.
    names = [model.name for model in suggest_models(6.0)]
    assert names[0] == "qwen3.5:4b"


def test_suggest_models_falls_back_to_smallest_when_nothing_fits() -> None:
    suggestions = suggest_models(0.1)
    assert [model.name for model in suggestions] == ["qwen2.5:0.5b"]


def test_suggest_models_respects_limit() -> None:
    assert len(suggest_models(100.0, limit=2)) == 2


def test_detect_gpu_vram_gb_parses_nvidia_smi_output() -> None:
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="24576\n", stderr="")
    with patch("subprocess.run", return_value=fake):
        assert detect_gpu_vram_gb() == 24.0


def test_detect_gpu_vram_gb_returns_none_when_no_tool_is_available(tmp_path: Path) -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert detect_gpu_vram_gb(sysfs_base=tmp_path) is None


def test_nvidia_smi_command_prefers_bundled_copy(tmp_path: Path) -> None:
    bundled = tmp_path / "nvml" / "nvidia-smi"
    bundled.parent.mkdir()
    bundled.touch()
    with patch.object(hardware, "_BUNDLED_NVIDIA_SMI", bundled):
        assert _nvidia_smi_command() == str(bundled)


def test_nvidia_smi_command_falls_back_to_path_lookup() -> None:
    missing = Path("/nonexistent/nvml/nvidia-smi")
    with patch.object(hardware, "_BUNDLED_NVIDIA_SMI", missing):
        assert _nvidia_smi_command() == "nvidia-smi"


def test_detect_gpu_vram_gb_invokes_bundled_nvidia_smi(tmp_path: Path) -> None:
    bundled = tmp_path / "nvml" / "nvidia-smi"
    bundled.parent.mkdir()
    bundled.touch()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="16384\n", stderr="")
    with (
        patch.object(hardware, "_BUNDLED_NVIDIA_SMI", bundled),
        patch("subprocess.run", return_value=fake) as run,
    ):
        assert detect_gpu_vram_gb(sysfs_base=tmp_path) == 16.0
    assert run.call_args.args[0][0] == str(bundled)


def test_sample_gpu_usage_invokes_bundled_nvidia_smi(tmp_path: Path) -> None:
    bundled = tmp_path / "nvml" / "nvidia-smi"
    bundled.parent.mkdir()
    bundled.touch()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="99, 15100, 16384\n", stderr="")
    with (
        patch.object(hardware, "_BUNDLED_NVIDIA_SMI", bundled),
        patch("subprocess.run", return_value=fake) as run,
    ):
        usage = sample_gpu_usage()
    assert usage is not None
    assert usage.utilization_percent == 99.0
    assert run.call_args.args[0][0] == str(bundled)


def test_detect_system_ram_gb_parses_meminfo(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16777216 kB\nMemFree:        1000 kB\n", encoding="utf-8")
    assert detect_system_ram_gb(meminfo) == 16.0


def test_sample_gpu_usage_parses_nvidia_smi_csv_line() -> None:
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="37, 4096, 24576\n", stderr="")
    with patch("subprocess.run", return_value=fake):
        usage = sample_gpu_usage()
    assert usage is not None
    assert usage.utilization_percent == 37.0
    assert usage.memory_used_gb == 4.0
    assert usage.memory_total_gb == 24.0


def test_sample_gpu_usage_returns_none_when_no_tool_is_available() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert sample_gpu_usage() is None


def test_sample_gpu_usage_returns_none_on_malformed_output() -> None:
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="not,a,number\n", stderr="")
    with patch("subprocess.run", return_value=fake):
        assert sample_gpu_usage() is None


def test_context_tokens_walks_the_ladder_down_to_the_smallest_rung() -> None:
    # The same memory figure suggest_models sizes against: as the weights eat
    # the machine, the context the app asks for walks down the ladder instead
    # of leaving Ollama to pay for its own default context.
    ladder = [context_tokens(24.0, weights) for weights in (0.0, 20.0, 21.0, 22.5, 23.5, 23.75)]
    assert ladder == [8192, 8192, 4096, 2048, 1024, 512]


def test_hardware_limit_uses_the_same_headroom_rule_as_suggestions() -> None:
    # qwen2.5:7b needs 4.7 * 1.3 + 1 = 7.1 GB: fine at 8 GB, over the line at 7.
    assert hardware_limit(8.0, "GPU VRAM", "qwen2.5:7b", 4.7) is None
    message = hardware_limit(7.0, "GPU VRAM", "qwen2.5:7b", 4.7)
    assert message is not None
    assert "qwen2.5:7b" in message and "7.0 GB" in message and "GPU VRAM" in message
    # With no suggestions the catalog's smallest model is the one to name.
    assert "qwen2.5:0.5b" in message
