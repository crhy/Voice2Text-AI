from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from voice2text.hardware import (
    detect_gpu_vram_gb,
    detect_system_ram_gb,
    suggest_models,
)


def test_suggest_models_picks_largest_that_fits() -> None:
    names = [model.name for model in suggest_models(6.0)]
    assert names[0] == "llama3.2:3b"


def test_suggest_models_falls_back_to_smallest_when_nothing_fits() -> None:
    suggestions = suggest_models(0.1)
    assert [model.name for model in suggestions] == ["qwen2.5:0.5b"]


def test_suggest_models_respects_limit() -> None:
    assert len(suggest_models(100.0, limit=2)) == 2


def test_detect_gpu_vram_gb_parses_nvidia_smi_output() -> None:
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="24576\n", stderr="")
    with patch("subprocess.run", return_value=fake):
        assert detect_gpu_vram_gb() == 24.0


def test_detect_gpu_vram_gb_returns_none_when_no_tool_is_available() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert detect_gpu_vram_gb() is None


def test_detect_system_ram_gb_parses_meminfo(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16777216 kB\nMemFree:        1000 kB\n", encoding="utf-8")
    assert detect_system_ram_gb(meminfo) == 16.0
