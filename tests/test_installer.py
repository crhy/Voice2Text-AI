from __future__ import annotations

import io
import threading
from unittest.mock import patch

import pytest

from voxa import installer


class FakeStdin:
    def __init__(self) -> None:
        self.written = ""
        self.closed = False

    def write(self, data: str) -> None:
        self.written += data

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self, output: str, returncode: int = 0) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(output)
        self._returncode = returncode
        self.terminated = False

    def wait(self) -> int:
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True


def test_privileged_shell_command_wraps_with_flatpak_spawn_when_sandboxed() -> None:
    with patch.object(installer, "is_flatpak", return_value=True):
        assert installer._privileged_shell_command() == ["flatpak-spawn", "--host", "pkexec", "sh", "-s"]


def test_privileged_shell_command_runs_directly_outside_flatpak() -> None:
    with patch.object(installer, "is_flatpak", return_value=False):
        assert installer._privileged_shell_command() == ["pkexec", "sh", "-s"]


def test_feed_stdin_writes_script_and_closes() -> None:
    class Process:
        pass

    process = Process()
    process.stdin = FakeStdin()

    installer._feed_stdin(process, "echo hi")

    assert process.stdin.written == "echo hi"
    assert process.stdin.closed is True


def test_install_ollama_streams_output_and_returns_exit_code() -> None:
    fake_process = FakeProcess("Downloading...\nInstalling...\n", returncode=0)
    lines: list[str] = []
    with (
        patch.object(installer, "_fetch_install_script", return_value="echo hi"),
        patch("subprocess.Popen", return_value=fake_process),
    ):
        exit_code = installer.install_ollama(on_output=lines.append, cancel_event=threading.Event())
    assert exit_code == 0
    assert lines == ["Downloading...", "Installing..."]


def test_install_ollama_raises_a_clear_error_when_pkexec_is_missing() -> None:
    with (
        patch.object(installer, "_fetch_install_script", return_value="echo hi"),
        patch("subprocess.Popen", side_effect=FileNotFoundError),
        pytest.raises(installer.InstallerError),
    ):
        installer.install_ollama(on_output=lambda _line: None, cancel_event=threading.Event())


def test_install_ollama_stops_reading_and_terminates_when_cancelled() -> None:
    fake_process = FakeProcess("line one\nline two\nline three\n", returncode=-15)
    cancel_event = threading.Event()
    lines: list[str] = []

    def capture(line: str) -> None:
        lines.append(line)
        if line == "line one":
            cancel_event.set()

    with (
        patch.object(installer, "_fetch_install_script", return_value="echo hi"),
        patch("subprocess.Popen", return_value=fake_process),
    ):
        installer.install_ollama(on_output=capture, cancel_event=cancel_event)

    assert lines == ["line one"]
    assert fake_process.terminated is True
