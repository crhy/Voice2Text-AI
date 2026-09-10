from __future__ import annotations

import contextlib
import subprocess
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

INSTALL_SCRIPT_URL = "https://ollama.com/install.sh"


class InstallerError(RuntimeError):
    pass


def is_flatpak() -> bool:
    return Path("/.flatpak-info").exists()


def _fetch_install_script(timeout: float = 15.0) -> str:
    request = urllib.request.Request(INSTALL_SCRIPT_URL, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise InstallerError(f"Could not download the Ollama install script: {exc}") from exc


def _privileged_shell_command() -> list[str]:
    # "-s" reads the script from stdin, so the elevated process only ever
    # sees bytes we already fetched ourselves (see install_ollama).
    base = ["pkexec", "sh", "-s"]
    if is_flatpak():
        # flatpak-spawn --host hands the command to the *host* pkexec, since
        # a sandboxed process cannot show a PolicyKit prompt or install onto
        # the host itself. Requires --talk-name=org.freedesktop.Flatpak.
        return ["flatpak-spawn", "--host", *base]
    return base


def _feed_stdin(process: subprocess.Popen, script: str) -> None:
    assert process.stdin is not None
    try:
        process.stdin.write(script)
    except (BrokenPipeError, OSError):
        pass
    finally:
        with contextlib.suppress(OSError):
            process.stdin.close()


def install_ollama(
    *,
    on_output: Callable[[str], None],
    cancel_event: threading.Event,
) -> int:
    """Fetch the official Ollama install script and run it as root via pkexec.

    Fetching happens from inside this (network-enabled) app process; only the
    already-downloaded script text is handed to the privileged process, so
    the elevated shell never reaches the network itself. The user sees a
    native PolicyKit password prompt from ``pkexec`` rather than a terminal.

    Requires ``pkexec`` on the host, and under Flatpak also requires
    ``--talk-name=org.freedesktop.Flatpak`` so ``flatpak-spawn --host`` can
    reach the host's ``pkexec``.
    """
    script = _fetch_install_script()
    command = _privileged_shell_command()
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise InstallerError(
            "Could not find pkexec. Install Ollama manually from a terminal: "
            "curl -fsSL https://ollama.com/install.sh | sh"
        ) from exc

    assert process.stdout is not None
    threading.Thread(target=_feed_stdin, args=(process, script), daemon=True).start()

    for line in process.stdout:
        if cancel_event.is_set():
            process.terminate()
            break
        stripped = line.rstrip("\n")
        if stripped:
            on_output(stripped)

    return process.wait()
