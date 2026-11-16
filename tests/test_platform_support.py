from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import voice2text.audio as audio
from voice2text.audio import AudioCapture
from voice2text.platform_support import (
    FLATPAK_HOST_RELEASE,
    OS_RELEASE_PATHS,
    PARTIAL,
    SUPPORTED,
    UNSUPPORTED,
    AudioBackends,
    Distribution,
    audio_backends,
    classify,
    os_release_search_paths,
    read_os_release,
    support_sentence,
)


class FakeGstProperties(dict):
    """Stand-in for Gst.Structure, matching tests/test_audio.py's duck-typed bag."""

    def __init__(self, values: dict | None = None) -> None:
        super().__init__(values or {})

    def has_field(self, key: str) -> bool:
        return key in self

    def get_value(self, key: str):
        return self.get(key)


def write_release(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "os-release"
    path.write_text(body, encoding="utf-8")
    return path


def devuan(tmp_path: Path) -> Path:
    return write_release(
        tmp_path,
        "# /etc/os-release\n"
        'NAME="Devuan GNU+Linux"\n'
        'VERSION="5 (cyclone)"\n'
        "ID=devuan\n"
        'ID_LIKE="debian"\n'
        'VERSION_ID="5"\n'
        'PRETTY_NAME="Devuan GNU+Linux 5"\n',
    )


def debian(tmp_path: Path) -> Path:
    return write_release(
        tmp_path,
        'PRETTY_NAME="Debian 12 (bookworm)"\nNAME="Debian GNU/Linux"\nID=debian\nVERSION_ID="12"\n',
    )


def ubuntu(tmp_path: Path) -> Path:
    return write_release(
        tmp_path,
        'NAME="Ubuntu"\nID=ubuntu\nID_LIKE=debian\nPRETTY_NAME="Ubuntu 24.04 LTS"\nVERSION_ID="24.04"\n',
    )


def linux_mint(tmp_path: Path) -> Path:
    return write_release(
        tmp_path,
        'NAME="Linux Mint"\nID=linuxmint\nID_LIKE="ubuntu debian"\nPRETTY_NAME="Linux Mint 13"\nVERSION_ID="13"\n',
    )


def bare_host(tmp_path: Path) -> Path:
    """A non-DistroKit host: a release file exists, but it names no ID."""
    return write_release(tmp_path, 'NAME="Some Embedded Shell"\nVERSION="1.0"\nSUPPORT_URL="https://example.invalid"\n')


def executable(directory: Path, *names: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        tool = directory / name
        tool.write_text("#!/bin/sh\n", encoding="utf-8")
        tool.chmod(0o755)
    return str(directory)


def test_read_os_release_parses_the_first_readable_file(tmp_path: Path) -> None:
    missing = tmp_path / "etc" / "os-release"
    distribution = read_os_release((missing, debian(tmp_path)))
    assert distribution.id == "debian"
    assert distribution.version_id == "12"
    assert distribution.pretty_name == "Debian 12 (bookworm)"
    assert distribution.label == "Debian 12 (bookworm)"


def test_read_os_release_returns_an_empty_distribution_when_nothing_is_readable(tmp_path: Path) -> None:
    distribution = read_os_release((tmp_path / "nope", tmp_path / "also-missing"))
    assert distribution == Distribution()
    assert distribution.identified is False
    assert distribution.label == ""


def test_read_os_release_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    distribution = read_os_release((devuan(tmp_path),))
    assert distribution.id == "devuan"
    assert distribution.id_like == "debian"
    assert distribution.label == "Devuan GNU+Linux 5"


def test_read_os_release_treats_a_release_file_without_an_id_as_unidentified(tmp_path: Path) -> None:
    distribution = read_os_release((bare_host(tmp_path),))
    assert distribution.identified is False
    assert distribution.id == ""
    assert distribution.label == ""


def test_family_includes_id_and_id_like(tmp_path: Path) -> None:
    mint = read_os_release((linux_mint(tmp_path),))
    assert mint.family == frozenset({"linuxmint", "ubuntu", "debian"})
    devuan_release = read_os_release((devuan(tmp_path),))
    assert devuan_release.family == frozenset({"devuan", "debian"})


def test_os_release_search_paths_prefer_the_host_copy_inside_the_flatpak() -> None:
    assert os_release_search_paths(flatpak=True)[0] == FLATPAK_HOST_RELEASE
    assert os_release_search_paths(flatpak=False) == OS_RELEASE_PATHS


def test_audio_backends_finds_pipewire_and_pulseaudio_tools(tmp_path: Path) -> None:
    bin_dir = executable(tmp_path / "bin", "pwls", "pulseaudio")
    backends = audio_backends(path=bin_dir, runtime_dir=tmp_path, dev_sound=tmp_path / "dev-snd")
    assert backends.pipewire is True
    assert backends.pulseaudio is True
    assert backends.alsa is False
    assert backends.names == ("PipeWire", "PulseAudio")
    assert backends.session_backend is True


def test_audio_backends_counts_pactl_as_a_session_without_naming_pipewire(tmp_path: Path) -> None:
    bin_dir = executable(tmp_path / "bin", "pactl")
    backends = audio_backends(path=bin_dir, runtime_dir=tmp_path, dev_sound=tmp_path / "dev-snd")
    assert backends.pipewire is False
    assert backends.pulseaudio is True


def test_audio_backends_ignores_a_non_executable_file_of_the_same_name(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "pwls").write_text("not executable\n", encoding="utf-8")
    backends = audio_backends(path=str(bin_dir), runtime_dir=tmp_path, dev_sound=tmp_path / "dev-snd")
    assert backends.any_backend is False


def test_audio_backends_falls_back_to_the_pulse_socket_directory(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    (runtime / "pulse").mkdir(parents=True)
    backends = audio_backends(path="", runtime_dir=runtime, dev_sound=tmp_path / "dev-snd")
    assert backends.pulseaudio is True
    assert backends.session_backend is True


def test_audio_backends_reports_plain_alsa_from_the_device_directory(tmp_path: Path) -> None:
    dev = tmp_path / "dev" / "snd"
    dev.mkdir(parents=True)
    backends = audio_backends(path="", runtime_dir=tmp_path, dev_sound=dev)
    assert backends.names == ("ALSA",)
    assert backends.any_backend is True
    assert backends.session_backend is False


def test_audio_backends_reports_nothing_on_a_bare_host(tmp_path: Path) -> None:
    backends = audio_backends(path="", runtime_dir=tmp_path, dev_sound=tmp_path / "dev-snd")
    assert backends == AudioBackends()
    assert backends.any_backend is False
    assert backends.names == ()


def test_classify_supports_a_tested_distribution_with_a_session_backend(tmp_path: Path) -> None:
    report = classify(read_os_release((linux_mint(tmp_path),)), AudioBackends(pipewire=True))
    assert report.level == SUPPORTED
    assert "Linux Mint 13" in report.reason
    assert "PipeWire" in report.reason


def test_classify_keeps_an_unlisted_linux_desktop_at_partial(tmp_path: Path) -> None:
    arch = write_release(tmp_path, "ID=arch\nPRETTY_NAME=Arch\nNAME=Arch\n")
    report = classify(read_os_release((arch,)), AudioBackends(pulseaudio=True))
    assert report.level == PARTIAL
    assert "not in the tested matrix" in report.reason


def test_classify_keeps_an_alsa_only_tested_distribution_at_partial(tmp_path: Path) -> None:
    report = classify(read_os_release((devuan(tmp_path),)), AudioBackends(alsa=True))
    assert report.level == PARTIAL
    assert "expects PipeWire or PulseAudio" in report.reason


def test_classify_supports_a_debian_derived_host_through_id_like(tmp_path: Path) -> None:
    derived = write_release(
        tmp_path,
        'ID=spaced\nID_LIKE=debian\nPRETTY_NAME="Spaced Linux 11.26.1"\nVERSION_ID="11.26.1"\n',
    )
    report = classify(read_os_release((derived,)), AudioBackends(pulseaudio=True))
    assert report.level == SUPPORTED
    assert "Spaced Linux 11.26.1" in report.reason


def test_classify_calls_a_host_with_no_audio_backend_unsupported(tmp_path: Path) -> None:
    report = classify(read_os_release((debian(tmp_path),)), AudioBackends())
    assert report.level == UNSUPPORTED
    assert "no audio backend at all" in report.reason


def test_classify_calls_a_host_that_names_no_distribution_unsupported(tmp_path: Path) -> None:
    report = classify(read_os_release((bare_host(tmp_path),)), AudioBackends(pipewire=True))
    assert report.level == UNSUPPORTED
    assert "no distribution ID in os-release" in report.reason


def test_support_sentence_states_the_level_and_names_the_distribution(tmp_path: Path) -> None:
    supported = support_sentence(classify(read_os_release((linux_mint(tmp_path),)), AudioBackends(pipewire=True)))
    assert supported.startswith("Supported here")
    assert "Linux Mint 13" in supported

    partial = support_sentence(classify(read_os_release((ubuntu(tmp_path),)), AudioBackends(alsa=True)))
    assert partial.startswith("Partially supported here")
    assert "Ubuntu 24.04 LTS" in partial

    unsupported = support_sentence(classify(read_os_release((bare_host(tmp_path),)), AudioBackends()))
    assert unsupported.startswith("Not supported here")


def test_support_sentence_never_claims_parity(tmp_path: Path) -> None:
    cases = (
        (read_os_release((linux_mint(tmp_path),)), AudioBackends(pipewire=True)),
        (read_os_release((debian(tmp_path),)), AudioBackends(alsa=True)),
        (read_os_release((bare_host(tmp_path),)), AudioBackends()),
    )
    for distribution, backends in cases:
        sentence = support_sentence(classify(distribution, backends))
        lowered = sentence.casefold()
        for claim in ("parity", "works on", "every platform", "cross-platform", "guaranteed", "anywhere"):
            assert claim not in lowered, sentence
        assert "windows, macos and android are not supported yet" in lowered, sentence


def test_start_names_the_platform_when_no_backend_exists(tmp_path: Path, monkeypatch) -> None:
    """With no backend at all, capture must say which distribution it found and
    what is missing — before GStreamer's silent autoaudiosrc fallback is asked for.
    """
    release = devuan(tmp_path)
    gst = MagicMock(name="gst")
    made: list[str] = []
    gst.ElementFactory.make.side_effect = lambda name: made.append(name) or MagicMock()
    monkeypatch.setattr(audio, "_ensure_gstreamer", lambda: gst)
    monkeypatch.setattr(audio, "read_os_release", lambda *args, **kwargs: read_os_release((release,)))
    monkeypatch.setattr(audio, "audio_backends", lambda *args, **kwargs: AudioBackends())

    capture = AudioCapture()
    try:
        capture.start("no-such-microphone", lambda *_: None, None, lambda _error: None)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("capture must fail when the host exposes no audio backend")

    assert "Devuan GNU+Linux 5" in message
    assert "no PipeWire, no PulseAudio and no ALSA devices" in message
    assert made == []
    assert capture.pipeline is None


def test_start_still_falls_back_to_autoaudiosrc_when_a_backend_exists(monkeypatch) -> None:
    """A host that does expose a backend keeps today's behaviour untouched."""
    gst = MagicMock(name="gst")
    made: list[str] = []
    gst.ElementFactory.make.side_effect = lambda name: made.append(name) or MagicMock()
    monkeypatch.setattr(audio, "_ensure_gstreamer", lambda: gst)
    monkeypatch.setattr(audio, "audio_backends", lambda *args, **kwargs: AudioBackends(pulseaudio=True))

    capture = AudioCapture()
    capture.start("", lambda *_: None, None, lambda _error: None)

    assert made[0] == "autoaudiosrc"
    assert capture.pipeline is not None
