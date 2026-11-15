from __future__ import annotations

import pytest
from test_audio import FakeGstProperties, pipewire_mics

from voice2text.audio import AudioCapture, AudioDevice


class FakeElement:
    def __init__(self, name: str) -> None:
        self.name = name

    def link(self, other) -> bool:
        return True

    def set_property(self, *_args) -> None:
        return None

    def find_property(self, _name):
        return None

    def connect(self, *_args) -> None:
        return None


class FakeBus:
    def add_signal_watch(self) -> None:
        return None

    def connect(self, *_args) -> None:
        return None


class FakePipeline:
    def __init__(self) -> None:
        self.elements: list[FakeElement] = []

    def add(self, element: FakeElement) -> None:
        self.elements.append(element)

    def get_bus(self) -> FakeBus:
        return FakeBus()

    def set_state(self, state) -> str:
        return state


class FakeElementFactory:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def make(self, name: str) -> FakeElement:
        self.log.append(name)
        return FakeElement(name)


class FakeGst:
    """Enough of the Gst module surface for start() to build a pipeline offline."""

    class State:
        NULL = "null"
        PLAYING = "playing"

    class StateChangeReturn:
        FAILURE = "failure"

    class Caps:
        @staticmethod
        def from_string(_spec: str) -> str:
            return "caps"

    class Pipeline:
        @staticmethod
        def new(_name: str) -> FakePipeline:
            return FakePipeline()

    def __init__(self) -> None:
        self.made: list[str] = []
        self.ElementFactory = FakeElementFactory(self.made)


class FakeSourceDevice:
    """A microphone that hands out its own source element, like Gst's AudioDevice."""

    def __init__(self, element_name: str) -> None:
        self.element_name = element_name

    def create_element(self, _profile) -> FakeElement:
        return FakeElement(self.element_name)


def headset_device(identifier: str = "node.name:Headset Microphone") -> AudioDevice:
    return AudioDevice(
        identifier=identifier,
        name="Headset Microphone",
        device=FakeSourceDevice("headset-source"),
        properties=pipewire_mics()["headset"],
    )


def monitor_device() -> AudioDevice:
    return AudioDevice(
        identifier="node.name:monitor of Blue Microphone",
        name="Monitor of Blue Microphone",
        device=FakeSourceDevice("monitor-source"),
        properties=pipewire_mics()["blue_monitor"],
    )


def test_a_drifted_identifier_still_opens_the_chosen_microphone(monkeypatch) -> None:
    """A microphone saved without a stable field keeps an identifier that is a
    property dump, so it drifts between launches. The saved name must still
    select it, and autoaudiosrc must never be asked for."""
    gst = FakeGst()
    monkeypatch.setattr("voice2text.audio._ensure_gstreamer", lambda: gst)
    capture = AudioCapture()
    capture.list_devices = lambda: capture._devices
    capture._devices = [headset_device()]
    drifted_id = "fakeGstStructure(display-name='Headset Microphone')"

    capture.start(drifted_id, lambda *_: None, None, lambda _e: None, device_name="Headset Microphone")

    assert "autoaudiosrc" not in gst.made
    assert capture.pipeline is not None
    assert capture.pipeline.elements[0].name == "headset-source"


def test_a_chosen_microphone_that_is_gone_fails_loudly(monkeypatch) -> None:
    """A saved choice that is genuinely absent must raise before any pipeline
    exists, instead of quietly capturing the machine's own audio."""
    gst = FakeGst()
    monkeypatch.setattr("voice2text.audio._ensure_gstreamer", lambda: gst)
    capture = AudioCapture()
    capture.list_devices = lambda: [headset_device()]
    capture._devices = [headset_device()]

    with pytest.raises(RuntimeError) as raised:
        capture.start(
            "device.serial:00:1E:06:DE:AD:00",
            lambda *_: None,
            None,
            lambda _e: None,
            device_name="Desk Microphone",
        )

    assert "no longer on this machine" in str(raised.value)
    assert gst.made == []
    assert capture.pipeline is None


def test_no_choice_yet_still_uses_the_default_source(monkeypatch) -> None:
    """Only an empty profile may use autoaudiosrc; that is the documented default."""
    gst = FakeGst()
    monkeypatch.setattr("voice2text.audio._ensure_gstreamer", lambda: gst)
    capture = AudioCapture()
    capture.list_devices = lambda: [headset_device()]

    capture.start("", lambda *_: None, None, lambda _e: None)

    assert gst.made[0] == "autoaudiosrc"
    assert capture.pipeline is not None
    assert capture.pipeline.elements[0].name == "autoaudiosrc"


def test_a_monitor_selection_is_still_refused(monkeypatch) -> None:
    """The loopback guard keeps working now that resolution goes through
    resolve_device: a monitor saved by an older build is never opened."""
    gst = FakeGst()
    monkeypatch.setattr("voice2text.audio._ensure_gstreamer", lambda: gst)
    capture = AudioCapture()
    capture.list_devices = lambda: []
    capture._devices = [monitor_device()]

    with pytest.raises(RuntimeError) as raised:
        capture.start(
            "node.name:monitor of Blue Microphone",
            lambda *_: None,
            None,
            lambda _e: None,
            device_name="Monitor of Blue Microphone",
        )

    assert "monitor" in str(raised.value).casefold()
    assert gst.made == []
    assert capture.pipeline is None


def test_resolve_device_rescans_once_when_the_cache_is_stale() -> None:
    """An empty or stale cache gets exactly one re-scan, then matches by id."""
    capture = AudioCapture()
    capture._devices = []
    scans: list[int] = []

    def rescan() -> list[AudioDevice]:
        scans.append(len(scans) + 1)
        return [headset_device()]

    capture.list_devices = rescan

    found = capture.resolve_device("node.name:Headset Microphone")
    assert found is not None
    assert found.name == "Headset Microphone"
    assert scans == [1]
    # A device that is really absent costs one extra scan, not a scan per call.
    assert capture.resolve_device("device.serial:99:99:99:99:99:99", "Gone Microphone") is None
    assert scans == [1, 2]


def test_resolve_device_matches_the_saved_name_case_insensitively() -> None:
    """A name saved by another build or keyboard layout still selects the device."""
    capture = AudioCapture()
    capture._devices = [headset_device(identifier="node.name:Headset Microphone")]
    capture.list_devices = lambda: []

    found = capture.resolve_device("alsa.card_name:stale", "HEADSET microphone")
    assert found is not None
    assert found.identifier == "node.name:Headset Microphone"
    # No choice ever saved: nothing to resolve, and no scan is wasted.
    assert capture.resolve_device("") is None
    assert capture.resolve_device("", "") is None


def test_identifier_drift_is_visible_in_the_saved_pair() -> None:
    """Documents the drift the fix has to survive: the same microphone, two ids."""
    props = FakeGstProperties({"display-name": "Headset Microphone"})
    before = AudioCapture._device_identifier(props, "Microphone 1")
    renamed = FakeGstProperties({"display-name": "Headset Microphone (desk)"})
    after = AudioCapture._device_identifier(renamed, "Microphone 1")
    assert before != after
    assert before == "fakeGstStructure(display-name='Headset Microphone')"
