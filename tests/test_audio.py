from __future__ import annotations

from typing import Any

from voice2text.audio import AudioCapture, AudioDevice, _stable_field, is_monitor_source


class FakeGstProperties(dict):
    """Stand-in for Gst.Structure: duck-typed key/value bag."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        super().__init__(values or {})

    def has_field(self, key: str) -> bool:
        return key in self

    def get_value(self, key: str) -> Any:
        return self.get(key)

    def to_string(self) -> str:
        return f"fakeGstStructure({'; '.join(f'{k}={v!r}' for k, v in self.items())})"


def pipewire_mics() -> dict[str, FakeGstProperties]:
    return {
        "blue": FakeGstProperties(
            {
                "object.stable-id": "pipewire_device_3.n",
                "device.serial": "00:1E:06:11:44:44",
                "node.name": "Blue Microphone",
                "display-name": "Blue Microphone",
            }
        ),
        "blue_monitor": FakeGstProperties(
            {
                "object.stable-id": "pipewire_device_3.n.monitor.",
                "node.name": "monitor of Blue Microphone",
                "display-name": "Monitor of Blue Microphone",
                "alsa.card_name": "Blue Microphone",
            }
        ),
        "headset": FakeGstProperties(
            {
                "object.stable-id": "pipewire_device_9.n",
                "node.name": "Headset Microphone",
                "display-name": "Headset Microphone",
            }
        ),
    }


def test_monitor_source_detected_by_stable_id() -> None:
    props, _ = next(iter(pipewire_mics().items()))
    assert is_monitor_source(props, "Blue Microphone") is False
    monitor = pipewire_mics()["blue_monitor"]
    assert is_monitor_source(monitor, "Monitor of Blue Microphone") is True
    # The display name would mislead a weaker heuristic, but the stable id
    # keeps the verdict even if the user renames or the label is missing.
    assert is_monitor_source(monitor, "Blue Microphone") is True


def test_monitor_source_detected_by_display_name_for_alsa_monitors() -> None:
    props = FakeGstProperties({"display-name": "Monitor of Blue Microphone"})
    assert is_monitor_source(props, "Monitor of Blue Microphone") is True
    plain = FakeGstProperties({"display-name": "Blue Microphone"})
    assert is_monitor_source(plain, "Blue Microphone") is False


def test_monitor_detection_is_case_insensitive_and_none_safe() -> None:
    props = FakeGstProperties({"display-name": "MONITOR OF Blue Microphone"})
    assert is_monitor_source(props, props.get_value("display-name")) is True
    assert is_monitor_source(None, "monitor of anything") is True
    assert is_monitor_source(None, "Blue Microphone") is False
    assert is_monitor_source(FakeGstProperties(), "") is False


def test_stable_field_tolerates_non_structure_properties() -> None:
    assert _stable_field(FakeGstProperties({"device.serial": "A1"}), "device.serial") == "A1"
    assert _stable_field(FakeGstProperties(), "device.serial") == ""
    assert _stable_field(None, "device.serial") == ""
    assert _stable_field(object(), "device.serial") == ""


def test_device_identifier_prefers_stable_fields() -> None:
    blue = pipewire_mics()["blue"]
    assert AudioCapture._device_identifier(blue, "Microphone 1") == "device.serial:00:1E:06:11:44:44"
    # With no stable field, the identifier falls back to the structure dump.
    assert AudioCapture._device_identifier(FakeGstProperties(), "Fallback") == "fakeGstStructure()"
    assert AudioCapture._device_identifier(None, "Fallback") == "Fallback"


def test_start_refuses_a_stale_monitor_selection() -> None:
    """A monitor selected by an old profile must be refused with a clear error,
    before any GStreamer pipeline is built."""
    monitor = pipewire_mics()["blue_monitor"]
    device = AudioDevice(
        identifier="node.name:monitor of Blue Microphone",
        name="Monitor of Blue Microphone",
        device=None,
        properties=monitor,
    )
    capture = AudioCapture()
    capture._devices = [device]
    try:
        capture.start("node.name:monitor of Blue Microphone", lambda *_: None, None, lambda _e: None)
    except RuntimeError as exc:
        assert "monitor" in str(exc).casefold()
    else:
        raise AssertionError("opening a monitor source must fail")
    assert capture.pipeline is None
