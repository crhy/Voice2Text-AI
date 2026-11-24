from __future__ import annotations

from voice2text.audio import AudioDevice, select_device


def mics() -> list[AudioDevice]:
    # Monitor nodes are filtered out of list_devices(), so a real scan of a
    # machine with one monitor holds exactly the real microphones.
    return [
        AudioDevice(identifier="device.serial:builtin", name="Built-in Microphone", device=None),
        AudioDevice(identifier="device.serial:usb04E8", name="Blue Microphones", device=None),
    ]


def test_saved_id_beats_the_first_device() -> None:
    saved = mics()
    selected = select_device(saved, "device.serial:usb04E8", "")
    assert selected is saved[1]


def test_saved_name_matches_case_insensitively() -> None:
    saved = mics()
    selected = select_device(saved, "stale-id-that-went-away", "blue MICROPHONES")
    assert selected is saved[1]
    assert select_device(saved, "", "Built-in microphone") is saved[0]


def test_unresolvable_choice_returns_none_not_the_first_device() -> None:
    # The old fallback replaced a valid saved choice with devices[0]; a miss
    # must now return None so the caller keeps the stored choice untouched.
    assert select_device(mics(), "device.serial:gone", "Headset Microphone") is None


def test_empty_saved_choice_returns_none() -> None:
    assert select_device(mics(), "", "") is None


def test_empty_device_list_returns_none() -> None:
    assert select_device([], "device.serial:builtin", "Built-in Microphone") is None
