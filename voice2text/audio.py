from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any

Gst: Any = None  # Initialized lazily so importing this module needs no GStreamer or PyGObject.


def _ensure_gstreamer() -> Any:
    global Gst
    if Gst is None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst as _Gst

        _Gst.init(None)
        Gst = _Gst
    return Gst


@dataclass(slots=True)
class AudioDevice:
    identifier: str
    name: str
    device: Any
    properties: Any = None


def _stable_field(properties, key: str) -> str:
    """Read one property in a backend-agnostic way; never raises."""
    if properties is None:
        return ""
    with contextlib.suppress(Exception):
        if properties.has_field(key):
            value = properties.get_value(key)
            return value if value is not None else ""
    return ""


def is_monitor_source(properties, display_name: str = "") -> bool:
    """Return True when a source device is a loopback monitor, not a real microphone.

    Pipewire and PulseAudio expose "Monitor of …" nodes through the same
    Audio/Source device class as real microphones. Opening one captures the
    machine's own playback instead of the user's voice, so monitors are
    excluded from the microphone picker and refused at capture start.
    """
    stable = _stable_field(properties, "object.stable-id").casefold()
    if "monitor" in stable:
        return True
    return "monitor of" in (display_name or "").casefold()


class AudioCapture:
    """Native GStreamer microphone capture producing 16 kHz mono signed PCM."""

    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self._devices: list[AudioDevice] = []
        self._on_audio = None
        self._on_error = None
        self._last_level_emit = 0.0

    def list_devices(self) -> list[AudioDevice]:
        gst = _ensure_gstreamer()
        monitor = gst.DeviceMonitor.new()
        monitor.add_filter("Audio/Source", None)
        if not monitor.start():
            return []
        try:
            found: list[AudioDevice] = []
            seen: set[str] = set()
            for index, device in enumerate(monitor.get_devices()):
                name = device.get_display_name() or f"Microphone {index + 1}"
                props = device.get_properties()
                if is_monitor_source(props, name):
                    continue
                stable = self._device_identifier(props, name)
                if stable in seen:
                    continue
                seen.add(stable)
                found.append(AudioDevice(identifier=stable, name=name, device=device, properties=props))
            self._devices = found
            return list(found)
        finally:
            monitor.stop()

    @staticmethod
    def _device_identifier(properties, fallback: str) -> str:
        if properties is None:
            return fallback
        for key in (
            "device.serial",
            "device.path",
            "node.name",
            "object.path",
            "alsa.card_name",
        ):
            value = _stable_field(properties, key)
            if value:
                return f"{key}:{value}"
        with contextlib.suppress(Exception):
            return properties.to_string()
        return fallback

    def resolve_device(self, device_id: str, device_name: str = "") -> AudioDevice | None:
        """Return the device a saved microphone choice points at, or None.

        Identifiers drift between launches: `_device_identifier` falls back to a
        dump of every property when a device exposes no stable field, so a saved
        id can miss on the next session even though the same microphone is
        present. Matching the saved display name as well, and re-scanning once
        when the cache is stale, is what keeps the user's choice from quietly
        turning into the sound server's default source.
        """
        if not device_id and not device_name:
            return None
        match = self._match_choice(device_id, device_name, self._devices)
        if match is not None:
            return match
        # One re-scan covers a microphone that appeared, changed index or was
        # renamed after the cache was built; list_devices refreshes self._devices.
        return self._match_choice(device_id, device_name, self.list_devices())

    @staticmethod
    def _match_choice(device_id: str, device_name: str, devices: list[AudioDevice]) -> AudioDevice | None:
        if device_id:
            for device in devices:
                if device.identifier == device_id:
                    return device
        if device_name:
            wanted = device_name.casefold()
            for device in devices:
                if device.name.casefold() == wanted:
                    return device
        return None

    def start(self, device_id: str, on_audio, on_level, on_error, *, device_name: str = "") -> None:
        self.stop()
        self._on_audio = on_audio
        self._on_error = on_error

        selected = self.resolve_device(device_id, device_name)
        # A profile saved by an older build could still point at a monitor
        # node; refuse to open it rather than capture the machine's own audio.
        if selected is not None and is_monitor_source(selected.properties, selected.name):
            raise RuntimeError("The selected source is a monitor stream and cannot capture your voice.")
        if selected is None and (device_id or device_name):
            # Never fall back to autoaudiosrc here: on a PipeWire desktop that
            # is usually the monitor of the default sink, i.e. the machine's
            # own audio. Say what happened instead, before any pipeline exists.
            raise RuntimeError("The microphone you chose is no longer on this machine. Pick one from the list.")
        gst = _ensure_gstreamer()
        if selected is not None:
            source = selected.device.create_element(None)
        else:
            source = gst.ElementFactory.make("autoaudiosrc")
        convert = gst.ElementFactory.make("audioconvert")
        resample = gst.ElementFactory.make("audioresample")
        capsfilter = gst.ElementFactory.make("capsfilter")
        sink = gst.ElementFactory.make("appsink")
        if not all((source, convert, resample, capsfilter, sink)):
            raise RuntimeError("Required GStreamer audio elements are unavailable.")

        capsfilter.set_property("caps", gst.Caps.from_string("audio/x-raw,format=S16LE,channels=1,rate=16000"))
        sink.set_property("emit-signals", True)
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 12)
        if sink.find_property("drop") is not None:
            sink.set_property("drop", True)
        sink.connect("new-sample", self._on_sample, on_level)

        pipeline = gst.Pipeline.new("voice2text-capture")
        for element in (source, convert, resample, capsfilter, sink):
            pipeline.add(element)
        if not source.link(convert) or not convert.link(resample) or not resample.link(capsfilter) or not capsfilter.link(sink):
            pipeline.set_state(gst.State.NULL)
            raise RuntimeError("Could not connect the GStreamer audio pipeline.")

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        self.pipeline = pipeline
        result = pipeline.set_state(gst.State.PLAYING)
        if result == gst.StateChangeReturn.FAILURE:
            self.stop()
            raise RuntimeError("The selected microphone could not be opened.")

    def _on_sample(self, sink, on_level):
        gst = _ensure_gstreamer()
        sample = sink.emit("pull-sample")
        if sample is None:
            return gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        success, mapped = buffer.map(gst.MapFlags.READ)
        if not success:
            return gst.FlowReturn.ERROR
        try:
            pcm = bytes(mapped.data)
        finally:
            buffer.unmap(mapped)

        level = self._rms(pcm)
        try:
            if self._on_audio is not None:
                self._on_audio(pcm, level)
            now = time.monotonic()
            if on_level is not None and now - self._last_level_emit >= 0.05:
                self._last_level_emit = now
                on_level(level)
        except Exception as exc:  # noqa: BLE001 - callback boundary
            if self._on_error is not None:
                self._on_error(str(exc))
            return gst.FlowReturn.ERROR
        return gst.FlowReturn.OK

    @staticmethod
    def _rms(pcm: bytes) -> float:
        import numpy as np

        samples = np.frombuffer(pcm, dtype="<i2")
        if not samples.size:
            return 0.0
        stride = max(1, samples.size // 2048)
        chosen = samples[::stride].astype(np.float64)
        return float(np.sqrt(np.mean(chosen * chosen)))

    def _on_bus_error(self, _bus, message) -> None:
        error, debug = message.parse_error()
        detail = f"{error.message}"
        if debug:
            detail = f"{detail} ({debug})"
        if self._on_error is not None:
            self._on_error(detail)
        self.stop()

    def stop(self) -> None:
        gst = _ensure_gstreamer() if self.pipeline is not None else None
        pipeline, self.pipeline = self.pipeline, None
        if pipeline is not None:
            pipeline.set_state(gst.State.NULL)
