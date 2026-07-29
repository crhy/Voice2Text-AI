from __future__ import annotations

import contextlib
import math
import sys
import time
from array import array
from dataclasses import dataclass
from typing import Any

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)


@dataclass(slots=True)
class AudioDevice:
    identifier: str
    name: str
    device: Any


class AudioCapture:
    """Native GStreamer microphone capture producing 16 kHz mono signed PCM."""

    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self._devices: list[AudioDevice] = []
        self._on_audio = None
        self._on_error = None
        self._last_level_emit = 0.0

    def list_devices(self) -> list[AudioDevice]:
        monitor = Gst.DeviceMonitor.new()
        monitor.add_filter("Audio/Source", None)
        if not monitor.start():
            return []
        try:
            found: list[AudioDevice] = []
            for index, device in enumerate(monitor.get_devices()):
                name = device.get_display_name() or f"Microphone {index + 1}"
                props = device.get_properties()
                stable = self._device_identifier(props, name)
                found.append(AudioDevice(identifier=stable, name=name, device=device))
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
            with contextlib.suppress(Exception):
                if properties.has_field(key):
                    value = properties.get_value(key)
                    if value:
                        return f"{key}:{value}"
        with contextlib.suppress(Exception):
            return properties.to_string()
        return fallback

    def start(self, device_id: str, on_audio, on_level, on_error) -> None:
        self.stop()
        self._on_audio = on_audio
        self._on_error = on_error

        selected = next((item for item in self._devices if item.identifier == device_id), None)
        source = selected.device.create_element(None) if selected else Gst.ElementFactory.make("autoaudiosrc")
        convert = Gst.ElementFactory.make("audioconvert")
        resample = Gst.ElementFactory.make("audioresample")
        capsfilter = Gst.ElementFactory.make("capsfilter")
        sink = Gst.ElementFactory.make("appsink")
        if not all((source, convert, resample, capsfilter, sink)):
            raise RuntimeError("Required GStreamer audio elements are unavailable.")

        capsfilter.set_property("caps", Gst.Caps.from_string("audio/x-raw,format=S16LE,channels=1,rate=16000"))
        sink.set_property("emit-signals", True)
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 12)
        if sink.find_property("drop") is not None:
            sink.set_property("drop", True)
        sink.connect("new-sample", self._on_sample, on_level)

        pipeline = Gst.Pipeline.new("voice2text-capture")
        for element in (source, convert, resample, capsfilter, sink):
            pipeline.add(element)
        if not source.link(convert) or not convert.link(resample) or not resample.link(capsfilter) or not capsfilter.link(sink):
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Could not connect the GStreamer audio pipeline.")

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        self.pipeline = pipeline
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            self.stop()
            raise RuntimeError("The selected microphone could not be opened.")

    def _on_sample(self, sink, on_level):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        success, mapped = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
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
            return Gst.FlowReturn.ERROR
        return Gst.FlowReturn.OK

    @staticmethod
    def _rms(pcm: bytes) -> float:
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            return 0.0
        stride = max(1, len(samples) // 2048)
        chosen = samples[::stride]
        return math.sqrt(sum(value * value for value in chosen) / len(chosen))

    def _on_bus_error(self, _bus, message) -> None:
        error, debug = message.parse_error()
        detail = f"{error.message}"
        if debug:
            detail = f"{detail} ({debug})"
        if self._on_error is not None:
            self._on_error(detail)
        self.stop()

    def stop(self) -> None:
        pipeline, self.pipeline = self.pipeline, None
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
