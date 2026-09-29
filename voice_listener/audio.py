"""Audio plumbing: capture sources, resampling to 16 kHz, and a ring buffer
addressed by absolute sample index.

Everything downstream works on 16 kHz mono int16. Time inside the engine is
measured in samples since the stream started, so behaviour is identical whether
audio comes from a live mic or a WAV file played faster than real time.
"""

from __future__ import annotations

import logging
import threading
import time
import wave
from math import gcd
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
AudioCallback = Callable[[np.ndarray], None]


class RingBuffer:
    """Fixed-size int16 ring buffer indexed by absolute sample position."""

    def __init__(self, seconds: float, sample_rate: int = SAMPLE_RATE):
        self.capacity = int(seconds * sample_rate)
        self._buf = np.zeros(self.capacity, dtype=np.int16)
        self.total = 0  # samples ever written == absolute index of the next sample

    def write(self, x: np.ndarray) -> None:
        n = len(x)
        if n >= self.capacity:
            self._buf[:] = x[-self.capacity:]
            self.total += n
            return
        start = self.total % self.capacity
        end = start + n
        if end <= self.capacity:
            self._buf[start:end] = x
        else:
            first = self.capacity - start
            self._buf[start:] = x[:first]
            self._buf[: n - first] = x[first:]
        self.total += n

    @property
    def oldest(self) -> int:
        return max(0, self.total - self.capacity)

    def read(self, start: int, end: int) -> np.ndarray:
        """Copy samples [start, end) in absolute positions, clamped to what's held."""
        start = max(start, self.oldest)
        end = min(end, self.total)
        if end <= start:
            return np.zeros(0, dtype=np.int16)
        s = start % self.capacity
        n = end - start
        if s + n <= self.capacity:
            return self._buf[s : s + n].copy()
        first = self.capacity - s
        return np.concatenate([self._buf[s:], self._buf[: n - first]])


class Resampler:
    """Streaming resampler to 16 kHz.

    Integer ratios (48k, 32k) use a built-in FIR low-pass + decimation that keeps
    filter state across blocks. Other rates (44.1k) need the optional `soxr`
    package.
    """

    def __init__(self, in_rate: int, out_rate: int = SAMPLE_RATE):
        self.in_rate, self.out_rate = in_rate, out_rate
        self._mode = "passthrough"
        if in_rate == out_rate:
            return
        if in_rate % out_rate == 0:
            from scipy.signal import firwin

            self.factor = in_rate // out_rate
            self._taps = firwin(24 * self.factor + 1, 0.9 / self.factor).astype(np.float32)
            self._zi = np.zeros(len(self._taps) - 1, dtype=np.float32)
            self._offset = 0  # absolute input index of the next sample
            self._mode = "decimate"
            return
        try:
            import soxr  # type: ignore
        except ImportError as e:
            g = gcd(in_rate, out_rate)
            raise RuntimeError(
                f"Capture rate {in_rate} Hz isn't an integer multiple of {out_rate} Hz "
                f"(ratio {in_rate // g}:{out_rate // g}). Install `soxr` (pip install soxr), "
                "pick a 16/32/48 kHz device rate, or capture through PipeWire/PulseAudio "
                "which resamples for you."
            ) from e
        self._soxr = soxr.ResampleStream(in_rate, out_rate, 1, dtype="int16")
        self._mode = "soxr"

    def process(self, x: np.ndarray) -> np.ndarray:
        if self._mode == "passthrough":
            return x
        if self._mode == "soxr":
            return self._soxr.resample_chunk(x)
        from scipy.signal import lfilter

        xf = x.astype(np.float32)
        y, self._zi = lfilter(self._taps, 1.0, xf, zi=self._zi)
        first = (-self._offset) % self.factor
        out = y[first :: self.factor]
        self._offset += len(x)
        return np.clip(np.round(out), -32768, 32767).astype(np.int16)


def list_devices() -> str:
    import sounddevice as sd

    return str(sd.query_devices())


def _resolve_device(device: str | int):
    """Accept "", an index (int or numeric string) or a name substring."""
    if device == "" or device is None:
        return None
    if isinstance(device, int):
        return device
    if device.isdigit():
        return int(device)
    import sounddevice as sd

    matches = [
        i
        for i, d in enumerate(sd.query_devices())
        if device.lower() in d["name"].lower() and d["max_input_channels"] > 0
    ]
    if not matches:
        raise RuntimeError(f"No input device matching {device!r}. Run with --list-devices.")
    return matches[0]


class MicSource:
    """Live microphone capture via PortAudio (sounddevice)."""

    def __init__(self, device: str | int = "", sample_rate: int = 0, channels: int = 1,
                 channel: int = 0, block_ms: int = 32):
        self.device = _resolve_device(device)
        self.channels = channels
        self.channel = channel
        self.block_ms = block_ms
        self.requested_rate = sample_rate
        self._stream = None

    def _pick_rate(self) -> int:
        import sounddevice as sd

        if self.requested_rate:
            return self.requested_rate
        try:
            sd.check_input_settings(device=self.device, channels=self.channels,
                                    dtype="int16", samplerate=SAMPLE_RATE)
            return SAMPLE_RATE
        except Exception:
            info = sd.query_devices(self.device, "input")
            rate = int(info["default_samplerate"])
            log.info("Device doesn't accept 16 kHz; capturing at %d Hz and resampling", rate)
            return rate

    def start(self, on_audio: AudioCallback) -> None:
        import sounddevice as sd

        rate = self._pick_rate()
        resampler = Resampler(rate)
        ch = self.channel

        def callback(indata, frames, time_info, status):  # PortAudio thread
            if status:
                log.debug("audio status: %s", status)
            mono = indata[:, ch] if indata.ndim == 2 else indata
            # Copy: PortAudio reuses `indata` after the callback returns.
            out = resampler.process(mono.astype(np.int16, copy=True))
            if len(out):
                on_audio(out)

        self._stream = sd.InputStream(
            device=self.device,
            channels=self.channels,
            samplerate=rate,
            dtype="int16",
            blocksize=int(rate * self.block_ms / 1000),
            callback=callback,
        )
        self._stream.start()
        name = sd.query_devices(self._stream.device)["name"]
        log.info("Capturing from %r at %d Hz (%d ch, using ch %d)", name, rate, self.channels, ch)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class FileSource:
    """Feed a WAV file as if it were a microphone. Used for tests and tuning.

    speed=1.0 plays in real time, 0 plays as fast as possible. `on_done` fires
    after the last block (plus optional trailing silence).
    """

    def __init__(self, path: str, speed: float = 1.0, block_ms: int = 32,
                 trailing_silence_s: float = 2.0, on_done: Callable[[], None] | None = None):
        self.path = path
        self.speed = speed
        self.block_ms = block_ms
        self.trailing_silence_s = trailing_silence_s
        self.on_done = on_done
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @staticmethod
    def load(path: str) -> np.ndarray:
        with wave.open(path, "rb") as w:
            if w.getsampwidth() != 2:
                raise RuntimeError(f"{path}: only 16-bit PCM WAV is supported")
            rate, nch = w.getframerate(), w.getnchannels()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if nch > 1:
            data = data.reshape(-1, nch)[:, 0]
        return Resampler(rate).process(data.copy())

    def start(self, on_audio: AudioCallback) -> None:
        audio = self.load(self.path)
        audio = np.concatenate([audio, np.zeros(int(self.trailing_silence_s * SAMPLE_RATE), np.int16)])
        block = int(SAMPLE_RATE * self.block_ms / 1000)

        def run():
            t0 = time.monotonic()
            for i in range(0, len(audio), block):
                if self._stop.is_set():
                    return
                on_audio(audio[i : i + block].copy())
                if self.speed > 0:
                    due = t0 + (i + block) / SAMPLE_RATE / self.speed
                    delay = due - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
            if self.on_done:
                self.on_done()

        self._thread = threading.Thread(target=run, name="file-source", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()
