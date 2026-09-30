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
from collections import deque
from math import gcd
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# on_audio(mic_16k, ref_16k_or_None): `ref` is what the speaker played during the
# same block (only for sources that play audio), used for echo cancellation.
AudioCallback = Callable[..., None]


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
    """Exact streaming polyphase resampler for any rational ratio (48k->16k,
    44.1k->16k, 22.05k->48k, ...).

    Output is sample-for-sample identical to scipy.signal.upfirdn over the whole
    signal, however the input is split into blocks, so there are no seams at
    block boundaries. Group delay is about `taps_per_phase / 2` input samples.
    """

    def __init__(self, in_rate: int, out_rate: int = SAMPLE_RATE, taps_per_phase: int = 24):
        self.in_rate, self.out_rate = in_rate, out_rate
        self.passthrough = in_rate == out_rate
        if self.passthrough:
            return
        from scipy.signal import firwin

        g = gcd(in_rate, out_rate)
        self.up, self.down = out_rate // g, in_rate // g
        T = taps_per_phase
        h = firwin(T * self.up, 0.92 / max(self.up, self.down)) * self.up
        self.h = h
        self._phases = h.reshape(T, self.up).T.astype(np.float64)  # [phase, tap] = h[phase + tap*up]
        self._T = T
        self._hist = np.zeros(T - 1, dtype=np.float64)  # last T-1 input samples
        self._n_in = 0  # input samples consumed so far
        self._k = 0  # index of the next output sample

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.passthrough:
            return x
        T, L, M = self._T, self.up, self.down
        buf = np.concatenate([self._hist, x.astype(np.float64)])
        n_in = self._n_in + len(x)
        k_end = (n_in * L - 1) // M + 1 if n_in else 0  # outputs whose newest input sample exists
        ks = np.arange(self._k, k_end)
        out = np.zeros(0, dtype=np.int16)
        if len(ks):
            n = ks * M
            phase = n % L
            newest = n // L - (self._n_in - (T - 1))  # index into buf of the newest input used
            idx = newest[:, None] - np.arange(T)[None, :]
            y = np.einsum("kt,kt->k", self._phases[phase], buf[idx])
            out = np.clip(np.round(y), -32768, 32767).astype(np.int16)
        self._k = k_end
        self._n_in = n_in
        self._hist = buf[len(buf) - (T - 1):]
        return out


def list_devices() -> str:
    import sounddevice as sd

    return str(sd.query_devices())


def _resolve_device(device: str | int, kind: str = "input"):
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
        if device.lower() in d["name"].lower() and d[f"max_{kind}_channels"] > 0
    ]
    if not matches:
        raise RuntimeError(f"No {kind} device matching {device!r}. Run with --list-devices.")
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
                 trailing_silence_s: float = 2.0, on_done: Callable[[], None] | None = None,
                 room: "EchoRoom | None" = None):
        self.path = path
        self.room = room  # if set: plays through a simulated speaker->mic echo path
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
                chunk = audio[i : i + block].copy()
                if self.room is not None:
                    on_audio(*self.room.process(chunk))
                else:
                    on_audio(chunk)
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


class _Item:
    __slots__ = ("id", "text", "then", "timeout_s", "client", "resampler", "chunks",
                 "queued", "played", "ended", "started")

    def __init__(self, id_, src_rate, out_rate, text, then, timeout_s, client):
        self.id, self.text, self.then, self.timeout_s, self.client = id_, text, then, timeout_s, client
        self.resampler = Resampler(src_rate, out_rate)
        self.chunks: deque[np.ndarray] = deque()
        self.queued = 0  # samples at the output rate
        self.played = 0
        self.ended = False
        self.started = False


class Player:
    """Plays agent audio (TTS) through the listener's own output stream.

    The agent streams PCM in; the audio callback pulls it out with `read()`.
    Because the listener owns playback it knows exactly what went to the
    speaker (the echo-cancellation reference), can duck and stop it instantly
    on barge-in, and knows when it really finished.

    Notifications are passed to `on_event(dict)` from the audio thread:
      {"kind": "started",  "id", "text"}
      {"kind": "finished", "id", "played_s", "then", "timeout_s", "more"}
    """

    def __init__(self, out_rate: int, on_event: Callable[[dict], None] | None = None,
                 ramp_ms: float = 15.0):
        self.out_rate = out_rate
        self.on_event = on_event or (lambda ev: None)
        self._lock = threading.Lock()
        self._items: deque[_Item] = deque()
        self._gain = 1.0
        self._target = 1.0
        self._step = 1.0 / max(1, int(out_rate * ramp_ms / 1000))

    # -- producer side (server thread) --------------------------------------

    def start_item(self, id_: str, src_rate: int, text: str = "", then: str = "listen",
                   timeout_s: float | None = None, client: str | None = None) -> None:
        with self._lock:
            for it in self._items:  # a new item implicitly completes the previous ones
                if it.client == client:
                    it.ended = True
            self._items.append(_Item(id_, src_rate, self.out_rate, text, then, timeout_s, client))

    def _find(self, id_: str | None, client: str | None) -> _Item | None:
        for it in reversed(self._items):
            if (id_ is None or it.id == id_) and (client is None or it.client == client) and not it.ended:
                return it
        return None

    def add_audio(self, pcm: np.ndarray, id_: str | None = None, client: str | None = None) -> bool:
        with self._lock:
            it = self._find(id_, client)
            if it is None:
                return False
            resampler = it.resampler
        data = resampler.process(pcm.astype(np.int16, copy=False))  # outside the lock
        with self._lock:
            if it in self._items and len(data):
                it.chunks.append(data)
                it.queued += len(data)
        return True

    def end_item(self, id_: str | None = None, client: str | None = None) -> bool:
        with self._lock:
            it = self._find(id_, client)
            if it is None:
                return False
            it.ended = True
            return True

    # -- control (engine thread) --------------------------------------------

    def stop(self, reason: str) -> list[dict]:
        """Drop everything queued. Returns what was cut off, for events."""
        with self._lock:
            cut = [{"playback_id": it.id, "reason": reason,
                    "played_s": round(it.played / self.out_rate, 2),
                    "total_s": round(it.queued / self.out_rate, 2) if it.ended else None,
                    "text": it.text}
                   for it in self._items if it.started or it.queued]
            self._items.clear()
            self._gain = self._target = 1.0
        return cut

    def duck(self, db: float) -> None:
        self._target = 10 ** (db / 20)

    def unduck(self) -> None:
        self._target = 1.0

    @property
    def active(self) -> bool:
        with self._lock:
            return any(it.started or it.queued for it in self._items)

    def current(self) -> dict | None:
        with self._lock:
            for it in self._items:
                if it.started:
                    return {"playback_id": it.id, "played_s": round(it.played / self.out_rate, 2),
                            "total_s": round(it.queued / self.out_rate, 2) if it.ended else None,
                            "text": it.text}
        return None

    # -- consumer side (audio thread) ---------------------------------------

    def read(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float64)
        events = []
        with self._lock:
            pos = 0
            while pos < n and self._items:
                it = self._items[0]
                if not it.chunks:
                    if it.ended:
                        self._items.popleft()
                        if it.started:
                            more = any(x.queued or not x.ended for x in self._items)
                            events.append({"kind": "finished", "id": it.id,
                                           "played_s": round(it.played / self.out_rate, 2),
                                           "then": it.then, "timeout_s": it.timeout_s, "more": more})
                        continue
                    break  # underrun: the agent hasn't sent the rest yet
                if not it.started:
                    it.started = True
                    events.append({"kind": "started", "id": it.id, "text": it.text})
                chunk = it.chunks[0]
                take = min(n - pos, len(chunk))
                out[pos : pos + take] = chunk[:take]
                if take == len(chunk):
                    it.chunks.popleft()
                else:
                    it.chunks[0] = chunk[take:]
                it.played += take
                pos += take
            # smooth gain changes (ducking) to avoid clicks
            if self._gain != self._target:
                lo, hi = sorted((self._gain, self._target))
                direction = 1.0 if self._target > self._gain else -1.0
                g = np.clip(self._gain + direction * self._step * np.arange(1, n + 1), lo, hi)
                self._gain = float(g[-1])
                out *= g
            elif self._gain != 1.0:
                out *= self._gain
        for ev in events:
            self.on_event(ev)
        return np.clip(np.round(out), -32768, 32767).astype(np.int16)


class EchoRoom:
    """Simulated speaker -> room -> mic path, for tests and for trying echo
    handling without hardware. Pulls the speaker signal from a Player running
    at 16 kHz and returns (mic, ref) = (user + echo + noise, what was played)."""

    def __init__(self, player: Player, delay_ms: float = 80, gain_db: float = -6,
                 rt60_ms: float = 300, distortion: float = 1.5, noise_db: float = -65, seed: int = 0,
                 echo: bool = True):
        from scipy.signal import lfilter  # noqa: F401  (fail early if scipy is missing)

        rng = np.random.default_rng(seed)
        n = int(SAMPLE_RATE * (delay_ms + rt60_ms) / 1000)
        d = int(SAMPLE_RATE * delay_ms / 1000)
        rir = np.zeros(n)
        rir[d] = 10 ** (gain_db / 20)
        tail = np.arange(n - d - 1)
        rir[d + 1 :] += rng.standard_normal(len(tail)) * np.exp(-tail / (SAMPLE_RATE * rt60_ms / 6900)) \
            * 0.16 * 10 ** (gain_db / 20)
        self.player = player
        self.echo = echo  # False: a perfectly isolated speaker (headset), for comparison
        self.rir = rir
        self._zi = np.zeros(len(rir) - 1)
        self.distortion = distortion
        self.noise_amp = 32768 * 10 ** (noise_db / 20)
        self._rng = rng

    def process(self, user: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from scipy.signal import lfilter

        ref = self.player.read(len(user))
        if not self.echo:
            return user, ref
        x = ref.astype(np.float64)
        if self.distortion:  # a little speaker nonlinearity, which linear AEC can't remove
            x = 32767 * np.tanh(x / 32767 * self.distortion) / self.distortion
        echo, self._zi = lfilter(self.rir, 1.0, x, zi=self._zi)
        mic = user.astype(np.float64) + echo + self._rng.standard_normal(len(user)) * self.noise_amp
        return np.clip(np.round(mic), -32768, 32767).astype(np.int16), ref


class DuplexSource:
    """Mic capture and agent playback on one full-duplex PortAudio stream.

    Input and output share a clock and a callback, so the block written to the
    speaker and the block read from the mic stay aligned: that is what lets the
    echo canceller use the played audio as its reference.
    """

    def __init__(self, player_factory: Callable[[int], Player], device: str | int = "",
                 output_device: str | int = "", sample_rate: int = 0, channels: int = 1,
                 channel: int = 0, output_channels: int = 1, block_ms: int = 32):
        self.in_dev = _resolve_device(device, "input")
        self.out_dev = _resolve_device(output_device, "output")
        self.channels, self.channel, self.out_channels = channels, channel, output_channels
        self.block_ms = block_ms
        self.rate = self._pick_rate(sample_rate)
        self.player = player_factory(self.rate)
        self.round_trip_ms = 0
        self._stream = None

    def _pick_rate(self, requested: int) -> int:
        import sounddevice as sd

        if requested:
            return requested
        for rate in (48000, SAMPLE_RATE):
            try:
                sd.check_input_settings(device=self.in_dev, channels=self.channels, dtype="int16", samplerate=rate)
                sd.check_output_settings(device=self.out_dev, channels=self.out_channels, dtype="int16",
                                         samplerate=rate)
                return rate
            except Exception:
                continue
        return int(sd.query_devices(self.in_dev, "input")["default_samplerate"])

    def start(self, on_audio: AudioCallback) -> None:
        import sounddevice as sd

        mic_rs, ref_rs = Resampler(self.rate), Resampler(self.rate)  # identical delay on both
        ch, player = self.channel, self.player

        def callback(indata, outdata, frames, time_info, status):  # PortAudio thread
            if status:
                log.debug("audio status: %s", status)
            played = player.read(frames)
            outdata[:] = played[:, None]
            mic = indata[:, ch].astype(np.int16, copy=True)
            m, r = mic_rs.process(mic), ref_rs.process(played)
            if len(m):
                on_audio(m, r)

        self._stream = sd.Stream(
            device=(self.in_dev, self.out_dev),
            channels=(self.channels, self.out_channels),
            samplerate=self.rate,
            dtype="int16",
            blocksize=int(self.rate * self.block_ms / 1000),
            callback=callback,
        )
        self._stream.start()
        ind, outd = self._stream.device
        lat = self._stream.latency
        self.round_trip_ms = int((lat[0] + lat[1]) * 1000 + self.block_ms)
        log.info("Duplex audio: in %r / out %r at %d Hz, latency in %.0f ms / out %.0f ms",
                 sd.query_devices(ind)["name"], sd.query_devices(outd)["name"], self.rate,
                 lat[0] * 1000, lat[1] * 1000)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()
