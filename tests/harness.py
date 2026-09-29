"""Test harness: drive the Engine with synthetic audio built from real speech clips,
using the real Silero VAD, a scripted wake detector and a fake STT backend."""

from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice_listener.aec import EchoStage, SpeexAEC, load_speexdsp  # noqa: E402
from voice_listener.audio import SAMPLE_RATE, EchoRoom, FileSource, Player  # noqa: E402
from voice_listener.config import Config  # noqa: E402
from voice_listener.engine import Engine  # noqa: E402
from voice_listener.stt import FakeBackend  # noqa: E402
from voice_listener.vad import SileroVAD  # noqa: E402
from voice_listener.wake import ScriptedWakeDetector  # noqa: E402

DATA = ROOT / "tests" / "data"
_rng = np.random.default_rng(0)


def noise(seconds: float, db: float = -65.0) -> np.ndarray:
    amp = 32768 * 10 ** (db / 20)
    return (_rng.standard_normal(int(seconds * SAMPLE_RATE)) * amp).astype(np.int16)


def clip(name: str, start: float = 0.0, end: float | None = None) -> np.ndarray:
    a = FileSource.load(str(DATA / name))
    s = int(start * SAMPLE_RATE)
    e = len(a) if end is None else int(end * SAMPLE_RATE)
    return a[s:e]


def wake_phrase() -> np.ndarray:
    return clip("hey_mycroft_test.wav")


def command() -> np.ndarray:
    """~2.4 s of speech, isolated by silence in the source clip."""
    return clip("office_lights.wav", 26.0, 28.7)


def short_reply() -> np.ndarray:
    return clip("office_lights.wav", 31.0, 32.6)


def agent_voice(seconds: float = 8.0) -> np.ndarray:
    """Continuous speech used as the agent's TTS output."""
    return clip("office_lights.wav", 2.0, 2.0 + seconds)


class Harness:
    def __init__(self, cfg: Config | None = None, wake_at: list[float] | None = None,
                 stt_fn=None, stt_delay: float = 0.0, playback: bool = False, echo: bool = True,
                 aec: bool = True, room: dict | None = None):
        """playback: the listener plays agent audio through a simulated room
        (EchoRoom) whose echo reaches the mic; aec: cancel it with SpeexDSP."""
        self.cfg = cfg or Config()
        self.cfg.debug.level_interval_ms = 0
        self.events: queue.Queue = queue.Queue()
        self.log: list[dict] = []
        vad = SileroVAD(str(ROOT / "models" / "silero_vad.onnx"))
        self.stt = FakeBackend(stt_fn, delay_s=stt_delay)
        self.player = self.room = None
        echo_stage = None
        if playback:
            self.player = Player(SAMPLE_RATE)
            self.room = EchoRoom(self.player, echo=echo, **(room or {}))
            canceller = None
            if aec:
                lib = load_speexdsp()
                if lib is None:
                    import unittest

                    raise unittest.SkipTest("libspeexdsp not installed (apt install libspeexdsp1)")
                a = self.cfg.aec
                canceller = SpeexAEC(lib, a.filter_ms, a.residual_suppression, a.noise_suppress_db)
            echo_stage = EchoStage(canceller, self.cfg.aec.hold_ms)
        self.engine = Engine(self.cfg, vad, ScriptedWakeDetector(wake_at or []), self.stt, self.events.put,
                             player=self.player, echo=echo_stage)
        self.engine.start()
        self.fed = 0

    def close(self) -> None:
        self.engine.stop()

    def feed(self, audio: np.ndarray, block: int = 512) -> None:
        """Feed audio and wait until the engine has processed it."""
        if self.room is None:
            for i in range(0, len(audio), block):
                self.engine.feed_audio(audio[i : i + block])
            self.fed += len(audio)
            self._wait_engine()
        else:
            # Lock-step, like a real audio device: the speaker output for the next
            # block depends on what the engine decided (duck/stop) after this one.
            for i in range(0, len(audio), block):
                b = audio[i : i + block]
                self.engine.feed_audio(*self.room.process(b))
                self.fed += len(b)
                self._wait_engine()
        self._drain()

    def _wait_engine(self) -> None:
        target = self.fed - (self.fed % 512)
        deadline = time.monotonic() + 30
        while self.engine.pos < target or not self.engine.inbox.empty():
            if time.monotonic() > deadline:
                raise TimeoutError("engine stalled")
            time.sleep(0.0005)

    def play(self, audio: np.ndarray, text: str = "", then: str = "listen", pid: str = "pb1") -> None:
        """What the server does for: play{...} + binary audio + play_end."""
        self.player.start_item(pid, SAMPLE_RATE, text, then, None, "test")
        self.player.add_audio(audio, pid, "test")
        self.player.end_item(pid, "test")

    def command(self, cmd: dict) -> None:
        self.engine.post_command(cmd, "test")
        time.sleep(0.02)
        self._drain()

    def _drain(self) -> None:
        while True:
            try:
                self.log.append(self.events.get_nowait())
            except queue.Empty:
                return

    def wait_for(self, type_: str, timeout: float = 5.0, **match) -> dict:
        deadline = time.monotonic() + timeout
        seen = 0
        while True:
            self._drain()
            for ev in self.log[seen:]:
                if ev["type"] == type_ and all(ev.get(k) == v for k, v in match.items()):
                    return ev
            if time.monotonic() > deadline:
                raise AssertionError(f"no {type_} {match} in {self.types()}")
            time.sleep(0.01)

    def types(self) -> list[str]:
        self._drain()
        return [e["type"] for e in self.log if e["type"] != "state_changed"]

    def states(self) -> list[str]:
        self._drain()
        return [e["to"] for e in self.log if e["type"] == "state_changed"]
