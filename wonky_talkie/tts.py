"""Text-to-speech backends, and the service that streams them into the Player.

The agent sends `say` with text; wonky-talkie synthesizes it and plays it on its
own output device. That is what keeps barge-in honest: because the service owns
both the speaker and the synthesizer, it has the exact playback reference for
echo cancellation and it can stop mid-sentence without asking anyone.

- kitten:  KittenTTS (ONNX, CPU, ~25-80 MB). The default. Verified on 3.13; see
           INSTALL_HINT for why it needs constraints rather than a plain install.
- piper:   the `piper` binary, raw PCM on stdout. No Python deps.
- command: any shell command, text on stdin -> WAV or raw PCM on stdout
           (e.g. `espeak-ng --stdout`). The zero-install fallback.
- openai:  HTTP to any OpenAI-compatible /v1/audio/speech server, so a Jetson
           can use the desktop's voice (mirrors the `openai` STT backend).

All return 16-bit mono PCM at the backend's own `sample_rate`; the Player
resamples to the device rate, so a backend never has to care about the device.

Long replies are synthesized sentence by sentence: the first audio reaches the
speaker after one sentence instead of after the whole paragraph, and a barge-in
throws away work that was never spoken.
"""

from __future__ import annotations

import ctypes.util
import io
import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.request
import uuid
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import redirect_stdout
from typing import Callable, Protocol

import numpy as np

from .config import TtsConfig

log = logging.getLogger(__name__)

# KittenTTS' own metadata is unhelpful on Python 3.13: misaki 0.9.4 is marked
# `<3.13` on PyPI although upstream has raised it to `<3.14`, and letting pip
# backtrack lands it on spacy 4.0.0.dev3 -> thinc 9 -> blis, which has no 3.13
# wheel and fails to compile. Pinning spacy<4 keeps it on the wheeled stack.
INSTALL_HINT = (
    'pip install --no-deps --ignore-requires-python "misaki==0.9.4"\n'
    '  pip install addict regex num2words "spacy<4" phonemizer-fork espeakng-loader soundfile\n'
    '  pip install --no-deps https://github.com/KittenML/KittenTTS/releases/download/0.8.1/'
    'kittentts-0.8.1-py3-none-any.whl'
)

# KittenTTS' friendly voice names, for a helpful error message. The real list
# comes from the model at load time (config.json carries the aliases).
KITTEN_VOICE_HINT = ("Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo, "
                     "or a raw name like expr-voice-5-m")


def _to_int16(audio: np.ndarray) -> np.ndarray:
    """Flatten to mono int16, accepting float [-1, 1] or already-int16 input."""
    a = np.asarray(audio)
    if a.ndim > 1:  # (1, n) from ONNX, or (n, channels)
        a = a.reshape(a.shape[0], -1)[:, 0] if a.shape[0] > a.shape[-1] else a.reshape(-1)
    if a.dtype == np.int16:
        return a
    if np.issubdtype(a.dtype, np.floating):
        return (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16)
    return a.astype(np.int16)


def decode_audio(data: bytes, fallback_rate: int) -> tuple[np.ndarray, int]:
    """Decode WAV bytes, or treat the bytes as raw int16 at `fallback_rate`."""
    if data[:4] == b"RIFF":
        with wave.open(io.BytesIO(data), "rb") as w:
            if w.getsampwidth() != 2:
                raise RuntimeError(f"expected 16-bit WAV, got {w.getsampwidth() * 8}-bit")
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            if w.getnchannels() > 1:
                pcm = pcm.reshape(-1, w.getnchannels())[:, 0]
            return pcm, w.getframerate()
    return np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2"), fallback_rate


_SENTENCE_END = re.compile(r"(?<=[.!?\u2026])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")


def split_sentences(text: str, max_chars: int, first_max_chars: int,
                    growth: float = 1.5) -> list[str]:
    """Break text into speakable chunks that playback can keep up with.

    Two rules, both learned the hard way:

    Only break where speech naturally pauses - a sentence end, else a clause
    (comma, semicolon, colon), and mid-clause only for a run-on longer than
    `max_chars`. KittenTTS appends a comma to any chunk that does not end in
    punctuation, so a chunk cut mid-clause is *spoken* as a stumble.

    Grow the chunks. The first is small so the first word comes quickly, but
    each chunk must be synthesizable in less time than the already-queued audio
    takes to play, or the speaker runs dry mid-reply. Synthesis produces
    `rtf` seconds of audio per second of work, so sizes may grow by up to `rtf`
    each step; the 1.5 default sits just under KittenTTS' measured ~1.6.
    """
    text = " ".join(text.split())
    if not text:
        return []
    limit = max(1, max_chars)
    budget = max(1, min(first_max_chars, limit))
    chunks: list[str] = []
    pending = ""

    def flush() -> None:
        nonlocal pending, budget
        if pending:
            chunks.append(pending)
            pending = ""
            budget = max(1, min(limit, int(budget * growth)))

    for atom in _atoms(text, limit):
        if not pending:
            pending = atom
        elif len(pending) + 1 + len(atom) <= budget:
            pending = f"{pending} {atom}"
        else:
            flush()
            pending = atom
        if len(pending) >= budget:
            flush()
    flush()
    return chunks


def _atoms(text: str, max_chars: int):
    """Yield the smallest pieces we are willing to end a chunk on."""
    for sentence in _SENTENCE_END.split(text):
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            yield sentence
            continue
        for clause in _CLAUSE_END.split(sentence):
            if not clause:
                continue
            if len(clause) <= max_chars:
                yield clause
            else:  # a run-on with no punctuation to hide behind
                yield from _wrap_words(clause, max_chars)


def _wrap_words(text: str, limit: int) -> list[str]:
    out: list[str] = []
    current = ""
    for word in text.split(" "):
        if current and len(current) + 1 + len(word) > limit:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        out.append(current)
    return out


def _ensure_espeak(preference: str = "") -> str | None:
    """Point phonemizer at a working libespeak-ng. Call *after* importing kittentts.

    The `espeakng_loader` wheel that KittenTTS depends on bundles a library whose
    compiled-in data directory does not exist ("Error processing file
    '/home/runner/work/.../espeak-ng-data/phontab'"), and the native library calls
    exit() on that failure, taking the whole process with it. A system espeak-ng
    knows where its own data lives, so prefer it.

    `misaki` pins the bundled library at import time, which beats the
    PHONEMIZER_ESPEAK_LIBRARY environment variable, so this has to run after
    kittentts is imported and has to call set_library() directly.
    """
    if preference == "off":
        return None
    candidates = [preference] if preference else []
    env = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
    if env:
        candidates.append(env)
    found = ctypes.util.find_library("espeak-ng")
    if found:
        candidates.append(found)
    candidates += [
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",   # Debian/Ubuntu amd64
        "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",  # Jetson
        "/usr/lib/libespeak-ng.so.1",
        "/usr/local/lib/libespeak-ng.so.1",
    ]
    for c in candidates:
        if not c:
            continue
        if "/" in c and not os.path.exists(c):
            continue
        try:
            ctypes.CDLL(c)
        except OSError:
            continue
        try:
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
        except ImportError:
            return None
        EspeakWrapper.set_library(c)
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = c
        log.info("phonemizing with espeak-ng at %s", c)
        return c
    log.warning("no usable system libespeak-ng found; KittenTTS may abort while "
                "phonemizing (install it: sudo apt install espeak-ng)")
    return None


class TtsBackend(Protocol):
    name: str
    sample_rate: int

    def synth(self, text: str) -> np.ndarray: ...

    def warmup(self) -> None: ...

    @property
    def info(self) -> dict: ...


class KittenBackend:
    """KittenTTS: a 15M-parameter ONNX model, fast enough on CPU to stay ahead
    of playback, which leaves the GPU entirely to Whisper and the agent."""

    name = "kitten"

    def __init__(self, cfg: TtsConfig):
        try:
            from kittentts import KittenTTS  # imports misaki, which pins espeak
        except ImportError as e:
            raise RuntimeError(
                "kittentts isn't installed. A plain `pip install <wheel>` fails: the\n"
                "published misaki metadata still says python <3.13, and backtracking\n"
                "then lands on spacy 4.0.0.dev / thinc 9 / blis, which has no wheels.\n"
                "Pinning the stack that does have wheels works on 3.13:\n"
                "  " + INSTALL_HINT
            ) from e
        _ensure_espeak(cfg.kitten.espeak_library)
        k = cfg.kitten
        self.cfg = cfg
        self.k = k
        self.sample_rate = k.sample_rate
        log.info("Loading KittenTTS %s (voice %s)", k.model, k.voice)
        # KittenTTS prints on every generate(); stdout carries our JSON events.
        with redirect_stdout(io.StringIO()):
            self.model = KittenTTS(k.model, cache_dir=k.cache_dir or None)
        voices = list(getattr(self.model, "available_voices", []) or [])
        inner = getattr(self.model, "model", None)
        raw = list(getattr(inner, "available_voices", []) or [])
        aliases = dict(getattr(inner, "voice_aliases", {}) or {})
        if k.voice not in voices and k.voice not in raw and k.voice not in aliases:
            known = ", ".join(voices + raw) or KITTEN_VOICE_HINT
            raise RuntimeError(f"tts.kitten.voice {k.voice!r} is not available. Choose from: {known}")
        self._voices = voices or raw

    def synth(self, text: str) -> np.ndarray:
        with redirect_stdout(io.StringIO()):
            audio = self.model.generate(text, voice=self.k.voice, speed=self.k.speed,
                                        clean_text=self.k.clean_text)
        return _to_int16(audio)

    def warmup(self) -> None:
        self.synth("Ready.")

    @property
    def info(self) -> dict:
        return {"model": self.k.model, "voice": self.k.voice, "voices": self._voices,
                "speed": self.k.speed}


class PiperBackend:
    """The piper binary, one process per chunk, raw int16 on stdout."""

    name = "piper"

    def __init__(self, cfg: TtsConfig):
        self.p = cfg.piper
        self.sample_rate = cfg.piper.sample_rate
        if not self.p.model:
            raise RuntimeError("tts.piper.model is required (path to a .onnx voice)")

    def synth(self, text: str) -> np.ndarray:
        cmd = [self.p.binary, "-m", self.p.model, "--output-raw"]
        if self.p.speaker:
            cmd += ["--speaker", str(self.p.speaker)]
        if self.p.extra_args:
            cmd += shlex.split(self.p.extra_args)
        out = _run(cmd, text.encode(), self.p.timeout_s, shell=False)
        pcm, rate = decode_audio(out, self.sample_rate)
        self.sample_rate = rate
        return pcm

    def warmup(self) -> None:
        self.synth("Ready.")

    @property
    def info(self) -> dict:
        return {"model": self.p.model, "binary": self.p.binary}


class CommandBackend:
    """Any shell command: text on stdin, WAV or raw PCM on stdout.

    This is the escape hatch that needs nothing installed, e.g.
    `espeak-ng --stdout` or a script wrapping a cloud voice.
    """

    name = "command"

    def __init__(self, cfg: TtsConfig):
        self.c = cfg.command
        self.sample_rate = cfg.command.sample_rate
        if not self.c.command:
            raise RuntimeError("tts.command.command is required, e.g. \"espeak-ng --stdout\"")

    def synth(self, text: str) -> np.ndarray:
        out = _run(self.c.command, text.encode(), self.c.timeout_s, shell=True)
        pcm, rate = decode_audio(out, self.sample_rate)
        self.sample_rate = rate  # a WAV header overrides the configured rate
        return pcm

    def warmup(self) -> None:
        self.synth("Ready.")

    @property
    def info(self) -> dict:
        return {"command": self.c.command}


class HttpTtsBackend:
    """An OpenAI-compatible /v1/audio/speech server (openedai-speech, Kokoro,
    or another wonky-talkie host), so a Jetson can borrow the desktop's voice."""

    name = "openai"

    def __init__(self, cfg: TtsConfig):
        self.h = cfg.http
        self.sample_rate = cfg.http.sample_rate

    def synth(self, text: str) -> np.ndarray:
        body = json.dumps({"model": self.h.model, "input": text, "voice": self.h.voice,
                           "response_format": self.h.response_format,
                           "speed": self.h.speed}).encode()
        headers = {"Content-Type": "application/json"}
        if self.h.api_key:
            headers["Authorization"] = f"Bearer {self.h.api_key}"
        req = urllib.request.Request(self.h.url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=self.h.timeout_s) as r:
            data = r.read()
        pcm, rate = decode_audio(data, self.sample_rate)
        self.sample_rate = rate
        return pcm

    def warmup(self) -> None:
        pass  # don't make a network call (and a bill) at startup

    @property
    def info(self) -> dict:
        return {"url": self.h.url, "model": self.h.model, "voice": self.h.voice}


class NoTtsBackend:
    """tts.backend = "none": `say` is refused, `play` still works."""

    name = "none"
    sample_rate = 24000

    def __init__(self, cfg: TtsConfig | None = None):
        pass

    def synth(self, text: str) -> np.ndarray:
        raise RuntimeError("tts.backend is \"none\"; send `play` with your own audio instead")

    def warmup(self) -> None:
        pass

    @property
    def info(self) -> dict:
        return {}


class FakeTtsBackend:
    """Test backend: a tone whose length is proportional to the text."""

    name = "fake"

    def __init__(self, sample_rate: int = 24000, ms_per_char: float = 20.0,
                 fn: Callable[[str], np.ndarray] | None = None):
        self.sample_rate = sample_rate
        self.ms_per_char = ms_per_char
        self.fn = fn
        self.calls: list[str] = []

    def synth(self, text: str) -> np.ndarray:
        self.calls.append(text)
        if self.fn is not None:
            return _to_int16(self.fn(text))
        n = max(1, int(self.sample_rate * self.ms_per_char * max(1, len(text)) / 1000))
        t = np.arange(n) / self.sample_rate
        return (np.sin(2 * np.pi * 220 * t) * 0.3 * 32767).astype(np.int16)

    def warmup(self) -> None:
        pass

    @property
    def info(self) -> dict:
        return {"ms_per_char": self.ms_per_char}


def _run(cmd, stdin: bytes, timeout_s: float, shell: bool) -> bytes:
    try:
        p = subprocess.run(cmd, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout_s, shell=shell, check=False)
    except FileNotFoundError as e:
        raise RuntimeError(f"TTS command not found: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"TTS command timed out after {timeout_s}s") from e
    if p.returncode != 0 or not p.stdout:
        err = p.stderr.decode(errors="replace").strip()[:400]
        raise RuntimeError(f"TTS command failed (exit {p.returncode}): {err}")
    return p.stdout


def make_backend(cfg: TtsConfig) -> TtsBackend:
    if cfg.backend == "kitten":
        return KittenBackend(cfg)
    if cfg.backend == "piper":
        return PiperBackend(cfg)
    if cfg.backend == "command":
        return CommandBackend(cfg)
    if cfg.backend == "openai":
        return HttpTtsBackend(cfg)
    if cfg.backend == "none":
        return NoTtsBackend(cfg)
    raise ValueError(f"unknown tts.backend {cfg.backend!r}")


class TtsService:
    """Turns `say` into queued playback, off the engine thread.

    One worker thread synthesizes; the Player is thread-safe, so audio goes
    straight to it. `Player.add_audio` returns False once an item is gone,
    which is exactly the barge-in signal: if the engine stopped playback, the
    remaining sentences are never synthesized.
    """

    def __init__(self, cfg: TtsConfig, backend: TtsBackend, player, emit: Callable[..., None] | None = None):
        self.cfg = cfg
        self.backend = backend
        self.player = player
        self.emit = emit or (lambda *a, **k: None)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self._lock = threading.Lock()
        self._cancelled: set[str] = set()
        self._futures: dict[str, Future] = {}

    @property
    def info(self) -> dict:
        return {"backend": self.backend.name, "sample_rate": self.backend.sample_rate,
                **self.backend.info}

    def say(self, text: str, then: str = "listen", timeout_s: float | None = None,
            client: str | None = None, playback_id: str | None = None) -> str:
        """Queue text for synthesis and playback. Returns the playback id."""
        pid = playback_id or f"pb_{uuid.uuid4().hex[:8]}"
        chunks = split_sentences(text, self.cfg.max_chars, self.cfg.first_chunk_chars,
                                 self.cfg.chunk_growth)
        self.player.start_item(pid, self.backend.sample_rate, text, then, timeout_s, client)
        with self._lock:
            self._futures[pid] = self._pool.submit(self._work, pid, chunks, client)
        return pid

    def wait(self, pid: str, timeout: float | None = None) -> None:
        """Block until everything for `pid` has been synthesized and queued."""
        with self._lock:
            fut = self._futures.get(pid)
        if fut is not None:
            fut.result(timeout)

    def cancel(self, pid: str) -> None:
        with self._lock:
            self._cancelled.add(pid)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _work(self, pid: str, chunks: list[str], client: str | None) -> None:
        t0 = time.monotonic()
        spoken = 0
        complete = False
        try:
            for i, chunk in enumerate(chunks):
                with self._lock:
                    if pid in self._cancelled:
                        return
                audio = self.backend.synth(chunk)
                if not self.player.add_audio(audio, pid, client):
                    log.debug("tts %s: dropped after %d chunk(s) (playback gone)", pid, i)
                    return  # barged in, stopped, or superseded
                spoken += len(audio)
                if i == 0:
                    log.info("tts %s: first audio in %.0f ms (%d chunk(s))",
                             pid, (time.monotonic() - t0) * 1000, len(chunks))
            complete = True
        except Exception as e:  # a bad voice, a dead binary, a 500
            log.error("tts %s failed: %s", pid, e)
            self.emit("tts_failed", playback_id=pid, message=str(e), _to=client)
        finally:
            with self._lock:
                self._cancelled.discard(pid)
                self._futures.pop(pid, None)
            self.player.end_item(pid, client)
            # Only a run that finished says anything useful about throughput: a
            # cancelled one spent its wall clock on audio nobody heard.
            if spoken and complete:
                audio_s = spoken / max(1, self.backend.sample_rate)
                wall = time.monotonic() - t0
                log.info("tts %s: %.1fs of audio in %.1fs (%.2fx realtime)",
                         pid, audio_s, wall, audio_s / wall if wall else 0.0)
            elif spoken:
                log.info("tts %s: cut off after %.1fs of audio",
                         pid, spoken / max(1, self.backend.sample_rate))
