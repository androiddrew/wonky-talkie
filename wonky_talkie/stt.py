"""Speech-to-text backends.

- faster_whisper: in-process CTranslate2. Best on the 4090.
- whisper_cpp:    HTTP to a whisper.cpp `whisper-server`. Best on Jetson Orin,
                  where it builds cleanly with CUDA and needs no PyTorch.
- openai:         HTTP to any OpenAI-compatible /v1/audio/transcriptions server
                  (e.g. speaches on the desktop, for a Jetson "satellite").

All take 16 kHz mono int16 and return an SttResult.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from .audio import SAMPLE_RATE, to_wav_bytes
from .config import SttConfig

log = logging.getLogger(__name__)


@dataclass
class SttResult:
    text: str
    language: str | None = None
    extra: dict = field(default_factory=dict)


class SttBackend(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, partial: bool = False) -> SttResult: ...

    def warmup(self) -> None: ...


class FasterWhisperBackend:
    name = "faster_whisper"

    def __init__(self, cfg: SttConfig):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError("faster-whisper isn't installed: pip install faster-whisper") from e
        fw = cfg.faster_whisper
        self.cfg = cfg
        self.fw = fw
        log.info("Loading faster-whisper %s on %s (%s)", fw.model, fw.device, fw.compute_type)
        self.model = WhisperModel(
            fw.model,
            device=fw.device,
            compute_type=fw.compute_type,
            cpu_threads=fw.cpu_threads,
            download_root=fw.download_root or None,
        )

    def transcribe(self, audio: np.ndarray, partial: bool = False) -> SttResult:
        x = audio.astype(np.float32) / 32768.0
        kwargs = dict(
            language=self.cfg.language or None,
            beam_size=self.fw.partial_beam_size if partial else self.fw.beam_size,
            vad_filter=False,  # already gated by our own VAD
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=self.cfg.initial_prompt or None,
            hotwords=self.fw.hotwords or None,
        )
        if partial:
            kwargs["temperature"] = 0.0
        segments, info = self.model.transcribe(x, **kwargs)
        segs = list(segments)
        # Drop segments Whisper itself thinks are non-speech.
        kept = [s for s in segs if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
        text = " ".join(s.text.strip() for s in kept).strip()
        extra = {"no_speech_prob": round(max((s.no_speech_prob for s in segs), default=0.0), 3)}
        return SttResult(text=text, language=info.language, extra=extra)

    def warmup(self) -> None:
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.int16), partial=True)


def _multipart(fields: dict[str, str], file_field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode()
    )
    parts.append(data)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class HttpBackend:
    """whisper.cpp server (`flavor="whisper_cpp"`) or OpenAI-compatible (`"openai"`)."""

    def __init__(self, cfg: SttConfig, flavor: str):
        self.cfg = cfg
        self.http = cfg.http
        self.flavor = flavor
        self.name = flavor

    def transcribe(self, audio: np.ndarray, partial: bool = False) -> SttResult:
        fields = {"response_format": "json"}
        if self.cfg.language:
            fields["language"] = self.cfg.language
        if self.cfg.initial_prompt:
            fields["prompt"] = self.cfg.initial_prompt
        if self.flavor == "whisper_cpp":
            fields["temperature"] = "0.0"
            fields["no_timestamps"] = "true"
        else:
            fields["model"] = self.http.model or "whisper-1"
        body, ctype = _multipart(fields, "file", "audio.wav", to_wav_bytes(audio))
        req = urllib.request.Request(self.http.url, data=body, method="POST")
        req.add_header("Content-Type", ctype)
        if self.http.api_key:
            req.add_header("Authorization", f"Bearer {self.http.api_key}")
        # Local servers shouldn't go through an HTTP proxy set in the environment.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=self.http.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if "error" in payload:
            raise RuntimeError(f"STT server error: {payload['error']}")
        text = " ".join(str(payload.get("text", "")).split())
        return SttResult(text=text, language=payload.get("language") or self.cfg.language or None)

    def warmup(self) -> None:
        t = time.monotonic()
        self.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.int16))
        log.info("STT server at %s reachable (%.0f ms)", self.http.url, (time.monotonic() - t) * 1000)


class NoSttBackend:
    """Transcription disabled: utterances are still captured and emitted as audio."""

    name = "none"

    def transcribe(self, audio: np.ndarray, partial: bool = False) -> SttResult:
        return SttResult(text="")

    def warmup(self) -> None:
        pass


class FakeBackend:
    """Test backend: returns fn(audio) or a description of the audio length."""

    name = "fake"

    def __init__(self, fn: Callable[[np.ndarray], str] | None = None, delay_s: float = 0.0):
        self.fn = fn
        self.delay_s = delay_s

    def transcribe(self, audio: np.ndarray, partial: bool = False) -> SttResult:
        if self.delay_s:
            time.sleep(self.delay_s)
        text = self.fn(audio) if self.fn else f"[{len(audio) / SAMPLE_RATE:.2f}s of audio]"
        return SttResult(text=text, language="en")

    def warmup(self) -> None:
        pass


def make_backend(cfg: SttConfig) -> SttBackend:
    if cfg.backend == "faster_whisper":
        return FasterWhisperBackend(cfg)
    if cfg.backend == "whisper_cpp":
        return HttpBackend(cfg, "whisper_cpp")
    if cfg.backend == "openai":
        return HttpBackend(cfg, "openai")
    return NoSttBackend()
