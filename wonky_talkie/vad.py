"""Silero VAD (ONNX, no PyTorch) and a hysteresis endpointer built on it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VAD_CHUNK = 512  # Silero v5 at 16 kHz takes exactly 512 samples (32 ms)
_CONTEXT = 64  # samples of the previous chunk that Silero v5 expects prepended


class SileroVAD:
    """Streaming Silero VAD v5 via onnxruntime.

    Mirrors the reference OnnxWrapper in snakers4/silero-vad, minus the torch
    dependency (which matters on Jetson, where torch wheels are a chore).
    """

    def __init__(self, model_path: str, threads: int = 1):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = threads
        self._sess = ort.InferenceSession(model_path, sess_options=opts,
                                          providers=["CPUExecutionProvider"])
        self._sr = np.array(16000, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        if len(chunk) != VAD_CHUNK:
            raise ValueError(f"Silero VAD needs {VAD_CHUNK} samples, got {len(chunk)}")
        x = (chunk.astype(np.float32) / 32768.0)[None, :]
        x = np.concatenate([self._context, x], axis=1)
        out, self._state = self._sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -_CONTEXT:]
        return float(out[0, 0])


@dataclass
class VadEvent:
    kind: str  # "onset" | "offset"
    sample: int  # onset: first speech sample; offset: first sample of the trailing silence


class Endpointer:
    """Turns per-chunk speech probabilities into onset / offset events.

    Onset: `min_speech_ms` of speech (prob >= threshold) without a clear
    silence frame (prob < neg_threshold) in between. Offset: `end_silence_ms`
    of continuous silence after an onset.
    """

    def __init__(self, threshold: float, neg_threshold: float, min_speech_ms: int,
                 end_silence_ms: int, sample_rate: int = 16000):
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.chunk_ms = VAD_CHUNK * 1000 / sample_rate
        self.min_speech_ms = min_speech_ms
        self.end_silence_ms = end_silence_ms
        self.reset()

    def reset(self) -> None:
        self.in_speech = False
        self._speech_chunks = 0
        self._speech_start = 0
        self._silence_chunks = 0
        self._silence_start = 0

    def force_speech(self) -> None:
        """Enter the in-speech state now (capture started by something other than
        our own onset, e.g. barge-in), so the next offset is measured from here."""
        self.in_speech = True
        self._silence_chunks = 0

    def update(self, prob: float, chunk_start: int) -> VadEvent | None:
        if not self.in_speech:
            if prob >= self.threshold:
                if self._speech_chunks == 0:
                    self._speech_start = chunk_start
                self._speech_chunks += 1
                if self._speech_chunks * self.chunk_ms >= self.min_speech_ms:
                    self.in_speech = True
                    self._silence_chunks = 0
                    return VadEvent("onset", self._speech_start)
            elif prob < self.neg_threshold:
                self._speech_chunks = 0
            return None

        if prob < self.neg_threshold:
            if self._silence_chunks == 0:
                self._silence_start = chunk_start
            self._silence_chunks += 1
            if self._silence_chunks * self.chunk_ms >= self.end_silence_ms:
                self.in_speech = False
                self._speech_chunks = 0
                return VadEvent("offset", self._silence_start)
        elif prob >= self.threshold:
            self._silence_chunks = 0
        return None


class SustainedSpeech:
    """Fires once when speech has been continuously present for `min_ms`.

    Used for barge-in, where we want a stricter test than normal onset.
    """

    def __init__(self, threshold: float, min_ms: int, sample_rate: int = 16000):
        self.threshold = threshold
        self.chunk_ms = VAD_CHUNK * 1000 / sample_rate
        self.min_ms = min_ms
        self.reset()

    def reset(self) -> None:
        self._chunks = 0
        self._start = 0

    def update(self, prob: float, chunk_start: int) -> int | None:
        """Return the start sample of the run when it becomes long enough."""
        if prob >= self.threshold:
            if self._chunks == 0:
                self._start = chunk_start
            self._chunks += 1
            if self._chunks * self.chunk_ms >= self.min_ms:
                start = self._start
                self.reset()
                return start
        else:
            self._chunks = 0
        return None
