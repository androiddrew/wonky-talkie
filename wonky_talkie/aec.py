"""Acoustic echo cancellation with SpeexDSP (via ctypes, no build step).

The listener plays the agent's TTS itself, so it knows exactly what went to the
speaker. That signal is the "reference": the echo canceller learns the
speaker -> room -> mic path and subtracts the predicted echo from the mic, and
the preprocessor then suppresses what's left.

Needs the system library:  sudo apt install libspeexdsp1
(on Ubuntu 24.04 desktop and JetPack 7.2 it is often already installed).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os

import numpy as np

log = logging.getLogger(__name__)

FRAME = 256  # 16 ms at 16 kHz; two frames per 512-sample VAD chunk

# speex_echo.h / speex_preprocess.h
_ECHO_SET_SAMPLING_RATE = 24
_PP_SET_DENOISE = 0
_PP_SET_NOISE_SUPPRESS = 18
_PP_SET_ECHO_SUPPRESS = 20
_PP_SET_ECHO_SUPPRESS_ACTIVE = 22
_PP_SET_ECHO_STATE = 24

_I16P = ctypes.POINTER(ctypes.c_int16)


def load_speexdsp(path: str = "") -> ctypes.CDLL | None:
    candidates = [path, os.environ.get("WONKY_TALKIE_SPEEXDSP", ""),
                  ctypes.util.find_library("speexdsp") or "",
                  "libspeexdsp.so.1", "libspeexdsp.so", "libspeexdsp.dylib"]
    for c in candidates:
        if not c:
            continue
        try:
            lib = ctypes.CDLL(c)
        except OSError:
            continue
        lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.speex_echo_state_init.restype = ctypes.c_void_p
        lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
        lib.speex_echo_state_reset.argtypes = [ctypes.c_void_p]
        lib.speex_echo_cancellation.argtypes = [ctypes.c_void_p, _I16P, _I16P, _I16P]
        lib.speex_echo_cancellation.restype = None
        lib.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.speex_echo_ctl.restype = ctypes.c_int
        lib.speex_preprocess_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.speex_preprocess_state_init.restype = ctypes.c_void_p
        lib.speex_preprocess_state_destroy.argtypes = [ctypes.c_void_p]
        lib.speex_preprocess_run.argtypes = [ctypes.c_void_p, _I16P]
        lib.speex_preprocess_run.restype = ctypes.c_int
        lib.speex_preprocess_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.speex_preprocess_ctl.restype = ctypes.c_int
        return lib
    return None


class SpeexAEC:
    """Frame-by-frame echo canceller + residual echo suppressor (16 kHz, 256 samples)."""

    def __init__(self, lib: ctypes.CDLL, filter_ms: int = 300, residual_suppression: bool = True,
                 noise_suppress_db: int = -10, echo_suppress_db: int = -40,
                 echo_suppress_active_db: int = -15, sample_rate: int = 16000):
        self._lib = lib
        self._echo = lib.speex_echo_state_init(FRAME, int(sample_rate * filter_ms / 1000))
        rate = ctypes.c_int(sample_rate)
        lib.speex_echo_ctl(self._echo, _ECHO_SET_SAMPLING_RATE, ctypes.byref(rate))
        self._pp = None
        if residual_suppression:
            # Speex only applies its residual-echo gains with its denoiser on,
            # so keep denoising but mild.
            self._pp = lib.speex_preprocess_state_init(FRAME, sample_rate)
            for req, val in ((_PP_SET_DENOISE, 1),
                             (_PP_SET_NOISE_SUPPRESS, noise_suppress_db),
                             (_PP_SET_ECHO_SUPPRESS, echo_suppress_db),
                             (_PP_SET_ECHO_SUPPRESS_ACTIVE, echo_suppress_active_db)):
                v = ctypes.c_int(val)
                lib.speex_preprocess_ctl(self._pp, req, ctypes.byref(v))
            lib.speex_preprocess_ctl(self._pp, _PP_SET_ECHO_STATE, ctypes.c_void_p(self._echo))
        self._out = np.zeros(FRAME, dtype=np.int16)

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Cancel echo of `ref` (what the speaker played) from `mic`. 256 samples each."""
        mic = np.ascontiguousarray(mic, dtype=np.int16)
        ref = np.ascontiguousarray(ref, dtype=np.int16)
        out = np.empty(FRAME, dtype=np.int16)
        self._lib.speex_echo_cancellation(self._echo, mic.ctypes.data_as(_I16P),
                                          ref.ctypes.data_as(_I16P), out.ctypes.data_as(_I16P))
        if self._pp is not None:
            self._lib.speex_preprocess_run(self._pp, out.ctypes.data_as(_I16P))
        return out

    def reset(self) -> None:
        self._lib.speex_echo_state_reset(self._echo)

    def close(self) -> None:
        if self._pp is not None:
            self._lib.speex_preprocess_state_destroy(self._pp)
            self._pp = None
        if self._echo is not None:
            self._lib.speex_echo_state_destroy(self._echo)
            self._echo = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class EchoStage:
    """Sits in front of the engine: aligns mic/reference into 256-sample frames,
    runs the canceller only while there is (or recently was) something playing,
    and keeps rough statistics of how much echo it removed."""

    def __init__(self, aec: SpeexAEC | None, hold_ms: int, sample_rate: int = 16000):
        self.aec = aec
        self.hold = int(sample_rate * hold_ms / 1000)
        self._mic = np.zeros(0, dtype=np.int16)
        self._ref = np.zeros(0, dtype=np.int16)
        self._since_ref = 10**9  # samples since the reference last had signal
        self.e_in = 0.0
        self.e_out = 0.0

    def reset_stats(self) -> None:
        self.e_in = self.e_out = 0.0

    def erle_db(self) -> float | None:
        """Echo return loss enhancement over the frames processed since reset_stats()."""
        if self.e_in <= 0 or self.e_out <= 0:
            return None
        return round(float(10 * np.log10(self.e_in / self.e_out)), 1)

    def process(self, mic: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (cleaned mic, the matching reference samples)."""
        self._mic = np.concatenate([self._mic, mic])
        self._ref = np.concatenate([self._ref, ref])
        n = min(len(self._mic), len(self._ref)) // FRAME * FRAME
        if n == 0:
            return np.zeros(0, dtype=np.int16), np.zeros(0, dtype=np.int16)
        out = []
        for i in range(0, n, FRAME):
            m, r = self._mic[i : i + FRAME], self._ref[i : i + FRAME]
            if np.max(np.abs(r)) > 8:
                self._since_ref = 0
            else:
                self._since_ref += FRAME
            if self.aec is not None and self._since_ref <= self.hold:
                o = self.aec.process(m, r)
                self.e_in += float(np.sum(m.astype(np.float64) ** 2))
                self.e_out += float(np.sum(o.astype(np.float64) ** 2))
                out.append(o)
            else:
                out.append(m)
        ref_out = self._ref[:n]
        self._mic, self._ref = self._mic[n:], self._ref[n:]
        return np.concatenate(out), ref_out
