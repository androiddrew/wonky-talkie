"""Wake word detection.

`OpenWakeWordDetector` wraps openWakeWord with the ONNX runtime (no tflite,
which has no wheels for Python 3.12). Detectors share one small interface so
the engine doesn't care which one is in use:

    process(chunk_1280_int16) -> list[(model_name, score)]
    reset()
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

WAKE_CHUNK = 1280  # openWakeWord's native 80 ms frame


class WakeDetector(Protocol):
    names: list[str]

    def process(self, chunk: np.ndarray) -> list[tuple[str, float]]: ...

    def reset(self) -> None: ...


class OpenWakeWordDetector:
    def __init__(self, models: list[str], auto_download: bool = True,
                 melspec_model_path: str = "", embedding_model_path: str = ""):
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as e:
            raise RuntimeError(
                "openWakeWord isn't installed. On Python 3.12 install it without its "
                "tflite dependency: pip install --no-deps openwakeword && "
                "pip install onnxruntime scipy scikit-learn tqdm requests"
            ) from e

        if auto_download:
            self._ensure_models(openwakeword, models, bool(melspec_model_path and embedding_model_path))

        kwargs = {}
        if melspec_model_path:
            kwargs["melspec_model_path"] = melspec_model_path
        if embedding_model_path:
            kwargs["embedding_model_path"] = embedding_model_path
        self._model = Model(wakeword_models=list(models), inference_framework="onnx", **kwargs)
        self.names = list(self._model.models.keys())
        log.info("openWakeWord loaded: %s", ", ".join(self.names))

    @staticmethod
    def _ensure_models(openwakeword, models: list[str], have_feature_models: bool) -> None:
        from openwakeword.utils import download_models

        pretrained = [m for m in models if not os.path.exists(m)]
        onnx_paths = openwakeword.get_pretrained_model_paths("onnx")
        res_dir = os.path.dirname(onnx_paths[0])
        missing = [m for m in pretrained
                   if not any(m.replace(" ", "_") in os.path.basename(p) and os.path.exists(p)
                              for p in onnx_paths)]
        features_missing = not have_feature_models and not all(
            os.path.exists(os.path.join(res_dir, f)) for f in ("melspectrogram.onnx", "embedding_model.onnx"))
        if missing or features_missing:
            log.info("Downloading openWakeWord models: %s", missing or "(feature models)")
            # download_models([]) would fetch *every* model; a name that matches
            # nothing fetches just the shared feature models.
            download_models(model_names=missing or ["__feature_models_only__"])

    def process(self, chunk: np.ndarray) -> list[tuple[str, float]]:
        scores = self._model.predict(chunk)
        return [(name, float(score)) for name, score in scores.items()]

    def reset(self) -> None:
        self._model.reset()


class NullWakeDetector:
    """Never fires. Used when wake.enabled = false (push-to-talk via `listen`)."""

    names: list[str] = []

    def process(self, chunk: np.ndarray) -> list[tuple[str, float]]:
        return []

    def reset(self) -> None:
        pass


class ScriptedWakeDetector:
    """Fires at given stream positions (seconds). For tests and demos without models."""

    def __init__(self, times_s: list[float], name: str = "scripted", sample_rate: int = 16000):
        self.names = [name]
        self._pending = sorted(int(t * sample_rate) for t in times_s)
        self._pos = 0

    def process(self, chunk: np.ndarray) -> list[tuple[str, float]]:
        self._pos += len(chunk)
        if self._pending and self._pos >= self._pending[0]:
            self._pending.pop(0)
            return [(self.names[0], 1.0)]
        return [(self.names[0], 0.0)]

    def reset(self) -> None:
        pass
