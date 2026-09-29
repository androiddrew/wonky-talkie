"""Fetch the openWakeWord models (ONNX) and check the Silero VAD model is present.

    python scripts/download_models.py                  # hey_jarvis (default)
    python scripts/download_models.py alexa hey_mycroft

Whisper models are fetched separately:
  - faster-whisper downloads on first run (to ~/.cache/huggingface)
  - whisper.cpp: ./models/download-ggml-model.sh small.en   (inside the whisper.cpp repo)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SILERO_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"


def main() -> int:
    names = sys.argv[1:] or ["hey_jarvis"]

    vad = ROOT / "models" / "silero_vad.onnx"
    if vad.exists():
        print(f"Silero VAD: {vad} ({vad.stat().st_size // 1024} KB)")
    else:
        import requests

        print(f"Downloading Silero VAD -> {vad}")
        vad.parent.mkdir(exist_ok=True)
        r = requests.get(SILERO_URL, timeout=60)
        r.raise_for_status()
        vad.write_bytes(r.content)

    import openwakeword
    from openwakeword.utils import download_models

    download_models(model_names=names)
    onnx = [p for p in openwakeword.get_pretrained_model_paths("onnx")
            if any(n in Path(p).name for n in names)]
    res = Path(onnx[0]).parent if onnx else None
    for p in onnx + [str(res / "melspectrogram.onnx"), str(res / "embedding_model.onnx")]:
        print(("ok      " if Path(p).exists() else "MISSING ") + p)

    # Smoke test: load the models and run one frame of silence.
    import numpy as np
    from openwakeword.model import Model

    m = Model(wakeword_models=names, inference_framework="onnx")
    print("openWakeWord loaded:", m.predict(np.zeros(1280, dtype=np.int16)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
