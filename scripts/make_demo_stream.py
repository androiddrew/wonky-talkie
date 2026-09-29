"""Build tests/data/demo_stream.wav: two wake-word sessions from real speech clips.

Lets you check the whole pipeline (VAD, endpointing, STT, your agent) without a mic:

    python -m voice_listener -c configs/desktop.toml --print-events \
        --input-file tests/data/demo_stream.wav --wake-at 1.95 --wake-at 13.80

Timeline (s): 1.0 wake phrase ends 1.95 | command 2.05-4.75 | pause | reply 8.25-9.85 |
              second wake ends 13.80 | command 13.90-16.60 | pause | speech again 19.10-21.80
The speech clips come from the openWakeWord repo's test data.
"""

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from harness import command, noise, short_reply, wake_phrase  # noqa: E402

parts = [noise(1.0), wake_phrase(), noise(0.1), command(), noise(3.5), short_reply(), noise(3.0),
         wake_phrase(), noise(0.1), command(), noise(2.5), command(), noise(4.0)]
out = ROOT / "tests" / "data" / "demo_stream.wav"
with wave.open(str(out), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(np.concatenate(parts).tobytes())
print(f"wrote {out}")
