import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice_listener.audio import SAMPLE_RATE, Resampler, RingBuffer  # noqa: E402
from voice_listener.config import load_config  # noqa: E402
from voice_listener.textutil import is_hallucination, strip_wake_phrase  # noqa: E402
from voice_listener.vad import Endpointer, SileroVAD  # noqa: E402

PHRASES = ["hey jarvis", "jarvis"]


class TextUtil(unittest.TestCase):
    def test_strip(self):
        cases = {
            "Hey Jarvis, turn off the lights.": "Turn off the lights.",
            "Hey, Jervis. What time is it?": "What time is it?",
            "Jarvis turn it up": "Turn it up",
            "jarvis, jarvis, are you there?": "Are you there?",
            "Hey Jarvis.": "",
            "Turn off the lights, Jarvis.": "Turn off the lights, Jarvis.",
            "Hey there, how are you?": "Hey there, how are you?",
            "Harvest the data": "Harvest the data",
        }
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertEqual(strip_wake_phrase(text, PHRASES), want)

    def test_hallucination(self):
        drop = ["thank you", "thanks for watching"]
        self.assertTrue(is_hallucination("", drop))
        self.assertTrue(is_hallucination(" Thank you. ", drop))
        self.assertTrue(is_hallucination("Thanks for watching!", drop))
        self.assertFalse(is_hallucination("Thank you, that's all.", drop))


class Audio(unittest.TestCase):
    def test_ring_buffer_wraps(self):
        rb = RingBuffer(1.0)
        data = np.arange(40000, dtype=np.int32).astype(np.int16)
        for i in range(0, len(data), 700):
            rb.write(data[i : i + 700])
        self.assertEqual(rb.total, 40000)
        self.assertEqual(rb.oldest, 40000 - 16000)
        np.testing.assert_array_equal(rb.read(30000, 39000), data[30000:39000])
        # Reads before the oldest sample are clamped.
        self.assertEqual(len(rb.read(0, 25000)), 25000 - rb.oldest)

    def test_resampler_48k_blockwise_matches_tone(self):
        r = Resampler(48000)
        t = np.arange(48000 * 2) / 48000
        x = (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        out = np.concatenate([r.process(x[i : i + 1234]) for i in range(0, len(x), 1234)])
        self.assertEqual(len(out), 32000)
        seg = out[2000:18000].astype(float)
        # Dominant frequency survives, amplitude roughly preserved, no block seams.
        freq = np.argmax(np.abs(np.fft.rfft(seg))) * SAMPLE_RATE / len(seg)
        self.assertAlmostEqual(freq, 440, delta=2)
        self.assertAlmostEqual(np.max(np.abs(seg)), 8000, delta=400)
        self.assertLess(np.max(np.abs(np.diff(seg))), 8000 * 2 * np.pi * 440 / 16000 * 1.1)

    def test_resampler_rejects_441_without_soxr(self):
        try:
            import soxr  # noqa: F401
            self.skipTest("soxr installed")
        except ImportError:
            with self.assertRaises(RuntimeError):
                Resampler(44100)


class Vad(unittest.TestCase):
    def test_endpointer_hysteresis(self):
        ep = Endpointer(0.5, 0.35, min_speech_ms=96, end_silence_ms=160)
        probs = [0.1, 0.9, 0.4, 0.9, 0.9, 0.9, 0.4, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
        events = []
        for i, p in enumerate(probs):
            ev = ep.update(p, i * 512)
            if ev:
                events.append((ev.kind, ev.sample // 512))
        # Onset after 3 speech chunks (0.4 is in the hysteresis band, doesn't reset);
        # the lone 0.9 resets the silence run; offset after 5 silent chunks.
        self.assertEqual(events, [("onset", 1), ("offset", 10)])

    def test_silero_on_silence_and_speech(self):
        vad = SileroVAD(str(ROOT / "models" / "silero_vad.onnx"))
        silence = [vad(np.zeros(512, np.int16)) for _ in range(20)]
        self.assertLess(max(silence), 0.1)
        from harness import command

        vad.reset()
        a = command()
        speech = [vad(a[i : i + 512]) for i in range(0, len(a) - 511, 512)]
        self.assertGreater(np.mean(np.array(speech) > 0.5), 0.5)


class ConfigFiles(unittest.TestCase):
    def test_shipped_profiles_load(self):
        for name in ("desktop.toml", "jetson.toml"):
            with self.subTest(name=name):
                cfg = load_config(ROOT / "configs" / name)
                self.assertTrue(cfg.wake.models)

    def test_unknown_key_is_an_error(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write("[vad]\nend_silense_ms = 500\n")
        with self.assertRaisesRegex(ValueError, "end_silense_ms"):
            load_config(f.name)


if __name__ == "__main__":
    unittest.main()
