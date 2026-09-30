import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wonky_talkie.audio import SAMPLE_RATE, Resampler, RingBuffer  # noqa: E402
from wonky_talkie.config import load_config  # noqa: E402
from wonky_talkie.textutil import is_hallucination, strip_wake_phrase  # noqa: E402
from wonky_talkie.vad import Endpointer, SileroVAD  # noqa: E402

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

    def test_resampler_is_exact_for_any_ratio_and_block_split(self):
        from scipy.signal import upfirdn

        rng = np.random.default_rng(1)
        for a, b in [(44100, 16000), (22050, 48000), (16000, 48000), (24000, 16000)]:
            with self.subTest(ratio=f"{a}->{b}"):
                r = Resampler(a, b)
                x = (rng.standard_normal(a // 2) * 3000).astype(np.int16)
                parts, i = [], 0
                while i < len(x):
                    n = int(rng.integers(1, 1500))
                    parts.append(r.process(x[i : i + n]))
                    i += n
                y = np.concatenate(parts)
                self.assertEqual(len(y), len(x) * b // a)
                ref = np.clip(np.round(upfirdn(r.h, x.astype(float), r.up, r.down)), -32768, 32767)
                np.testing.assert_array_equal(y, ref[: len(y)].astype(np.int16))


class PlayerTest(unittest.TestCase):
    def test_items_events_duck_and_stop(self):
        from wonky_talkie.audio import Player

        events = []
        p = Player(16000, events.append)
        self.assertFalse(p.read(160).any())
        p.start_item("a", 16000, "hi", then="end", client="c")
        p.add_audio(np.full(1600, 1000, np.int16), client="c")
        out = p.read(800)
        self.assertEqual(events[-1]["kind"], "started")
        self.assertEqual(int(out[-1]), 1000)
        p.duck(-20)
        out = p.read(800)
        self.assertEqual(int(out[-1]), 100)  # ramped down to -20 dB
        p.end_item(client="c")
        p.read(100)
        fin = events[-1]
        self.assertEqual((fin["kind"], fin["then"], fin["more"]), ("finished", "end", False))
        self.assertAlmostEqual(fin["played_s"], 0.1, places=3)
        p.start_item("b", 8000, client="c")  # other rates are resampled
        p.add_audio(np.full(8000, 500, np.int16), client="c")
        p.read(4000)
        cut = p.stop("interrupted")
        self.assertEqual(cut[0]["playback_id"], "b")
        self.assertAlmostEqual(cut[0]["played_s"], 0.25, places=2)
        self.assertIsNone(cut[0]["total_s"])  # play_end never arrived
        self.assertFalse(p.active)
        self.assertFalse(p.add_audio(np.ones(10, np.int16), "b", "c"))  # late audio is dropped


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
        # Every shipped config, so a new one can't go stale unnoticed.
        for name in sorted(q.name for q in (ROOT / "configs").glob("*.toml")):
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
