"""In-process TTS: chunking, the backends' contract, and the property that
matters most — a barge-in must stop synthesis, not just stop the speaker."""

import threading
import time
import unittest
import wave

import numpy as np
from harness import Harness, agent_voice

from wonky_talkie.audio import Player, to_wav_bytes
from wonky_talkie.config import Config, TtsConfig, load_config
from wonky_talkie.tts import (
    CommandBackend,
    FakeTtsBackend,
    NoTtsBackend,
    TtsService,
    _to_int16,
    decode_audio,
    make_backend,
    split_sentences,
)


class Splitting(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(split_sentences("   ", 300, 100), [])

    def test_first_chunk_is_short_for_latency(self):
        text = ("This is the first sentence. This is the second one. "
                "And here is a third sentence to push us past the budget.")
        chunks = split_sentences(text, max_chars=300, first_max_chars=40)
        self.assertEqual(chunks[0], "This is the first sentence.")
        self.assertGreater(len(chunks), 1)
        # Nothing is lost or duplicated.
        self.assertEqual(" ".join(chunks).split(), text.split())

    def test_sentences_stay_whole_when_they_fit(self):
        text = "Short one. Short two."
        self.assertEqual(split_sentences(text, 300, 300), ["Short one. Short two."])

    def test_long_sentence_is_wrapped_at_word_boundaries(self):
        text = "word " * 200
        chunks = split_sentences(text, max_chars=100, first_max_chars=100)
        self.assertTrue(all(len(c) <= 100 for c in chunks), [len(c) for c in chunks])
        self.assertTrue(all("word" in c for c in chunks))
        self.assertEqual(" ".join(chunks).split(), text.split())

    def test_question_and_ellipsis_end_sentences(self):
        chunks = split_sentences("Are you there? Yes… Good.", 300, 14)
        self.assertEqual(chunks[0], "Are you there?")


class ChunkPacing(unittest.TestCase):
    """The two properties that make a long reply sound continuous."""

    # The example agent's story, which is what exposed both bugs.
    STORY = ("Once upon a time there was a listener that never slept. It heard every "
             "word, but it only paid attention when someone called its name. ") * 3

    def test_never_breaks_mid_clause(self):
        """KittenTTS appends a comma to an unpunctuated chunk, so a chunk that
        ends mid-clause is spoken as a stumble ("...that never,")."""
        for chunk in split_sentences(self.STORY, 320, 48):
            self.assertRegex(chunk, r"[.!?,;:\u2026]$",
                             f"chunk ends mid-clause: {chunk!r}")

    def test_chunks_ramp_so_playback_never_starves(self):
        """Each chunk must synthesize faster than the queued audio plays out.

        Modelled with this machine's measured KittenTTS rates: 0.062 s of work
        and 0.100 s of audio per character (~1.6x realtime).
        """
        synth_per_char, audio_per_char = 0.062, 0.100
        chunks = split_sentences(self.STORY, 320, 48)
        self.assertGreater(len(chunks), 2)
        # Chunk k is ready at sum(synth[0..k]); audio lasts until
        # synth[0] + sum(audio[0..k-1]). Cancelling synth[0] (that is latency,
        # not a gap) leaves: sum(synth[1..k]) <= sum(audio[0..k-1]).
        cum_synth = cum_audio = 0.0
        for i, c in enumerate(chunks):
            if i:
                cum_synth += synth_per_char * len(c)
                self.assertLessEqual(
                    round(cum_synth, 6), round(cum_audio, 6),
                    f"chunk {i} ({len(c)} chars) starves playback")
            cum_audio += audio_per_char * len(c)

    def test_chunk_sizes_do_not_leap(self):
        chunks = split_sentences("word " * 400, max_chars=320, first_max_chars=48,
                                 growth=1.5)
        for prev, nxt in zip(chunks, chunks[1:]):
            self.assertLessEqual(len(nxt), max(320, int(len(prev) * 1.5) + 40),
                                 f"{len(prev)} -> {len(nxt)} is too big a jump")

    def test_growth_of_one_keeps_every_chunk_small(self):
        chunks = split_sentences(self.STORY, 320, 48, growth=1.0)
        self.assertTrue(all(len(c) <= 130 for c in chunks), [len(c) for c in chunks])

    def test_a_run_on_with_no_punctuation_is_still_wrapped(self):
        text = "word " * 200  # no sentence or clause boundary anywhere
        chunks = split_sentences(text, max_chars=100, first_max_chars=48)
        self.assertTrue(all(len(c) <= 100 for c in chunks), [len(c) for c in chunks])
        self.assertEqual(" ".join(chunks).split(), text.split())

    def test_nothing_is_lost_or_duplicated(self):
        for fmc in (10, 48, 120):
            chunks = split_sentences(self.STORY, 320, fmc)
            self.assertEqual(" ".join(chunks).split(), self.STORY.split(), fmc)

    def test_growth_below_one_is_rejected(self):
        from wonky_talkie.config import Config, validate
        cfg = Config()
        cfg.tts.chunk_growth = 0.9
        with self.assertRaises(ValueError):
            validate(cfg)


class Conversions(unittest.TestCase):
    def test_float_to_int16(self):
        out = _to_int16(np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32))
        self.assertEqual(out.dtype, np.int16)
        self.assertEqual(list(out), [0, 32767, -32767, 32767])  # clipped

    def test_int16_passes_through(self):
        a = np.array([1, 2, 3], dtype=np.int16)
        self.assertIs(_to_int16(a), a)

    def test_onnx_shape_is_flattened(self):
        out = _to_int16(np.zeros((1, 400), dtype=np.float32))
        self.assertEqual(out.shape, (400,))

    def test_decode_wav_roundtrip(self):
        pcm = (np.sin(np.arange(800) / 8) * 10000).astype(np.int16)
        got, rate = decode_audio(to_wav_bytes(pcm, 22050), fallback_rate=1)
        self.assertEqual(rate, 22050)
        np.testing.assert_array_equal(got, pcm)

    def test_decode_raw_uses_fallback_rate(self):
        pcm = np.array([5, -5, 7], dtype=np.int16)
        got, rate = decode_audio(pcm.tobytes(), fallback_rate=24000)
        self.assertEqual(rate, 24000)
        np.testing.assert_array_equal(got, pcm)

    def test_decode_odd_byte_count_does_not_crash(self):
        got, _ = decode_audio(b"\x01\x02\x03", fallback_rate=16000)
        self.assertEqual(len(got), 1)


class Backends(unittest.TestCase):
    def test_config_validation_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            make_backend(TtsConfig(backend="nope"))

    def test_none_backend_refuses_to_synthesize(self):
        be = make_backend(TtsConfig(backend="none"))
        self.assertIsInstance(be, NoTtsBackend)
        with self.assertRaises(RuntimeError):
            be.synth("hello")

    def test_command_backend_needs_a_command(self):
        with self.assertRaises(RuntimeError):
            make_backend(TtsConfig(backend="command"))

    def test_command_backend_reads_wav_from_stdout(self):
        """A stand-in for `espeak-ng --stdout`: python writing a WAV."""
        cfg = TtsConfig(backend="command")
        cfg.command.command = (
            'python -c "'
            "import sys,wave,numpy as np;"
            "t=sys.stdin.read();"
            "pcm=(np.sin(np.arange(len(t)*100)/8)*8000).astype(np.int16);"
            "w=wave.open(sys.stdout.buffer,'wb');w.setnchannels(1);w.setsampwidth(2);"
            "w.setframerate(22050);w.writeframes(pcm.tobytes());w.close()"
            '"')
        be = make_backend(cfg)
        pcm = be.synth("hello there")
        self.assertEqual(pcm.dtype, np.int16)
        self.assertEqual(len(pcm), len("hello there") * 100)
        self.assertEqual(be.sample_rate, 22050)

    def test_command_backend_reports_a_failing_command(self):
        cfg = TtsConfig(backend="command")
        cfg.command.command = "exit 3"
        with self.assertRaises(RuntimeError) as cm:
            make_backend(cfg).synth("hi")
        self.assertIn("exit 3", str(cm.exception))

    def test_piper_backend_needs_a_model(self):
        with self.assertRaises(RuntimeError):
            make_backend(TtsConfig(backend="piper"))

    def test_tts_section_parses_from_toml(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.toml"
            p.write_text('[tts]\nbackend = "command"\nmax_chars = 90\n'
                         '[tts.command]\ncommand = "espeak-ng --stdout"\n')
            cfg = load_config(p)
        self.assertEqual(cfg.tts.backend, "command")
        self.assertEqual(cfg.tts.max_chars, 90)
        self.assertEqual(cfg.tts.command.command, "espeak-ng --stdout")

    def test_unknown_tts_key_is_an_error(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.toml"
            p.write_text('[tts]\nvoice = "Bella"\n')  # voice lives under [tts.kitten]
            with self.assertRaises(ValueError):
                load_config(p)


class Service(unittest.TestCase):
    """TtsService against a real Player, with no audio device."""

    def setUp(self):
        self.player = Player(16000)
        self.events: list[tuple] = []
        self.backend = FakeTtsBackend(sample_rate=16000, ms_per_char=10.0)
        self.svc = TtsService(TtsConfig(first_chunk_chars=30, max_chars=60), self.backend,
                              self.player, emit=lambda t, **f: self.events.append((t, f)))

    def tearDown(self):
        self.svc.shutdown()

    def _drain(self, pid, timeout=10.0):
        """Wait for the worker, then read everything the player queued."""
        self.svc.wait(pid, timeout)
        out = []
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            out.append(self.player.read(4096))
            if not self.player.active:
                break
        return np.concatenate(out) if out else np.zeros(0, np.int16)

    def test_say_synthesizes_every_chunk_and_plays_it(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        pid = self.svc.say(text, client="c1")
        audio = self._drain(pid)
        self.assertEqual(" ".join(self.backend.calls).split(), text.split())
        expected = sum(int(16000 * 10.0 * len(c) / 1000) for c in self.backend.calls)
        self.assertEqual(int(np.count_nonzero(audio != 0) > 0), 1)
        self.assertGreaterEqual(len(audio), expected * 0.9)
        self.assertTrue(pid.startswith("pb_"))

    def test_say_uses_the_backend_sample_rate(self):
        self.backend.sample_rate = 24000
        pid = self.svc.say("Hello.", client="c1")
        self._drain(pid)
        # 24 kHz in, 16 kHz out: the Player resampled, so it is shorter than the source.
        self.assertEqual(self.svc.info["sample_rate"], 24000)

    def test_stopping_playback_abandons_the_remaining_sentences(self):
        """The barge-in guarantee: stop() must cut synthesis, not just audio."""
        gate = threading.Event()
        calls: list[str] = []

        def slow(text: str) -> np.ndarray:
            calls.append(text)
            gate.wait(2.0)  # hold the worker inside the first chunk
            return np.zeros(1600, np.int16)

        self.backend.fn = slow
        text = ". ".join(f"Sentence number {i} is here" for i in range(12)) + "."
        self.svc.say(text, client="c1")
        for _ in range(100):  # let the worker enter the first synth
            if calls:
                break
            time.sleep(0.01)
        self.player.stop("interrupted")
        gate.set()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and self.player.active:
            self.player.read(1024)
        self.assertLessEqual(len(calls), 2, f"kept synthesizing after stop: {calls}")
        self.assertGreater(len(split_sentences(text, 60, 30)), 6)

    def test_a_failing_backend_emits_tts_failed_and_ends_the_item(self):
        def boom(text: str):
            raise RuntimeError("voice exploded")

        self.backend.fn = boom
        with self.assertLogs("wonky_talkie.tts", "ERROR"):
            self.svc.say("This will fail.", client="c1")
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not self.events:
                time.sleep(0.01)
        self.assertEqual(self.events[0][0], "tts_failed")
        self.assertIn("voice exploded", self.events[0][1]["message"])
        self.assertEqual(self.events[0][1]["_to"], "c1")
        # The item must be closed, or the engine would wait for audio forever.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and self.player.active:
            self.player.read(1024)
        self.assertFalse(self.player.active)

    def test_info_describes_the_backend(self):
        self.assertEqual(self.svc.info["backend"], "fake")
        self.assertEqual(self.svc.info["sample_rate"], 16000)


class EngineIntegration(unittest.TestCase):
    """`say` audio must drive the same state machine `play` audio does."""

    def test_spoken_reply_enters_speaking_then_reopens_the_session(self):
        h = Harness(stt_fn=lambda a: "Hello there.", playback=True, echo=False)
        svc = TtsService(TtsConfig(), FakeTtsBackend(sample_rate=16000, ms_per_char=8.0),
                         h.player, emit=h.engine.post_event)
        try:
            h.say(svc, "Hi, I am wonky talkie.", then="listen")
            h.feed(np.zeros(int(16000 * 2.0), np.int16))  # let the room play it out
            h.wait_for("playback_started")
            fin = h.wait_for("playback_finished")
            self.assertGreater(fin["played_s"], 0.05)
            # then="listen" reopens a follow-up window, so no wake word is needed.
            h.wait_for("listening", trigger="followup")
        finally:
            svc.shutdown()
            h.close()

    def test_barge_in_stops_a_spoken_reply(self):
        """A person talking over synthesized speech must still win."""
        h = Harness(stt_fn=lambda a: "Stop please.", playback=True, echo=False)
        svc = TtsService(TtsConfig(), FakeTtsBackend(sample_rate=16000, ms_per_char=40.0),
                         h.player, emit=h.engine.post_event)
        try:
            h.say(svc, "I am going to talk for quite a while about many things.")
            h.feed(np.zeros(int(16000 * 0.3), np.int16))
            h.wait_for("playback_started")
            h.feed(agent_voice(2.0))  # a real voice over the top of it
            h.wait_for("interruption_detected", timeout=10)
            self.assertFalse(h.player.active)
        finally:
            svc.shutdown()
            h.close()


if __name__ == "__main__":
    unittest.main()
