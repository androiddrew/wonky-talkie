"""End-to-end engine scenarios on real speech audio with the real Silero VAD.

Run: python -m unittest discover -s tests -v   (or: pytest tests)
"""

import unittest

from harness import Harness, command, noise, short_reply, wake_phrase

from wonky_talkie.config import Config

WAKE_T = 1.95  # end of the wake phrase in the streams below


def wake_then_command(h: Harness) -> dict:
    h.feed(noise(1.0))
    h.feed(wake_phrase())
    h.feed(noise(0.1))
    h.feed(command())
    h.feed(noise(1.5))
    return h.wait_for("transcript")


class WakeAndCapture(unittest.TestCase):
    def test_wake_then_command_same_breath(self):
        h = Harness(wake_at=[WAKE_T], stt_fn=lambda a: "Hey Jarvis, turn on the office lights.")
        try:
            tr = wake_then_command(h)
            self.assertEqual(tr["text"], "Turn on the office lights.")
            self.assertEqual(tr["trigger"], "wake_word")
            self.assertEqual(tr["end_reason"], "silence")
            self.assertEqual(
                h.types(),
                ["session_started", "wake_detected", "listening", "speech_started",
                 "silence_detected", "transcript"],
            )
            self.assertEqual(h.states(), ["listening", "capturing", "processing", "waiting"])
            # Utterance audio starts at/after the wake point minus the small preroll,
            # so the wake phrase itself is not re-transcribed.
            self.assertLess(tr["audio_s"], 3.3)
        finally:
            h.close()

    def test_wake_then_pause_then_command(self):
        h = Harness(wake_at=[WAKE_T], stt_fn=lambda a: "turn on the office lights")
        try:
            h.feed(noise(1.0))
            h.feed(wake_phrase())
            h.feed(noise(2.0))  # user pauses after the wake word
            self.assertEqual(h.engine.state, "listening")
            h.feed(command())
            h.feed(noise(1.5))
            tr = h.wait_for("transcript")
            self.assertEqual(tr["text"], "turn on the office lights")
            self.assertEqual(h.types().count("speech_started"), 1)
        finally:
            h.close()

    def test_no_speech_after_wake_times_out(self):
        cfg = Config()
        cfg.session.after_wake_timeout_s = 2.0
        h = Harness(cfg, wake_at=[WAKE_T])
        try:
            h.feed(noise(1.0))
            h.feed(wake_phrase())
            h.feed(noise(2.5))
            h.wait_for("listen_timeout")
            ended = h.wait_for("session_ended")
            self.assertEqual(ended["reason"], "timeout")
            self.assertEqual(h.engine.state, "idle")
        finally:
            h.close()

    def test_speech_without_wake_is_ignored(self):
        h = Harness()
        try:
            h.feed(noise(0.5))
            h.feed(command())
            h.feed(noise(1.5))
            self.assertEqual(h.types(), [])
            self.assertEqual(h.engine.state, "idle")
        finally:
            h.close()

    def test_hallucinated_text_is_discarded_and_session_keeps_listening(self):
        h = Harness(wake_at=[WAKE_T], stt_fn=lambda a: "Thank you.")
        try:
            h.feed(noise(1.0))
            h.feed(wake_phrase())
            h.feed(noise(0.1))
            h.feed(command())
            h.feed(noise(1.5))
            ev = h.wait_for("utterance_discarded")
            self.assertEqual(ev["reason"], "no_text")
            h.feed(noise(0.1))
            self.assertEqual(h.engine.state, "listening")
            self.assertNotIn("transcript", h.types())
        finally:
            h.close()

    def test_max_utterance_length(self):
        cfg = Config()
        cfg.session.max_utterance_s = 1.0
        h = Harness(cfg, wake_at=[WAKE_T])
        try:
            wake_then_command(h)
            sil = h.wait_for("silence_detected")
            self.assertEqual(sil["reason"], "max_duration")
        finally:
            h.close()

    def test_partial_transcripts_while_talking(self):
        cfg = Config()
        cfg.stt.partials = True
        cfg.stt.partial_interval_ms = 400
        h = Harness(cfg, wake_at=[WAKE_T], stt_fn=lambda a: f"Jarvis heard {len(a) // 1600} tenths")
        try:
            h.feed(noise(1.0))
            h.feed(wake_phrase())
            h.feed(noise(0.1))
            for i in range(0, 43200, 3200):  # feed the command slowly so partials can run
                h.feed(command()[i : i + 3200])
                __import__("time").sleep(0.05)
            h.feed(noise(1.5))
            tr = h.wait_for("transcript")
            partials = [e for e in h.log if e["type"] == "partial_transcript"]
            self.assertGreaterEqual(len(partials), 2)
            self.assertTrue(all(p["utterance_id"] == tr["utterance_id"] for p in partials))
            self.assertTrue(partials[0]["text"].startswith("Heard"))  # wake word stripped
            types = h.types()
            self.assertLess(types.index("partial_transcript"), types.index("silence_detected"))
        finally:
            h.close()

    def test_push_to_talk_listen_command(self):
        h = Harness(stt_fn=lambda a: "what's the weather")
        try:
            h.feed(noise(0.5))
            h.command({"type": "listen", "timeout_s": 5})
            started = h.wait_for("session_started")
            self.assertEqual(started["trigger"], "command")
            h.feed(command())
            h.feed(noise(1.5))
            tr = h.wait_for("transcript")
            self.assertEqual(tr["trigger"], "command")
        finally:
            h.close()


class Conversation(unittest.TestCase):
    def test_clarifying_question_followup_without_wake_word(self):
        replies = iter(["Turn on the lights.", "The office."])
        h = Harness(wake_at=[WAKE_T], stt_fn=lambda a: next(replies))
        try:
            wake_then_command(h)
            # Agent: "Which room?"
            h.command({"type": "speaking_started", "text": "Which room?"})
            self.assertEqual(h.engine.state, "speaking")
            h.feed(noise(1.0))
            h.command({"type": "speaking_ended", "then": "listen", "timeout_s": 6})
            self.assertEqual(h.engine.state, "listening")
            h.feed(noise(0.5))
            h.feed(short_reply())
            h.feed(noise(1.5))
            tr = h.wait_for("transcript", text="The office.")
            self.assertEqual(tr["trigger"], "followup")
            self.assertEqual(h.types().count("session_started"), 1)
            h.command({"type": "end_session"})
            self.assertEqual(h.wait_for("session_ended")["reason"], "client")
        finally:
            h.close()

    def test_followup_window_times_out(self):
        h = Harness(wake_at=[WAKE_T])
        try:
            wake_then_command(h)
            h.command({"type": "speaking_started"})
            h.feed(noise(0.5))
            h.command({"type": "speaking_ended", "then": "listen", "timeout_s": 1.5})
            h.feed(noise(2.0))
            self.assertEqual(h.wait_for("session_ended")["reason"], "timeout")
        finally:
            h.close()

    def test_speaking_ended_then_end(self):
        h = Harness(wake_at=[WAKE_T])
        try:
            wake_then_command(h)
            h.command({"type": "speaking_started"})
            h.command({"type": "speaking_ended", "then": "end"})
            self.assertEqual(h.wait_for("session_ended")["reason"], "agent")
        finally:
            h.close()

    def test_agent_can_start_a_session_by_speaking(self):
        h = Harness()
        try:
            h.command({"type": "speaking_started", "text": "Your timer is done."})
            self.assertEqual(h.wait_for("session_started")["trigger"], "agent")
            self.assertEqual(h.engine.state, "speaking")
        finally:
            h.close()


class BargeIn(unittest.TestCase):
    def test_speech_interrupts_agent(self):
        texts = iter(["Tell me about the weather.", "Stop, that's enough."])
        h = Harness(wake_at=[WAKE_T], stt_fn=lambda a: next(texts))
        try:
            wake_then_command(h)
            h.command({"type": "speaking_started", "text": "A long-winded answer..."})
            h.feed(noise(0.6))
            h.feed(command())  # user talks over the agent
            ev = h.wait_for("interruption_detected")
            self.assertEqual(ev["trigger"], "speech")
            self.assertEqual(ev["agent_state"], "speaking")
            h.feed(noise(1.5))
            tr = h.wait_for("transcript", text="Stop, that's enough.")
            self.assertEqual(tr["trigger"], "barge_in")
            # The interruption audio includes the start of the speech (preroll).
            self.assertGreater(tr["audio_s"], 2.0)
            # A late speaking_ended from the agent is harmless.
            h.command({"type": "speaking_ended", "then": "listen"})
            self.assertEqual(h.engine.state, "waiting")
        finally:
            h.close()

    def test_fast_barge_in_captures_whole_interruption(self):
        # Barge-in fires before the normal endpointer would register an onset;
        # the capture must still run until the user stops, not end immediately.
        cfg = Config()
        cfg.barge_in.min_speech_ms = 96
        cfg.vad.min_speech_ms = 400
        texts = iter(["Tell me a story.", "Stop."])
        h = Harness(cfg, wake_at=[WAKE_T], stt_fn=lambda a: next(texts))
        try:
            wake_then_command(h)
            h.command({"type": "speaking_started"})
            h.feed(noise(0.6))
            h.feed(command())
            h.feed(noise(1.5))
            tr = h.wait_for("transcript", text="Stop.")
            self.assertGreater(tr["audio_s"], 2.0)
        finally:
            h.close()

    def test_grace_period_ignores_speech_right_after_agent_starts(self):
        cfg = Config()
        cfg.barge_in.grace_ms = 5000
        h = Harness(cfg, wake_at=[WAKE_T])
        try:
            wake_then_command(h)
            h.command({"type": "speaking_started"})
            h.feed(command())
            h.feed(noise(1.0))
            self.assertNotIn("interruption_detected", h.types())
            self.assertEqual(h.engine.state, "speaking")
        finally:
            h.close()

    def test_wake_word_only_barge_in(self):
        cfg = Config()
        cfg.barge_in.triggers = ["wake_word"]  # e.g. no echo cancellation available
        speak_start = 1.0 + 0.95 + 0.1 + 2.7 + 1.5  # after wake_then_command()
        wake2 = speak_start + 0.3 + 2.7 + 0.2 + 0.95
        texts = iter(["Tell me a story.", "Hey Jarvis, stop."])
        h = Harness(cfg, wake_at=[WAKE_T, wake2], stt_fn=lambda a: next(texts))
        try:
            wake_then_command(h)
            h.command({"type": "speaking_started"})
            h.feed(noise(0.3))
            h.feed(command())  # echo of the agent's voice: must NOT interrupt
            h.feed(noise(0.2))
            self.assertNotIn("interruption_detected", h.types())
            h.feed(wake_phrase())  # "hey jarvis" does
            h.feed(noise(0.3))
            ev = h.wait_for("interruption_detected")
            self.assertEqual(ev["trigger"], "wake_word")
            h.feed(short_reply())
            h.feed(noise(1.5))
            tr = h.wait_for("transcript", text="Stop.")
            self.assertEqual(tr["trigger"], "barge_in")
        finally:
            h.close()

    def test_barge_in_disabled_is_half_duplex(self):
        cfg = Config()
        cfg.barge_in.enabled = False
        h = Harness(cfg, wake_at=[WAKE_T])
        try:
            wake_then_command(h)
            h.command({"type": "speaking_started"})
            h.feed(noise(0.5))
            h.feed(command())
            h.feed(noise(1.0))
            self.assertNotIn("interruption_detected", h.types())
        finally:
            h.close()

    def test_user_speaks_while_agent_is_thinking(self):
        texts = iter(["Turn on the lights.", "Actually, never mind."])
        h = Harness(wake_at=[WAKE_T], stt_fn=lambda a: next(texts))
        try:
            wake_then_command(h)
            self.assertEqual(h.engine.state, "waiting")
            h.feed(noise(0.5))
            h.feed(short_reply())
            ev = h.wait_for("interruption_detected")
            self.assertEqual(ev["agent_state"], "thinking")
            h.feed(noise(1.5))
            self.assertEqual(h.wait_for("transcript", text="Actually, never mind.")["trigger"], "barge_in")
        finally:
            h.close()

    def test_user_keeps_talking_while_transcribing(self):
        texts = iter(["first part", "second part"])
        h = Harness(wake_at=[WAKE_T], stt_fn=lambda a: next(texts), stt_delay=1.0)
        try:
            h.feed(noise(1.0))
            h.feed(wake_phrase())
            h.feed(noise(0.1))
            h.feed(command())
            h.feed(noise(0.9))
            h.wait_for("silence_detected")
            h.feed(short_reply())  # continues during transcription
            h.feed(noise(1.2))
            first = h.wait_for("transcript", text="first part", timeout=5)
            second = h.wait_for("transcript", text="second part", timeout=5)
            self.assertEqual(first["trigger"], "wake_word")
            self.assertEqual(second["trigger"], "continuation")
            self.assertEqual(h.engine.state, "waiting")
        finally:
            h.close()


if __name__ == "__main__":
    unittest.main()
