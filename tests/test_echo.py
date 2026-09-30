"""The agent's own voice coming back through the speaker must not count as the
user interrupting, and the user talking over the agent still must.

The listener plays the agent's audio into a simulated room (80 ms delay, reverb,
a little speaker distortion, echo 6 dB below the source); the mic hears that
echo plus whatever the user says.
"""

import unittest

from harness import Harness, agent_voice, command, noise, wake_phrase

from wonky_talkie.config import Config

WAKE_T = 1.95


def start_conversation(h: Harness) -> None:
    h.feed(noise(1.0))
    h.feed(wake_phrase())
    h.feed(noise(0.1))
    h.feed(command())
    h.feed(noise(1.5))
    h.wait_for("transcript")


def texts(*items):
    it = iter(items)
    return lambda audio: next(it)


class SelfInterruption(unittest.TestCase):
    def test_reproduces_the_bug_without_protection(self):
        """Echo with no cancellation and no verification = the agent interrupts itself."""
        cfg = Config()
        cfg.barge_in.verify = False
        h = Harness(cfg, wake_at=[WAKE_T], stt_fn=texts("Tell me a story.", "One second."),
                    playback=True, aec=False)
        try:
            start_conversation(h)
            h.play(agent_voice(), "Once upon a time...")
            h.feed(noise(8.5))
            self.assertIn("interruption_detected", h.types())
        finally:
            h.close()

    def _agent_talks_alone(self, **kw):
        h = Harness(Config(), wake_at=[WAKE_T], stt_fn=texts("Tell me a story."), playback=True, **kw)
        try:
            start_conversation(h)
            h.play(agent_voice(), "Once upon a time...", then="listen")
            h.feed(noise(9.0))
            types = h.types()
            self.assertNotIn("interruption_detected", types)
            fin = h.wait_for("playback_finished")
            self.assertAlmostEqual(fin["played_s"], 8.0, delta=0.05)
            self.assertEqual(h.wait_for("listening", trigger="followup")["trigger"], "followup")
            return h, types
        finally:
            h.close()

    def test_aec_and_verify_no_self_interruption(self):
        h, types = self._agent_talks_alone()
        started = h.wait_for("playback_started")
        self.assertEqual(started["playback_id"], "pb1")

    def test_verify_alone_rejects_echo(self):
        """Without libspeexdsp, duck-and-verify still catches the echo."""
        h, types = self._agent_talks_alone(aec=False)
        self.assertIn("interruption_suspected", types)
        self.assertIn("interruption_rejected", types)
        rej = h.wait_for("interruption_rejected")
        self.assertLess(rej["excess_db"], 6)  # the mic followed the speaker down

    def test_aec_alone_is_not_enough_at_first(self):
        """Echo cancellation needs ~1 s to converge; without verification the
        start of the first reply still self-interrupts. (Why verify is on.)"""
        cfg = Config()
        cfg.barge_in.verify = False
        h = Harness(cfg, wake_at=[WAKE_T], stt_fn=texts("Tell me a story.", "x"), playback=True)
        try:
            start_conversation(h)
            h.play(agent_voice(), "Once upon a time...")
            h.feed(noise(8.5))
            self.assertIn("interruption_detected", h.types())
        finally:
            h.close()

    def test_headset_no_echo_no_suspicion(self):
        h, types = self._agent_talks_alone(echo=False)
        self.assertNotIn("interruption_suspected", types)


class RealInterruption(unittest.TestCase):
    def _interrupt(self, **kw):
        h = Harness(Config(), wake_at=[WAKE_T], stt_fn=texts("Tell me a story.", "Stop, that's enough."),
                    playback=True, **kw)
        try:
            start_conversation(h)
            h.play(agent_voice(8.0), "Once upon a time...")
            h.feed(noise(3.0))  # agent talks alone for 3 s
            h.feed(command())  # user talks over it
            ev = h.wait_for("interruption_detected")
            self.assertEqual(ev["trigger"], "speech")
            self.assertEqual(ev["playback_id"], "pb1")
            self.assertGreaterEqual(ev["excess_db"], 6)
            self.assertTrue(2.9 < ev["played_s"] < 4.5, ev)
            stopped = h.wait_for("playback_stopped")
            self.assertEqual(stopped["reason"], "interrupted")
            self.assertFalse(h.player.active)  # the listener stopped the audio itself
            h.feed(noise(1.5))
            tr = h.wait_for("transcript", text="Stop, that's enough.")
            self.assertEqual(tr["trigger"], "barge_in")
            self.assertGreater(tr["audio_s"], 2.0)
            self.assertNotIn("playback_finished", h.types())
            return h
        finally:
            h.close()

    def test_user_interrupts_with_aec(self):
        self._interrupt()

    def test_user_interrupts_without_aec(self):
        self._interrupt(aec=False)

    def test_interrupt_right_at_the_start_of_the_reply(self):
        """AEC not yet converged + user speaks immediately: still an interruption."""
        h = Harness(Config(), wake_at=[WAKE_T], stt_fn=texts("Tell me a story.", "No, wait."),
                    playback=True)
        try:
            start_conversation(h)
            h.play(agent_voice(8.0), "Once upon a time...")
            h.feed(noise(0.4))
            h.feed(command())
            h.feed(noise(1.5))
            self.assertEqual(h.wait_for("transcript", text="No, wait.")["trigger"], "barge_in")
        finally:
            h.close()

    def test_wake_word_stops_playback_immediately(self):
        cfg = Config()
        cfg.barge_in.triggers = ["wake_word"]
        wake2 = 1.0 + 0.952 + 0.1 + 2.7 + 1.5 + 2.0 + 0.952
        h = Harness(cfg, wake_at=[WAKE_T, wake2], stt_fn=texts("Tell me a story.", "Stop."),
                    playback=True)
        try:
            start_conversation(h)
            h.play(agent_voice(8.0), "Once upon a time...")
            h.feed(noise(2.0))
            h.feed(wake_phrase())
            h.feed(noise(0.2))
            ev = h.wait_for("interruption_detected")
            self.assertEqual(ev["trigger"], "wake_word")
            self.assertFalse(h.player.active)
        finally:
            h.close()


class PlaybackFlow(unittest.TestCase):
    def test_then_end_closes_session_after_audio(self):
        h = Harness(Config(), wake_at=[WAKE_T], stt_fn=texts("What time is it?"), playback=True)
        try:
            start_conversation(h)
            h.play(agent_voice(2.0), "It's three o'clock.", then="end")
            h.feed(noise(1.0))
            self.assertEqual(h.engine.state, "speaking")
            h.feed(noise(1.5))
            self.assertEqual(h.wait_for("session_ended")["reason"], "agent")
        finally:
            h.close()

    def test_agent_stop_playback(self):
        h = Harness(Config(), wake_at=[WAKE_T], stt_fn=texts("Tell me a story."), playback=True)
        try:
            start_conversation(h)
            h.play(agent_voice(6.0))
            h.feed(noise(1.0))
            h.command({"type": "stop_playback"})
            self.assertEqual(h.wait_for("playback_stopped")["reason"], "client")
            self.assertEqual(h.engine.state, "waiting")
        finally:
            h.close()

    def test_streamed_audio_with_underrun(self):
        """Audio arriving slower than real time: gaps, then finish once play_end is seen."""
        h = Harness(Config(), wake_at=[WAKE_T], stt_fn=texts("Hi."), playback=True)
        try:
            start_conversation(h)
            v = agent_voice(3.0)
            h.player.start_item("s1", 16000, "", "wait", None, "test")
            h.player.add_audio(v[:16000], "s1", "test")
            h.feed(noise(1.5))  # 0.5 s underrun
            self.assertEqual(h.engine.state, "speaking")
            h.player.add_audio(v[16000:], "s1", "test")
            h.player.end_item("s1", "test")
            h.feed(noise(2.5))
            fin = h.wait_for("playback_finished")
            self.assertAlmostEqual(fin["played_s"], 3.0, delta=0.05)
            self.assertEqual(h.engine.state, "waiting")
        finally:
            h.close()


if __name__ == "__main__":
    unittest.main()
