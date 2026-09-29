"""Protocol test over a real WebSocket: an agent client subscribes, gets the
wake -> transcript flow, speaks, gets interrupted, and receives the audio."""

import asyncio
import base64
import io
import json
import time
import unittest
import wave

import numpy as np
from harness import Harness, agent_voice, command, noise, wake_phrase

from voice_listener.server import EventServer

try:
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidStatus
except ImportError:  # pragma: no cover
    connect = None


@unittest.skipIf(connect is None, "websockets not installed")
class ServerProtocol(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(asyncio.wait_for(coro, 60))

    def _setup(self, token="", **harness_kw):
        texts = iter(["Hey Jarvis, tell me a long story.", "Stop."])
        h = Harness(wake_at=[1.95], stt_fn=lambda a: next(texts), **harness_kw)
        server = EventServer(h.engine, "127.0.0.1", 0, token=token)
        orig = h.events.put

        def emit(ev):
            orig(ev)
            server.publish(ev)

        h.engine._emit_cb = emit
        return h, server

    def test_agent_flow(self):
        h, server = self._setup()

        async def scenario():
            started, stop = asyncio.Event(), asyncio.Event()
            task = asyncio.create_task(server.serve_forever(started, stop))
            await started.wait()
            loop = asyncio.get_running_loop()
            got: list[dict] = []

            async with connect(f"ws://127.0.0.1:{server.port}/") as ws:
                hello = json.loads(await ws.recv())
                self.assertEqual(hello["type"], "hello")
                self.assertEqual(hello["state"], "idle")
                self.assertIn("interruption_detected", hello["events"])

                await ws.send(json.dumps({"type": "subscribe", "events": [
                    "session_started", "wake_detected", "speech_started", "silence_detected",
                    "transcript", "interruption_detected", "state"], "include_audio": True}))
                self.assertEqual(json.loads(await ws.recv())["type"], "subscribed")

                async def next_of(type_):
                    while True:
                        ev = json.loads(await ws.recv())
                        got.append(ev)
                        if ev["type"] == type_:
                            return ev

                def feed_first():
                    h.feed(noise(1.0)); h.feed(wake_phrase()); h.feed(noise(0.1))
                    h.feed(command()); h.feed(noise(1.5))

                await loop.run_in_executor(None, feed_first)
                tr = await next_of("transcript")
                self.assertEqual(tr["text"], "Tell me a long story.")
                with wave.open(io.BytesIO(base64.b64decode(tr["audio_wav_b64"]))) as w:
                    self.assertEqual(w.getframerate(), 16000)
                    self.assertAlmostEqual(w.getnframes() / 16000, tr["audio_s"], delta=0.05)

                # Agent starts talking, user interrupts.
                await ws.send(json.dumps({"type": "speaking_started", "text": "Once upon a time..."}))
                await asyncio.sleep(0.05)
                await loop.run_in_executor(None, lambda: (h.feed(noise(0.6)), h.feed(command())))
                intr = await next_of("interruption_detected")
                self.assertEqual(intr["agent_state"], "speaking")
                await loop.run_in_executor(None, lambda: h.feed(noise(1.5)))
                tr2 = await next_of("transcript")
                self.assertEqual((tr2["text"], tr2["trigger"]), ("Stop.", "barge_in"))

                # Targeted reply goes only to the asking client.
                await ws.send(json.dumps({"type": "get_state"}))
                st = await next_of("state")
                self.assertEqual(st["state"], "waiting")

                # Bad input yields an error, not a disconnect.
                await ws.send("not json")
                err = json.loads(await ws.recv())
                self.assertEqual(err["type"], "error")

            types = [e["type"] for e in got]
            self.assertNotIn("state_changed", types)  # filtered by the subscription
            stop.set()
            await task

        try:
            self.run_async(scenario())
        finally:
            h.close()

    def test_play_audio_through_the_listener(self):
        """play + binary PCM + play_end; then the user interrupts the second reply."""
        h, server = self._setup(playback=True)

        async def scenario():
            started, stop = asyncio.Event(), asyncio.Event()
            task = asyncio.create_task(server.serve_forever(started, stop))
            await started.wait()
            loop = asyncio.get_running_loop()
            async with connect(f"ws://127.0.0.1:{server.port}/") as ws:
                hello = json.loads(await ws.recv())
                self.assertTrue(hello["playback"])

                async def next_of(type_):
                    while True:
                        ev = json.loads(await ws.recv())
                        if ev["type"] == type_:
                            return ev

                await loop.run_in_executor(None, lambda: (
                    h.feed(noise(1.0)), h.feed(wake_phrase()), h.feed(noise(0.1)),
                    h.feed(command()), h.feed(noise(1.5))))
                await next_of("transcript")

                # Reply 1: 22.05 kHz audio in three binary chunks, then listen.
                reply = np.repeat(agent_voice(2.0), 2)[: int(2.0 * 22050)]  # any rate works
                await ws.send(json.dumps({"type": "play", "sample_rate": 22050,
                                          "text": "Which room?", "then": "listen"}))
                acc = await next_of("play_accepted")
                for chunk in np.array_split(reply, 3):
                    await ws.send(chunk.astype("<i2").tobytes())
                await ws.send(json.dumps({"type": "play_end"}))
                await asyncio.sleep(0.1)
                await loop.run_in_executor(None, lambda: h.feed(noise(2.6)))
                self.assertEqual((await next_of("playback_started"))["playback_id"], acc["playback_id"])
                fin = await next_of("playback_finished")
                self.assertAlmostEqual(fin["played_s"], 2.0, delta=0.05)
                self.assertEqual((await next_of("listening"))["trigger"], "followup")

                # Reply 2 is interrupted; audio still arriving afterwards is dropped.
                await ws.send(json.dumps({"type": "play", "sample_rate": 16000, "text": "Once..."}))
                await next_of("play_accepted")
                await ws.send(agent_voice(6.0).astype("<i2").tobytes())
                await asyncio.sleep(0.1)
                await loop.run_in_executor(None, lambda: (h.feed(noise(2.0)), h.feed(command())))
                intr = await next_of("interruption_detected")
                self.assertLess(intr["played_s"], 6.0)
                stopped = await next_of("playback_stopped")
                self.assertEqual(stopped["reason"], "interrupted")
                await ws.send(agent_voice(1.0).astype("<i2").tobytes())  # late, ignored
                await ws.send(json.dumps({"type": "play_end"}))
                await loop.run_in_executor(None, lambda: h.feed(noise(1.5)))
                tr = await next_of("transcript")
                self.assertEqual(tr["trigger"], "barge_in")
                self.assertFalse(h.player.active)

                await ws.send(json.dumps({"type": "play"}))  # missing sample_rate
                self.assertEqual((await next_of("error"))["type"], "error")
            stop.set()
            await task

        try:
            self.run_async(scenario())
        finally:
            h.close()

    def test_token_required(self):
        h, server = self._setup(token="s3cret")

        async def scenario():
            started, stop = asyncio.Event(), asyncio.Event()
            task = asyncio.create_task(server.serve_forever(started, stop))
            await started.wait()
            with self.assertRaises(InvalidStatus):
                async with connect(f"ws://127.0.0.1:{server.port}/"):
                    pass
            async with connect(f"ws://127.0.0.1:{server.port}/?token=s3cret") as ws:
                self.assertEqual(json.loads(await ws.recv())["type"], "hello")
            async with connect(f"ws://127.0.0.1:{server.port}/",
                               additional_headers={"Authorization": "Bearer s3cret"}) as ws:
                self.assertEqual(json.loads(await ws.recv())["type"], "hello")
            stop.set()
            await task

        try:
            self.run_async(scenario())
        finally:
            h.close()


if __name__ == "__main__":
    unittest.main()
