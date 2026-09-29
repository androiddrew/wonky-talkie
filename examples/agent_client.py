"""Minimal agent client for the voice listener.

What an agent has to do:
  1. react to `transcript` events (the user's turn),
  2. send its speech to the listener with `play` + binary PCM + `play_end`
     (the listener plays it, cancels its echo, arms barge-in, and opens the
     follow-up window afterwards if you asked for `then: "listen"`),
  3. stop generating speech when `interruption_detected` arrives (the
     listener has already stopped the audio).

The "brain" is a stub: replace `think()` with your LLM call.

    python examples/agent_client.py                                   # simulated, no audio
    python examples/agent_client.py --tts-cmd "espeak-ng --stdout"    # WAV on stdout
    python examples/agent_client.py --tts-rate 22050 \\
        --tts-cmd "piper -m en_US-lessac-medium.onnx --output-raw"      # raw PCM on stdout

The TTS command reads the text on stdin and writes audio (WAV, or raw 16-bit
mono PCM at --tts-rate) to stdout; it's streamed to the listener as it arrives.

--local-playback keeps the old behaviour (the command plays audio itself and the
agent sends speaking_started / speaking_ended). Speech barge-in then only works
with a headset or system-level echo cancellation.

Requires: pip install websockets numpy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal

import numpy as np
from websockets.asyncio.client import connect


async def think(history: list[dict]) -> tuple[str, bool]:
    """Return (reply, expects_reply). Stand-in for an LLM call."""
    last = history[-1]["content"].lower()
    if "light" in last and not any(r in last for r in ("kitchen", "office", "bedroom", "living")):
        return "Sure. Which room?", True
    if len(last.split()) <= 2 and len(history) > 1:
        return f"Okay, {history[-1]['content'].rstrip('.')}. Done.", False
    if "story" in last:
        return ("Once upon a time there was a listener that never slept. It heard every "
                "word, but it only paid attention when someone called its name. " * 3), False
    return f"You said: {history[-1]['content']}", False


async def read_audio_format(stream: asyncio.StreamReader, raw_rate: int) -> tuple[int, int, bytes]:
    """Parse a WAV header if there is one. Returns (rate, channels, leftover bytes)."""
    head = await stream.read(4)
    if head != b"RIFF":
        return raw_rate, 1, head  # raw PCM
    head += await stream.readexactly(8)
    if head[8:12] != b"WAVE":
        raise RuntimeError("TTS output is RIFF but not WAVE")
    rate, channels = raw_rate, 1
    while True:
        cid = await stream.readexactly(4)
        size = int.from_bytes(await stream.readexactly(4), "little")
        if cid == b"data":
            return rate, channels, b""
        body = await stream.readexactly(size + (size & 1))
        if cid == b"fmt ":
            channels = int.from_bytes(body[2:4], "little")
            rate = int.from_bytes(body[4:8], "little")
            if int.from_bytes(body[14:16], "little") != 16:
                raise RuntimeError("TTS output must be 16-bit PCM")


class Agent:
    def __init__(self, ws, hello: dict, args):
        self.ws = ws
        self.tts_cmd = args.tts_cmd
        self.tts_rate = args.tts_rate
        self.words_per_s = args.words_per_s
        self.listener_playback = bool(args.tts_cmd) and hello.get("playback") and not args.local_playback
        self.history: list[dict] = []
        self.turn: asyncio.Task | None = None
        self._midline = False  # simulated speech is printed word by word

    async def send(self, **cmd) -> None:
        await self.ws.send(json.dumps(cmd))

    async def run(self) -> None:
        await self.send(type="subscribe", events=[
            "session_started", "session_ended", "wake_detected", "listening", "speech_started",
            "partial_transcript", "silence_detected", "transcript", "utterance_discarded",
            "interruption_suspected", "interruption_rejected", "interruption_detected",
            "playback_started", "playback_finished", "playback_stopped", "listen_timeout", "error"])
        async for raw in self.ws:
            await self.on_event(json.loads(raw))

    async def on_event(self, ev: dict) -> None:
        t = ev["type"]
        if t == "partial_transcript":
            print(f"   … {ev['text']}", flush=True)
            return
        if t not in ("subscribed", "hello"):
            if self._midline:
                print(flush=True)
                self._midline = False
            detail = ev.get("text") or ev.get("reason") or ev.get("trigger") or ""
            if "excess_db" in ev:
                detail = f"{detail} (voice {ev['excess_db']:+.1f} dB over echo)"
            print(f"<- {t:<22} {detail}", flush=True)

        if t == "transcript":
            if self.turn and not self.turn.done():
                self.turn.cancel()
            self.history.append({"role": "user", "content": ev["text"]})
            self.turn = asyncio.create_task(self.take_turn())
        elif t == "interruption_detected":
            # The listener already stopped the audio and is capturing what the
            # user says; stop producing speech. The transcript follows.
            if self.turn and not self.turn.done():
                self.turn.cancel()
            if self.history and self.history[-1]["role"] == "assistant":
                heard = ev.get("played_s")
                total = ev.get("total_s")
                note = f" [interrupted after {heard:.1f}s" + (f" of {total:.1f}s]" if total else "]") \
                    if heard is not None else " [interrupted]"
                self.history[-1]["content"] += note
            print("   (agent stopped talking)", flush=True)
        elif t == "session_ended":
            self.history.clear()

    async def take_turn(self) -> None:
        reply, expects_reply = await think(self.history)
        self.history.append({"role": "assistant", "content": reply})
        then = "listen" if expects_reply else "end"
        if self.listener_playback:
            await self.speak_via_listener(reply, then)  # the listener applies `then`
            return
        await self.send(type="speaking_started", text=reply)
        await self.speak_locally(reply)
        await self.send(type="speaking_ended", then=then)

    async def speak_via_listener(self, text: str, then: str) -> None:
        print(f"-> {text}", flush=True)
        proc = await asyncio.create_subprocess_shell(
            self.tts_cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            start_new_session=True)
        try:
            proc.stdin.write(text.encode() + b"\n")
            await proc.stdin.drain()
            proc.stdin.close()
            rate, channels, pending = await read_audio_format(proc.stdout, self.tts_rate)
            await self.send(type="play", sample_rate=rate, text=text, then=then)
            frame = 2 * channels
            while True:
                chunk = await proc.stdout.read(16384)
                if not chunk:
                    break
                data = pending + chunk
                usable = len(data) - len(data) % frame
                pending = data[usable:]
                pcm = np.frombuffer(data[:usable], dtype="<i2")
                if channels > 1:
                    pcm = pcm.reshape(-1, channels)[:, 0]
                await self.ws.send(pcm.astype("<i2").tobytes())
            await self.send(type="play_end")
            await proc.wait()
        except asyncio.CancelledError:
            _kill(proc)
            raise

    async def speak_locally(self, text: str) -> None:
        if not self.tts_cmd:
            print("-> ", end="", flush=True)
            self._midline = True
            for word in text.split():
                print(word, end=" ", flush=True)
                await asyncio.sleep(1 / self.words_per_s)
            print(flush=True)
            self._midline = False
            return
        proc = await asyncio.create_subprocess_shell(
            self.tts_cmd, stdin=asyncio.subprocess.PIPE, start_new_session=True)
        try:
            proc.stdin.write(text.encode() + b"\n")
            await proc.stdin.drain()
            proc.stdin.close()
            await proc.wait()
        except asyncio.CancelledError:
            _kill(proc)
            raise


def _kill(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)  # the whole pipeline, e.g. piper | aplay
    except ProcessLookupError:
        pass


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8765/")
    p.add_argument("--tts-cmd", help="shell command: text on stdin -> WAV or raw PCM on stdout")
    p.add_argument("--tts-rate", type=int, default=22050, help="sample rate of raw PCM output")
    p.add_argument("--local-playback", action="store_true",
                   help="the TTS command plays audio itself (old mode, no echo cancellation)")
    p.add_argument("--words-per-s", type=float, default=3.0, help="simulated speaking rate")
    args = p.parse_args()
    async with connect(args.url) as ws:
        hello = json.loads(await ws.recv())
        agent = Agent(ws, hello, args)
        mode = "listener playback" if agent.listener_playback else \
            ("local playback" if args.tts_cmd else "simulated speech")
        print(f"connected: state={hello['state']} wake={hello['wake_models']} "
              f"stt={hello['stt_backend']} aec={hello.get('aec')} mode={mode}")
        await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
