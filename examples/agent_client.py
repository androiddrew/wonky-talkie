"""Minimal agent client for the voice listener.

Shows the three things an agent has to do:
  1. react to `transcript` events (the user's turn),
  2. bracket its own speech with `speaking_started` / `speaking_ended`
     (that is what arms barge-in and opens the follow-up window),
  3. stop talking the moment `interruption_detected` arrives.

The "brain" is a stub: replace `think()` with your LLM call.

    python examples/agent_client.py                       # simulated speech (prints words)
    python examples/agent_client.py --tts-cmd "espeak-ng" # real, interruptible TTS
    python examples/agent_client.py --tts-cmd \
      "piper -m en_US-lessac-medium.onnx --output-raw | aplay -q -r 22050 -f S16_LE -t raw -"

Requires: pip install websockets
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal

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


class Agent:
    def __init__(self, ws, tts_cmd: str | None, words_per_s: float):
        self.ws = ws
        self.tts_cmd = tts_cmd
        self.words_per_s = words_per_s
        self.history: list[dict] = []
        self.turn: asyncio.Task | None = None
        self._midline = False  # simulated speech is printed word by word

    async def send(self, **cmd) -> None:
        await self.ws.send(json.dumps(cmd))

    async def run(self) -> None:
        await self.send(type="subscribe", events=[
            "session_started", "session_ended", "wake_detected", "listening", "speech_started",
            "partial_transcript", "silence_detected", "transcript", "utterance_discarded",
            "interruption_detected", "listen_timeout", "error"])
        async for raw in self.ws:
            ev = json.loads(raw)
            await self.on_event(ev)

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
            print(f"<- {t:<22} {detail}", flush=True)

        if t == "transcript":
            if self.turn and not self.turn.done():
                self.turn.cancel()
            self.history.append({"role": "user", "content": ev["text"]})
            self.turn = asyncio.create_task(self.take_turn())
        elif t == "interruption_detected":
            # The listener is already capturing what the user is saying; our job
            # is just to shut up immediately. The transcript follows.
            if self.turn and not self.turn.done():
                self.turn.cancel()
                print("   (agent stopped talking)", flush=True)
        elif t == "session_ended":
            self.history.clear()

    async def take_turn(self) -> None:
        reply, expects_reply = await think(self.history)
        self.history.append({"role": "assistant", "content": reply})
        await self.send(type="speaking_started", text=reply)
        try:
            await self.speak(reply)
        except asyncio.CancelledError:
            # Interrupted. The listener already moved on; nothing else to send.
            self.history[-1]["content"] += " [interrupted]"
            raise
        await self.send(type="speaking_ended", then="listen" if expects_reply else "end")

    async def speak(self, text: str) -> None:
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
            try:
                os.killpg(proc.pid, signal.SIGTERM)  # kills the whole pipeline (piper | aplay)
            except ProcessLookupError:
                pass
            raise


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8765/")
    p.add_argument("--tts-cmd", help="shell command that speaks text from stdin")
    p.add_argument("--words-per-s", type=float, default=3.0, help="simulated speaking rate")
    args = p.parse_args()
    async with connect(args.url) as ws:
        hello = json.loads(await ws.recv())
        print(f"connected: state={hello['state']} wake={hello['wake_models']} stt={hello['stt_backend']}")
        await Agent(ws, args.tts_cmd, args.words_per_s).run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
