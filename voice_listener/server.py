"""WebSocket pub/sub front end.

Clients (your agent, a debug UI, a logger) connect to ws://host:port/, receive
a `hello` event, and then get every engine event as one JSON object per
message. They can narrow what they receive with `subscribe`, and send commands
(`speaking_started`, `listen`, ...) that are forwarded to the engine.
See PROTOCOL.md for the full list.
"""

from __future__ import annotations

import asyncio
import base64
import http
import json
import logging
import uuid
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from .audio import to_wav_bytes
from .engine import Engine

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 1

EVENT_TYPES = [
    "hello", "state_changed", "session_started", "session_ended", "wake_detected",
    "listening", "speech_started", "partial_transcript", "silence_detected",
    "transcript", "utterance_discarded", "interruption_detected", "listen_timeout",
    "level", "state", "pong", "subscribed", "error",
]
COMMAND_TYPES = ["subscribe", "speaking_started", "speaking_ended", "listen",
                 "end_session", "get_state", "ping"]
HIGH_RATE = {"level"}


@dataclass
class _Client:
    id: str
    ws: ServerConnection
    events: set[str] | None = None  # None = everything except HIGH_RATE
    include_audio: bool = False
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=2000))

    def wants(self, type_: str) -> bool:
        if self.events is None:
            return type_ not in HIGH_RATE
        return "*" in self.events or type_ in self.events


class EventServer:
    def __init__(self, engine: Engine, host: str = "127.0.0.1", port: int = 8765, token: str = ""):
        self.engine = engine
        self.host, self.port, self.token = host, port, token
        self._clients: dict[str, _Client] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    # Called from the engine thread.
    def publish(self, event: dict) -> None:
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._dispatch, event)

    def _dispatch(self, event: dict) -> None:
        event = dict(event)
        target = event.pop("_to", None)
        audio = event.pop("_audio", None)
        plain = None
        with_audio = None
        for c in list(self._clients.values()):
            if target is not None:
                if c.id != target:
                    continue
            elif not c.wants(event["type"]):
                continue
            if audio is not None and c.include_audio:
                if with_audio is None:
                    b64 = base64.b64encode(to_wav_bytes(audio)).decode("ascii")
                    with_audio = json.dumps({**event, "audio_wav_b64": b64})
                msg = with_audio
            else:
                if plain is None:
                    plain = json.dumps(event)
                msg = plain
            try:
                c.queue.put_nowait(msg)
            except asyncio.QueueFull:
                log.warning("client %s is not reading; dropping %s", c.id, event["type"])

    def _check_auth(self, connection: ServerConnection, request):
        if not self.token:
            return None
        q = parse_qs(urlparse(request.path).query)
        auth = request.headers.get("Authorization", "")
        if q.get("token", [""])[0] == self.token or auth == f"Bearer {self.token}":
            return None
        return connection.respond(http.HTTPStatus.UNAUTHORIZED, "bad or missing token\n")

    async def _handler(self, ws: ServerConnection) -> None:
        client = _Client(id=f"cli_{uuid.uuid4().hex[:6]}", ws=ws)
        self._clients[client.id] = client
        log.info("client %s connected from %s", client.id, ws.remote_address)
        hello = {"type": "hello", "protocol": PROTOCOL_VERSION, "client_id": client.id,
                 **self.engine.snapshot(), "events": EVENT_TYPES, "commands": COMMAND_TYPES}
        await ws.send(json.dumps(hello))

        async def writer():
            while True:
                await ws.send(await client.queue.get())

        writer_task = asyncio.create_task(writer())
        try:
            async for raw in ws:
                try:
                    cmd = json.loads(raw)
                    if not isinstance(cmd, dict) or "type" not in cmd:
                        raise ValueError("expected a JSON object with a 'type'")
                except ValueError as e:
                    client.queue.put_nowait(json.dumps({"type": "error", "message": f"bad message: {e}"}))
                    continue
                if cmd["type"] == "subscribe":
                    ev = cmd.get("events")
                    client.events = None if ev is None else set(ev)
                    client.include_audio = bool(cmd.get("include_audio", False))
                    client.queue.put_nowait(json.dumps({
                        "type": "subscribed",
                        "events": "default" if client.events is None else sorted(client.events),
                        "include_audio": client.include_audio}))
                else:
                    self.engine.post_command(cmd, client.id)
        except ConnectionClosed:
            pass
        finally:
            writer_task.cancel()
            self._clients.pop(client.id, None)
            log.info("client %s disconnected", client.id)

    async def serve_forever(self, started: asyncio.Event | None = None,
                            stop: asyncio.Event | None = None) -> None:
        """Serve until `stop` is set, then close connections politely (1001)."""
        self._loop = asyncio.get_running_loop()
        stop = stop or asyncio.Event()
        async with serve(self._handler, self.host, self.port, process_request=self._check_auth,
                         max_size=2**20) as server:
            self.port = next(iter(server.sockets)).getsockname()[1]  # real port if 0 was asked
            log.info("Listening for agents on ws://%s:%d", self.host, self.port)
            if started is not None:
                started.set()
            await stop.wait()
            server.close()  # sends "going away" to clients
            await server.wait_closed()
