"""Command line entry point: python -m wonky_talkie --config configs/desktop.toml"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time

from .config import load_config

log = logging.getLogger("wonky_talkie")


def _print_event(ev: dict) -> None:
    fields = {k: v for k, v in ev.items() if k not in ("type", "ts", "t", "_audio", "_to")}
    if fields.get("session_id") is None:
        fields.pop("session_id", None)
    print(f"[{ev['t']:8.2f}] {ev['type']:<22} {json.dumps(fields, ensure_ascii=False)}", flush=True)


def _say_once(cfg, text: str, make_tts_backend) -> int:
    """--say: synthesize one line and play it with no mic, server or engine."""
    import numpy as np
    import sounddevice as sd

    backend = make_tts_backend(cfg.tts)
    t0 = time.monotonic()
    pcm = backend.synth(text)
    dur = len(pcm) / backend.sample_rate
    wall = time.monotonic() - t0
    log.info("%s: %.2fs of audio in %.2fs (%.1fx realtime) at %d Hz",
             backend.name, dur, wall, dur / wall if wall else 0.0, backend.sample_rate)
    device = cfg.audio.output_device or None
    sd.play(np.asarray(pcm), backend.sample_rate, device=device, blocking=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wonky_talkie", description=__doc__)
    p.add_argument("--config", "-c", help="TOML config (see configs/)")
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.add_argument("--host", help="override server.host")
    p.add_argument("--port", type=int, help="override server.port")
    p.add_argument("--device", help="override audio.device")
    p.add_argument("--stt-backend", choices=["faster_whisper", "whisper_cpp", "openai", "none"],
                   help="override stt.backend")
    p.add_argument("--tts-backend", choices=["kitten", "piper", "command", "openai", "none"],
                   help="override tts.backend")
    p.add_argument("--voice", help="override the backend's voice (tts.<backend>.voice)")
    p.add_argument("--say", metavar="TEXT",
                   help="speak this once at startup and exit (to check TTS on its own)")
    p.add_argument("--print-events", action="store_true", help="also print events to stdout")
    p.add_argument("--print-levels", action="store_true", help="include the 10 Hz level meter when printing")
    p.add_argument("--input-file", help="read audio from a WAV file instead of the mic")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed for --input-file (0 = max)")
    p.add_argument("--simulate-echo", action="store_true",
                   help="with --input-file: pass the listener's playback through a simulated "
                        "speaker->room->mic echo path (to try echo handling without hardware)")
    p.add_argument("--exit-when-done", type=float, metavar="SECONDS",
                   help="with --input-file: exit this many seconds after the file ends")
    p.add_argument("--wake-at", type=float, action="append", metavar="SECONDS",
                   help="simulate a wake word at this stream time instead of running openWakeWord "
                        "(repeatable; for testing without a mic or models)")
    p.add_argument("--log-level", help="override debug.log_level")
    args = p.parse_args(argv)

    if args.list_devices:
        from .audio import list_devices

        print(list_devices())
        return 0

    cfg = load_config(args.config)
    if args.host:
        cfg.server.host = args.host
    if args.port is not None:
        cfg.server.port = args.port
    if args.device is not None:
        cfg.audio.device = args.device
    if args.stt_backend:
        cfg.stt.backend = args.stt_backend
    if args.tts_backend:
        cfg.tts.backend = args.tts_backend
    if args.voice:
        for section in (cfg.tts.kitten, cfg.tts.http):
            if hasattr(section, "voice"):
                section.voice = args.voice
    logging.basicConfig(
        level=(args.log_level or cfg.debug.log_level).upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Heavy imports after arg parsing so --help and --list-devices stay fast.
    from .aec import EchoStage, SpeexAEC, load_speexdsp
    from .audio import DuplexSource, EchoRoom, FileSource, MicSource, Player
    from .engine import Engine
    from .server import EventServer
    from .stt import make_backend
    from .tts import TtsService, make_backend as make_tts_backend
    from .vad import SileroVAD
    from .wake import NullWakeDetector, OpenWakeWordDetector, ScriptedWakeDetector

    vad_path = cfg.resolve_path(cfg.vad.model_path)
    if not vad_path.exists():
        log.error("Silero VAD model not found at %s (see README: Models)", vad_path)
        return 2
    vad = SileroVAD(str(vad_path), threads=cfg.vad.threads)

    if args.wake_at:
        cfg.wake.enabled = True
        wake = ScriptedWakeDetector(args.wake_at)
    elif cfg.wake.enabled:
        wake = OpenWakeWordDetector(cfg.wake.models, cfg.wake.auto_download,
                                    cfg.wake.melspec_model_path, cfg.wake.embedding_model_path)
    else:
        wake = NullWakeDetector()

    stt = make_backend(cfg.stt)
    t0 = time.monotonic()
    try:
        stt.warmup()
    except Exception as e:
        log.warning("STT warm-up failed (%s); will retry on first utterance", e)
    log.info("STT backend %s ready in %.1fs", stt.name, time.monotonic() - t0)

    if args.say:
        return _say_once(cfg, args.say, make_tts_backend)

    server_holder: dict = {}

    def emit(ev: dict) -> None:
        if args.print_events and (ev["type"] != "level" or args.print_levels) and "_to" not in ev:
            _print_event(ev)
        srv = server_holder.get("server")
        if srv is not None:
            srv.publish(ev)

    # Audio I/O. With playback on, the listener owns the speaker as well as the mic.
    a = cfg.audio
    player = room = None
    if args.input_file:
        if a.playback:
            player = Player(16000)
            room = EchoRoom(player, echo=args.simulate_echo)
        source = FileSource(args.input_file, speed=args.speed, room=room)
    elif a.playback:
        source = DuplexSource(Player, a.device, a.output_device, a.sample_rate, a.channels,
                              a.channel, a.output_channels, a.block_ms)
        player = source.player
    else:
        source = MicSource(a.device, a.sample_rate, a.channels, a.channel, a.block_ms)

    echo = None
    if player is not None:
        aec = None
        if cfg.aec.enabled:
            lib = load_speexdsp(cfg.aec.library)
            if lib is None:
                log.warning("libspeexdsp not found: echo cancellation OFF (sudo apt install "
                            "libspeexdsp1). Duck-and-verify barge-in still works.")
            else:
                aec = SpeexAEC(lib, cfg.aec.filter_ms, cfg.aec.residual_suppression,
                               cfg.aec.noise_suppress_db)
                log.info("Echo cancellation on (SpeexDSP, %d ms filter)", cfg.aec.filter_ms)
        echo = EchoStage(aec, cfg.aec.hold_ms)

    engine = Engine(cfg, vad, wake, stt, emit, player=player, echo=echo)

    # In-process TTS: wonky-talkie speaks, so it owns the reference signal that
    # echo cancellation and duck-and-verify barge-in depend on.
    tts_service = None
    if player is not None and cfg.tts.backend != "none":
        t0 = time.monotonic()
        try:
            tts_backend = make_tts_backend(cfg.tts)
            if cfg.tts.warmup:
                tts_backend.warmup()
            tts_service = TtsService(cfg.tts, tts_backend, player, emit=engine.post_event)
            log.info("TTS backend %s ready in %.1fs (%d Hz)", tts_backend.name,
                     time.monotonic() - t0, tts_backend.sample_rate)
        except Exception as e:
            log.error("TTS unavailable (%s); `say` will be refused, `play` still works", e)
    elif cfg.tts.backend != "none":
        log.warning("audio.playback is false, so there is no speaker to synthesize to; "
                    "`say` will be refused")

    server = EventServer(engine, cfg.server.host, cfg.server.port, cfg.server.token,
                         tts=tts_service)
    server_holder["server"] = server

    async def run() -> None:
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:  # Windows
                pass

        started = asyncio.Event()
        server_task = asyncio.create_task(server.serve_forever(started, stop))
        await started.wait()
        engine.start()

        if args.input_file:
            def done():
                log.info("input file finished")
                if args.exit_when_done is not None:
                    loop.call_soon_threadsafe(loop.call_later, args.exit_when_done, stop.set)

            source.on_done = done
        source.start(engine.feed_audio)
        engine.io_latency_ms = getattr(source, "round_trip_ms", 0)
        try:
            await stop.wait()
        finally:
            stop.set()
            source.stop()
            engine.stop()
            try:
                await asyncio.wait_for(server_task, 5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
