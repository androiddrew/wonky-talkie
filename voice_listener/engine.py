"""The listener state machine.

One thread owns all state. Audio blocks, client commands and finished STT jobs
arrive through a single inbox queue and are handled strictly in order, so there
are no locks and behaviour is deterministic for a given input.

States (reported in `state_changed` events):

    idle        no session; only the wake word is listened for
    listening   session open, waiting for the user to start talking
                (no wake word needed); times out
    capturing   user is talking; recording until end-of-speech silence
    processing  utterance finished; transcription running
    waiting     transcript delivered; the agent is thinking
    speaking    the agent says its TTS is playing; barge-in armed

Time is counted in samples of the 16 kHz stream (the `t` field on events).
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .audio import SAMPLE_RATE, RingBuffer, to_wav_bytes
from .config import Config
from .stt import SttBackend, SttResult
from .textutil import is_hallucination, strip_wake_phrase
from .vad import VAD_CHUNK, Endpointer, SustainedSpeech
from .wake import WAKE_CHUNK, WakeDetector

log = logging.getLogger(__name__)

IDLE, LISTENING, CAPTURING, PROCESSING, WAITING, SPEAKING = (
    "idle", "listening", "capturing", "processing", "waiting", "speaking")

Emit = Callable[[dict], None]


def _ms(n_samples: int) -> int:
    return int(round(n_samples * 1000 / SAMPLE_RATE))


def _samples(ms: float) -> int:
    return int(ms * SAMPLE_RATE / 1000)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class Utterance:
    id: str
    session_id: str
    trigger: str  # wake_word | followup | command | barge_in | continuation
    start: int  # first sample of the audio slice
    speech_start: int
    speech_chunks: int = 0
    end: int = 0
    end_reason: str = ""
    ended_wall: float = 0.0


@dataclass
class SttJob:
    kind: str  # "final" | "partial"
    utt: Utterance
    audio: np.ndarray
    result: SttResult | None = None
    error: str | None = None
    elapsed_ms: int = 0
    extra: dict = field(default_factory=dict)


class Engine:
    def __init__(self, cfg: Config, vad: Callable[[np.ndarray], float], wake: WakeDetector,
                 stt: SttBackend, emit: Emit):
        self.cfg = cfg
        self.vad = vad
        self.wake = wake
        self.stt = stt
        self._emit_cb = emit
        self.inbox: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

        v, b = cfg.vad, cfg.barge_in
        self.ring = RingBuffer(max(60.0, cfg.session.max_utterance_s * 2 + 10))
        self.endpointer = Endpointer(v.threshold, v.neg_threshold, v.min_speech_ms, v.end_silence_ms)
        self.barge = SustainedSpeech(b.vad_threshold, b.min_speech_ms)

        self.state = IDLE
        self.session_id: str | None = None
        self.session_trigger = ""
        self.utt: Utterance | None = None
        self.pos = 0  # samples processed by the VAD so far (the engine's clock)

        self._vad_buf = np.zeros(0, dtype=np.int16)
        self._wake_buf = np.zeros(0, dtype=np.int16)
        self._wake_pos = 0
        self._wake_block_until = 0

        self._listen_deadline = 0
        self._listen_timeout_s = cfg.session.followup_timeout_s
        self._listen_lower_bound = 0
        self._listen_not_before = 0
        self._listen_trigger = "followup"
        self._speaking_since = 0
        self._speaking_text = ""
        self._waiting_since = 0

        self._stt_busy = False
        self._stt_queue: deque[SttJob] = deque()
        self._last_partial_pos = 0

        self._level_every = _samples(cfg.debug.level_interval_ms) if cfg.debug.level_interval_ms else 0
        self._level_acc = 0
        self._level_sq = 0.0
        self._level_vad = 0.0

        self._save_dir = cfg.resolve_path(cfg.debug.save_audio_dir) if cfg.debug.save_audio_dir else None
        if self._save_dir:
            self._save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ API
    # These are safe to call from any thread.

    def feed_audio(self, block: np.ndarray) -> None:
        self.inbox.put(("audio", block))

    def post_command(self, cmd: dict, client_id: str | None = None) -> None:
        self.inbox.put(("cmd", cmd, client_id))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.inbox.put(("stop",))
        if self._thread:
            self._thread.join(timeout=5)
        self._stt_pool.shutdown(wait=False, cancel_futures=True)

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "session_id": self.session_id,
            "utterance_id": self.utt.id if self.utt else None,
            "t": round(self.pos / SAMPLE_RATE, 3),
            "wake_models": list(self.wake.names),
            "stt_backend": self.stt.name,
            "barge_in": {
                "enabled": self.cfg.barge_in.enabled,
                "triggers": list(self.cfg.barge_in.triggers),
            },
        }

    # ------------------------------------------------------------ main loop

    def _run(self) -> None:
        while True:
            item = self.inbox.get()
            kind = item[0]
            if kind == "stop":
                return
            try:
                if kind == "audio":
                    self._on_audio(item[1])
                elif kind == "cmd":
                    self._on_command(item[1], item[2])
                elif kind == "stt":
                    self._on_stt(item[1])
            except Exception as e:  # keep the listener alive whatever happens
                log.exception("engine error")
                self._emit("error", where=kind, message=f"{type(e).__name__}: {e}")

    # ---------------------------------------------------------------- audio

    def _on_audio(self, block: np.ndarray) -> None:
        self.ring.write(block)

        self._vad_buf = np.concatenate([self._vad_buf, block])
        while len(self._vad_buf) >= VAD_CHUNK:
            chunk, self._vad_buf = self._vad_buf[:VAD_CHUNK], self._vad_buf[VAD_CHUNK:]
            start = self.pos
            self.pos += VAD_CHUNK
            prob = self.vad(chunk)
            self._meter(chunk, prob)
            self._on_vad(prob, start)
            self._check_timers()

        if self.cfg.wake.enabled:
            self._wake_buf = np.concatenate([self._wake_buf, block])
            while len(self._wake_buf) >= WAKE_CHUNK:
                chunk, self._wake_buf = self._wake_buf[:WAKE_CHUNK], self._wake_buf[WAKE_CHUNK:]
                self._wake_pos += WAKE_CHUNK
                self._on_wake_scores(self.wake.process(chunk), self._wake_pos)

    def _meter(self, chunk: np.ndarray, prob: float) -> None:
        if not self._level_every:
            return
        self._level_sq += float(np.sum(chunk.astype(np.float64) ** 2))
        self._level_acc += len(chunk)
        self._level_vad = max(self._level_vad, prob)
        if self._level_acc >= self._level_every:
            rms = math.sqrt(self._level_sq / self._level_acc) / 32768.0
            db = 20 * math.log10(rms) if rms > 1e-9 else -120.0
            self._emit("level", rms_db=round(db, 1), vad=round(self._level_vad, 3), state=self.state)
            self._level_acc, self._level_sq, self._level_vad = 0, 0.0, 0.0

    def _on_vad(self, prob: float, start: int) -> None:
        ev = self.endpointer.update(prob, start)
        st = self.state

        if st == CAPTURING:
            utt = self.utt
            if prob >= self.cfg.vad.threshold:
                utt.speech_chunks += 1
            if ev is not None and ev.kind == "offset":
                self._end_utterance(ev.sample, "silence")
            elif self.pos - utt.start >= self.cfg.session.max_utterance_s * SAMPLE_RATE:
                self._end_utterance(self.pos, "max_duration")
            else:
                self._maybe_partial()

        elif st == LISTENING:
            if ev is not None and ev.kind == "onset" and ev.sample >= self._listen_not_before:
                self._begin_capture(self._listen_trigger, ev.sample, self._listen_lower_bound)
            elif ev is not None and ev.kind == "onset":
                self.endpointer.reset()  # echo tail right after the agent stopped talking

        elif st == PROCESSING:
            # The user kept talking after the endpoint: capture it as a new utterance.
            if ev is not None and ev.kind == "onset":
                self._begin_capture("continuation", ev.sample, 0)

        elif st == WAITING:
            if ev is not None and ev.kind == "onset" and self.cfg.barge_in.during_thinking:
                self._emit("interruption_detected", trigger="speech", agent_state="thinking",
                           speech_ms=_ms(self.pos - ev.sample))
                self._begin_capture("barge_in", ev.sample, 0)

        elif st == SPEAKING:
            b = self.cfg.barge_in
            if (b.enabled and "speech" in b.triggers
                    and start >= self._speaking_since + _samples(b.grace_ms)):
                run_start = self.barge.update(prob, start)
                if run_start is not None:
                    self._emit("interruption_detected", trigger="speech", agent_state="speaking",
                               speech_ms=_ms(self.pos - run_start))
                    self._begin_capture("barge_in", run_start, 0)

    def _on_wake_scores(self, scores: list[tuple[str, float]], pos: int) -> None:
        if not scores or pos < self._wake_block_until:
            return
        name, score = max(scores, key=lambda s: s[1])
        if score < self.cfg.wake.threshold:
            return
        st = self.state
        if st == CAPTURING:
            return  # the wake word inside an utterance is just words
        if st == SPEAKING and not (self.cfg.barge_in.enabled and "wake_word" in self.cfg.barge_in.triggers):
            return

        self._wake_block_until = pos + int(self.cfg.wake.refractory_s * SAMPLE_RATE)
        self.wake.reset()
        if self.session_id is None:
            self._start_session("wake_word")
        self._emit("wake_detected", model=name, score=round(score, 3))
        if st in (WAITING, SPEAKING):
            self._emit("interruption_detected", trigger="wake_word",
                       agent_state="speaking" if st == SPEAKING else "thinking")
        lower = pos - _samples(self.cfg.wake.command_preroll_ms)
        trigger = "barge_in" if st in (WAITING, SPEAKING) else "wake_word"
        self._enter_listening(self.cfg.session.after_wake_timeout_s, trigger, lower_bound=lower)

    def _check_timers(self) -> None:
        if self.state == LISTENING and self.pos >= self._listen_deadline:
            self._emit("listen_timeout", waited_s=self._listen_timeout_s)
            self._end_session("timeout")
        elif (self.state == WAITING and self.cfg.session.agent_timeout_s > 0
              and self.pos - self._waiting_since >= self.cfg.session.agent_timeout_s * SAMPLE_RATE):
            self._end_session("agent_timeout")

    # ---------------------------------------------------- state transitions

    def _set_state(self, new: str) -> None:
        if new != self.state:
            old, self.state = self.state, new
            self._emit("state_changed", **{"from": old, "to": new})

    def _start_session(self, trigger: str) -> None:
        self.session_id = _new_id("ses")
        self.session_trigger = trigger
        self._emit("session_started", trigger=trigger)

    def _end_session(self, reason: str) -> None:
        if self.session_id is None:
            return
        sid = self.session_id
        self.utt = None
        self._set_state(IDLE)
        self.session_id = None
        self.endpointer.reset()
        self._emit("session_ended", session_id=sid, reason=reason)

    def _enter_listening(self, timeout_s: float, trigger: str, lower_bound: int,
                         not_before: int = 0, end_silence_ms: int | None = None) -> None:
        self.utt = None
        self.endpointer.reset()
        self.endpointer.end_silence_ms = end_silence_ms or self.cfg.vad.end_silence_ms
        self._listen_timeout_s = timeout_s
        self._listen_deadline = self.pos + int(timeout_s * SAMPLE_RATE)
        self._listen_lower_bound = lower_bound
        self._listen_not_before = not_before
        self._listen_trigger = trigger
        self._set_state(LISTENING)
        self._emit("listening", trigger=trigger, timeout_s=timeout_s)

    def _begin_capture(self, trigger: str, speech_start: int, lower_bound: int) -> None:
        start = max(speech_start - _samples(self.cfg.vad.speech_preroll_ms), lower_bound, self.ring.oldest)
        self.utt = Utterance(id=_new_id("utt"), session_id=self.session_id, trigger=trigger,
                             start=start, speech_start=speech_start,
                             speech_chunks=max(1, (self.pos - speech_start) // VAD_CHUNK))
        if self.state != LISTENING:  # per-window overrides only apply to `listen` windows
            self.endpointer.end_silence_ms = self.cfg.vad.end_silence_ms
        self.endpointer.force_speech()  # make sure an offset will follow
        self.barge.reset()
        self._last_partial_pos = self.pos
        self._set_state(CAPTURING)
        self._emit("speech_started", utterance_id=self.utt.id, trigger=trigger,
                   speech_start_t=round(speech_start / SAMPLE_RATE, 3))

    def _end_utterance(self, end_sample: int, reason: str) -> None:
        utt = self.utt
        utt.end = min(end_sample + _samples(200), self.pos)  # keep a little trailing audio
        utt.end_reason = reason
        utt.ended_wall = time.time()
        speech_ms = utt.speech_chunks * VAD_CHUNK * 1000 // SAMPLE_RATE
        self._emit("silence_detected", utterance_id=utt.id, reason=reason,
                   audio_s=round((utt.end - utt.start) / SAMPLE_RATE, 2),
                   speech_s=round(speech_ms / 1000, 2))
        if speech_ms < self.cfg.vad.min_utterance_ms:
            self._emit("utterance_discarded", utterance_id=utt.id, reason="too_short")
            self._listen_again()
            return
        audio = self.ring.read(utt.start, utt.end)
        if self._save_dir:
            (self._save_dir / f"{utt.id}.wav").write_bytes(to_wav_bytes(audio))
        self._set_state(PROCESSING)
        self._submit_stt(SttJob("final", utt, audio))

    def _listen_again(self) -> None:
        """After a discarded utterance, keep the session open and listen again."""
        timeout = self._listen_timeout_s if self.utt and self.utt.trigger in ("wake_word", "followup", "command") \
            else self.cfg.session.followup_timeout_s
        trigger = self.utt.trigger if self.utt and self.utt.trigger in ("wake_word", "command") else "followup"
        self._enter_listening(timeout, trigger, lower_bound=self.pos)

    # ------------------------------------------------------------------ STT

    def _submit_stt(self, job: SttJob) -> None:
        if self._stt_busy:
            if job.kind == "final":
                self._stt_queue.append(job)
            return
        self._stt_busy = True
        self._stt_pool.submit(self._run_stt, job)

    def _run_stt(self, job: SttJob) -> None:  # runs on the STT thread
        t0 = time.monotonic()
        try:
            job.result = self.stt.transcribe(job.audio, partial=job.kind == "partial")
        except Exception as e:
            log.exception("transcription failed")
            job.error = f"{type(e).__name__}: {e}"
        job.elapsed_ms = int((time.monotonic() - t0) * 1000)
        self.inbox.put(("stt", job))

    def _maybe_partial(self) -> None:
        if not self.cfg.stt.partials or self._stt_busy:
            return
        if self.pos - self._last_partial_pos < _samples(self.cfg.stt.partial_interval_ms):
            return
        self._last_partial_pos = self.pos
        self._submit_stt(SttJob("partial", self.utt, self.ring.read(self.utt.start, self.pos)))

    def _on_stt(self, job: SttJob) -> None:
        self._stt_busy = False
        if self._stt_queue:
            self._submit_stt(self._stt_queue.popleft())

        utt = job.utt
        text = job.result.text if job.result else ""
        clean = strip_wake_phrase(text, self.cfg.wake.strip_phrases) if text else ""

        if job.kind == "partial":
            if self.state == CAPTURING and self.utt is utt and clean:
                self._emit("partial_transcript", utterance_id=utt.id, text=clean)
            return

        is_current = self.state == PROCESSING and self.utt is utt
        if job.error:
            self._emit("error", where="stt", utterance_id=utt.id, message=job.error)
        if utt.session_id != self.session_id:
            self._emit("utterance_discarded", session_id=utt.session_id, utterance_id=utt.id,
                       reason="session_ended", raw_text=text)
            return
        if self.stt.name != "none" and is_hallucination(clean, self.cfg.stt.drop_phrases):
            self._emit("utterance_discarded", utterance_id=utt.id,
                       reason="stt_error" if job.error else "no_text", raw_text=text)
            if is_current:
                self._listen_again()
            return

        self._emit(
            "transcript",
            _audio=job.audio,
            utterance_id=utt.id,
            text=clean,
            raw_text=text,
            language=job.result.language if job.result else None,
            trigger=utt.trigger,
            end_reason=utt.end_reason,
            audio_s=round(len(job.audio) / SAMPLE_RATE, 2),
            stt_ms=job.elapsed_ms,
            latency_ms=int((time.time() - utt.ended_wall) * 1000),
            backend=self.stt.name,
            **(job.result.extra if job.result else {}),
        )
        if is_current:
            self._waiting_since = self.pos
            self._set_state(WAITING)

    # ------------------------------------------------------------- commands

    def _on_command(self, cmd: dict, client_id: str | None) -> None:
        t = cmd.get("type")
        reply = {"_to": client_id} if client_id else {}

        if t == "speaking_started":
            if self.session_id is None:
                self._start_session("agent")
            if self.state == CAPTURING:  # agent talks over the user: close out what we have
                self._end_utterance(self.pos, "agent_spoke")
            self._speaking_text = str(cmd.get("text", ""))
            if self.state != SPEAKING:  # repeated notices (e.g. per sentence) don't restart grace
                self._speaking_since = self.pos
                self.barge.reset()
                self._set_state(SPEAKING)

        elif t == "speaking_ended":
            if self.state != SPEAKING:
                # Usually a barge-in already moved us on (to capturing, or to waiting
                # with the interruption's transcript); the late notice changes nothing.
                log.debug("speaking_ended ignored in state %s", self.state)
                return
            then = cmd.get("then", "listen")
            if then == "listen":
                self._listen_command(cmd, trigger="followup")
            elif then == "end":
                self._end_session("agent")
            elif then == "wait":
                self._waiting_since = self.pos
                self._set_state(WAITING)
            else:
                self._emit("error", message=f"speaking_ended: unknown then={then!r}", **reply)

        elif t == "listen":
            if self.state in (CAPTURING, PROCESSING):
                self._emit("error", message=f"listen ignored: already {self.state}", **reply)
                return
            trigger = "followup"
            if self.session_id is None:
                self._start_session("command")
                trigger = "command"
            self._listen_command(cmd, trigger)

        elif t == "end_session":
            self._end_session(cmd.get("reason", "client"))

        elif t == "get_state":
            self._emit("state", **self.snapshot(), **reply)

        elif t == "ping":
            self._emit("pong", echo=cmd.get("echo"), **reply)

        else:
            self._emit("error", message=f"unknown command {t!r}", **reply)

    def _listen_command(self, cmd: dict, trigger: str) -> None:
        holdoff = _samples(self.cfg.session.post_speech_holdoff_ms) if self.state == SPEAKING else 0
        self._enter_listening(
            float(cmd.get("timeout_s") or self.cfg.session.followup_timeout_s),
            trigger,
            lower_bound=self.pos,
            not_before=self.pos + holdoff,
            end_silence_ms=cmd.get("end_silence_ms"),
        )

    # --------------------------------------------------------------- events

    def _emit(self, type_: str, **fields) -> None:
        ev = {"type": type_, "ts": round(time.time(), 3), "t": round(self.pos / SAMPLE_RATE, 3),
              "session_id": self.session_id}
        ev.update(fields)
        self._emit_cb(ev)
