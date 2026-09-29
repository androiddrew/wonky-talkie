# Listener ↔ agent protocol (v1)

Connect a WebSocket to `ws://HOST:PORT/` (default `ws://127.0.0.1:8765/`; add
`?token=...` or an `Authorization: Bearer ...` header if `server.token` is set).
Every message in both directions is one JSON object with a `type`.

Any number of clients may connect. Events go to every subscribed client;
replies to `get_state` / `ping` / bad commands go only to the client that asked.

## Events (listener → agent)

Every event carries:

| field | meaning |
|---|---|
| `type` | event name |
| `ts` | Unix wall-clock time (s) |
| `t` | stream time (s since the listener started): the listener's own clock |
| `session_id` | current session, or `null` |

| event | when | extra fields |
|---|---|---|
| `hello` | on connect | `protocol`, `client_id`, `state`, `session_id`, `utterance_id`, `wake_models`, `stt_backend`, `barge_in`, `events`, `commands` |
| `subscribed` | reply to `subscribe` | `events` (`"default"` or list), `include_audio` |
| `state_changed` | every state transition | `from`, `to` |
| `session_started` | a session opens | `trigger`: `wake_word` \| `command` \| `agent` |
| `wake_detected` | wake word heard (not while the user is mid-utterance) | `model`, `score` |
| `listening` | a listening window opens; no wake word needed until it closes | `trigger`, `timeout_s` |
| `speech_started` | the user started talking; capture begins | `utterance_id`, `trigger`, `speech_start_t` |
| `partial_transcript` | interim text while talking (`stt.partials = true`) | `utterance_id`, `text` |
| `silence_detected` | end of the user's turn | `utterance_id`, `reason`: `silence` \| `max_duration` \| `agent_spoke`, `audio_s`, `speech_s` |
| `transcript` | final text for an utterance | `utterance_id`, `text`, `raw_text`, `language`, `trigger`, `end_reason`, `audio_s`, `stt_ms`, `latency_ms`, `backend`, `no_speech_prob`*, `audio_wav_b64`** |
| `utterance_discarded` | an utterance produced nothing usable | `utterance_id`, `reason`: `too_short` \| `no_text` \| `stt_error` \| `session_ended`, `raw_text`* |
| `interruption_detected` | the user cut in on the agent's turn | `trigger`: `speech` \| `wake_word`, `agent_state`: `speaking` \| `thinking`, `speech_ms`* |
| `listen_timeout` | a listening window closed with no speech | `waited_s` (a `session_ended` follows) |
| `session_ended` | session closed, back to idle | `reason`: `timeout` \| `agent` \| `client` \| `agent_timeout` \| your own reason |
| `level` | ~10 Hz meter, **only if subscribed explicitly** | `rms_db`, `vad`, `state` |
| `state` | reply to `get_state` | same fields as `hello`'s snapshot |
| `pong` | reply to `ping` | `echo` |
| `error` | bad command, STT failure, internal error | `message`, `where`*, `utterance_id`* |

\* only when relevant. \*\* only for clients that subscribed with `include_audio: true`
(16 kHz mono 16-bit WAV, base64).

`trigger` on `speech_started` / `transcript` says why the listener was capturing:
`wake_word`, `followup` (after the agent asked something), `command` (after a
`listen` command), `barge_in` (the user interrupted), or `continuation` (the user
kept talking while the previous utterance was being transcribed).

`text` has the wake phrase stripped ("Hey Jarvis, lights off" → "Lights off");
`raw_text` is Whisper's output as-is.

## Commands (agent → listener)

| command | fields | effect |
|---|---|---|
| `subscribe` | `events`: list of event types, or `["*"]` for everything; omit for the default (all but `level`). `include_audio`: bool | choose what this client receives |
| `speaking_started` | `text` (optional, informational) | Send when TTS audio **starts playing**. Arms barge-in. Opens a session (`trigger: agent`) if none, so the agent can speak first. If the user was mid-utterance, that utterance is closed and transcribed (`end_reason: agent_spoke`). Repeats while already speaking are harmless. |
| `speaking_ended` | `then`: `"listen"` (default) \| `"end"` \| `"wait"`; `timeout_s`; `end_silence_ms` | Send when the last TTS audio **has finished playing**. `listen` opens a follow-up window (no wake word needed), `end` closes the session, `wait` keeps it open for more agent output. Ignored unless the state is `speaking` (a barge-in may already have moved on). |
| `listen` | `timeout_s`, `end_silence_ms` | Open a listening window now: push-to-talk, or "listen without the wake word". Opens a session (`trigger: command`) if none. |
| `end_session` | `reason` (optional) | Close the session and go idle. |
| `get_state` | | Reply with a `state` event. |
| `ping` | `echo` (optional) | Reply with `pong`. |

## States

```
idle        ── wake word, or listen ───────────▶ listening
idle        ── speaking_started ───────────────▶ speaking
listening   ── speech ───────────────────────────▶ capturing
listening   ── timeout ──────────────────────────▶ idle
capturing   ── end-of-speech silence ────────────▶ processing
processing  ── transcript delivered ─────────────▶ waiting
processing  ── user keeps talking ───────────────▶ capturing  (continuation)
waiting     ── speaking_started ───────────────▶ speaking
waiting     ── speech / wake word ───────────────▶ capturing / listening  (interruption)
speaking    ── barge-in (speech / wake word) ────▶ capturing / listening  (interruption)
speaking    ── speaking_ended then=listen ─────▶ listening
speaking    ── speaking_ended then=end ────────▶ idle
any         ── end_session, agent_timeout ─────▶ idle
```

- **idle**: only the wake word (or a `listen` / `speaking_started` command) does anything.
- **listening**: the next speech is captured without a wake word; times out.
- **capturing**: recording until `vad.end_silence_ms` of silence or `session.max_utterance_s`.
- **processing**: transcribing. If the user starts talking again, a new
  `continuation` utterance is captured.
- **waiting**: the transcript is out; it's the agent's turn. Speech here counts as an
  interruption (`agent_state: thinking`) if `barge_in.during_thinking`. The wake
  word always works here.
- **speaking**: the agent's TTS is playing. Barge-in by `speech` and/or `wake_word`
  per `barge_in.triggers`; everything else is ignored.

## Typical exchanges

Command, answer, done:

```
<- session_started {trigger: wake_word}
<- wake_detected {model: hey_jarvis, score: 0.93}
<- listening {trigger: wake_word, timeout_s: 5}
<- speech_started
<- silence_detected {reason: silence}
<- transcript {text: "Turn off the lights."}
-> speaking_started {text: "Done."}
-> speaking_ended {then: "end"}
<- session_ended {reason: agent}
```

Clarifying question (no wake word for the answer):

```
<- transcript {text: "Turn on the lights."}
-> speaking_started {text: "Which room?"}
-> speaking_ended {then: "listen", timeout_s: 8}
<- listening {trigger: followup}
<- speech_started {trigger: followup}
<- transcript {text: "The office.", trigger: followup}
```

Interruption:

```
-> speaking_started {text: "Once upon a time..."}
<- interruption_detected {trigger: speech, agent_state: speaking}   # stop TTS now
<- speech_started {trigger: barge_in}                              # already capturing
<- silence_detected
<- transcript {text: "Stop, that's enough.", trigger: barge_in}
```

After `interruption_detected` the agent only needs to stop its audio. It doesn't
have to send `speaking_ended` (it's ignored if it does). It's worth recording in
the conversation history how much of the reply was actually heard.
