# wonky-talkie ↔ agent protocol (v3)

Connect a WebSocket to `ws://HOST:PORT/` (default `ws://127.0.0.1:8765/`; add
`?token=...` or an `Authorization: Bearer ...` header if `server.token` is set).
Every text message in both directions is one JSON object with a `type`. Binary
messages from the agent are audio for wonky-talkie to play (see `play`), which
most agents never need.

Any number of clients may connect. Events go to every subscribed client;
replies to `get_state` / `ping` / `say` / `play` / bad commands go only to the
client that sent them.

**v3 moves the voice into wonky-talkie.** The agent sends `say` with *text* and
wonky-talkie synthesizes it, plays it, and cuts it off when interrupted. The
`hello` event carries a `tts` object (backend, voice, sample rate), or `null`
when no TTS is configured. v2's `play` + binary PCM and v1's
`speaking_started` / `speaking_ended` both still work, for an agent that wants
to own its own voice.

## Why wonky-talkie owns the voice

If the agent plays its own TTS, wonky-talkie can't tell the agent's voice
coming back through the mic from the user talking, so speech barge-in only
works with a headset or system echo cancellation. Owning the speaker — and, from
v3, the synthesizer — gives it:

- **Echo cancellation.** It knows exactly what the speaker played and cancels it
  from the mic (SpeexDSP).
- **Duck-and-verify.** When speech is heard during playback, it lowers the
  playback volume (−20 dB) for about 350 ms and compares the mic level with what
  pure echo would produce at that volume. The agent's echo follows the volume
  down; a person keeps talking at the same level. Only then does it stop.
- **Instant stop.** On a real interruption it stops the audio itself, with no
  round trip to the agent — mid-word if need be.
- **Nothing wasted.** Because it also synthesizes, an interruption cancels the
  *unsaid* sentences instead of throwing away audio the agent already made.
- **Accurate end of speech.** It knows when the audio really finished, which is
  when the follow-up listening window should open.

A reply is synthesized sentence by sentence (`tts.first_chunk_chars`,
`tts.max_chars`), so the first words start playing after one sentence rather
than after the whole paragraph.

## Events (wonky-talkie → agent)

Every event carries:

| field | meaning |
|---|---|
| `type` | event name |
| `ts` | Unix wall-clock time (s) |
| `t` | stream time (s since start): wonky-talkie's own clock |
| `session_id` | current session, or `null` |

| event | when | extra fields |
|---|---|---|
| `hello` | on connect | `protocol`, `client_id`, `state`, `session_id`, `utterance_id`, `wake_models`, `stt_backend`, `tts` (object, or `null`), `barge_in`, `playback` (bool), `playback_rate`, `aec` (bool), `events`, `commands` |
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
| `say_accepted` | reply to `say` | `playback_id`, `backend`, `sample_rate`, `voice`* and other backend details |
| `play_accepted` | reply to `play` | `playback_id` |
| `playback_started` | the first sample of the reply went to the speaker | `playback_id` |
| `playback_finished` | the item played to the end | `playback_id`, `played_s`, `aec_erle_db`* (echo removed, dB) |
| `playback_stopped` | the item was cut off | `playback_id`, `reason`: `interrupted` \| `client` \| `session_ended`, `played_s`, `total_s` (`null` if `play_end` hadn't arrived) |
| `interruption_suspected` | speech heard during playback; the agent is ducked while it's checked | `speech_ms` |
| `interruption_rejected` | it was the agent's own echo; volume restored | `reason: echo`, `excess_db`, `voiced` |
| `interruption_detected` | the user cut in on the agent's turn | `trigger`: `speech` \| `wake_word`, `agent_state`: `speaking` \| `thinking`, `speech_ms`*, `excess_db`*, `playback_id`*, `played_s`*, `total_s`* |
| `listen_timeout` | a listening window closed with no speech | `waited_s` (a `session_ended` follows) |
| `session_ended` | session closed, back to idle | `reason`: `timeout` \| `agent` \| `client` \| `agent_timeout` \| your own reason |
| `level` | ~10 Hz meter, **only if subscribed explicitly** | `rms_db`, `vad`, `state` |
| `state` | reply to `get_state` | same fields as `hello`'s snapshot |
| `pong` | reply to `ping` | `echo` |
| `tts_failed` | synthesis failed for a `say` (sender only) | `playback_id`, `message` |
| `error` | bad command, STT/TTS failure, internal error | `message`, `where`*, `utterance_id`* |

\* only when relevant. \*\* only for clients that subscribed with `include_audio: true`
(16 kHz mono 16-bit WAV, base64).

`trigger` on `speech_started` / `transcript` says why it was capturing:
`wake_word`, `followup` (after the agent asked something), `command` (after a
`listen` command), `barge_in` (the user interrupted), or `continuation` (the user
kept talking while the previous utterance was being transcribed).

`text` has the wake phrase stripped ("Hey Jarvis, lights off" → "Lights off");
`raw_text` is Whisper's output as-is.

`excess_db` is how much louder the mic was, after ducking, than the reply's own
echo alone would have been. Echo lands near 0 dB and a person talking over the
reply lands well above that. The cut-off is `barge_in.verify_min_excess_db`
(default 6 dB).

## Commands (agent → wonky-talkie)

| command | fields | effect |
|---|---|---|
| `subscribe` | `events`: list of event types, or `["*"]` for everything; omit for the default (all but `level`). `include_audio`: bool | choose what this client receives |
| `say` | `text` (required), `then`: `"listen"` (default) \| `"end"` \| `"wait"`, `timeout_s`, `playback_id` | **The normal way to speak.** wonky-talkie synthesizes `text` with its configured TTS backend and plays it. Replies `say_accepted` with the `playback_id` and the backend in use. Sentence by sentence, so it starts speaking quickly; a barge-in cancels whatever is left unsaid. Opens a session (`trigger: agent`) if none. `then` is applied when it has finished, exactly as for `play`. A new `say` completes the previous item. Refused with an `error` if `audio.playback` is false or `tts.backend = "none"`. |
| `play` | `sample_rate` (required, 8000–96000), `text`, `then`: `"listen"` (default) \| `"end"` \| `"wait"`, `timeout_s`, `playback_id`, `audio_b64` (optional: whole clip inline) | **Legacy / bring your own voice.** Start an item of agent-synthesized speech. Then send the audio as **binary messages of 16-bit little-endian mono PCM** at `sample_rate` (any chunk size, as fast as you generate it), then `play_end`. wonky-talkie plays it, handles barge-in, and when it has finished applies `then`: `listen` opens a follow-up window, `end` closes the session, `wait` keeps it open. Opens a session (`trigger: agent`) if none. A new `play` completes the previous item. |
| `play_end` | `playback_id` (optional) | No more audio for this item. |
| `stop_playback` | `reason` (optional) | Cut off playback now (e.g. the agent changed its mind). The state goes to `waiting`. |
| `speaking_started` | `text` | **Legacy; only if the agent plays audio itself.** Send when its audio starts playing. Arms barge-in (immediate, with no ducking; needs a headset or system AEC for speech barge-in). |
| `speaking_ended` | `then`, `timeout_s`, `end_silence_ms` | **Legacy; only if the agent plays audio itself.** Send when its audio has finished playing. Same `then` as `play`. Ignored unless the state is `speaking`. |
| `listen` | `timeout_s`, `end_silence_ms` | Open a listening window now: push-to-talk, or "listen without the wake word". Opens a session (`trigger: command`) if none. |
| `end_session` | `reason` (optional) | Close the session (stopping any playback) and go idle. |
| `get_state` | | Reply with a `state` event. |
| `ping` | `echo` (optional) | Reply with `pong`. |

Use one `say` (or one `play`) per reply. `then` is applied when the last item
has finished. Audio that arrives for an item after it was interrupted is dropped
silently, so an agent streaming `play` audio can simply stop sending when it
sees `interruption_detected`.

Choosing between them:

| | `say` | `play` |
|---|---|---|
| the agent sends | text | `sample_rate` + binary PCM |
| who synthesizes | wonky-talkie | the agent |
| voice configured in | `[tts]` on the host | the agent |
| interruption cancels | the unsaid sentences | queued audio only |
| needs | `tts.backend` ≠ `"none"` | nothing |

If `say` fails (a dead TTS binary, a bad voice, an HTTP error) the sender gets
`tts_failed` with a `message`, and the playback item is closed so the state
machine doesn't wait for audio that will never come.

## States

```
idle        ─ wake word, or listen ─────────────▶ listening
idle        ─ say / play / speaking_started ────▶ speaking
listening   ─ speech ───────────────────────────▶ capturing
listening   ─ timeout ──────────────────────────▶ idle
capturing   ─ end-of-speech silence ────────────▶ processing
processing  ─ transcript delivered ─────────────▶ waiting
processing  ─ user keeps talking ───────────────▶ capturing   (continuation)
waiting     ─ say / play / speaking_started ────▶ speaking
waiting     ─ speech / wake word ───────────────▶ capturing / listening   (interruption)
speaking    ─ barge-in confirmed ───────────────▶ capturing / listening   (interruption)
speaking    ─ audio done, then=listen ──────────▶ listening
speaking    ─ audio done, then=end ─────────────▶ idle
speaking    ─ stop_playback ────────────────────▶ waiting
any         ─ end_session, agent_timeout ───────▶ idle
```

- **idle**: only the wake word (or a `listen` / `say` / `play` / `speaking_started` command) does anything.
- **listening**: the next speech is captured without a wake word; times out.
- **capturing**: recording until `vad.end_silence_ms` of silence or `session.max_utterance_s`.
- **processing**: transcribing. If the user starts talking again, a new
  `continuation` utterance is captured.
- **waiting**: the transcript is out; it's the agent's turn. Speech here counts as an
  interruption (`agent_state: thinking`) if `barge_in.during_thinking`. The wake
  word always works here.
- **speaking**: the reply is playing (ours from `say`, or the agent's from `play`). Barge-in by `speech` and/or `wake_word`
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
-> say {text: "Done.", then: "end"}
<- say_accepted {playback_id: pb_1, backend: kitten, voice: expr-voice-5-m}
<- playback_started {playback_id: pb_1}
<- playback_finished {playback_id: pb_1, played_s: 0.6}
<- session_ended {reason: agent}
```

Clarifying question (no wake word for the answer):

```
<- transcript {text: "Turn on the lights."}
-> say {text: "Which room?", then: "listen", timeout_s: 8}
<- say_accepted {playback_id: pb_2}
<- playback_finished
<- listening {trigger: followup}
<- speech_started {trigger: followup}
<- transcript {text: "The office.", trigger: followup}
```

Interruption:

```
-> say {text: "Once upon a time there was a listener that never slept. ..."}
<- say_accepted {playback_id: pb_3}
<- playback_started
<- interruption_suspected {speech_ms: 320}          # agent ducked, checking
<- interruption_rejected {reason: echo}             # just the agent's echo; volume back
   ...
<- interruption_suspected {speech_ms: 320}
<- interruption_detected {trigger: speech, excess_db: 14.2, played_s: 3.9, total_s: 21.0}
<- playback_stopped {reason: interrupted}           # already stopped, mid-sentence
<- speech_started {trigger: barge_in}               # and capturing what the user says
<- transcript {text: "Stop, that's enough.", trigger: barge_in}
```

After `interruption_detected` the agent only has to stop generating. With `say`
there is no audio to stop sending: the unsaid sentences are simply never
synthesized. `played_s` / `total_s` tell it how much of the reply was actually
heard, which is worth recording in the conversation history — the user did not
hear the rest.
