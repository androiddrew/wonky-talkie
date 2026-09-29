# voice-listener: context for Claude Code

An always-on voice front end for an agent: mic → wake word → utterance capture → Whisper,
published as JSON events over a WebSocket. The agent is a client: it subscribes to events
and sends back commands plus its TTS audio. Targets: a desktop with an RTX 4090 (Ubuntu), then
Jetson Orin on JetPack 7.2 (Ubuntu 24.04, Python 3.12, CUDA 13). Xavier and older are out of scope.

Built in a Claude chat session. This file carries over the decisions and state from it.
README.md (setup, tuning) and PROTOCOL.md (event/command contract, v2) are the user-facing docs.

## Run / test

```bash
sudo apt install libportaudio2 libspeexdsp1
pip install -r requirements-desktop.txt && pip install --no-deps openwakeword   # tflite dep has no py3.12 wheel
python scripts/download_models.py                      # openWakeWord ONNX models
python -m voice_listener -c configs/desktop.toml --print-events
python examples/agent_client.py --tts-cmd "espeak-ng --stdout"
python -m unittest discover -s tests -v                # 46 tests; AEC tests skip without libspeexdsp
```

No-hardware checks:
- Recorded conversation: `--input-file tests/data/demo_stream.wav --wake-at 1.95 --wake-at 13.80`
  (a scripted wake word stands in for openWakeWord).
- Simulated speaker→mic echo: add `--simulate-echo`. This needs real-time speed (the default
  `--speed 1`) so that ducking stays causal.
- `VOICE_LISTENER_SPEEXDSP=/path/libspeexdsp.so.1` points at a non-standard lib.

## Layout

- `voice_listener/engine.py`: the state machine (idle / listening / capturing / processing /
  waiting / speaking). A single thread owns all state. Audio, commands, STT results and player
  events arrive through one inbox queue, so there are no locks. Time is counted in 16 kHz
  samples (`t` on events), which keeps file and mic input deterministic.
- `audio.py`: RingBuffer (60 s, absolute sample index); `Resampler` (exact streaming
  polyphase, identical to scipy upfirdn for any ratio or block split); `Player` (agent audio
  queue, duck/stop, started/finished events); `DuplexSource` (one full-duplex sounddevice
  stream, speaker block = AEC reference); `MicSource` (capture only); `FileSource`;
  `EchoRoom` (simulated room echo for tests and `--simulate-echo`).
- `aec.py`: SpeexDSP echo canceller plus residual suppression via ctypes (no build step).
  `EchoStage` aligns mic and reference into 256-sample frames and runs only while playback is
  active (+1 s hold).
- `vad.py`: Silero VAD v5 via onnxruntime (no torch; `models/silero_vad.onnx` is bundled),
  hysteresis `Endpointer`, `SustainedSpeech` for barge-in.
- `wake.py`: openWakeWord with the ONNX runtime; Null and Scripted detectors.
- `stt.py`: faster_whisper (in-process, desktop), whisper_cpp HTTP (`whisper-server
  /inference`, Jetson), OpenAI-compatible HTTP (Jetson as a satellite of the desktop), Fake.
- `server.py`: WebSocket pub/sub. `play`/`play_end` and binary PCM go straight to the Player;
  everything else becomes an engine command.
- `config.py`: nested dataclasses from TOML; unknown keys are errors; every key has a default.

## Key decisions (and why)

- **openWakeWord for the trigger, Whisper after it.** A "contains keyword" check on Whisper
  output was considered and rejected: too costly always-on, and prone to false wakes.
- **The utterance starts at the wake point** (−100 ms), so the wake phrase isn't
  re-transcribed. Leftovers are fuzzily stripped from text (`raw_text` keeps the original).
  Whisper's silence hallucinations ("Thank you.") are dropped.
- **Sessions:** after a transcript the session stays open; `then: "listen"` opens a
  follow-up window with no wake word needed (for clarifying questions); timeouts return to idle.
- **The listener owns the speaker (protocol v2).** v1 had the agent play TTS itself and send
  `speaking_started`/`ended`. On open speakers the agent's voice came back through the mic and
  it interrupted itself: in the user's log "Once upon a…" was transcribed as "One second." and
  the agent answered it. Now the agent streams PCM with `play`, and the listener:
  1. cancels the echo (Speex AEC, exact reference);
  2. duck-and-verify: on sustained speech during playback (VAD ≥ 0.6 for 300 ms), it ducks
     playback −20 dB and waits `max(150 ms, device latency + 40 ms)`, then measures for 200 ms.
     `excess_db = 10·log10(mic_post / (mic_pre · ref_post/ref_pre + noise_floor))`.
     Interruption is confirmed if excess ≥ 6 dB and ≥ 50 % of frames are voiced; otherwise it
     is rejected as echo, with a 700 ms cooldown;
  3. stops playback itself and captures from the start of the speech run.
- **Energy, not VAD, decides echo versus person.** Silero is nearly level-invariant: echo
  20 dB down still scores as speech.
- **Speex AEC takes ~1–1.5 s of agent speech to converge.** Duck-and-verify covers that window,
  and also works with no libspeexdsp (then the volume "pumps" about once per second on echo).
- The legacy `speaking_started`/`ended` path still exists for agents that must play audio
  themselves (barge-in is then immediate and needs a headset or system AEC;
  `scripts/pipewire/60-echo-cancel.conf`).

Simulated room (echo −6 dB, 80 ms delay, 300 ms reverb, tanh distortion): echo scored ≤ +3.9 dB
excess; a person talking over the agent scored +9 to +24 dB.

## Verified vs not

Verified in a sandbox with no audio hardware or model downloads:
- the engine with real speech clips and real Silero VAD;
- Speex AEC (built from source; same API as `libspeexdsp1`);
- the WebSocket protocol, including `play` + binary PCM;
- the HTTP STT client against a whisper.cpp stand-in;
- the full CLI + `examples/agent_client.py` with `--simulate-echo`.

**Not run yet:** live PortAudio duplex in a real room; the openWakeWord models;
faster-whisper inference; a real whisper.cpp server; the Jetson.

## Where things stand / next steps

- The user applied the v2 patch with `--exclude=configs/desktop.toml` because that file has
  local edits. New settings default on in code. Check that `[barge_in] triggers` includes
  `"speech"` (it may still be `["wake_word"]` from the earlier workaround).
- First live test: run with `--print-events` and watch:
  - `excess_db` on `interruption_rejected`/`interruption_detected`: tune
    `barge_in.verify_min_excess_db`;
  - `aec_erle_db` on `playback_finished`: how much echo the canceller removed;
  - the startup log line with duplex latency: the round trip must be well under
    `aec.filter_ms` (250); `barge_in.echo_delay_ms` overrides the delay used for prediction.
- Possible follow-ups:
  - WebRTC AEC3 if Speex isn't enough in the real room;
  - smarter cooldown to reduce pumping when AEC is off;
  - calibrate the echo delay with a chirp;
  - Jetson bring-up per README (whisper.cpp with `-DGGML_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES=87`).

## Conventions

- Keep the engine single-threaded; anything from another thread goes through `engine.inbox`.
- Every engine event goes through `_emit` (adds `ts`, `t`, `session_id`). `_to` makes an event
  client-private and `_audio` attaches a WAV for `include_audio` subscribers; the server strips
  both.
- Tests use `tests/harness.py`: real Silero VAD, scripted wake, fake STT. With `playback=True`
  it steps the simulated room in lock-step with the engine, like a real device.
