# wonky-talkie: context for Claude Code

An always-on voice front end **and voice** for an agent: mic → wake word → utterance capture
→ Whisper → events, and text → speech → speaker. Published as JSON events over a WebSocket.
The agent is a client: it subscribes to events and replies with `say {text}`. Targets: a
desktop with an RTX 4090 (Ubuntu), then Jetson Orin on JetPack 7.2 (Ubuntu 24.04, Python 3.12,
CUDA 13). Xavier and older are out of scope.

Named `voice-listener` until 2026-09-29, when TTS moved in-process and it stopped being only a
listener. The Python package is `wonky_talkie`; the env var is `WONKY_TALKIE_SPEEXDSP`.

Built in a Claude chat session. This file carries over the decisions and state from it.
README.md (setup, tuning) and PROTOCOL.md (event/command contract, v3) are the user-facing docs.

## Run / test

```bash
make system-deps && make setup          # the whole install; refuses outside an active venv
make doctor                             # imports + system libs + models, changes nothing
# or by hand:
sudo apt install libportaudio2 libspeexdsp1 espeak-ng
pip install -r requirements-desktop.txt && pip install --no-deps openwakeword   # tflite dep has no py3.12 wheel
pip install https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl
python scripts/download_models.py                      # openWakeWord ONNX models
python -m wonky_talkie -c configs/desktop.toml --say "Testing one two."   # TTS on its own
python -m wonky_talkie -c configs/desktop.toml --print-events
python examples/agent_client.py                        # sends `say`; no audio code in the agent
python -m unittest discover -s tests -v                # 72 tests; AEC tests skip without libspeexdsp
```

**KittenTTS installs on 3.13 but needs constraints.** A plain `pip install <wheel>` fails with
`No matching distribution found for misaki>=0.9.4`, for two stacked packaging problems: misaki
0.9.4's *published* metadata says `<3.13` (upstream `pyproject.toml` has raised it to `<3.14`),
and once pip backtracks it picks `spacy 4.0.0.dev3` -> `thinc 9` -> `blis`, which has no 3.13
wheel and fails to Cythonize. `spacy<4` keeps it on the wheeled stack. Installed and verified
in the repo `.venv` (3.13.4); `--no-deps` on the wheel also avoids torch + transformers, which
`misaki[en]` pulls in and nothing here uses. The command lives in `tts.INSTALL_HINT`, the README
and both requirements files.

No-hardware checks:
- Recorded conversation: `--input-file tests/data/demo_stream.wav --wake-at 1.95 --wake-at 13.80`
  (a scripted wake word stands in for openWakeWord).
- Simulated speaker→mic echo: add `--simulate-echo`. This needs real-time speed (the default
  `--speed 1`) so that ducking stays causal.
- `WONKY_TALKIE_SPEEXDSP=/path/libspeexdsp.so.1` points at a non-standard lib.

## Layout

- `tts.py`: TTS backends (`kitten` KittenTTS/ONNX, `piper` binary, `command` shell command,
  `openai` HTTP `/v1/audio/speech`, `none`, `fake`) behind one `TtsBackend` Protocol, mirroring
  `stt.py`. `TtsService` synthesizes on one worker thread and feeds the `Player` directly:
  `Player.add_audio` returns False once the item is gone, which *is* the cancellation signal,
  so a barge-in stops synthesis with no extra plumbing. Long replies are split by sentence
  (`split_sentences`) so the first audio is quick and an interruption wastes little.
- `wonky_talkie/engine.py`: the state machine (idle / listening / capturing / processing /
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
- `server.py`: WebSocket pub/sub. `say` goes to the TtsService; `play`/`play_end` and binary
  PCM go straight to the Player; everything else becomes an engine command.
- `config.py`: nested dataclasses from TOML; unknown keys are errors; every key has a default.

## Key decisions (and why)

- **openWakeWord for the trigger, Whisper after it.** A "contains keyword" check on Whisper
  output was considered and rejected: too costly always-on, and prone to false wakes.
- **The utterance starts at the wake point** (−100 ms), so the wake phrase isn't
  re-transcribed. Leftovers are fuzzily stripped from text (`raw_text` keeps the original).
  Whisper's silence hallucinations ("Thank you.") are dropped.
- **Sessions:** after a transcript the session stays open; `then: "listen"` opens a
  follow-up window with no wake word needed (for clarifying questions); timeouts return to idle.
- **The service owns the voice (protocol v3).** v2 had the agent synthesize and stream PCM;
  v3 has it send text. Same reasoning taken one step further: whoever holds the audio can cut
  it off, and whoever holds the *text* can also stop making the audio. An interruption now
  drops the unsaid sentences instead of discarding work already done. `play` + PCM and the v1
  `speaking_started`/`ended` both still work.
- **KittenTTS as the default voice**, on the user's request: 15M params, ONNX/CPU, 25-80 MB, so
  the GPU stays free for Whisper. Measured ~1.6-1.7x real time on this desktop CPU, which is
  enough to stay ahead of playback given sentence chunking. Two traps, both handled in `tts.py`:
  its `generate()` **prints to stdout** (which `--print-events` uses for JSON, so it is wrapped
  in `redirect_stdout`), and its bundled `espeakng_loader` library has a **broken data path**
  that makes the native code call `exit()` and kill the process - `_ensure_espeak()` points
  phonemizer at the system `libespeak-ng` *after* importing kittentts, because `misaki` pins
  the bundled one at import time and that beats the env var.
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
- the full CLI + `examples/agent_client.py` with `--simulate-echo`;
- KittenTTS end to end in the repo `.venv` on **Python 3.13.4** (load 1.4 s, 24 kHz int16,
  1.6-1.7x real time, 8 voices, no stdout leak), with faster-whisper still warming up on CUDA
  and all 72 tests still passing afterwards;
- `say` over a real WebSocket with the `command`/espeak-ng backend: first audio 31 ms after the
  command, then a genuine barge-in (`trigger: speech`, `excess_db` 22.3) cutting the synthesized
  reply at 3.9 s of 6.9 s and capturing the user;
- openWakeWord models load (`hey_jarvis`).

**Not run yet:** live PortAudio duplex in a real room; KittenTTS through a real speaker or on
aarch64 (spacy/thinc/blis wheels there are untested); `--say` (it needs an output device);
openWakeWord actually firing on a spoken wake word; faster-whisper transcribing real audio
(warmup only); a real whisper.cpp server; the Jetson.

## Where things stand / next steps

- `[barge_in] triggers` in `configs/desktop.toml` was checked and already includes `"speech"`;
  that old workaround is gone.
- `tts.first_chunk_chars` defaults to 48, not 120: the first chunk is reply latency, and
  KittenTTS costs ~0.07 s per character here, so 120 meant ~8 s of silence before the first
  word. `split_sentences` hard-bounds the first chunk (breaking at a clause boundary, then a
  space) - an earlier version let a long opening sentence through whole.
- The v3 rename + TTS work is uncommitted on `main` (the user committed the v2 work as
  `72bb3e4 "Working test"` mid-session). The directory is still `~/experiments/voice-listener`;
  only the package was renamed.
- First live test: run with `--print-events` and watch:
  - `excess_db` on `interruption_rejected`/`interruption_detected`: tune
    `barge_in.verify_min_excess_db`;
  - `aec_erle_db` on `playback_finished`: how much echo the canceller removed;
  - the startup log line with duplex latency: the round trip must be well under
    `aec.filter_ms` (250); `barge_in.echo_delay_ms` overrides the delay used for prediction.
- Possible follow-ups:
  - drop the pip constraints once misaki publishes metadata matching its `<3.14` pyproject;
  - `tts.kitten.model`: `mini-0.8` is the default; drop to `nano-0.8-int8` if the Orin is slow;
  - streaming TTS *within* a sentence, if first-word latency still feels long;
  - WebRTC AEC3 if Speex isn't enough in the real room;
  - smarter cooldown to reduce pumping when AEC is off;
  - calibrate the echo delay with a chirp;
  - Jetson bring-up per README (whisper.cpp with `-DGGML_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES=87`).

## Conventions

- `Makefile` is the entry point for setup and the common runs (`make`, `make setup`, `make
  doctor`, `make demo`, `make say`, `make test`). It uses `.ONESHELL` so recipes are single
  bash scripts - which also strips leading whitespace from every recipe line, so no recipe may
  contain indentation-sensitive Python (that is why `doctor` is a flat `python -c`).
- Keep the engine single-threaded; anything from another thread goes through `engine.inbox`
  (`Engine.post_event` exists for exactly this: the TTS worker emitting `tts_failed`).
- Every engine event goes through `_emit` (adds `ts`, `t`, `session_id`). `_to` makes an event
  client-private and `_audio` attaches a WAV for `include_audio` subscribers; the server strips
  both.
- Tests use `tests/harness.py`: real Silero VAD, scripted wake, fake STT. With `playback=True`
  it steps the simulated room in lock-step with the engine, like a real device.
