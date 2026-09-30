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
make system-deps && make setup          # picks setup-cpu or setup-gpu; refuses outside a venv
make setup-cpu | make setup-gpu         # choose explicitly; either can be run over the other
make doctor                             # onnxruntime build, resolved device, libs, models
make docker-cpu && make docker-cuda     # the two images; make docker-sources lists host mics
make docker-run-cuda PULSE_SOURCE=<mic> # run one on the host's mic + speakers
python -m wonky_talkie -c configs/desktop.toml --say "Testing one two."   # TTS on its own
python -m wonky_talkie -c configs/desktop.toml --print-events
python examples/agent_client.py                        # sends `say`; no audio code in the agent
python -m unittest discover -s tests -v                # 98 tests; AEC tests skip without libspeexdsp
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

- `compute.py`: CPU or CUDA, decided once per process by `resolve(compute.device)`. `auto`
  = CUDA iff onnxruntime has the CUDA provider (i.e. the GPU install) AND the driver reports a
  device (ctypes `cuDeviceGetCount`, no toolkit needed). `prepare()` calls
  `ort.preload_dlls()`, which also lets CTranslate2 find the pip cuBLAS/cuDNN by name - verified
  in the CUDA image, which has no system CUDA libraries at all. `use_providers()` sets and then
  *checks* a session's provider, because ORT only warns when CUDA is unusable. Every model takes
  a `compute=` argument: SileroVAD, OpenWakeWordDetector (feature models via `device="gpu"`, the
  classifier heads via `set_providers` since the library pins them to CPU), KittenBackend
  (`set_providers`), FasterWhisperBackend (device, plus `auto` model/precision).
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

- **Two installs, one config.** The user's requirement: a CPU install that runs everything on
  the CPU, and a GPU install that runs *everything it can* on CUDA - no per-model switches.
  `[compute] device = "auto"` follows the install; `stt.faster_whisper.model/compute_type =
  "auto"` pick large-v3-turbo/float16 on GPU and small.en/int8 on CPU (measured: large-v3-turbo
  is 60 ms/utterance on the 4090 but 5.6 s on the CPU; small.en is 1.3 s). The old
  `stt.faster_whisper.device` key raises a "moved to [compute] device" error (`config._MOVED`).
- **One CUDA 12 stack.** `onnxruntime-gpu>=1.26,<1.27`: 1.27+ ships CUDA 13 wheels, which would
  sit beside CTranslate2's CUDA 12 ones. CUDA 12 wheels run under the user's CUDA 13 driver.
- **The onnxruntime trap.** faster-whisper depends on the CPU `onnxruntime`; both packages write
  one `onnxruntime/` dir. A plain install of the GPU requirements leaves a mix that *lists*
  onnxruntime-gpu and *runs* CPU-only (seen: version 1.30.0, CPU providers, the CUDA `.so` on
  disk unused). `make onnx-runtime` uninstalls both and installs one; `make doctor` flags a mix.
- **KittenTTS default is nano-0.8**, changed from mini-0.8: it is the only float Kitten model.
  mini/micro/nano-int8 use `MatMulInteger`/`DynamicQuantizeLinear`, which have no CUDA kernel,
  so on "CUDA" they still ran on the CPU (mini: 1.6x CPU vs 1.8x "GPU"). nano-0.8: 21x on CPU,
  57x on the 4090; first sentence 240 ms / 110 ms vs mini's ~3 s. Quality is the trade.
- **VAD on the GPU is slower** (0.38 vs 0.10 ms per 32 ms block: transfer overhead) but both are
  <2 % of the budget, so the GPU install runs it there anyway per the user's rule.
- **Docker audio goes through the host's Pulse socket** (PipeWire serves it): PortAudio -> ALSA
  default -> `docker/asound.conf` -> pulse plugin -> `PULSE_SERVER`. **`PULSE_SOURCE` is
  required in practice**: this desktop's default source is the speaker *monitor*; unset, the
  container recorded rms 0.0 (deaf), with the mic named, 3438. `make docker-sources` lists them.
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

Verified 2026-09-30 on the desktop (and in both Docker images): real duplex audio through
PipeWire from inside the containers (48 kHz, 64/96 ms), a `say` reply heard by the user through
the real speakers; every model on CUDA with no system CUDA libs in the image; Whisper on real
speech (GPU 55-91 ms, CPU small.en ~1.3 s); openWakeWord firing on real speech (hey_mycroft
0.998, same on CPU and GPU); first real-room AEC readings, 3.5 and 3.9 dB ERLE on single 5 s
replies (barely converged - not a tuned result).

**Not run yet:** barge-in with a person talking over real speakers; the `--say` CLI path;
KittenTTS on aarch64; a real whisper.cpp server; the Jetson; Docker without a desktop sound
server.

## Where things stand / next steps

- `[barge_in] triggers` in `configs/desktop.toml` was checked and already includes `"speech"`;
  that old workaround is gone.
- **Chunking is pacing, not just splitting.** Two bugs the user heard in the example agent's
  story reply: (1) the first chunk was cut mid-clause at "...that never", and KittenTTS appends
  a comma to any chunk lacking terminal punctuation, so it was *spoken* as a stumble; (2) chunk
  sizes leapt 48 -> 274 chars, and a 274-char chunk needs ~17 s to synthesize while only 4.8 s
  of audio was queued, so the speaker went silent for ~12 s mid-story. `split_sentences` now
  breaks only on sentence -> clause -> (run-ons only) word boundaries, and ramps chunk sizes by
  `tts.chunk_growth` (1.5, which must stay under the backend's realtime factor of ~1.6).
  Measured after: 5 chunks, first audio 2.8 s, 34.5 s of audio in 20.1 s, no gap.
  `tests/test_tts.py::ChunkPacing` asserts both properties against measured per-character rates.
- The rename + TTS work is committed (`aa5a2af` "Adding TTS" through `ac55593` "better chunking
  for tts"). The CPU/GPU installs + Docker work (compute.py, requirements-{base,cpu,gpu}.txt,
  Dockerfile, configs/docker.toml) are uncommitted. The directory is still
  `~/experiments/voice-listener`.
- **The repo `.venv` is the CPU install** (plain onnxruntime, plus CUDA 12 wheels left over for
  CTranslate2). Under the new rules that means Whisper drops to small.en on the CPU next time
  the service starts there. `make setup-gpu` in that venv restores GPU Whisper and puts the
  ONNX models on the GPU too. A scratch GPU venv verified this; the repo venv was left alone.
- `configs/desktop.toml` has `audio.device = "7"`, which on 2026-09-30 was PortAudio's `hdmi`
  with 0 input channels (indices shift as devices come and go). Left as the user's setting.
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
