# voice-listener

An always-on voice front end for an agent. It listens to the mic, waits for a
wake word (openWakeWord), captures what you say until you stop talking (Silero
VAD), transcribes it (Whisper), and publishes all of it as events over a
WebSocket. Your agent is just a client: it subscribes to events like
`wake_detected`, `speech_started`, `silence_detected`, `transcript` and
`interruption_detected`, and tells the listener when its own TTS is playing so
the user can cut it off.

The same code runs on a desktop GPU (faster-whisper in-process) and on a Jetson
Orin with JetPack 7.2 (a whisper.cpp server built with CUDA). Only the config
changes.

```
 agent ── play + PCM ──▶ player (duck / stop) ──▶ speaker
                            │ what was played = echo reference
                            ▼
 mic ──▶ echo canceller ──▶ ring buffer ─┬─▶ Silero VAD ──▶ endpointer ─┐
                                         └─▶ openWakeWord ──────────────┤
                                                                        ▼
 agent ◀── WebSocket events ◀──────────── session state machine ──▶ Whisper
```

- **Wake word, then the command.** Say it in one breath ("Hey Jarvis, lights
  off") or pause after the wake word. The utterance starts at the wake point, so
  the wake phrase isn't re-transcribed; any leftover is stripped from the text.
- **Sessions.** After a transcript the session stays open. When the agent asks
  a clarifying question it sends `speaking_ended {then: "listen"}`, and the
  user's answer is captured with no wake word. The window times out back to idle.
- **Interruptions.** The listener plays the agent's TTS itself, so it can
  cancel the echo and tell the agent's voice from yours (see "Interruptions and
  echo"). When you talk over the agent it stops the audio, raises
  `interruption_detected`, and captures what you're saying, including the audio
  just before the interruption was detected. Speech while the agent is still
  thinking counts too.
- **Pre-roll.** Utterances include ~300 ms from before speech onset, so first
  syllables aren't clipped.

The full event and command reference is in [PROTOCOL.md](PROTOCOL.md).

## Quick start: desktop (RTX 4090, Ubuntu)

```bash
sudo apt install libportaudio2 libspeexdsp1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-desktop.txt
pip install --no-deps openwakeword        # its Linux tflite dependency often has no wheel; we use ONNX
python scripts/download_models.py         # hey_jarvis + feature models; checks Silero VAD
```

If faster-whisper complains about missing `libcublas` / `libcudnn`, install
the CUDA 12 runtime libraries with pip (faster-whisper's README describes this),
then set `LD_LIBRARY_PATH` as shown there:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
```

**Try it without a mic first.** This feeds a recorded two-session conversation
and simulates the wake word at the right moments, so only VAD and Whisper are
exercised:

```bash
python -m voice_listener -c configs/desktop.toml --print-events \
    --input-file tests/data/demo_stream.wav --wake-at 1.95 --wake-at 13.80
```

**Then live:**

```bash
python -m voice_listener --list-devices                  # find your mic
python -m voice_listener -c configs/desktop.toml --print-events
# in another terminal:
python examples/agent_client.py                                    # simulated speech, no audio
python examples/agent_client.py --tts-cmd "espeak-ng --stdout"     # real speech via the listener
python examples/agent_client.py --tts-rate 22050 \
    --tts-cmd "piper -m en_US-lessac-medium.onnx --output-raw"      # or Piper, raw PCM
```

Say "Hey Jarvis, turn on the lights". The stub agent asks "Which room?", and
you answer without the wake word. Say "Hey Jarvis, tell me a story" and talk
over it to interrupt.

## Jetson Orin (JetPack 7.2)

JetPack 7.2 brings Ubuntu 24.04 and Python 3.12 to Orin. The listener itself
needs no GPU: wake word and VAD are small ONNX models on the CPU. Transcription
runs in whisper.cpp's HTTP server, built with CUDA. That avoids PyTorch and
CTranslate2 on the board entirely (CTranslate2's PyPI wheels for aarch64 are
CPU-only).

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks       # max performance; avoids latency spikes
sudo apt install -y libportaudio2 libspeexdsp1 python3-venv build-essential cmake git

# 1. whisper.cpp with CUDA (Orin is compute capability 8.7)
export PATH=/usr/local/cuda/bin:$PATH
git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp
cmake -B build -DGGML_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j$(nproc) --config Release
./models/download-ggml-model.sh small.en
./build/bin/whisper-server -m models/ggml-small.en.bin --host 127.0.0.1 --port 8080 -fa
cd ..

# 2. the listener
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-jetson.txt
pip install --no-deps openwakeword
python scripts/download_models.py
python -m voice_listener -c configs/jetson.toml --print-events
```

Model choice: start with `small.en` and measure `stt_ms` in the transcript
events. On an AGX Orin, try `large-v3-turbo-q5_0` next. On an Orin Nano, drop to
`base.en` if `small.en` is too slow. Once STT is comfortably fast, set
`stt.partials = true`. Run `whisper-server` as a systemd service so it stays
loaded.

**Satellite mode.** The Jetson can also capture and endpoint locally but send
audio to the 4090 for transcription. Run an OpenAI-compatible Whisper server on
the desktop and use `backend = "openai"` (commented example in
`configs/jetson.toml`). The agent can run anywhere: set `server.host =
"0.0.0.0"` and a `server.token`.

## Interruptions and echo

Speech barge-in only works if the listener can tell the agent's own voice
(coming back from the speaker) from the user talking. That's why **the listener
plays the agent's audio itself**: the agent sends its TTS as PCM over the same
WebSocket (`play`, binary frames, `play_end`; see PROTOCOL.md), and the
listener:

1. **Cancels the echo.** SpeexDSP's echo canceller uses the exact audio sent to
   the speaker as its reference, plus residual-echo suppression. It takes about
   a second of agent speech to converge, and stays converged afterwards.
2. **Ducks and verifies.** When it hears speech during playback it lowers the
   playback volume by 20 dB for about 350 ms (`interruption_suspected`). It then
   compares the mic level with what the agent's echo alone would be at that
   volume. Echo follows the volume down and gets `interruption_rejected`, with
   the volume restored. A person talking over the agent stays loud, so playback
   stops and `interruption_detected` fires. This is what covers the
   not-yet-converged start of the first reply, and it works even without
   libspeexdsp.
3. **Stops the audio itself** on a real interruption. The user's words, including
   the ones that triggered it, are already being captured.

Simulated in the tests (speaker echo 6 dB below the source, 80 ms delay,
reverb, some distortion), the agent's echo scored at most +4 dB over the echo
prediction. A person talking over it scored +9 to +24 dB. The threshold is 6 dB.

| setup | what to do |
|---|---|
| Speakers (default) | agent sends audio with `play`; `sudo apt install libspeexdsp1` for echo cancellation |
| Headset | same; there's just no echo to deal with |
| Agent must play audio itself | `speaking_started` / `speaking_ended`, plus a headset or system AEC (`scripts/pipewire/60-echo-cancel.conf`), or `triggers = ["wake_word"]` |
| Capture only | `audio.playback = false` |

Duplex audio: with `audio.playback = true` the listener opens one full-duplex
stream (`audio.device` in, `audio.output_device` out, default devices if
empty). The echo canceller needs the input and output to stay in step, so the
best setups are one device with a shared clock, or both going through
PipeWire's default device. At startup it logs the stream latency. The
speaker→mic round trip must stay under `aec.filter_ms` (250 ms).

Check it without hardware: this plays the agent's replies through a
simulated echoing room and feeds a recorded conversation as the mic:

```bash
python -m voice_listener -c configs/desktop.toml --print-events --simulate-echo \
    --input-file tests/data/demo_stream.wav --wake-at 1.95 --wake-at 13.80
python examples/agent_client.py --tts-cmd "espeak-ng --stdout"   # in another terminal
```

Tuning (`[barge_in]`):

| setting | default | meaning |
|---|---|---|
| `vad_threshold`, `min_speech_ms` | 0.6, 300 | speech needed to *suspect* an interruption |
| `duck_db` | -20 | how far the agent is turned down while checking |
| `verify_delay_ms`, `verify_window_ms` | 150, 200 | wait for the ducked audio to reach the mic (raised automatically to the device latency + 40 ms), then measure |
| `verify_min_excess_db` | 6 | how far above the predicted echo the mic must be. Raise it if the agent still interrupts itself; lower it if it's slow to yield to you |
| `reject_cooldown_ms` | 700 | after rejecting echo, don't check again this soon (limits volume "pumping" when there's no AEC) |
| `grace_ms` | 250 | ignore the first moments of each reply |

`interruption_rejected` / `interruption_detected` report `excess_db`, and
`playback_finished` reports `aec_erle_db` (how much echo the canceller
removed). Watch them with `--print-events` while you tune.

## Tuning the rest

| setting | default | what it does |
|---|---|---|
| `wake.threshold` | 0.5 | higher = fewer false wakes, more misses |
| `wake.command_preroll_ms` | 100 | audio kept before the wake detection point; raise if a same-breath command loses its first word |
| `vad.end_silence_ms` | 800 | pause that ends a turn. Per window: `listen {end_silence_ms}` (e.g. longer for dictation) |
| `vad.speech_preroll_ms` | 300 | audio kept before speech onset |
| `vad.min_utterance_ms` | 250 | shorter speech is discarded without running STT |
| `session.after_wake_timeout_s` | 5 | wait for the command after the wake word |
| `session.followup_timeout_s` | 8 | default follow-up window |
| `stt.initial_prompt`, `stt.faster_whisper.hotwords` | | bias Whisper toward names and jargon |
| `stt.drop_phrases` | "thank you", ... | Whisper's classic silence hallucinations, dropped when they're the whole transcript |
| `debug.save_audio_dir` | | save every utterance as WAV; useful for tuning |

## Tests

```bash
pip install websockets     # plus the base requirements; libspeexdsp1 for the AEC tests
python -m unittest discover -s tests -v
```

The 46 tests drive the real state machine with real speech clips and the real
Silero VAD:

- wake → command in one breath or with a pause
- timeouts
- follow-up answers without the wake word
- barge-in by speech and by wake word
- grace period and half-duplex
- speaking while the agent thinks, and continuing to talk during transcription
- partial transcripts
- the agent's echo in a simulated room (delay, reverb, distortion): no
  self-interruption with echo cancellation plus duck-and-verify, or with either
  one alone once it has converged; real interruptions still detected, including
  right at the start of a reply
- the exact streaming resampler and the player (ducking, stop, underruns)
- the WebSocket protocol, including auth, audio attachments and `play` with
  binary PCM
- the HTTP STT client against a whisper.cpp stand-in

## Status: what has and hasn't been run

Verified in a sandbox without audio hardware or model downloads:

- the engine with real speech and Silero VAD
- echo cancellation with SpeexDSP (built from source here; the system
  `libspeexdsp1` exposes the same API) in a simulated echoing room
- the WebSocket server
- the HTTP STT client
- the CLI running end to end with `--simulate-echo` against
  `examples/agent_client.py` streaming TTS audio (clarifying question plus
  barge-in), with a stand-in whisper.cpp server

Not run yet:

- live duplex audio through PortAudio: a real room, speaker and mic (the echo
  numbers above are from a simulated room)
- the openWakeWord models (the scripted `--wake-at` detector stood in)
- faster-whisper inference
- a real whisper.cpp server
- a Jetson

The first run on your machine should be `scripts/download_models.py`, then the
demo-stream command above, then the live mic.

## Layout

```
voice_listener/
  engine.py     state machine: sessions, endpointing, barge-in (duck-and-verify), STT scheduling
  server.py     WebSocket pub/sub, play commands + binary audio
  audio.py      duplex/mic/file sources, player, resampler, ring buffer, simulated echo room
  aec.py        SpeexDSP echo cancellation (ctypes)
  vad.py        Silero VAD (ONNX, no torch) + endpointer
  wake.py       openWakeWord (ONNX) + null/scripted detectors
  stt.py        faster-whisper, whisper.cpp HTTP, OpenAI-compatible HTTP
  textutil.py   wake-phrase stripping, hallucination filter
  config.py     TOML config with defaults and validation
configs/        desktop.toml, jetson.toml
examples/       agent_client.py
scripts/        download_models.py, make_demo_stream.py, pipewire/60-echo-cancel.conf
models/         silero_vad.onnx (MIT, from snakers4/silero-vad)
```
