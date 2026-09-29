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
 mic ─▶ 16 kHz ring buffer (60 s) ─┬─▶ Silero VAD (32 ms) ──▶ endpointer ─┐
                                   └─▶ openWakeWord (80 ms) ──────────────┤
                                                                          ▼
   agent(s) ◀── WebSocket events ◀── session state machine ──▶ Whisper (faster-whisper
      │                                  ▲                     or whisper.cpp server)
      └── speaking_started / speaking_ended / listen / end_session ──┘
```

- **Wake word, then the command.** Say it in one breath ("Hey Jarvis, lights
  off") or pause after the wake word. The utterance starts at the wake point, so
  the wake phrase isn't re-transcribed; any leftover is stripped from the text.
- **Sessions.** After a transcript the session stays open. When the agent asks
  a clarifying question it sends `speaking_ended {then: "listen"}`, and the
  user's answer is captured with no wake word. The window times out back to idle.
- **Interruptions.** While the agent says it's speaking, sustained speech (or
  the wake word) raises `interruption_detected` and the listener immediately
  starts capturing what the user is saying, including the audio just before the
  interruption was detected. Speech while the agent is still thinking counts
  too.
- **Pre-roll.** Utterances include ~300 ms from before speech onset, so first
  syllables aren't clipped.

The full event and command reference is in [PROTOCOL.md](PROTOCOL.md).

## Quick start: desktop (RTX 4090, Ubuntu)

```bash
sudo apt install libportaudio2
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
python examples/agent_client.py                          # simulated speech
python examples/agent_client.py --tts-cmd "espeak-ng"     # real speech, interruptible
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
sudo apt install -y libportaudio2 python3-venv build-essential cmake git

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

Speech barge-in has one hard requirement: the mic must not hear the agent's own
voice as "the user talking". Pick one of these options:

| setup | barge-in config | notes |
|---|---|---|
| Headset | `triggers = ["speech", "wake_word"]` | simplest; no echo at all |
| PipeWire echo cancellation | `triggers = ["speech", "wake_word"]` | copy `scripts/pipewire/60-echo-cancel.conf`, use `ec_source` as the mic and play TTS to `ec_sink` (instructions are in the file) |
| USB mic array with onboard AEC | `triggers = ["speech", "wake_word"]` | good fit for the Jetson; the array must also be the playback device so it has the reference signal |
| Open speaker, no AEC | `triggers = ["wake_word"]` | say "Hey Jarvis" to interrupt; the agent's voice won't trigger it unless it says the wake word |
| Nothing | `enabled = false` | half-duplex: the mic is ignored while the agent talks |

Tuning (`[barge_in]`):

- `vad_threshold` (0.6) and `min_speech_ms` (300) are stricter than normal
  onset, so residual echo, coughs and "mm-hm" don't cut the agent off. Raise
  them if it's jumpy; lower them if it's slow to yield.
- `grace_ms` (250) ignores the first moments of agent speech while echo
  cancellation converges.

For the agent side:

- Send `speaking_started` when audio actually starts playing, and
  `speaking_ended` when the audio device has drained, not when synthesis
  finishes.
- If you stream TTS sentence by sentence, send `speaking_started` once per
  turn.

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
pip install websockets     # plus the base requirements
python -m unittest discover -s tests -v
```

The 32 tests drive the real state machine with real speech clips and the real
Silero VAD:

- wake → command in one breath or with a pause
- timeouts
- follow-up answers without the wake word
- barge-in by speech and by wake word
- grace period and half-duplex
- speaking while the agent thinks, and continuing to talk during transcription
- partial transcripts
- the WebSocket protocol, including auth and audio attachments
- the HTTP STT client against a whisper.cpp stand-in

## Status: what has and hasn't been run

Verified in a sandbox without audio hardware or model downloads:

- the engine with real speech and Silero VAD
- the WebSocket server
- the HTTP STT client
- the CLI running end to end against `examples/agent_client.py` (clarifying
  question plus barge-in), with a stand-in whisper.cpp server

Not run yet:

- live capture through PortAudio
- the openWakeWord models (the scripted `--wake-at` detector stood in)
- faster-whisper inference
- a real whisper.cpp server
- a Jetson

The first run on your machine should be `scripts/download_models.py`, then the
demo-stream command above, then the live mic.

## Layout

```
voice_listener/
  engine.py     state machine: sessions, endpointing, barge-in, STT scheduling
  server.py     WebSocket pub/sub
  audio.py      mic/file sources, resampling, ring buffer
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
