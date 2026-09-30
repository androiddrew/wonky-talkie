# wonky-talkie

An always-on voice front end **and voice** for an agent. It listens to the mic,
waits for a wake word (openWakeWord), captures what you say until you stop
talking (Silero VAD), transcribes it (Whisper), and publishes all of it as events
over a WebSocket. Your agent is just a client: it subscribes to events like
`wake_detected`, `speech_started`, `silence_detected`, `transcript` and
`interruption_detected`, and replies with a line of **text**. wonky-talkie
speaks that text itself (KittenTTS by default) and lets you cut it off
mid-sentence.

The agent therefore needs no audio code at all: no TTS, no output device, no
PCM. It sends `say {text: "..."}` and gets on with thinking.

The same code runs on a desktop GPU (faster-whisper in-process) and on a Jetson
Orin with JetPack 7.2 (a whisper.cpp server built with CUDA). Only the config
changes.

```
 agent ── say {text} ──▶ TTS (kitten/piper/…) ──▶ player (duck / stop) ──▶ speaker
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
- **It speaks.** `say {text: "...", then: "listen"}` is the whole reply path.
  Long replies are synthesized sentence by sentence, so it starts talking after
  the first sentence instead of after the last. Backends: `kitten` (default),
  `piper`, `command` (any shell command, e.g. `espeak-ng --stdout`), `openai`
  (any OpenAI-compatible `/v1/audio/speech` server), or `none`.
- **Sessions.** After a transcript the session stays open. When the agent asks
  a clarifying question it sends `say {..., then: "listen"}`, and the user's
  answer is captured with no wake word. The window times out back to idle.
- **Interruptions.** Because wonky-talkie owns both the synthesizer and the
  speaker, it can cancel its own echo and tell its voice from yours (see
  "Interruptions and echo"). When you talk over it, it stops mid-sentence,
  raises `interruption_detected`, never synthesizes the rest, and captures what
  you're saying — including the audio from just before the interruption was
  detected. Speech while the agent is still thinking counts too.
- **Pre-roll.** Utterances include ~300 ms from before speech onset, so first
  syllables aren't clipped.

The full event and command reference is in [PROTOCOL.md](PROTOCOL.md).

## Install: CPU or GPU

There are two installs, and the install is the only choice you make:

| | every model runs on | needs |
|---|---|---|
| **CPU** | the CPU | nothing but the system packages |
| **GPU** | an NVIDIA GPU | an NVIDIA driver for CUDA 12 or newer |

"Every model" means all of them: the wake word, the VAD, the voice (KittenTTS)
and Whisper. The config doesn't say which install it is: `[compute] device =
"auto"` follows whatever is installed, so the same config works on both, and so
do both Docker images (see [Docker](#docker)).

```bash
python3 -m venv .venv && source .venv/bin/activate
make system-deps      # apt: libportaudio2, libspeexdsp1, espeak-ng (uses sudo)
make setup            # CPU or GPU, chosen by whether an NVIDIA GPU is present
make doctor           # check it without changing anything
```

`make setup-cpu` and `make setup-gpu` choose explicitly, and either can be run
over the other in the same virtualenv to switch. `make setup*` refuses to run
unless a virtualenv is active, so it cannot install into the system Python by
accident. `make` on its own lists every target.

`make doctor` says which you ended up with, and why:

```
onnxruntime   onnxruntime-gpu 1.26.0
compute       CUDA - GPU install and a GPU is visible (1 device)
```

**A CUDA 13 driver is fine.** The GPU install brings the CUDA 12 libraries as
pip wheels (no CUDA toolkit install needed), and NVIDIA drivers run older CUDA
runtimes than their own. Everything is on one CUDA 12 stack on purpose:
`onnxruntime-gpu` is pinned below 1.27 because 1.27 moved to CUDA 13 wheels,
which would install a second CUDA stack beside Whisper's.

What the GPU buys, measured on an RTX 4090 against the same desktop's CPU:

| model | CPU install | GPU install |
|---|---|---|
| Whisper, per utterance | 1.3 s (`small.en`, int8) | **60 ms** (`large-v3-turbo`, float16) |
| voice, first sentence ready | 240 ms (21× realtime) | **110 ms** (57× realtime) |
| wake word, per 80 ms chunk | 1.5 ms | 1.2 ms |
| VAD, per 32 ms block | 0.10 ms | 0.38 ms |

Whisper is the one that matters, so `stt.faster_whisper.model = "auto"` picks
by device: `large-v3-turbo` takes 5.6 s per utterance on the CPU, which is too
slow to hold a conversation, so the CPU install uses `small.en` instead. Name a
model to override it. The VAD is honestly slower on the GPU (it is tiny, and
copying each block over costs more than the maths), but both numbers are under
2 % of the time budget, so the GPU install runs it there anyway.

<details>
<summary>The long way round, without make</summary>

```bash
sudo apt install libportaudio2 libspeexdsp1 espeak-ng
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt           # or requirements-gpu.txt
pip install --no-deps openwakeword            # its Linux tflite dependency often has no wheel; we use ONNX
python scripts/download_models.py             # hey_jarvis + feature models; checks Silero VAD
```

**For the GPU install, one more step, and it matters.** faster-whisper depends
on the CPU `onnxruntime`, so pip installs it alongside `onnxruntime-gpu`, and
both write into the same `onnxruntime/` directory. The result lists
`onnxruntime-gpu`, has the CUDA library on disk, and quietly runs everything on
the CPU. Clear both out and reinstall the GPU one:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install "onnxruntime-gpu[cuda,cudnn]>=1.26,<1.27"
```

`make doctor` flags that mixed state if you ever end up in it.
</details>

**The voice.** The default backend is [KittenTTS](https://github.com/KittenML/KittenTTS),
small ONNX models (25–80 MB). The default is `kitten-tts-nano-0.8`, because it
is the only one of the four that is float: the others are int8-quantized with
`MatMulInteger`, which ONNX Runtime has no CUDA kernel for, so on a GPU they run
their heavy layers on the CPU anyway. It's also the fastest on a CPU (21×
realtime, against `mini-0.8`'s 1.6×). `mini-0.8` sounds fuller, if you can
spare ~3 s to the first word; set `tts.kitten.model`. `make setup` installs it
all; by hand:

```bash
sudo apt install espeak-ng                # phonemization; see the note below

# Don't `pip install` the wheel on its own -- see why below.
pip install --no-deps --ignore-requires-python "misaki==0.9.4"
pip install addict regex num2words "spacy<4" phonemizer-fork espeakng-loader soundfile
pip install --no-deps https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl

python -m wonky_talkie -c configs/desktop.toml --say "Hello, I am wonky talkie."
```

Two things to know about KittenTTS:

- **Install it with those constraints, on any Python from 3.11 to 3.13.** A plain
  `pip install <wheel>` fails on 3.13 with `No matching distribution found for
  misaki>=0.9.4`. Two separate packaging problems stack up: misaki 0.9.4's
  *published* metadata still says `<3.13` even though upstream has raised it to
  `<3.14`, and once pip starts backtracking it settles on
  `spacy 4.0.0.dev3` → `thinc 9` → `blis`, which has no 3.13 wheel and fails to
  Cythonize. Pinning `spacy<4` keeps it on the stack that does ship wheels.
  `--no-deps` on the wheel itself skips the `torch` and `transformers` that
  `misaki[en]` would otherwise drag in; nothing in this path uses them.
- **Install `espeak-ng`.** The `espeakng_loader` wheel it pulls in bundles a
  library whose compiled-in data directory does not exist, and the native code
  calls `exit()` when it can't find it — taking the process with it. wonky-talkie
  detects a system `libespeak-ng` and points phonemizer at it instead
  (`tts.kitten.espeak_library` overrides the choice, `"off"` disables it).

Would rather not install any of that? Any of these work with nothing extra:

```toml
[tts]
backend = "command"                       # or "piper", or "openai"
[tts.command]
command = "espeak-ng --stdout"
```

```bash
python -m wonky_talkie -c configs/desktop.toml --tts-backend command --say "Testing."
```

**Try it without a mic first.** This feeds a recorded two-session conversation
and simulates the wake word at the right moments, so only VAD and Whisper are
exercised:

```bash
python -m wonky_talkie -c configs/desktop.toml --print-events \
    --input-file tests/data/demo_stream.wav --wake-at 1.95 --wake-at 13.80
```

**Then live:**

```bash
python -m wonky_talkie --list-devices                  # find your mic
python -m wonky_talkie -c configs/desktop.toml --print-events
# in another terminal:
python examples/agent_client.py                 # sends `say`; wonky-talkie speaks
python examples/agent_client.py --simulate      # no audio; the reply is printed word by word
```

The agent sends text and nothing else. If you would rather it own its own voice
(the v2 way), `--tts-cmd` still streams PCM with `play`:

```bash
python examples/agent_client.py --tts-cmd "espeak-ng --stdout"
python examples/agent_client.py --tts-rate 22050 \
    --tts-cmd "piper -m en_US-lessac-medium.onnx --output-raw"
```

Say "Hey Jarvis, turn on the lights". The stub agent asks "Which room?", and
you answer without the wake word. Say "Hey Jarvis, tell me a story" and talk
over it to interrupt.

## Docker

Two images from one `Dockerfile`, matching the two installs:

| image | build | models run on | the host needs |
|---|---|---|---|
| `wonky-talkie:cpu` (1.4 GB) | `make docker-cpu` | the CPU | Docker |
| `wonky-talkie:cuda` (4.6 GB) | `make docker-cuda` | an NVIDIA GPU | an NVIDIA driver for CUDA 12+ (CUDA 13 is fine) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) |

The CUDA image needs no CUDA toolkit on the host and has none of its own: the
CUDA 12 libraries are pip wheels inside it, loaded at startup.

### Your microphone and speakers

A container has no sound card. wonky-talkie uses the host's through its sound
server's socket: PulseAudio's, which PipeWire (the default on current Ubuntu,
Fedora and Debian desktops) serves as well. Inside, PortAudio opens ALSA's
default device, which the image routes to that socket.

**First find your microphone.** Many desktops set the default input to a
speaker *monitor* — a loopback of whatever is playing — and a container left on
that hears its own voice and never you. On the machine this was tested on, the
default input recorded silence while the real microphone was right beside it.
List what the container can see:

```bash
make docker-sources
```
```
sources (the default is marked *):
  * alsa_output.pci-0000_08_00.4.analog-stereo.monitor     <- a loopback, not a mic
    alsa_input.pci-0000_08_00.4.analog-stereo              <- the microphone
```

**Then run it**, naming the microphone:

```bash
make docker-run-cpu  PULSE_SOURCE=alsa_input.pci-0000_08_00.4.analog-stereo
make docker-run-cuda PULSE_SOURCE=alsa_input.pci-0000_08_00.4.analog-stereo
```

which is:

```bash
docker run --rm -it --name wonky-talkie \
  -v "$XDG_RUNTIME_DIR/pulse/native:/run/pulse/native" \
  -e PULSE_SERVER=unix:/run/pulse/native \
  -e PULSE_SOURCE=alsa_input.pci-0000_08_00.4.analog-stereo \
  -p 8765:8765 \
  -v wonky-talkie-models:/cache \
  wonky-talkie:cpu                      # or: --gpus all ... wonky-talkie:cuda
```

| flag | what it's for |
|---|---|
| `-v $XDG_RUNTIME_DIR/pulse/native:/run/pulse/native` | the host sound server's socket: the microphone in, the speakers out |
| `-e PULSE_SERVER=unix:/run/pulse/native` | tells the container's audio where that socket is |
| `-e PULSE_SOURCE=<mic>` | which input to record; see above. `PULSE_SINK=<name>` picks the output the same way (the default is usually right) |
| `-p 8765:8765` | the WebSocket your agent connects to (`ws://127.0.0.1:8765/`) |
| `-v wonky-talkie-models:/cache` | Whisper downloads here on first start (`large-v3-turbo`, 1.6 GB, on the CUDA image; `small.en`, 464 MB, on the CPU image), so only once |
| `--gpus all` | CUDA image only: hands it the GPU |

On startup the log should say where the models run and that the audio opened:

```
Compute: CUDA (GPU install and a GPU is visible (1 device))
Duplex audio: in 'default' / out 'default' at 48000 Hz, latency in 64 ms / out 96 ms
```

If the CUDA image says `Compute: CPU (no CUDA device is visible ...)`, it was
started without `--gpus all`. The in + out latency must stay well under
`aec.filter_ms` (250) for echo cancellation to work; through PipeWire it was
160 ms here.

**Check the speakers on their own** with `--say`; anything after the image
name is passed to wonky-talkie:

```bash
make docker-run-cpu PULSE_SOURCE=... ARGS='--say "Hello from the container."'
```

Then point an agent at it from the host as usual: `python examples/agent_client.py`.

Other things worth knowing:

- **Your own config:** mount it over the baked-in one,
  `-v ./my.toml:/app/configs/docker.toml:ro`.
- **Plain PulseAudio** (not PipeWire) also wants its auth cookie:
  `-v ~/.config/pulse/cookie:/root/.config/pulse/cookie:ro`.
- **Checking the connection:** `docker run --rm -v "$XDG_RUNTIME_DIR/pulse/native:/run/pulse/native"
  -e PULSE_SERVER=unix:/run/pulse/native --entrypoint pactl wonky-talkie:cpu info`
  should print your sound server's name.
- **Headless machines** without a desktop sound server: not tested. Passing the
  card with `--device /dev/snd` would also need the image's ALSA routing
  (`/etc/asound.conf`, which sends everything to PulseAudio) replaced, and the
  container would then own the card exclusively.

## Jetson Orin (JetPack 7.2)

JetPack 7.2 brings Ubuntu 24.04 and Python 3.12 to Orin. wonky-talkie itself
needs no GPU: wake word, VAD and TTS are small ONNX models on the CPU. Transcription
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

# 2. wonky-talkie
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-jetson.txt
pip install --no-deps openwakeword
python scripts/download_models.py
sudo apt install -y espeak-ng
pip install --no-deps --ignore-requires-python "misaki==0.9.4"
pip install addict regex num2words "spacy<4" phonemizer-fork espeakng-loader soundfile
pip install --no-deps https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl
python -m wonky_talkie -c configs/jetson.toml --say "Hello from the Orin."
python -m wonky_talkie -c configs/jetson.toml --print-events
```

Model choice: start with `small.en` and measure `stt_ms` in the transcript
events. On an AGX Orin, try `large-v3-turbo-q5_0` next. On an Orin Nano, drop to
`base.en` if `small.en` is too slow. Once STT is comfortably fast, set
`stt.partials = true`. Run `whisper-server` as a systemd service so it stays
loaded.

**Satellite mode.** The Jetson can also capture and endpoint locally but send
audio to the 4090 for transcription, and borrow its voice too: `stt.backend =
"openai"` plus `tts.backend = "openai"` (commented examples in
`configs/jetson.toml`). The agent can run anywhere: set `server.host =
"0.0.0.0"` and a `server.token`.

## Interruptions and echo

Speech barge-in only works if wonky-talkie can tell its own voice (coming back
from the speaker) from the user talking. That's why **wonky-talkie owns the
voice**: the agent sends text, and wonky-talkie synthesizes, plays and monitors
it. In order, it:

1. **Cancels the echo.** SpeexDSP's echo canceller uses the exact audio sent to
   the speaker as its reference, plus residual-echo suppression. It takes about
   a second of speech to converge, and stays converged afterwards.
2. **Ducks and verifies.** When it hears speech during playback it lowers the
   playback volume by 20 dB for about 350 ms (`interruption_suspected`). It then
   compares the mic level with what its own echo alone would be at that volume.
   Echo follows the volume down and gets `interruption_rejected`, with the volume
   restored. A person talking over it stays loud, so playback stops and
   `interruption_detected` fires. This is what covers the not-yet-converged start
   of the first reply, and it works even without libspeexdsp.
3. **Stops the audio itself** on a real interruption, and **stops synthesizing**:
   the sentences it had not reached are dropped rather than spoken later. The
   user's words, including the ones that triggered it, are already being captured.

Energy, not VAD, decides echo versus person: Silero is nearly level-invariant, so
echo 20 dB down still scores as speech.

Simulated in the tests (speaker echo 6 dB below the source, 80 ms delay,
reverb, some distortion), the agent's echo scored at most +4 dB over the echo
prediction. A person talking over it scored +9 to +24 dB. The threshold is 6 dB.

| setup | what to do |
|---|---|
| Speakers (default) | agent sends `say`; `sudo apt install libspeexdsp1` for echo cancellation |
| Headset | same; there's just no echo to deal with |
| Agent brings its own voice | `play` + binary PCM — still fully echo-cancelled, since we own the speaker |
| Agent must play audio itself | `speaking_started` / `speaking_ended`, plus a headset or system AEC (`scripts/pipewire/60-echo-cancel.conf`), or `triggers = ["wake_word"]` |
| Capture only | `audio.playback = false`; `say` is refused |

Duplex audio: with `audio.playback = true` wonky-talkie opens one full-duplex
stream (`audio.device` in, `audio.output_device` out, default devices if
empty). The echo canceller needs the input and output to stay in step, so the
best setups are one device with a shared clock, or both going through
PipeWire's default device. At startup it logs the stream latency. The
speaker→mic round trip must stay under `aec.filter_ms` (250 ms).

Check it without hardware: this plays the agent's replies through a
simulated echoing room and feeds a recorded conversation as the mic:

```bash
python -m wonky_talkie -c configs/desktop.toml --print-events --simulate-echo \
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
| `compute.device` | `auto` | `auto` follows the install (GPU install → CUDA). `cpu` forces the CPU; `cuda` refuses to start without one, rather than quietly running on the CPU. `--compute` overrides it for one run |
| `stt.faster_whisper.model` | `auto` | `large-v3-turbo` on a GPU, `small.en` on a CPU; any faster-whisper model name overrides it |
| `stt.faster_whisper.compute_type` | `auto` | `float16` on a GPU, `int8` on a CPU |
| `tts.backend` | `kitten` | `kitten` \| `piper` \| `command` \| `openai` \| `none` |
| `tts.first_chunk_chars` | 48 | how much is *packed* into the first chunk: this is your reply latency. A single sentence longer than it still goes out whole — breaking mid-sentence makes KittenTTS stumble, because it appends a comma to any chunk without terminal punctuation |
| `tts.max_chars` | 320 | ceiling for later chunks; larger sounds smoother, costs more on an interruption |
| `tts.chunk_growth` | 1.5 | how much each chunk may grow over the last. **Must stay below the backend's realtime factor** (KittenTTS measures ~1.6×) or the speaker runs dry mid-reply while a big chunk is still synthesizing. Lower it if you hear gaps |
| `tts.kitten.voice` | `expr-voice-5-m` | or Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo |
| `tts.kitten.model` | `…nano-0.8` | the only float model, so the only one a GPU can run, and the fastest on a CPU too. `mini-0.8` (78 MB) and `micro-0.8` (41) sound fuller but are int8 and CPU-bound, ~3 s to the first word |
| `tts.kitten.speed` | 1.0 | speech rate multiplier |
| `tts.warmup` | true | synthesize once at startup so the first real reply isn't slow |
| `debug.save_audio_dir` | | save every utterance as WAV; useful for tuning |

Chunks start small and ramp up, because a long reply has two competing needs:
the first word should arrive quickly, but every later chunk must be
synthesizable in less time than the already-queued audio takes to play. The
`tts` log lines report both — `first audio in N ms` and the realtime factor —
and anything above ~1.5× keeps comfortably ahead of playback. `--say TEXT` synthesizes one
line and exits, which is the quickest way to check a voice or a new backend.

## Tests

```bash
pip install websockets     # plus the base requirements; libspeexdsp1 for the AEC tests
python -m unittest discover -s tests -v
```

The 98 tests drive the real state machine with real speech clips and the real
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
- TTS: sentence chunking, WAV/raw decoding, the `command` backend against a real
  subprocess, a spoken reply driving the state machine, and the two properties
  that matter most — a barge-in must stop *synthesis* and not just the speaker,
  and a failing backend must still close the playback item
- compute: CPU vs CUDA resolution for every combination of install and GPU, that
  asking for CUDA without one refuses to start instead of falling back, and that
  a session which quietly lands on the CPU is reported

## Status: what has and hasn't been run

Verified on the development desktop (RTX 4090, driver 595 / CUDA 13.2, PipeWire
1.0.5), in both installs and both Docker images:

- **real audio, live:** full-duplex through PipeWire at 48 kHz (64 ms in, 96 ms
  out), from inside the containers. An agent on the host sent `say`, and the
  reply came out of the real speakers and was heard in the room.
- **every model on the GPU** in the GPU install and CUDA image, with no CUDA
  toolkit on the host or in the image: the pip CUDA 12 wheels under a CUDA 13
  driver. No session fell back to the CPU.
- **Whisper transcribing real speech:** `large-v3-turbo` in 55–91 ms per
  utterance on the GPU; `small.en` in ~1.3 s and `large-v3-turbo` in ~5.6 s on
  the CPU.
- **openWakeWord firing on real speech:** `hey_mycroft` in openWakeWord's own
  test recording, score 0.998, identically on CPU and GPU.
- **echo cancellation in a real room**, first readings: 3.5 and 3.9 dB of echo
  removed over single 5 s replies. That's only just past Speex's ~1–1.5 s
  convergence, so treat it as a first data point, not a tuned result.
- the engine with real speech clips and real Silero VAD; SpeexDSP in a simulated
  echoing room; the WebSocket protocol; the HTTP STT client against a
  whisper.cpp stand-in; `say` with a real barge-in (`excess_db` 22.3, cutting a
  reply at 3.9 s of 6.9 s); KittenTTS on Python 3.12 and 3.13.

Not run yet:

- **barge-in in a real room**: nobody has talked over a reply through real
  speakers yet; the numbers above are from the simulated room and recordings
- the `--say` CLI path itself (`say` over the WebSocket is what was tested)
- the Jetson: KittenTTS on aarch64 (spacy/thinc/blis wheels there are
  untested), a real whisper.cpp server, and the board itself
- Docker on a headless machine without a desktop sound server

The first run on a new machine: `make setup`, `make doctor`, `make demo`, then
the live mic (or the Docker equivalents: `make docker-sources`, then
`make docker-run-cpu PULSE_SOURCE=...`).

## Layout

```
wonky_talkie/
  engine.py     state machine: sessions, endpointing, barge-in (duck-and-verify), STT scheduling
  server.py     WebSocket pub/sub, say/play commands + binary audio
  audio.py      duplex/mic/file sources, player, resampler, ring buffer, simulated echo room
  aec.py        SpeexDSP echo cancellation (ctypes)
  vad.py        Silero VAD (ONNX, no torch) + endpointer
  wake.py       openWakeWord (ONNX) + null/scripted detectors
  stt.py        faster-whisper, whisper.cpp HTTP, OpenAI-compatible HTTP
  tts.py        KittenTTS, piper, any shell command, OpenAI-compatible HTTP; sentence streaming
  textutil.py   wake-phrase stripping, hallucination filter
  compute.py    CPU or CUDA, decided once from what's installed; puts every model there
  config.py     TOML config with defaults and validation
configs/        desktop.toml, jetson.toml, docker.toml (baked into both images)
Dockerfile      both images (--build-arg PROFILE=cpu|gpu); docker/asound.conf routes ALSA to Pulse
requirements-*  base (shared), cpu, gpu, jetson
examples/       agent_client.py
scripts/        download_models.py, make_demo_stream.py, pipewire/60-echo-cancel.conf
models/         silero_vad.onnx (MIT, from snakers4/silero-vad)
```
