"""Configuration: nested dataclasses loaded from a TOML file.

Every field has a default, so a config file only needs the values you want to
change. Unknown keys raise an error (typos in a config are a common source of
"why isn't my setting doing anything").
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AudioConfig:
    # Input device: "" = system default, an integer index, or a substring of the
    # device name (see `python -m voice_listener --list-devices`).
    device: str | int = ""
    # 0 = try 16 kHz first, fall back to the device's default rate and resample.
    sample_rate: int = 0
    # Channels to open on the device, and which one to use. Mic arrays often
    # expose several channels with the processed (beamformed / AEC) one first.
    channels: int = 1
    channel: int = 0
    # PortAudio block size in milliseconds (at the capture rate).
    block_ms: int = 32
    # The listener plays the agent's audio itself (agents send it with `play`).
    # This is what makes speech barge-in work on open speakers: the listener
    # knows exactly what the speaker is playing, cancels it from the mic, and
    # can duck/stop it instantly. Set false for capture-only operation.
    playback: bool = True
    output_device: str | int = ""
    output_channels: int = 1


@dataclass
class WakeConfig:
    enabled: bool = True
    # Pre-trained openWakeWord names ("hey_jarvis", "alexa", "hey_mycroft",
    # "hey_rhasspy") or paths to custom .onnx models.
    models: list[str] = field(default_factory=lambda: ["hey_jarvis"])
    threshold: float = 0.5
    # Ignore further detections for this long after one fires.
    refractory_s: float = 2.0
    # How much audio before the detection point may be kept at the start of the
    # command. Detection fires near the end of the wake phrase, so a small value
    # avoids clipping a command spoken in the same breath.
    command_preroll_ms: int = 100
    # Removed from the start of transcripts (case/punctuation-insensitive).
    strip_phrases: list[str] = field(default_factory=lambda: ["hey jarvis", "jarvis"])
    # Download pre-trained models on first use if they're missing.
    auto_download: bool = True
    # Optional overrides for openWakeWord's shared feature models.
    melspec_model_path: str = ""
    embedding_model_path: str = ""


@dataclass
class AecConfig:
    # Echo cancellation of the listener's own playback (needs libspeexdsp:
    # `sudo apt install libspeexdsp1`). Ignored when audio.playback is false.
    enabled: bool = True
    library: str = ""  # path to libspeexdsp if it isn't found automatically
    # Longest speaker->mic delay + reverb it can model. Round-trip latency of
    # the audio stack counts too; keep it well under this.
    filter_ms: int = 250
    # Suppress what the linear canceller leaves behind (uses Speex's denoiser).
    residual_suppression: bool = True
    noise_suppress_db: int = -10
    # Keep cancelling this long after playback stops (echo tail).
    hold_ms: int = 1000


@dataclass
class VadConfig:
    model_path: str = "models/silero_vad.onnx"
    threshold: float = 0.5
    # Hysteresis: a frame only counts as silence below this.
    neg_threshold: float = 0.35
    # Continuous speech needed before "speech_started" fires.
    min_speech_ms: int = 200
    # Continuous silence that ends an utterance.
    end_silence_ms: int = 800
    # Audio kept from before the detected speech onset.
    speech_preroll_ms: int = 300
    # Utterances with less detected speech than this are discarded unheard.
    min_utterance_ms: int = 250
    threads: int = 1


@dataclass
class SessionConfig:
    # After a wake word, how long to wait for the command to start.
    after_wake_timeout_s: float = 5.0
    # Default listening window when the agent asks for a reply.
    followup_timeout_s: float = 8.0
    # Hard cap on a single utterance.
    max_utterance_s: float = 20.0
    # End the session if the agent does nothing for this long after a
    # transcript (0 disables).
    agent_timeout_s: float = 120.0
    # After the agent stops speaking, ignore speech onsets for this long so
    # the tail of its own voice doesn't open a new utterance.
    post_speech_holdoff_ms: int = 150


@dataclass
class BargeInConfig:
    enabled: bool = True
    # "speech": sustained voice while the agent talks (needs echo cancellation
    #           or a headset, otherwise the agent's own voice will trigger it).
    # "wake_word": the wake word interrupts (works without echo cancellation).
    triggers: list[str] = field(default_factory=lambda: ["speech", "wake_word"])
    # Stricter VAD settings used only while the agent is speaking.
    vad_threshold: float = 0.6
    min_speech_ms: int = 300
    # Ignore speech for this long after the agent starts talking (lets echo
    # cancellation converge).
    grace_ms: int = 250
    # Also treat speech as an interruption while the agent is thinking
    # (after a transcript, before it starts speaking).
    during_thinking: bool = True
    # Duck-and-verify (only when the listener plays the audio itself): on
    # suspected speech, drop the playback volume by `duck_db` and keep
    # listening. The agent's echo falls with the volume, a real voice doesn't.
    # Playback is stopped only if speech is still there in the verify window.
    verify: bool = True
    duck_db: float = -20.0
    verify_delay_ms: int = 150  # let the ducked audio reach the mic first (raised to device latency + 40)
    verify_window_ms: int = 200
    verify_min_fraction: float = 0.5  # of VAD frames in the window that must be speech
    # After ducking, the mic must be at least this much louder than the echo
    # alone would be (predicted from the speaker signal) to count as a person.
    verify_min_excess_db: float = 6.0
    # Speaker -> mic delay used for that prediction; 0 = device latency, or 100 ms.
    echo_delay_ms: int = 0
    reject_cooldown_ms: int = 700  # after a rejected suspicion, don't re-check this soon


@dataclass
class FasterWhisperConfig:
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    partial_beam_size: int = 1
    download_root: str = ""
    # Space-separated words to bias recognition toward (names, jargon).
    hotwords: str = ""
    cpu_threads: int = 0


@dataclass
class HttpSttConfig:
    # whisper.cpp:  http://127.0.0.1:8080/inference
    # openai-style: http://host:8000/v1/audio/transcriptions
    url: str = "http://127.0.0.1:8080/inference"
    flavor: str = "whisper_cpp"  # "whisper_cpp" | "openai"
    model: str = ""  # sent as `model` for the openai flavor
    api_key: str = ""
    timeout_s: float = 30.0


@dataclass
class SttConfig:
    backend: str = "faster_whisper"  # faster_whisper | whisper_cpp | openai | none
    language: str = "en"
    initial_prompt: str = ""
    # Emit partial_transcript events while the user is still talking.
    partials: bool = False
    partial_interval_ms: int = 700
    # Whisper's classic hallucinations on near-silent audio. Transcripts that
    # consist only of one of these are dropped.
    drop_phrases: list[str] = field(
        default_factory=lambda: [
            "thank you",
            "thanks for watching",
            "thank you for watching",
            "you",
            "bye",
            "okay",
            "subtitles by the amara.org community",
        ]
    )
    faster_whisper: FasterWhisperConfig = field(default_factory=FasterWhisperConfig)
    http: HttpSttConfig = field(default_factory=HttpSttConfig)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    # If set, clients must connect with ?token=... or an
    # "Authorization: Bearer ..." header.
    token: str = ""


@dataclass
class DebugConfig:
    # Write every captured utterance to this folder as a WAV (for tuning).
    save_audio_dir: str = ""
    log_level: str = "INFO"
    # Interval for the high-rate "level" meter event (0 disables).
    level_interval_ms: int = 100


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    aec: AecConfig = field(default_factory=AecConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def resolve_path(self, p: str) -> Path:
        """Resolve a relative path in the config against the project root."""
        path = Path(p).expanduser()
        return path if path.is_absolute() else PACKAGE_ROOT / path


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in fields:
            valid = ", ".join(sorted(fields))
            raise ValueError(f"Unknown config key '{where}{key}'. Valid keys here: {valid}")
        f = fields[key]
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        if dataclasses.is_dataclass(default):
            if not isinstance(value, dict):
                raise ValueError(f"Config section '{where}{key}' must be a table")
            kwargs[key] = _build(type(default), value, f"{where}{key}.")
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path | None) -> Config:
    if path is None:
        return Config()
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    cfg = _build(Config, data, "")
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.stt.backend not in {"faster_whisper", "whisper_cpp", "openai", "none"}:
        raise ValueError(f"stt.backend must be faster_whisper, whisper_cpp, openai or none, not {cfg.stt.backend!r}")
    bad = set(cfg.barge_in.triggers) - {"speech", "wake_word"}
    if bad:
        raise ValueError(f"barge_in.triggers has unknown entries: {sorted(bad)}")
    if not 0 < cfg.vad.neg_threshold <= cfg.vad.threshold < 1:
        raise ValueError("vad thresholds must satisfy 0 < neg_threshold <= threshold < 1")
