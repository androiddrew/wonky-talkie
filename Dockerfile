# wonky-talkie in a container. One Dockerfile, two images:
#
#   docker build --build-arg PROFILE=cpu -t wonky-talkie:cpu .     (make docker-cpu)
#   docker build --build-arg PROFILE=gpu -t wonky-talkie:cuda .    (make docker-cuda)
#
# The CUDA image carries the CUDA 12 libraries as pip wheels, so the host needs
# only an NVIDIA driver for CUDA 12 or newer (a CUDA 13 driver is fine) and the
# NVIDIA Container Toolkit, and runs the image with `--gpus all`.
#
# Audio: the container has no sound card. It reaches the host's microphone and
# speakers through the PulseAudio socket, which PipeWire also serves; see
# README "Docker" for the `docker run` flags (or `make docker-run-cpu`).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf

# libasound2-plugins carries PortAudio's ALSA output to PulseAudio; espeak-ng is
# KittenTTS' phonemizer; pulseaudio-utils gives pactl/paplay for checking audio.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv make ca-certificates \
        libportaudio2 libspeexdsp1 espeak-ng \
        libasound2-plugins pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv "$VIRTUAL_ENV"
COPY docker/asound.conf /etc/asound.conf

WORKDIR /app

# Dependencies before code, so editing the code doesn't reinstall gigabytes.
# This is the same install the Makefile does on a desktop, including the
# onnxruntime swap that keeps the CUDA image from quietly running on the CPU.
# PROFILE is declared this late on purpose: a build arg is part of the cache key
# of every RUN after it, so both images share all the layers above.
ARG PROFILE=cpu
COPY Makefile requirements-*.txt ./
RUN make install-deps PROFILE=$PROFILE && rm -rf /root/.cache/pip

COPY . .
# Bake in the small models: wake word, VAD, and the voice. Whisper (up to 1.6 GB)
# downloads on first start into /cache, a volume, so it happens once.
RUN make models CONFIG=configs/docker.toml && make doctor CONFIG=configs/docker.toml

EXPOSE 8765
VOLUME /cache
ENTRYPOINT ["python", "-m", "wonky_talkie", "-c", "configs/docker.toml"]
CMD ["--print-events"]
