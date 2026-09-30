# wonky-talkie -- common tasks. Run `make` (or `make help`) to list them.
#
# Two installs: `make setup-cpu` runs every model on the CPU, `make setup-gpu`
# runs every model on an NVIDIA GPU. `make setup` picks one for you. They refuse
# to run outside an activated virtualenv, so they can never install into the
# system Python by accident.

SHELL := /bin/bash
# One shell per recipe (so multi-line scripts work), and -e so a failing
# command still fails the target. Note that .ONESHELL also strips leading
# whitespace from recipe lines, so no recipe may hold indented Python.
.ONESHELL:
.SHELLFLAGS := -ec

PY ?= python
CONFIG ?= configs/desktop.toml
PROFILE ?= cpu
REQS ?= requirements-$(PROFILE).txt
TEXT ?= Hello, I am wonky talkie.
IMAGE ?= wonky-talkie
PORT ?= 8765
ARGS ?=

# The GPU install's onnxruntime. 1.26 is the last release built on CUDA 12, the
# same CUDA as CTranslate2 (Whisper); 1.27+ moved to CUDA 13 wheels. A CUDA 13
# driver runs these fine: drivers run older runtimes.
ifeq ($(PROFILE),gpu)
ONNX_PKG := "onnxruntime-gpu[cuda,cudnn]>=1.26,<1.27"
else
ONNX_PKG := onnxruntime
endif

# KittenTTS cannot be installed with a plain `pip install`: misaki 0.9.4's
# published metadata still says python <3.13 (upstream has raised it to <3.14),
# and once pip backtracks it lands on spacy 4.0.0.dev3 -> thinc 9 -> blis,
# which has no 3.13 wheel and fails to compile. Pinning spacy<4 keeps it on the
# stack that does ship wheels; --no-deps also avoids torch and transformers,
# which misaki[en] would drag in and nothing here uses.
KITTEN_WHEEL := https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl
KITTEN_DEPS := addict regex num2words "spacy<4" phonemizer-fork espeakng-loader soundfile
APT_PACKAGES := libportaudio2 libspeexdsp1 espeak-ng

.DEFAULT_GOAL := help
.PHONY: help check-venv system-deps setup setup-cpu setup-gpu setup-jetson install
.PHONY: install-deps deps onnx-runtime wake-deps kitten models voice-model doctor
.PHONY: test run demo say agent devices clean
.PHONY: docker-cpu docker-cuda docker-run-cpu docker-run-cuda docker-sources

help:
	@echo "wonky-talkie"
	echo
	echo "Install (needs an activated virtualenv):"
	echo "  make system-deps    apt-get the system libraries (uses sudo)"
	echo "  make setup          pick CPU or GPU from whether an NVIDIA GPU is present"
	echo "  make setup-cpu      every model on the CPU"
	echo "  make setup-gpu      every model on the GPU (NVIDIA driver for CUDA 12+)"
	echo "  make setup-jetson   the Jetson install (requirements-jetson.txt)"
	echo "  make doctor         check the environment without changing it"
	echo
	echo "Run:"
	echo "  make run            start the service      (CONFIG=$(CONFIG))"
	echo "  make demo           run the recorded conversation, no mic needed"
	echo "  make say TEXT=...   synthesize one line and exit"
	echo "  make agent          run the example agent client"
	echo "  make devices        list audio devices"
	echo "  make test           run the test suite"
	echo "  make clean          remove caches and bytecode"
	echo
	echo "Docker (no virtualenv needed):"
	echo "  make docker-cpu     build $(IMAGE):cpu"
	echo "  make docker-cuda    build $(IMAGE):cuda"
	echo "  make docker-sources list the host's microphones and speakers"
	echo "  make docker-run-cpu PULSE_SOURCE=<mic>   (or docker-run-cuda) run it on the host's audio"
	echo "                       (extra arguments: ARGS='--say \"hi\"')"

# ----------------------------------------------------------------- guard rail

check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then
		echo "error: no activated virtualenv (VIRTUAL_ENV is unset)."
		echo
		echo "  python3 -m venv .venv && source .venv/bin/activate && make $(MAKECMDGOALS)"
		echo
		echo "Refusing to install into the system Python."
		exit 1
	fi
	if ! $(PY) -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)'; then
		echo "error: VIRTUAL_ENV is set, but '$(PY)' is not that virtualenv's interpreter."
		echo "       VIRTUAL_ENV=$$VIRTUAL_ENV"
		echo "       $(PY) -> $$(command -v $(PY) || echo 'not found')"
		exit 1
	fi
	echo "virtualenv: $$VIRTUAL_ENV ($$($(PY) -V))"

# --------------------------------------------------------------------- install

system-deps:
	sudo apt update
	sudo apt install -y $(APT_PACKAGES)

setup: check-venv
	@if nvidia-smi -L >/dev/null 2>&1; then
		profile=gpu; why="an NVIDIA GPU is present"
	else
		profile=cpu; why="no NVIDIA GPU was found"
	fi
	echo "make setup: $$why, so this is the $$profile install."
	echo "            (make setup-cpu / make setup-gpu to choose for yourself)"
	$(MAKE) --no-print-directory setup-$$profile

setup-cpu:
	@$(MAKE) --no-print-directory install PROFILE=cpu

setup-gpu:
	@$(MAKE) --no-print-directory install PROFILE=gpu

setup-jetson:
	@$(MAKE) --no-print-directory install PROFILE=cpu REQS=requirements-jetson.txt CONFIG=configs/jetson.toml

install: check-venv install-deps models
	@$(MAKE) --no-print-directory doctor CONFIG=$(CONFIG)
	echo
	echo "Ready ($(PROFILE) install). Try:  make say  |  make demo  |  make run"

# Everything pip-installable. The Dockerfile runs this on its own, so the heavy
# layer is cached apart from the source code.
install-deps: check-venv deps onnx-runtime wake-deps kitten

deps: check-venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r $(REQS)

# onnxruntime and onnxruntime-gpu install into the same onnxruntime/ directory,
# and faster-whisper depends on the CPU one. So after `deps` a GPU venv holds
# both, and the mix is the worst of each: pip lists onnxruntime-gpu, but the
# Python layer is the CPU build and quietly runs everything on the CPU. Remove
# both, then install exactly one. This also makes switching profiles safe.
onnx-runtime: check-venv
	$(PY) -m pip uninstall -y -q onnxruntime onnxruntime-gpu 2>/dev/null || true
	$(PY) -m pip install $(ONNX_PKG)

# openWakeWord's tflite dependency has no wheel for recent Pythons and we use
# the ONNX runtime instead, so its dependencies come from $(REQS).
wake-deps: check-venv
	$(PY) -m pip install --no-deps openwakeword

kitten: check-venv
	$(PY) -m pip install --no-deps --ignore-requires-python "misaki==0.9.4"
	$(PY) -m pip install $(KITTEN_DEPS)
	$(PY) -m pip install --no-deps $(KITTEN_WHEEL)

models: check-venv voice-model
	$(PY) scripts/download_models.py

# The voice model is fetched lazily on first use, which would otherwise mean a
# download during the first real startup. Pull it now instead.
voice-model: check-venv
	@if $(PY) -c "import sys; from wonky_talkie.config import load_config; sys.exit(0 if load_config('$(CONFIG)').tts.backend == 'kitten' else 1)"; then
		$(PY) -c "from wonky_talkie.config import load_config; from wonky_talkie.tts import make_backend; make_backend(load_config('$(CONFIG)').tts).warmup(); print('voice model ready')"
	else
		echo "tts.backend is not kitten in $(CONFIG); no voice model to fetch"
	fi

# ---------------------------------------------------------------------- checks

doctor:
	@$(PY) -c "import ctypes.util, importlib.util as u, importlib.metadata as md, pathlib, sys; \
	mods = 'numpy scipy onnxruntime websockets sounddevice sklearn faster_whisper openwakeword kittentts'.split(); \
	miss = [m for m in mods if u.find_spec(m) is None]; \
	libs = {n: ctypes.util.find_library(n) for n in ('portaudio', 'speexdsp', 'espeak-ng')}; \
	vad = pathlib.Path('models/silero_vad.onnx'); \
	names = {(d.metadata['Name'] or '').lower(): d.version for d in md.distributions()}; \
	ort = [f'{n} {names[n]}' for n in ('onnxruntime', 'onnxruntime-gpu') if n in names]; \
	print('python       ', sys.version.split()[0], '(venv)' if sys.prefix != sys.base_prefix else '(SYSTEM -- activate a venv)'); \
	print('onnxruntime  ', ', '.join(ort) or 'MISSING', '  <-- BOTH: they overwrite each other; rerun make setup-cpu or setup-gpu' if len(ort) > 1 else ''); \
	from wonky_talkie.compute import resolve; from wonky_talkie.config import load_config; \
	want = load_config('$(CONFIG)').compute.device; c = resolve('auto'); \
	print('compute      ', c.device.upper(), '-', c.reason, f'  [{\"$(CONFIG)\"}: compute.device = {want}]'); \
	clash = want == 'cuda' and not c.cuda; \
	print('               ^ compute.device = cuda would refuse to start') if clash else None; \
	[print(f'  {\"ok     \" if m not in miss else \"MISSING\"} {m}') for m in mods]; \
	[print(f'  {\"ok     \" if libs[n] else \"MISSING\"} lib{n}') for n in libs]; \
	print(f'  {\"ok     \" if vad.exists() else \"MISSING\"} models/silero_vad.onnx'); \
	bad = len(miss) + (len(ort) != 1) + clash + (0 if libs['portaudio'] else 1) + (0 if libs['espeak-ng'] else 1) + (0 if vad.exists() else 1); \
	print() or print(bad, 'problem(s). Try: make system-deps && make setup') if bad else print('\\neverything in place'); \
	sys.exit(1 if bad else 0)"
	@echo 'note: libspeexdsp is optional (echo cancellation); everything else is required'

test: check-venv
	$(PY) -m unittest discover -s tests -v

# ------------------------------------------------------------------------- run

run: check-venv
	$(PY) -m wonky_talkie -c $(CONFIG) --print-events

demo: check-venv
	$(PY) -m wonky_talkie -c $(CONFIG) --print-events --input-file tests/data/demo_stream.wav --wake-at 1.95 --wake-at 13.80

say: check-venv
	$(PY) -m wonky_talkie -c $(CONFIG) --say "$(TEXT)"

agent: check-venv
	$(PY) examples/agent_client.py

devices: check-venv
	$(PY) -m wonky_talkie --list-devices

clean:
	find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0r rm -rf
	rm -rf .pytest_cache .mypy_cache .ruff_cache

# ---------------------------------------------------------------------- docker
# Two images from one Dockerfile. These don't need a virtualenv. The run targets
# hand the container the host's PulseAudio socket (PipeWire serves one too), so
# it uses the host's microphone and speakers; README "Docker" explains each flag.

docker-cpu:
	docker build --build-arg PROFILE=cpu -t $(IMAGE):cpu .

docker-cuda:
	docker build --build-arg PROFILE=gpu -t $(IMAGE):cuda .

# PULSE_SOURCE matters: many desktops default the input to a speaker *monitor*,
# and a container on that hears its own playback instead of you (verified: rms
# 0.0 on the monitor, 3438 on the mic). `make docker-sources` lists the names.
PULSE_SOCKET = $${XDG_RUNTIME_DIR:-/run/user/$$(id -u)}/pulse/native
PULSE_FLAGS = -v "$(PULSE_SOCKET):/run/pulse/native" -e PULSE_SERVER=unix:/run/pulse/native \
	$(if $(PULSE_SOURCE),-e PULSE_SOURCE=$(PULSE_SOURCE)) $(if $(PULSE_SINK),-e PULSE_SINK=$(PULSE_SINK))
DOCKER_RUN = docker run --rm -it --name $(IMAGE) $(PULSE_FLAGS) \
	-p $(PORT):8765 \
	-v $(IMAGE)-models:/cache

docker-run-cpu:
	$(DOCKER_RUN) $(IMAGE):cpu $(ARGS)

docker-run-cuda:
	$(DOCKER_RUN) --gpus all $(IMAGE):cuda $(ARGS)

# Your microphones and speakers, as the container sees them.
docker-sources:
	@img=$$(docker image inspect $(IMAGE):cpu >/dev/null 2>&1 && echo $(IMAGE):cpu || echo $(IMAGE):cuda)
	for kind in sources sinks; do
		echo "$$kind (the default is marked *):"
		def=$$(docker run --rm $(PULSE_FLAGS) --entrypoint pactl $$img get-default-$${kind%s})
		docker run --rm $(PULSE_FLAGS) --entrypoint pactl $$img list short $$kind \
			| awk -v d="$$def" '{ printf "  %s %s\n", ($$2 == d ? "*" : " "), $$2 }'
	done
	echo
	echo "Pick your microphone, then:  make docker-run-cpu PULSE_SOURCE=<name>"
	echo "(names ending .monitor are loopbacks of a speaker, not microphones)"
