# wonky-talkie -- common tasks. Run `make` (or `make help`) to list them.
#
# `make setup` refuses to run outside an activated virtualenv, so it can never
# install this project's dependencies into the system Python by accident.

SHELL := /bin/bash
# One shell per recipe (so multi-line scripts and heredocs work), and -e so a
# failing command still fails the target.
.ONESHELL:
.SHELLFLAGS := -ec

PY ?= python
CONFIG ?= configs/desktop.toml
REQS ?= requirements-desktop.txt
TEXT ?= Hello, I am wonky talkie.

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
.PHONY: help check-venv system-deps setup setup-jetson deps wake-deps kitten
.PHONY: models voice-model doctor test run demo say agent devices clean

help:
	@echo "wonky-talkie"
	echo
	echo "Setup (needs an activated virtualenv):"
	echo "  make system-deps    apt-get the system libraries (uses sudo)"
	echo "  make setup          install every Python dependency + fetch the models"
	echo "  make setup-jetson   same, with requirements-jetson.txt"
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

setup: check-venv deps wake-deps kitten models
	@$(MAKE) --no-print-directory doctor
	echo
	echo "Ready. Try:  make say  |  make demo  |  make run"

setup-jetson:
	@$(MAKE) --no-print-directory setup REQS=requirements-jetson.txt CONFIG=configs/jetson.toml

deps: check-venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r $(REQS)

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
# ~80 MB download during the first real startup. Pull it now instead.
voice-model: check-venv
	@if $(PY) -c "import sys; from wonky_talkie.config import load_config; sys.exit(0 if load_config('$(CONFIG)').tts.backend == 'kitten' else 1)"; then
		$(PY) -c "from wonky_talkie.config import load_config; from wonky_talkie.tts import make_backend; make_backend(load_config('$(CONFIG)').tts).warmup(); print('voice model ready')"
	else
		echo "tts.backend is not kitten in $(CONFIG); no voice model to fetch"
	fi

# ---------------------------------------------------------------------- checks

doctor:
	@$(PY) -c "import ctypes.util, importlib.util as u, pathlib, sys; \
	mods = 'numpy scipy onnxruntime websockets sounddevice sklearn faster_whisper openwakeword kittentts'.split(); \
	miss = [m for m in mods if u.find_spec(m) is None]; \
	libs = {n: ctypes.util.find_library(n) for n in ('portaudio', 'speexdsp', 'espeak-ng')}; \
	vad = pathlib.Path('models/silero_vad.onnx'); \
	print('python       ', sys.version.split()[0], '(venv)' if sys.prefix != sys.base_prefix else '(SYSTEM -- activate a venv)'); \
	[print(f'  {\"ok     \" if m not in miss else \"MISSING\"} {m}') for m in mods]; \
	[print(f'  {\"ok     \" if libs[n] else \"MISSING\"} lib{n}') for n in libs]; \
	print(f'  {\"ok     \" if vad.exists() else \"MISSING\"} models/silero_vad.onnx'); \
	bad = len(miss) + (0 if libs['portaudio'] else 1) + (0 if libs['espeak-ng'] else 1) + (0 if vad.exists() else 1); \
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
