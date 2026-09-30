"""CPU or CUDA: decided once per process, from what is installed.

There are two installs. The CPU install has plain `onnxruntime`; the GPU install
has `onnxruntime-gpu` plus the CUDA 12 libraries as pip wheels (which run fine
under a CUDA 13 driver: NVIDIA drivers run older runtimes, and the wheels bring
their own). With `compute.device = "auto"` (the default) the install decides, and
nothing else in the config needs to know which one it is.

On CUDA, every model we load runs there:

- Silero VAD and KittenTTS: we choose the execution provider;
- openWakeWord: its feature models take `device="gpu"`, and its classifier
  head, which the library pins to CPU, is moved with `set_providers`;
- Whisper: CTranslate2's own `device` / `compute_type`.

ONNX Runtime fails *quietly* when CUDA is unusable: an unavailable provider only
logs a warning and the session runs on CPU. So this module checks for itself,
and `use_providers` confirms what each session actually got.
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

CUDA_EP = "CUDAExecutionProvider"
CPU_EP = "CPUExecutionProvider"
DEVICES = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class Compute:
    device: str  # "cuda" | "cpu"
    reason: str  # why, for the startup log

    @property
    def cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def onnx_providers(self) -> list[str]:
        return [CUDA_EP, CPU_EP] if self.cuda else [CPU_EP]

    def whisper_model(self, preference: str = "auto") -> str:
        """Whisper size. Measured per utterance on this project's desktop:
        large-v3-turbo takes ~60 ms on an RTX 4090 but ~5.6 s on the CPU, where
        small.en takes ~1.3 s - the difference between a conversation and not."""
        if preference and preference != "auto":
            return preference
        return "large-v3-turbo" if self.cuda else "small.en"

    def whisper_compute_type(self, preference: str = "auto") -> str:
        """CTranslate2 precision: float16 on a GPU, int8 on a CPU."""
        if preference and preference != "auto":
            return preference
        return "float16" if self.cuda else "int8"


CPU = Compute("cpu", "default")


def cuda_device_count() -> int:
    """Visible CUDA devices, straight from the driver (no CUDA toolkit needed)."""
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        count = ctypes.c_int(0)
        if lib.cuInit(0) != 0 or lib.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return 0
        return count.value
    return 0


def onnx_has_cuda() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return CUDA_EP in ort.get_available_providers()


def resolve(preference: str = "auto") -> Compute:
    """Turn `compute.device` into a concrete device, or explain why it can't."""
    if preference not in DEVICES:
        raise ValueError(f"compute.device must be one of {', '.join(DEVICES)}, not {preference!r}")
    if preference == "cpu":
        return Compute("cpu", "compute.device = cpu")
    runtime, gpus = onnx_has_cuda(), cuda_device_count()
    if runtime and gpus:
        why = "compute.device = cuda" if preference == "cuda" else "GPU install and a GPU is visible"
        return Compute("cuda", f"{why} ({gpus} device{'s' if gpus != 1 else ''})")
    missing = ("onnxruntime is the CPU build (this is the CPU install; `make setup-gpu` for the GPU one)"
               if not runtime else "no CUDA device is visible (no driver, or a container run without --gpus)")
    if preference == "cuda":
        raise RuntimeError(f"compute.device = cuda, but {missing}")
    return Compute("cpu", missing)


def prepare(compute: Compute) -> None:
    """Load the pip-installed CUDA libraries before any model needs them.

    `preload_dlls` finds the nvidia-*-cu12 wheels in site-packages. Loading them
    first also lets CTranslate2 find cuBLAS and cuDNN by name, so Whisper needs
    no LD_LIBRARY_PATH either.
    """
    if not compute.cuda:
        return
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    # On CUDA every session logs a "Memcpy nodes are added" performance hint at
    # WARNING. It's expected (a few small ops stay on the CPU) and it would bury
    # the startup log; use_providers reports what actually matters. Errors,
    # including a CUDA provider that fails to load, still come through.
    ort.set_default_logger_severity(3)


def use_providers(session, compute: Compute, what: str) -> None:
    """Point an ONNX Runtime session at our device and confirm it took."""
    if compute.cuda:
        session.set_providers(compute.onnx_providers)
    got = session.get_providers()[0]
    want = compute.onnx_providers[0]
    if got != want:  # ONNX Runtime only warns about this, so we say it louder
        log.warning("%s wanted %s but is running on %s", what, want, got)
