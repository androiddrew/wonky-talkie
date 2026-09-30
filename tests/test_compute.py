"""CPU vs CUDA: the install decides, and a wrong answer must be loud.

ONNX Runtime's own failure mode is silent (an unavailable provider is a warning,
and the session runs on the CPU), so these tests pin down that we never make the
same choice quietly.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wonky_talkie import compute
from wonky_talkie.compute import CPU_EP, CUDA_EP, Compute, resolve, use_providers
from wonky_talkie.config import Config, load_config, validate

ROOT = Path(__file__).resolve().parent.parent


def _machine(runtime: bool, gpus: int):
    """Pretend the install has (or lacks) onnxruntime-gpu and N visible GPUs."""
    return mock.patch.multiple(compute, onnx_has_cuda=lambda: runtime,
                               cuda_device_count=lambda: gpus)


class Resolve(unittest.TestCase):
    def test_auto_uses_cuda_on_the_gpu_install_with_a_gpu(self):
        with _machine(runtime=True, gpus=1):
            c = resolve("auto")
        self.assertEqual(c.device, "cuda")
        self.assertIn("GPU install", c.reason)

    def test_auto_falls_back_to_cpu_on_the_cpu_install(self):
        with _machine(runtime=False, gpus=1):  # a GPU in the box doesn't matter
            c = resolve("auto")
        self.assertEqual(c.device, "cpu")
        self.assertIn("CPU build", c.reason)

    def test_auto_falls_back_to_cpu_when_no_gpu_is_visible(self):
        """The GPU image run without --gpus: say so, rather than just 'cpu'."""
        with _machine(runtime=True, gpus=0):
            c = resolve("auto")
        self.assertEqual(c.device, "cpu")
        self.assertIn("--gpus", c.reason)

    def test_cpu_is_honoured_even_with_a_gpu(self):
        with _machine(runtime=True, gpus=2):
            self.assertEqual(resolve("cpu").device, "cpu")

    def test_cuda_refuses_to_start_without_the_runtime(self):
        with _machine(runtime=False, gpus=1), self.assertRaises(RuntimeError) as cm:
            resolve("cuda")
        self.assertIn("setup-gpu", str(cm.exception))

    def test_cuda_refuses_to_start_without_a_gpu(self):
        with _machine(runtime=True, gpus=0), self.assertRaises(RuntimeError):
            resolve("cuda")

    def test_unknown_device_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve("gpu")  # a plausible typo; the word is "cuda"


class Choices(unittest.TestCase):
    def test_onnx_providers(self):
        self.assertEqual(Compute("cuda", "").onnx_providers, [CUDA_EP, CPU_EP])
        self.assertEqual(Compute("cpu", "").onnx_providers, [CPU_EP])

    def test_whisper_precision_follows_the_device(self):
        self.assertEqual(Compute("cuda", "").whisper_compute_type(), "float16")
        self.assertEqual(Compute("cpu", "").whisper_compute_type("auto"), "int8")

    def test_whisper_size_follows_the_device(self):
        self.assertEqual(Compute("cuda", "").whisper_model(), "large-v3-turbo")
        self.assertEqual(Compute("cpu", "").whisper_model("auto"), "small.en")
        self.assertEqual(Compute("cpu", "").whisper_model("medium.en"), "medium.en")

    def test_an_explicit_precision_wins(self):
        self.assertEqual(Compute("cuda", "").whisper_compute_type("int8_float16"), "int8_float16")


class _FakeSession:
    def __init__(self, lands_on: str):
        self.lands_on = lands_on
        self.asked: list | None = None

    def set_providers(self, providers):
        self.asked = providers

    def get_providers(self):
        return [self.lands_on, CPU_EP]


class UseProviders(unittest.TestCase):
    def test_cuda_is_requested_on_the_session(self):
        s = _FakeSession(CUDA_EP)
        use_providers(s, Compute("cuda", ""), "thing")
        self.assertEqual(s.asked, [CUDA_EP, CPU_EP])

    def test_cpu_leaves_the_session_alone(self):
        s = _FakeSession(CPU_EP)
        use_providers(s, Compute("cpu", ""), "thing")
        self.assertIsNone(s.asked)

    def test_a_silent_cpu_fallback_is_reported(self):
        """What ONNX Runtime does when CUDA won't load: we must not hide it."""
        s = _FakeSession(CPU_EP)
        with self.assertLogs("wonky_talkie.compute", "WARNING") as logs:
            use_providers(s, Compute("cuda", ""), "KittenTTS")
        self.assertIn("KittenTTS wanted CUDAExecutionProvider", logs.output[0])


class ConfigKeys(unittest.TestCase):
    def test_default_is_auto(self):
        self.assertEqual(Config().compute.device, "auto")

    def test_bad_device_fails_validation(self):
        cfg = Config()
        cfg.compute.device = "gpu"
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_the_old_whisper_device_key_says_where_it_went(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "old.toml"
            p.write_text('[stt.faster_whisper]\ndevice = "cuda"\n')
            with self.assertRaises(ValueError) as cm:
                load_config(p)
        self.assertIn("moved", str(cm.exception))
        self.assertIn("[compute] device", str(cm.exception))


class RealSessions(unittest.TestCase):
    """Against the installed onnxruntime, whichever build it is."""

    def test_silero_vad_defaults_to_the_cpu(self):
        from wonky_talkie.vad import SileroVAD

        vad = SileroVAD(str(ROOT / "models" / "silero_vad.onnx"))
        self.assertEqual(vad._sess.get_providers()[0], CPU_EP)

    def test_silero_vad_follows_a_resolved_device(self):
        from wonky_talkie.vad import SileroVAD

        c = resolve("auto")
        vad = SileroVAD(str(ROOT / "models" / "silero_vad.onnx"), compute=c)
        self.assertEqual(vad._sess.get_providers()[0], c.onnx_providers[0])


if __name__ == "__main__":
    unittest.main()
