"""The HTTP STT backend against a stand-in for whisper.cpp's /inference endpoint."""

import io
import json
import sys
import threading
import unittest
import wave
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_listener.config import SttConfig  # noqa: E402
from voice_listener.stt import HttpBackend  # noqa: E402

RECEIVED: dict = {}


class FakeWhisperServer(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        msg = BytesParser(policy=default).parsebytes(
            b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
        fields, wav = {}, None
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if part.get_filename():
                wav = part.get_payload(decode=True)
            else:
                fields[name] = part.get_content().strip()
        with wave.open(io.BytesIO(wav)) as w:
            RECEIVED.update(fields=fields, path=self.path, rate=w.getframerate(), frames=w.getnframes(),
                            auth=self.headers.get("Authorization"))
        out = json.dumps({"text": " Turn on the office lights.\n"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class HttpBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), FakeWhisperServer)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_whisper_cpp_flavor(self):
        cfg = SttConfig(backend="whisper_cpp", language="en", initial_prompt="Jarvis")
        cfg.http.url = self.base + "/inference"
        res = HttpBackend(cfg, "whisper_cpp").transcribe(np.zeros(16000, np.int16))
        self.assertEqual(res.text, "Turn on the office lights.")
        self.assertEqual(RECEIVED["path"], "/inference")
        self.assertEqual(RECEIVED["rate"], 16000)
        self.assertEqual(RECEIVED["frames"], 16000)
        self.assertEqual(RECEIVED["fields"]["response_format"], "json")
        self.assertEqual(RECEIVED["fields"]["language"], "en")
        self.assertEqual(RECEIVED["fields"]["prompt"], "Jarvis")
        self.assertEqual(RECEIVED["fields"]["temperature"], "0.0")

    def test_openai_flavor(self):
        cfg = SttConfig(backend="openai")
        cfg.http.url = self.base + "/v1/audio/transcriptions"
        cfg.http.model = "whisper-large-v3"
        cfg.http.api_key = "k"
        HttpBackend(cfg, "openai").transcribe(np.zeros(8000, np.int16))
        self.assertEqual(RECEIVED["fields"]["model"], "whisper-large-v3")
        self.assertEqual(RECEIVED["auth"], "Bearer k")
        self.assertNotIn("temperature", RECEIVED["fields"])


if __name__ == "__main__":
    unittest.main()
