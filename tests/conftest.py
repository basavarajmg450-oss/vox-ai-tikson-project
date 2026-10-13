"""Shared fixtures: a Flask test app wired with fake AI services.

The fakes implement the *same contracts* as the real services (see
``services/speech_to_text.py``, ``llm_service.py``, ``text_to_speech.py``), so
the routes, pipeline, session store and validation are all exercised for real -
only the model inference is replaced.  Real-model checks live in
``tests/test_integration.py`` (marked ``integration``).
"""

from __future__ import annotations

import io
import math
import struct
import uuid
import wave
from pathlib import Path

import pytest

from app import create_app
from config import Config
from services import Services
from services.errors import ApiError
from services.session_store import SessionStore
from services.text_to_speech import TTSService
from services.voice_pipeline import VoicePipeline


# --------------------------------------------------------------------------- fakes
class FakeSTT:
    """faster-whisper stand-in with the same return contract."""

    name = "faster-whisper"

    def __init__(self) -> None:
        self.loaded = False
        self.calls = 0

    def load(self) -> None:
        self.loaded = True

    def status(self):
        return {
            "name": self.name,
            "model": "fake-whisper",
            "device": "cpu",
            "compute_type": "int8",
            "language": "auto",
            "loaded": self.loaded,
            "state": "ready" if self.loaded else "idle",
            "load_seconds": 0.0,
            "last_error": None,
        }

    def transcribe(self, audio_bytes: bytes, content_type=None):
        if not audio_bytes:
            raise ApiError("The uploaded recording is empty (0 bytes).", 400, "empty_audio")
        if b"SILENCE" in audio_bytes:
            raise ApiError(
                "No speech detected in the recording.",
                422,
                "no_speech",
                hint="Record a longer clip and try again.",
            )
        if b"BOOM" in audio_bytes:
            raise RuntimeError("fake whisper exploded")
        self.calls += 1
        self.loaded = True
        return {
            "text": "hello from the fake",
            "language": "en",
            "language_probability": 0.97,
            "audio_duration_seconds": 1.5,
            "stt_ms": 12.5,
            "stt_load_ms": 0.0,
        }


class FakeLLM:
    """Ollama stand-in: answers echo every user message it was given.

    That makes history handling observable: if the backend passes prior
    turns, they show up in the answer.
    """

    name = "ollama"

    def __init__(self) -> None:
        self.reachable = True
        self.model_present = True

    def status(self):
        return {
            "name": self.name,
            "host": "http://localhost:11434",
            "model": "fake-model",
            "reachable": self.reachable,
            "model_present": self.model_present,
            "latency_ms": 1.0,
            "error": None if self.reachable else "Connection refused",
            "hint": "Start the local server with:  ollama serve" if not self.reachable else None,
        }

    def generate(self, history, system_prompt=None, temperature=None, max_tokens=None):
        if not self.reachable:
            raise ApiError("Ollama is not reachable.", 503, "ollama_unreachable",
                           hint="Start it with: ollama serve")
        users = [m["content"] for m in history if m.get("role") == "user"]
        answer = "FAKE_ANSWER users=" + " | ".join(users)
        return answer, {
            "llm_ms": 5.0,
            "prompt_tokens": 10,
            "completion_tokens": 7,
            "model": "fake-model",
            "temperature": 0.6 if temperature is None else temperature,
            "max_tokens": max_tokens,
        }


class FakeTTS(TTSService):
    """Piper stand-in that writes a real (tiny) WAV through the real class.

    Subclassing keeps ``resolve_audio()`` - the production path-traversal
    protection - fully under test.  ``synthesize`` is overridden so no model
    is ever loaded.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.fail = False
        self.calls = 0

    def load(self, model_path=None):  # noqa: ANN001 - signature mirrors parent
        return object()

    def synthesize(self, text: str, language=None):  # noqa: ANN001
        if self.fail:
            raise ApiError("Piper is unavailable.", 503, "tts_unavailable",
                           hint="Check PIPER_MODEL_PATH in .env")
        if not (text or "").strip():
            raise ApiError("Nothing to speak.", 400, "empty_text")
        self.calls += 1
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        filename = uuid.uuid4().hex + ".wav"
        path = self.audio_dir / filename
        rate = 16000
        frames = int(rate * 0.1)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            payload = b"".join(
                struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / rate)))
                for i in range(frames)
            )
            wav.writeframes(payload)
        return {
            "filename": filename,
            "duration_seconds": round(frames / rate, 3),
            "tts_ms": 8.0,
            "tts_load_ms": 0.0,
            "voice": "fake",
        }


# --------------------------------------------------------------------------- fixtures
@pytest.fixture()
def cfg(tmp_path) -> Config:
    c = Config.from_env()
    c.audio_dir = Path(tmp_path) / "audio"
    c.max_audio_bytes = 64 * 1024  # small enough to hit 413 cheaply
    c.strict_startup = False
    return c


@pytest.fixture()
def svc(cfg) -> Services:
    sessions = SessionStore(cfg)
    stt = FakeSTT()
    llm = FakeLLM()
    tts = FakeTTS(cfg)
    pipeline = VoicePipeline(cfg, sessions=sessions, stt=stt, llm=llm, tts=tts)
    return Services(config=cfg, sessions=sessions, stt=stt, llm=llm, tts=tts, pipeline=pipeline)


@pytest.fixture()
def app(cfg, svc):
    return create_app(config=cfg, services=svc)


@pytest.fixture()
def client(app):
    app.testing = True
    return app.test_client()


def make_audio(content: bytes = b"RIFF-fake-webm-audio", name: str = "clip.webm",
               ctype: str = "audio/webm"):
    """Multipart payload helper for the audio endpoints."""
    return {"audio": (io.BytesIO(content), name, ctype)}
