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
import threading
import time
import uuid
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import create_app
from config import Config
from services import Services
from services.errors import ApiError
from services.session_store import SessionStore
from services.text_to_speech import TTSService
from services.voice_pipeline import VoicePipeline


# --------------------------------------------------------------------------- fakes
def gemini_model_entry(
    name: str,
    actions=("generateContent", "countTokens"),
    description: str = "",
):
    """A ``models.list()`` entry mirroring the SDK's ``Model`` fields.

    ``actions=None`` simulates a model whose capability metadata is missing.
    """
    entry = {"name": name if name.startswith("models/") else f"models/{name}"}
    if actions is not None:
        entry["supported_actions"] = list(actions)
    if description:
        entry["description"] = description
    return entry


class FakeGenAI:
    """Mock of the ``google-genai`` client (``models.list`` + generation).

    Lets model discovery, selection, caching and fallback run without any
    network access or API key.  Thread-safe so concurrency tests can use it.
    """

    def __init__(
        self,
        model_entries=None,
        list_error=None,
        generate_error=None,
        fail_models=None,
        response=None,
        list_delay=0.0,
    ) -> None:
        self.entries = list(model_entries or [])
        self.list_error = list_error
        self.generate_error = generate_error
        self.fail_models = dict(fail_models or {})
        self.response = response
        self.list_delay = list_delay
        self.list_calls = 0
        self.generate_calls = []
        self._lock = threading.Lock()
        self.models = self  # client.models.list() / client.models.generate_content()

    def list(self, **kwargs):
        with self._lock:
            self.list_calls += 1
        if self.list_delay:
            time.sleep(self.list_delay)
        if self.list_error is not None:
            raise self.list_error
        return list(self.entries)

    def generate_content(self, **kwargs):
        with self._lock:
            self.generate_calls.append(kwargs)
        model = kwargs.get("model")
        if model in self.fail_models:
            raise self.fail_models[model]
        if self.generate_error is not None:
            raise self.generate_error
        if self.response is not None:
            return self.response
        return SimpleNamespace(
            text="A concise answer.",
            model_version=model,
            usage_metadata=SimpleNamespace(
                prompt_token_count=12, candidates_token_count=5
            ),
        )


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
    """Gemini stand-in: answers echo every user message it was given.

    That makes history handling observable: if the backend passes prior
    turns, they show up in the answer.
    """

    name = "gemini"

    def __init__(self) -> None:
        self.configured = True

    def status(self):
        return {
            "name": self.name,
            "model": "fake-model",
            "configured": self.configured,
            "availability": "not_checked",
            "reachable": None,
            "hint": None if self.configured else "Set GEMINI_API_KEY in the backend .env file.",
        }

    def generate(self, history, system_prompt=None, temperature=None, max_tokens=None):
        if not self.configured:
            raise ApiError("Gemini is not configured.", 503, "gemini_missing_credentials",
                           hint="Set GEMINI_API_KEY in .env.")
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

    def status(self):
        return {
            "name": self.name,
            "model_path": str(self.config.piper_model_path),
            "model_present": True,
            "config_present": True,
            "hindi_model_path": str(self.config.piper_model_path_hi),
            "hindi_present": True,
            "length_scale": self.config.piper_length_scale,
            "loaded": self.loaded,
            "voices_loaded": [],
            "state": "ready",
            "last_error": None,
        }

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


class FakeWebSearch:
    """Mock web search service for testing."""

    name = "fake-search"

    def __init__(self, ok: bool = True, snippets=None, sources=None, error: str = "Search failed") -> None:
        self.ok = ok
        self.snippets = snippets if snippets is not None else [
            {
                "title": "Python 3.14 Release Notes",
                "body": "Python 3.14 was released with significant performance improvements.",
                "url": "https://www.python.org/downloads/release/python-3140/",
            }
        ]
        self.sources = sources if sources is not None else [
            {"title": "Python 3.14 Release Notes", "url": "https://www.python.org/downloads/release/python-3140/"}
        ]
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    def status(self) -> Dict[str, Any]:
        return {"name": self.name, "enabled": True}

    def search(self, query: str, max_results: Optional[int] = None) -> Dict[str, Any]:
        self.calls.append({"query": query, "max_results": max_results})
        if not self.ok:
            return {"ok": False, "error": self.error, "search_ms": 15.0}
        return {
            "ok": True,
            "snippets": self.snippets,
            "sources": self.sources,
            "search_ms": 25.0,
        }


# --------------------------------------------------------------------------- fixtures
@pytest.fixture()
def cfg(tmp_path) -> Config:
    c = Config.from_env()
    c.audio_dir = Path(tmp_path) / "audio"
    c.max_audio_bytes = 64 * 1024  # small enough to hit 413 cheaply
    c.strict_startup = False
    c.gemini_api_key = "unit-test-key"
    c.web_search_enabled = True
    return c


@pytest.fixture()
def svc(cfg) -> Services:
    sessions = SessionStore(cfg)
    stt = FakeSTT()
    llm = FakeLLM()
    tts = FakeTTS(cfg)
    web_search = FakeWebSearch()
    pipeline = VoicePipeline(
        cfg, sessions=sessions, stt=stt, llm=llm, tts=tts, web_search=web_search
    )
    return Services(
        config=cfg,
        sessions=sessions,
        stt=stt,
        llm=llm,
        tts=tts,
        pipeline=pipeline,
        web_search=web_search,
    )


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
