"""Integration checks against the REAL local models (Ollama + Whisper + Piper).

Skipped by default so the unit suite stays fast and dependency-free:

    set VOXAI_INTEGRATION=1          # PowerShell
    .venv\\Scripts\\python.exe -m pytest -m integration -v

Requirements: ``ollama serve`` running with the configured model pulled, and
the Whisper + Piper assets downloaded (``python scripts/prefetch_assets.py``).
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("VOXAI_INTEGRATION"),
        reason="set VOXAI_INTEGRATION=1 to run against the real local models",
    ),
]


@pytest.fixture(scope="module")
def real_app():
    from app import create_app

    application = create_app(warmup=False)
    application.testing = True
    return application


@pytest.fixture(scope="module")
def real_client(real_app):
    return real_app.test_client()


def test_health_real_services_ready(real_client):
    body = real_client.get("/api/health").get_json()
    assert body["ollama"]["reachable"] is True, "is `ollama serve` running?"
    assert body["ollama"]["model_present"] is True, "model missing: run `ollama pull qwen3:4b`"
    assert body["piper"]["model_present"] is True, "run `python scripts/prefetch_assets.py`"
    assert body["status"] == "ok", f"startup issues: {body['startup_issues']}"


def test_chat_real_llm_answers(real_client):
    resp = real_client.post("/api/chat", json={"message": "Reply with exactly: OK"})
    assert resp.status_code == 200
    answer = resp.get_json()["response"].strip()
    assert answer, "empty answer from the model"
    assert resp.get_json()["latency"]["llm_ms"] > 0


def test_multi_turn_memory_real_llm(real_client):
    sid = real_client.post("/api/session/new").get_json()["session_id"]
    first = real_client.post("/api/chat", json={"message": "My name is Trikon. Remember it.", "session_id": sid})
    assert first.status_code == 200
    second = real_client.post("/api/chat", json={"message": "What is my name?", "session_id": sid})
    assert second.status_code == 200
    answer = second.get_json()["response"]
    assert "Trikon" in answer, f"model forgot context: {answer!r}"


def test_tts_real_wav_is_valid(real_client, real_app):
    from routes._common import ctx  # noqa: F401 - documents intent

    svc = real_app.extensions["voxai"].services
    out = svc.tts.synthesize("Hello from VoxAI, this is a local voice.")
    path = Path(svc.tts.audio_dir) / out["filename"]
    assert path.exists()
    with wave.open(str(path), "rb") as wav:
        assert wav.getnframes() > 0
        assert wav.getframerate() > 8000
    assert out["duration_seconds"] > 0.2
    # And it is served by the hardened endpoint:
    resp = real_client.get(f"/api/audio/{out['filename']}")
    assert resp.status_code == 200
    assert resp.data[:4] == b"RIFF"


def test_stt_real_roundtrip_on_tts_audio(real_client, real_app):
    """Piper speaks -> Whisper transcribes: STT really works on real audio."""
    svc = real_app.extensions["voxai"].services
    out = svc.tts.synthesize("The quick brown fox jumps over the lazy dog.")
    path = Path(svc.tts.audio_dir) / out["filename"]
    with open(path, "rb") as fh:
        resp = real_client.post(
            "/api/transcribe",
            data={"audio": (fh, "roundtrip.wav", "audio/wav")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    text = resp.get_json()["transcript"].strip()
    assert len(text.split()) >= 3, f"transcript too short: {text!r}"


def test_voice_real_end_to_end(real_client, real_app):
    svc = real_app.extensions["voxai"].services
    out = svc.tts.synthesize("What is the capital of France?")
    path = Path(svc.tts.audio_dir) / out["filename"]
    with open(path, "rb") as fh:
        resp = real_client.post(
            "/api/voice",
            data={"audio": (fh, "question.wav", "audio/wav"), "recording_ms": "3000"},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["transcript"].strip()
    assert body["response"].strip()
    assert body["audio_url"]
    lat = body["latency"]
    assert lat["stt_ms"] > 0 and lat["llm_ms"] > 0 and lat["tts_ms"] > 0
    assert lat["streaming"] is False
    audio = real_client.get(body["audio_url"])
    assert audio.status_code == 200 and audio.data[:4] == b"RIFF"
