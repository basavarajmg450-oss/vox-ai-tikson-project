"""Integration checks against Gemini API, Whisper and Piper.

Skipped by default so the unit suite stays fast and dependency-free:

    set VOXAI_INTEGRATION=1          # PowerShell
    .venv\\Scripts\\python.exe -m pytest -m integration -v

Requirements: ``GEMINI_API_KEY`` configured, plus Whisper and Piper assets
downloaded (``python scripts/prefetch_assets.py``).
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
    assert body["gemini"]["configured"] is True, "set GEMINI_API_KEY in .env"
    assert body["gemini"]["availability"] == "not_checked"
    assert body["gemini"]["discovery"] == "not_attempted"  # health makes no API request
    assert body["piper"]["model_present"] is True, "run `python scripts/prefetch_assets.py`"
    assert body["status"] == "ok", f"startup issues: {body['startup_issues']}"


def test_model_discovery_selects_a_real_available_model(real_app):
    """Free ``models.list()`` check against the real key (no generation call)."""
    svc = real_app.extensions["voxai"].services

    discovered = svc.llm.discover_models(refresh=True)

    assert discovered, "no text-generation Gemini model available to this API key"
    assert svc.llm.selected_model in discovered
    status = svc.llm.status()
    assert status["discovery"] == "ok"
    assert status["availability"] == "available"  # verified, not assumed
    assert "unit-test-key" not in repr(status)
    # The preferred model, when it exists in the discovered list, comes first:
    assert discovered[0] == svc.llm.selected_model


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


# ------------------------------------------------------------------ voice consistency
# The configured pair is a verified same-gender pair (male):
#   en_US-ryan-medium  - male US-English single speaker (finetuned from lessac)
#   hi_IN-rohan-medium - male Hindi single speaker ("Hindi Mono Male" database)
# A mixed-gender pair causes gender switches between languages and must not be used.


def test_tts_same_voice_for_every_english_response(real_app):
    """Greeting-style, answer, follow-up and error-style text: one voice."""
    svc = real_app.extensions["voxai"].services
    expected = Path(svc.tts.config.piper_model_path).name
    texts = [
        "Hello! I am VoxAI, your local voice assistant.",   # greeting
        "The capital of France is Paris.",                   # normal answer
        "About two million people live there.",              # follow-up
        "Sorry, I could not complete that request.",         # error-style message
    ]
    voices = {svc.tts.synthesize(t)["voice"] for t in texts}
    assert voices == {expected}, f"voice changed between responses: {voices}"


def test_tts_same_voice_for_every_hindi_response(real_app):
    """Hindi responses all use the single configured Hindi voice."""
    svc = real_app.extensions["voxai"].services
    hi = Path(svc.tts.config.piper_model_path_hi)
    assert hi.exists(), "run .venv\\Scripts\\python.exe scripts\\prefetch_assets.py"
    texts = [
        "नमस्ते! मैं वॉक्सआई हूँ, आपकी स्थानीय सहायक।",
        "भारत की राजधानी नई दिल्ली है।",
        "वहाँ लगभग बीस लाख लोग रहते हैं।",
        "क्षमा करें, मैं वह अनुरोध पूरा नहीं कर सका।",
    ]
    voices = {svc.tts.synthesize(t, language="hi")["voice"] for t in texts}
    assert voices == {hi.name}, f"Hindi voice changed between responses: {voices}"


def test_tts_alternating_languages_use_fixed_language_models(real_app):
    """EN/HI/EN/HI turns: language selects the model, never a random speaker."""
    svc = real_app.extensions["voxai"].services
    en = Path(svc.tts.config.piper_model_path).name
    hi = Path(svc.tts.config.piper_model_path_hi).name
    for turn, lang in enumerate([None, "hi", "en", "hi", None, "hi"]):
        out = svc.tts.synthesize(f"Voice consistency check, turn {turn}.", language=lang)
        want = hi if (lang or "").startswith("hi") else en
        assert out["voice"] == want, (
            f"language={lang!r} selected {out['voice']!r}, expected {want!r}"
        )


def test_voice_pipeline_reports_voice_matching_detected_language(real_client, real_app):
    """Full /api/voice run: tts_voice always equals the configured model for
    the detected language.  A failed voice surfaces as tts_error with
    audio_url None - it can never fall back to a different voice."""
    svc = real_app.extensions["voxai"].services
    hi = Path(svc.tts.config.piper_model_path_hi)
    if not hi.exists():
        pytest.skip("Hindi voice not installed")
    en = Path(svc.tts.config.piper_model_path).name
    out = svc.tts.synthesize(
        "नमस्ते, मेरा नाम वॉक्सआई है। मैं आपकी क्या मदद कर सकती हूँ?", language="hi"
    )
    path = Path(svc.tts.audio_dir) / out["filename"]
    with open(path, "rb") as fh:
        resp = real_client.post(
            "/api/voice",
            data={"audio": (fh, "hindi.wav", "audio/wav")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    body = resp.get_json()
    want = hi.name if body["language"].startswith("hi") else en
    assert body["tts_voice"] == want
    # Backend timing instrumentation must survive the frontend cleanup.
    assert body["latency"]["stt_ms"] > 0 and body["latency"]["tts_ms"] > 0
