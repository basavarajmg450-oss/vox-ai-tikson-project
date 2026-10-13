"""API validation: request bodies, audio uploads and error status codes."""

from __future__ import annotations

import io

from tests.conftest import make_audio


# ------------------------------------------------------------------ /api/chat
def test_chat_requires_json_content_type(client):
    resp = client.post("/api/chat", data="message=hi")
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "invalid_content_type"


def test_chat_rejects_non_object_json(client):
    resp = client.post("/api/chat", json=["not", "an", "object"])
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "invalid_json"


def test_chat_rejects_missing_message(client):
    resp = client.post("/api/chat", json={"session_id": None})
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "empty_message"


def test_chat_rejects_blank_message(client):
    resp = client.post("/api/chat", json={"message": "   "})
    assert resp.status_code == 400


def test_chat_rejects_oversized_message(client, cfg):
    resp = client.post("/api/chat", json={"message": "x" * (cfg.max_message_chars + 1)})
    assert resp.status_code == 413


def test_chat_rejects_bad_temperature(client):
    for bad in ("hot", 9, -1):
        resp = client.post("/api/chat", json={"message": "hi", "temperature": bad})
        assert resp.status_code == 400, bad
        assert resp.get_json()["code"] == "invalid_temperature"


def test_chat_rejects_bad_max_tokens(client):
    for bad in ("lots", 0, 99999):
        resp = client.post("/api/chat", json={"message": "hi", "max_tokens": bad})
        assert resp.status_code == 400, bad
        assert resp.get_json()["code"] == "invalid_max_tokens"


# ------------------------------------------------------------- audio uploads
def test_transcribe_requires_a_file(client):
    resp = client.post("/api/transcribe", data={})
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "missing_audio"


def test_transcribe_rejects_empty_file(client):
    resp = client.post("/api/transcribe",
                       data={"audio": (io.BytesIO(b""), "clip.webm", "audio/webm")},
                       content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "empty_audio"


def test_transcribe_rejects_unsupported_type(client):
    resp = client.post("/api/transcribe",
                       data={"audio": (io.BytesIO(b"just text"), "notes.txt", "text/plain")},
                       content_type="multipart/form-data")
    assert resp.status_code == 415
    assert resp.get_json()["code"] == "unsupported_audio_type"


def test_transcribe_rejects_oversized_upload(client, cfg):
    big = b"a" * (cfg.max_audio_bytes + 1024)
    resp = client.post("/api/transcribe",
                       data={"audio": (io.BytesIO(big), "huge.webm", "audio/webm")},
                       content_type="multipart/form-data")
    assert resp.status_code == 413
    assert resp.get_json()["code"] == "audio_too_large"


def test_voice_rejects_missing_audio(client):
    resp = client.post("/api/voice", data={"session_id": "a" * 32},
                       content_type="multipart/form-data")
    assert resp.status_code == 400


def test_voice_rejects_non_numeric_recording_ms(client):
    data = make_audio()
    data["session_id"] = "a" * 32
    data["recording_ms"] = "yesterday"
    resp = client.post("/api/voice", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "invalid_parameter"


def test_voice_rejects_out_of_range_recording_ms(client):
    data = make_audio()
    data["session_id"] = "a" * 32
    data["recording_ms"] = "-5"
    resp = client.post("/api/voice", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "invalid_recording_ms"


# ------------------------------------------------------------- /api/audio/<file>
def test_audio_unknown_file_is_404(client):
    assert client.get("/api/audio/never-existed.wav").status_code == 404


def test_audio_path_traversal_is_blocked(client):
    attempts = [
        "..%5C..%5Capp.py",            # ..\..\app.py (backslash)
        "..%2F..%2Fapp.py",            # ../../app.py (encoded slash)
        "....//....//app.py",
        "%2e%2e%2f%2e%2e%2fconfig.py",
        "..%5C..%5C..%5CWindows%5Cwin.ini",
        "nonexistent.wav%00.png",
    ]
    for name in attempts:
        resp = client.get(f"/api/audio/{name}")
        assert resp.status_code == 404, (name, resp.status_code)


def test_audio_real_file_is_served_as_wav(client, svc):
    fake = svc.tts.synthesize("hello world")
    resp = client.get(f"/api/audio/{fake['filename']}")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("audio/wav")
    assert resp.data[:4] == b"RIFF"
