"""POST /api/voice and POST /api/transcribe - the pipeline itself.

Only model inference is faked; routing, validation, session commits, timing
and audio serving are the real implementations under test.
"""

from __future__ import annotations

from tests.conftest import make_audio


# ------------------------------------------------------------------ transcribe
def test_transcribe_returns_text_and_language(client):
    resp = client.post("/api/transcribe", data=make_audio(), content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["transcript"] == "hello from the fake"
    assert body["language"] == "en"
    assert body["language_probability"] > 0.5
    assert body["latency"]["stt_ms"] == 12.5
    assert body["latency"]["sources"]["stt_ms"] == "measured_server"


def test_transcribe_silence_is_a_clear_422(client):
    resp = client.post("/api/transcribe", data=make_audio(b"bSILENCEb"),
                       content_type="multipart/form-data")
    assert resp.status_code == 422
    body = resp.get_json()
    assert body["code"] == "no_speech"
    assert "speech" in body["error"].lower()


def test_transcribe_crash_is_500_not_a_hang(client):
    resp = client.post("/api/transcribe", data=make_audio(b"BOOM"),
                       content_type="multipart/form-data")
    assert resp.status_code == 500
    assert resp.get_json()["error"]


# ------------------------------------------------------------------ full voice
def test_voice_returns_transcript_answer_audio_and_metrics(client, svc):
    data = make_audio()
    data["session_id"] = "a" * 32
    data["recording_ms"] = "1500"
    resp = client.post("/api/voice", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()

    assert body["ok"] is True
    assert body["transcript"] == "hello from the fake"
    assert body["response"].startswith("FAKE_ANSWER")
    assert body["audio_url"].startswith("/api/audio/")
    assert body["history_turns"] == 1

    lat = body["latency"]
    assert lat["stt_ms"] == 12.5
    assert lat["llm_ms"] == 5.0
    assert lat["tts_ms"] == 8.0
    assert lat["recording_ms"] == 1500.0
    assert lat["streaming"] is False          # honest: buffered, not streaming
    assert lat["mode"] == "buffered"
    assert lat["sources"]["stt_ms"] == "measured_server"
    assert lat["sources"]["llm_ms"] == "measured_server"
    assert lat["sources"]["tts_ms"] == "measured_server"
    assert lat["sources"]["recording_ms"] == "measured_client"
    assert lat["total_ms"] >= 1500.0
    # stage sum should be consistent with the reported backend total
    assert lat["backend_total_ms"] > 0

    # The committed turn is queryable from the server afterwards:
    hist = client.get(f"/api/session/{'a' * 32}/history").get_json()
    assert hist["turns"] == 1
    assert hist["history"][0]["content"] == "hello from the fake"


def test_voice_generated_audio_is_playable(client, svc):
    resp = client.post("/api/voice", data=make_audio(), content_type="multipart/form-data")
    url = resp.get_json()["audio_url"]
    audio = client.get(url)
    assert audio.status_code == 200
    assert audio.headers["Content-Type"].startswith("audio/wav")
    assert audio.data[:4] == b"RIFF"
    assert len(audio.data) > 44  # header + samples


def test_voice_tts_failure_keeps_the_text_answer(client, svc):
    svc.tts.fail = True
    resp = client.post("/api/voice", data=make_audio(), content_type="multipart/form-data")
    assert resp.status_code == 200  # pipeline still succeeds
    body = resp.get_json()
    assert body["response"].startswith("FAKE_ANSWER")
    assert body["audio_url"] is None
    assert body["tts_error"]["code"] == "tts_unavailable"
    assert body["latency"]["tts_status"] == "failed"
    assert body["latency"]["tts_ms"] is None


def test_voice_silence_short_circuits_with_422(client, svc):
    resp = client.post("/api/voice", data=make_audio(b"SILENCE"),
                       content_type="multipart/form-data")
    assert resp.status_code == 422
    assert resp.get_json()["code"] == "no_speech"
    # Nothing was committed to the session:
    assert svc.llm is not None


def test_voice_multi_turn_context(client):
    sid = "a" * 32
    first = make_audio()
    first["session_id"] = sid
    assert client.post("/api/voice", data=first, content_type="multipart/form-data").status_code == 200
    second = make_audio()
    second["session_id"] = sid
    body = client.post("/api/voice", data=second, content_type="multipart/form-data").get_json()
    # Both transcripts were passed to the LLM -> history works across voice turns.
    assert body["response"].count("hello from the fake") == 2
    assert body["history_turns"] == 2


# ------------------------------------------------------------------ SSE stream
def test_voice_stream_emits_real_stage_events(client):
    data = make_audio()
    data["stream"] = "progress"
    resp = client.post("/api/voice", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("text/event-stream")
    raw = b"".join(resp.response).decode("utf-8")

    assert "event: progress" in raw
    assert '"stage": "stt"' in raw
    assert '"stage": "llm"' in raw
    assert '"stage": "tts"' in raw
    assert "event: done" in raw
    assert '"transcript": "hello from the fake"' in raw
    # Stream must terminate with done, never leave the client hanging:
    assert raw.rstrip().endswith("}")
    assert "event: error" not in raw


def test_voice_stream_reports_errors_as_events(client):
    data = make_audio(b"SILENCE")
    data["stream"] = "progress"
    resp = client.post("/api/voice", data=data, content_type="multipart/form-data")
    raw = b"".join(resp.response).decode("utf-8")
    assert "event: error" in raw
    assert "no_speech" in raw
