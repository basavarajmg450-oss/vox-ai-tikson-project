"""GET /api/health - server and dependency reporting."""

from __future__ import annotations

from google.genai import errors

from app import create_app
from services import Services
from services.llm_service import LLMService
from services.session_store import SessionStore
from services.voice_pipeline import VoicePipeline
from tests.conftest import FakeGenAI, FakeSTT, FakeTTS, gemini_model_entry


def make_real_llm_app(cfg, fake):
    """A test app whose LLM is the *real* LLMService over a mocked SDK."""
    cfg.gemini_model = "gemini-2.5-flash"  # deterministic preferred model
    sessions = SessionStore(cfg)
    stt = FakeSTT()
    llm = LLMService(cfg, client=fake)
    tts = FakeTTS(cfg)
    pipeline = VoicePipeline(cfg, sessions=sessions, stt=stt, llm=llm, tts=tts)
    svc = Services(config=cfg, sessions=sessions, stt=stt, llm=llm, tts=tts, pipeline=pipeline)
    application = create_app(config=cfg, services=svc)
    application.testing = True
    return application


def test_health_reports_all_dependencies(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["service"] == "voxai"
    assert body["status"] in {"ok", "degraded"}
    # Dependency blocks the assignment asks for:
    assert body["gemini"]["configured"] is True
    assert body["gemini"]["availability"] == "not_checked"
    assert body["whisper"]["loaded"] is False  # lazy until first use
    assert body["piper"]["model_present"] is True
    assert "active_sessions" in body["sessions"]
    assert "max_history_turns" in body["config"]
    assert "web_search" in body
    assert body["web_search"]["name"] == "fake-search"
    # Endpoint documentation is part of the contract:
    paths = {(e["method"], e["path"]) for e in body["endpoints"]}
    assert ("GET", "/api/health") in paths
    assert ("POST", "/api/voice") in paths
    assert ("DELETE", "/api/session/<session_id>") in paths


def test_health_degrades_when_gemini_credentials_are_missing(client, svc):
    svc.llm.configured = False
    svc.config.gemini_api_key = ""
    resp = client.get("/api/health")
    body = resp.get_json()
    assert resp.status_code == 200  # liveness still answers
    assert body["status"] == "degraded"
    assert body["gemini"]["configured"] is False
    assert "GEMINI_API_KEY" in body["gemini"]["hint"]


def test_health_does_not_claim_model_availability_without_api_request(client):
    body = client.get("/api/health").get_json()
    assert body["gemini"]["availability"] == "not_checked"


def test_health_never_returns_api_key(client):
    body = client.get("/api/health").get_json()
    assert "unit-test-key" not in repr(body)


def test_unknown_api_route_is_json_404(client):
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404
    assert resp.get_json()["error"]


# ------------------------------------------- health + real LLMService (mock SDK)
def test_health_keeps_not_checked_and_makes_no_api_request_before_a_chat(cfg):
    fake = FakeGenAI(model_entries=[gemini_model_entry("gemini-2.5-flash")])
    client = make_real_llm_app(cfg, fake).test_client()

    body = client.get("/api/health").get_json()

    assert body["gemini"]["configured"] is True
    assert body["gemini"]["availability"] == "not_checked"
    assert body["gemini"]["discovery"] == "not_attempted"
    assert body["gemini"]["selected_model"] is None
    assert body["gemini"]["reachable"] is None
    assert fake.list_calls == 0          # health never calls the Gemini API
    assert fake.generate_calls == []
    assert "unit-test-key" not in repr(body)


def test_health_reports_the_verified_model_after_the_first_chat(cfg):
    fake = FakeGenAI(model_entries=[gemini_model_entry("gemini-2.5-flash")])
    client = make_real_llm_app(cfg, fake).test_client()

    chat = client.post("/api/chat", json={"message": "hello"})
    assert chat.status_code == 200

    body = client.get("/api/health").get_json()
    assert body["gemini"]["availability"] == "available"   # actually verified
    assert body["gemini"]["discovery"] == "ok"
    assert body["gemini"]["selected_model"] == "gemini-2.5-flash"
    assert body["gemini"]["model"] == "gemini-2.5-flash"
    assert fake.list_calls == 1
    assert "unit-test-key" not in repr(body)


def test_health_reports_model_discovery_auth_failure_without_blaming_the_model(cfg):
    fake = FakeGenAI(
        list_error=errors.APIError(
            401, {"error": {"message": "API key not valid: unit-test-key"}}
        )
    )
    client = make_real_llm_app(cfg, fake).test_client()

    chat = client.post("/api/chat", json={"message": "hello"})
    assert chat.status_code == 502
    assert chat.get_json()["code"] == "gemini_auth_error"

    body = client.get("/api/health").get_json()
    assert body["status"] == "degraded"
    assert body["gemini"]["configured"] is True
    assert body["gemini"]["discovery"] == "failed"
    assert body["gemini"]["availability"] == "error"
    assert "GEMINI_API_KEY" in body["gemini"]["hint"]
    assert "unavailable" not in (body["gemini"]["discovery_error"] or "").lower()
    assert "unit-test-key" not in repr(body)


def test_health_reports_model_discovery_failure_for_a_missing_usable_model(cfg):
    fake = FakeGenAI(model_entries=[])
    client = make_real_llm_app(cfg, fake).test_client()

    chat = client.post("/api/chat", json={"message": "hello"})
    assert chat.status_code == 503
    assert chat.get_json()["code"] == "gemini_no_model_available"

    body = client.get("/api/health").get_json()
    assert body["gemini"]["configured"] is True
    assert body["gemini"]["discovery"] == "ok"
    assert body["gemini"]["availability"] == "unavailable"
    assert body["gemini"]["hint"]
