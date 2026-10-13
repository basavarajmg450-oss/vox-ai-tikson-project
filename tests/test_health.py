"""GET /api/health - server and dependency reporting."""

from __future__ import annotations


def test_health_reports_all_dependencies(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["service"] == "voxai"
    assert body["status"] in {"ok", "degraded"}
    # Dependency blocks the assignment asks for:
    assert body["ollama"]["reachable"] is True
    assert body["ollama"]["model_present"] is True
    assert body["whisper"]["loaded"] is False  # lazy until first use
    assert body["piper"]["model_present"] is True
    assert "active_sessions" in body["sessions"]
    assert "max_history_turns" in body["config"]
    # Endpoint documentation is part of the contract:
    paths = {(e["method"], e["path"]) for e in body["endpoints"]}
    assert ("GET", "/api/health") in paths
    assert ("POST", "/api/voice") in paths
    assert ("DELETE", "/api/session/<session_id>") in paths


def test_health_degrades_when_ollama_is_down(client, svc):
    svc.llm.reachable = False
    resp = client.get("/api/health")
    body = resp.get_json()
    assert resp.status_code == 200  # liveness still answers
    assert body["status"] == "degraded"
    assert body["ollama"]["reachable"] is False
    assert "ollama serve" in (body["ollama"]["hint"] or "")


def test_health_degrades_when_model_missing(client, svc):
    svc.llm.model_present = False
    body = client.get("/api/health").get_json()
    assert body["status"] == "degraded"
    assert body["ollama"]["model_present"] is False


def test_unknown_api_route_is_json_404(client):
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404
    assert resp.get_json()["error"]
