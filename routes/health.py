"""``GET /api/health`` - server and dependency status.

Always answers ``200`` when the Flask process itself is alive (liveness plus a
detailed dependency report); the JSON body's ``status`` field says whether the
AI dependencies are ready.  Ollama is pinged at most once every few seconds so
the endpoint stays cheap for polling frontends.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from flask import Blueprint, jsonify

from routes._common import config, ctx, services

health_bp = Blueprint("health", __name__)

#: Cache window for the (relatively slow) Ollama ping.
_OLLAMA_CACHE_SECONDS = 3.0
_ollama_cache = {"at": 0.0, "payload": None, "owner": None}
_ollama_lock = threading.Lock()


def _cached_ollama_status():
    now = time.time()
    # Keep a strong reference to the owning LLM service: a fresh app (tests)
    # gets a fresh cache, and the referenced object cannot be garbage-collected
    # and have its id() reused while cached.
    owner = services().llm
    with _ollama_lock:
        cached = _ollama_cache["payload"]
        if (
            cached is not None
            and _ollama_cache["owner"] == owner
            and now - _ollama_cache["at"] < _OLLAMA_CACHE_SECONDS
        ):
            return dict(cached), False
        payload = services().llm.status()
        _ollama_cache["payload"] = payload
        _ollama_cache["at"] = time.time()
        _ollama_cache["owner"] = owner
        return dict(payload), True


@health_bp.get("/api/health")
def health():
    started = time.perf_counter()
    cfg = config()
    svc = services()
    context = ctx()

    ollama_status, _fresh = _cached_ollama_status()
    whisper_status = svc.stt.status()
    piper_status = svc.tts.status()

    problems = [issue for issue in context.issues if issue["level"] == "error"]
    if not ollama_status["reachable"]:
        status = "degraded"
    elif not ollama_status["model_present"]:
        status = "degraded"
    elif problems:
        status = "degraded"
    else:
        status = "ok"

    body = {
        "status": status,
        "service": "voxai",
        "version": context.version,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": round(time.time() - context.started_at, 1),
        "ollama": ollama_status,
        "whisper": whisper_status,
        "piper": piper_status,
        "sessions": svc.sessions.stats(),
        "config": cfg.public(),
        "startup_issues": context.issues,
        "endpoints": [
            {"method": "GET", "path": "/api/health", "purpose": "server + dependency status"},
            {"method": "POST", "path": "/api/transcribe", "purpose": "audio -> transcript + language"},
            {"method": "POST", "path": "/api/chat", "purpose": "text -> LLM answer (session history kept server-side)"},
            {"method": "POST", "path": "/api/voice", "purpose": "audio -> transcript + answer + WAV audio"},
            {"method": "POST", "path": "/api/session/new", "purpose": "create an isolated session"},
            {"method": "GET", "path": "/api/session/<session_id>", "purpose": "session info (turn count, expiry)"},
            {"method": "GET", "path": "/api/session/<session_id>/history", "purpose": "this session's server-side history"},
            {"method": "DELETE", "path": "/api/session/<session_id>", "purpose": "clear a session's history"},
            {"method": "GET", "path": "/api/audio/<filename>", "purpose": "serve a generated WAV (sanitised names only)"},
        ],
        "checked_in_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }
    return jsonify(body), 200
