"""``GET /api/health`` - server and dependency status.

Always answers ``200`` when the Flask process itself is alive (liveness plus a
detailed dependency report). Gemini configuration is reported without making
any API request; model discovery runs on the first chat request, so until a
model has actually been verified ``availability`` stays ``not_checked``.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from flask import Blueprint, jsonify

from routes._common import config, ctx, services

health_bp = Blueprint("health", __name__)

@health_bp.get("/api/health")
def health():
    started = time.perf_counter()
    cfg = config()
    svc = services()
    context = ctx()

    gemini_status = svc.llm.status()
    whisper_status = svc.stt.status()
    piper_status = svc.tts.status()

    problems = [issue for issue in context.issues if issue["level"] == "error"]
    gemini_availability = gemini_status.get("availability")
    if not gemini_status["configured"]:
        status = "degraded"
    elif gemini_availability in {"error", "unavailable"}:
        # Discovery ran and verified a problem (auth, quota, network, or no
        # usable model) - report it instead of claiming everything is fine.
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
        "gemini": gemini_status,
        "whisper": whisper_status,
        "piper": piper_status,
        "web_search": svc.web_search.status() if svc.web_search else {"enabled": False},
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
