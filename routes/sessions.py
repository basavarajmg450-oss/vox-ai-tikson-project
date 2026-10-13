"""Session endpoints: create, inspect, read history and clear.

History is authoritative on the backend; the browser only ever holds the
session id.
"""

from __future__ import annotations

from flask import Blueprint, jsonify

from routes._common import config, services
from services.errors import not_found
from services.session_store import SessionStore

sessions_bp = Blueprint("sessions", __name__)


@sessions_bp.post("/api/session/new")
def new_session():
    session = services().sessions.create()
    return jsonify({
        "ok": True,
        "session_id": session.session_id,
        "created_at": session.created_at,
        "ttl_seconds": config().session_ttl_seconds,
    }), 201


@sessions_bp.get("/api/session/<session_id>")
def session_info(session_id: str):
    sid = SessionStore.validate_id(session_id)
    try:
        info = services().sessions.info(sid)
    except Exception:
        raise not_found("Unknown session id.", code="unknown_session",
                        hint="Create one with POST /api/session/new.") from None
    return jsonify({"ok": True, **info}), 200


@sessions_bp.get("/api/session/<session_id>/history")
def session_history(session_id: str):
    sid = SessionStore.validate_id(session_id)
    try:
        history = services().sessions.history(sid)
    except Exception:
        raise not_found("Unknown session id.", code="unknown_session",
                        hint="Create one with POST /api/session/new.") from None
    return jsonify({
        "ok": True,
        "session_id": sid,
        "turns": sum(1 for m in history if m["role"] == "user"),
        "history": history,
    }), 200


@sessions_bp.delete("/api/session/<session_id>")
def clear_session(session_id: str):
    sid = SessionStore.validate_id(session_id)
    try:
        cleared = services().sessions.clear(sid)
    except Exception:
        raise not_found("Unknown session id.", code="unknown_session",
                        hint="Create one with POST /api/session/new.") from None
    return jsonify({"ok": True, "session_id": sid, "cleared": True, **cleared}), 200
