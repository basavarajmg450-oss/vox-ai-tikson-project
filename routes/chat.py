"""``POST /api/chat`` - text turn against the local LLM.

The client sends only a message (and optionally its session id, temperature and
max_tokens).  Any ``history`` field a client includes is deliberately ignored:
conversation state lives exclusively on the backend.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from routes._common import coerce_session_id, services
from services.errors import bad_request

chat_bp = Blueprint("chat", __name__)


@chat_bp.post("/api/chat")
def chat():
    if not request.is_json:
        raise bad_request(
            "Send the message as JSON with Content-Type: application/json.",
            code="invalid_content_type",
        )
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise bad_request(
            "Request body must be a JSON object like "
            '{"message": "hello", "session_id": "<id>"}',
            code="invalid_json",
        )

    # Client-supplied history is ignored on purpose - see services/session_store.py.
    result = services().pipeline.chat(
        message=body.get("message"),
        session_id=coerce_session_id(body.get("session_id")),
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens"),
    )
    return jsonify({"ok": True, **result}), 200
