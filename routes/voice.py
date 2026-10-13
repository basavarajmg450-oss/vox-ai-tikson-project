"""``POST /api/transcribe``, ``POST /api/voice`` and ``GET /api/audio/<filename>``.

``/api/voice`` accepts an optional ``stream=progress`` form flag: instead of a
single JSON body it then emits honest stage events (``progress`` /
``done`` / ``error``) over Server-Sent Events before the final payload.  The
pipeline itself is still buffered - the whole recording is uploaded and
processed before the answer is produced.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Dict

from flask import Blueprint, Response, current_app, jsonify, render_template, send_file, stream_with_context

from routes._common import coerce_session_id, config, ctx, get_uploaded_audio, services
from services.errors import ApiError

voice_bp = Blueprint("voice", __name__)

STREAM_STAGES = {"stt", "llm", "tts", "done"}


@voice_bp.post("/api/transcribe")
def transcribe():
    """Audio in -> transcript + detected language (no history is touched)."""
    audio, ctype, _filename = get_uploaded_audio()
    result = services().pipeline.transcribe(audio, content_type=ctype)
    return jsonify({"ok": True, **result}), 200


def _sse(payload: Dict[str, Any], event: str) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _run_voice_stream(audio, ctype, session_id, recording_ms, temperature, max_tokens) -> Response:
    """Run the pipeline in a worker thread and stream progress events."""
    events: "queue.Queue[tuple]" = queue.Queue()
    # The worker thread has no request (or app) context of its own; capture the
    # app object here so services() works inside the thread.
    app = current_app._get_current_object()

    def worker() -> None:
        app_ctx = app.app_context()
        app_ctx.push()
        try:
            result = services().pipeline.voice(
                audio,
                content_type=ctype,
                session_id=session_id,
                recording_ms=recording_ms,
                temperature=temperature,
                max_tokens=max_tokens,
                progress=lambda payload: events.put(("progress", payload)),
            )
            events.put(("done", result))
        except ApiError as exc:
            events.put(("error", exc))
        except Exception as exc:  # noqa: BLE001 - never kill the stream silently
            events.put((
                "error",
                ApiError(f"Unexpected server error: {type(exc).__name__}: {exc}", 500, "internal_error"),
            ))
        finally:
            try:
                app_ctx.pop()
            except Exception:  # noqa: BLE001 - context may already be gone
                pass

    threading.Thread(target=worker, name="voice-pipeline", daemon=True).start()

    def generate():
        while True:
            kind, payload = events.get()
            if kind == "progress":
                yield _sse(payload, "progress")
            elif kind == "done":
                yield _sse({"ok": True, **payload}, "done")
                return
            elif kind == "error":
                yield _sse(payload.to_dict(), "error")
                return

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@voice_bp.post("/api/voice")
def voice():
    """Full pipeline: recording in -> transcript, answer, WAV URL, metrics."""
    audio, ctype, _filename = get_uploaded_audio()
    session_id = coerce_session_id(request_form("session_id"))
    recording_ms = request_form("recording_ms", cast=float)
    temperature = request_form("temperature")
    max_tokens = request_form("max_tokens")
    stream = (request_form("stream") or "").lower() in {"progress", "1", "true", "yes"}

    if stream:
        return _run_voice_stream(audio, ctype, session_id, recording_ms, temperature, max_tokens)

    result = services().pipeline.voice(
        audio,
        content_type=ctype,
        session_id=session_id,
        recording_ms=recording_ms,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return jsonify({"ok": True, **result}), 200


def request_form(name: str, cast=None):
    """Read an optional form field, tolerating malformed values."""
    from flask import request

    raw = request.form.get(name)
    if raw is None:
        return None
    raw = str(raw).strip()
    if raw == "":
        return None
    if cast is float:
        try:
            return float(raw)
        except ValueError:
            raise ApiError(
                f"Form field '{name}' must be a number, got {raw!r}.",
                400,
                "invalid_parameter",
            ) from None
    return raw


@voice_bp.get("/api/audio/<filename>")
def serve_audio(filename: str):
    """Serve a generated WAV. Only sanitised names inside var/audio work."""
    path = services().tts.resolve_audio(filename)
    return send_file(
        path,
        mimetype="audio/wav",
        download_name=filename,
        conditional=True,
        max_age=3600,
        etag=True,
        last_modified=True,
    )
