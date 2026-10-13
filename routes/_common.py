"""Shared helpers for the route blueprints."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Tuple

from flask import current_app, request

from config import ALLOWED_AUDIO_EXTENSIONS, ALLOWED_AUDIO_TYPES, Config
from services import Services
from services.errors import ApiError, bad_request
from services.session_store import SessionStore

#: Form / multipart field names accepted for the audio payload.
AUDIO_FIELD_NAMES = ("audio", "file", "audio_file")


def ctx() -> SimpleNamespace:
    """The per-app namespace holding config + services + metadata."""
    return current_app.extensions["voxai"]


def config() -> Config:
    return ctx().config


def services() -> Services:
    return ctx().services


def coerce_session_id(raw: Optional[str]) -> Optional[str]:
    """Normalise an optional session id from a form field or JSON body.

    Empty/absent means "create a fresh session"; anything malformed is a 400.
    """
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"", "null", "undefined", "none"}:
        return None
    return SessionStore.validate_id(value)


def get_uploaded_audio() -> Tuple[bytes, str, str]:
    """Extract, validate and read the uploaded recording.

    Returns ``(bytes, content_type, filename)``.  Raises :class:`ApiError`
    with 400 (missing/empty), 413 (too large) or 415 (unsupported type).
    """
    cfg = config()
    max_total = cfg.max_audio_bytes + 512 * 1024  # multipart overhead allowance
    if request.content_length and request.content_length > max_total:
        raise ApiError(
            f"Upload too large: {request.content_length} bytes "
            f"(limit {cfg.max_audio_bytes} bytes of audio).",
            413,
            "audio_too_large",
            hint="Record shorter clips or raise MAX_AUDIO_BYTES in .env.",
        )

    if not request.files:
        raise bad_request(
            "No audio received. Upload the recording as multipart form field 'audio'.",
            code="missing_audio",
            hint="curl -F audio=@recording.webm http://localhost:5000/api/transcribe",
        )

    storage = None
    for name in AUDIO_FIELD_NAMES:
        if name in request.files:
            storage = request.files[name]
            break
    if storage is None:
        raise bad_request(
            f"No audio received. Expected form field 'audio' (also accepted: {', '.join(AUDIO_FIELD_NAMES[1:])}).",
            code="missing_audio",
        )
    if not (storage.filename or "").strip():
        raise bad_request("The audio part has no filename.", code="missing_audio")

    data = storage.read(cfg.max_audio_bytes + 1)
    if len(data) > cfg.max_audio_bytes:
        raise ApiError(
            f"Recording too large: more than {cfg.max_audio_bytes} bytes.",
            413,
            "audio_too_large",
            hint="Record shorter clips or raise MAX_AUDIO_BYTES in .env.",
        )
    if not data:
        raise ApiError("The uploaded recording is empty (0 bytes).", 400, "empty_audio")

    ctype = (storage.content_type or "").split(";")[0].strip().lower()
    ext = Path(storage.filename).suffix.lower()
    if ctype not in ALLOWED_AUDIO_TYPES and ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise ApiError(
            f"Unsupported audio type '{storage.content_type or 'unknown'}' "
            f"(file '{storage.filename}').",
            415,
            "unsupported_audio_type",
            hint="Allowed types: " + ", ".join(sorted(t for t in ALLOWED_AUDIO_TYPES if t != "application/octet-stream")),
        )
    return data, ctype, storage.filename
