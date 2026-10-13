"""VoxAI - a fully local voice assistant (Flask application factory).

Run it with::

    .venv\\Scripts\\python.exe app.py

Models are loaded lazily (or warmed up in background threads when run this
way), so ``/api/health`` answers immediately even while the first request is
still warming the engines.
"""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from typing import Optional

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from config import Config
from services import Services, build_services
from services.errors import ApiError

VERSION = "1.0.0"

logger = logging.getLogger("voxai")


def create_app(
    config: Optional[Config] = None,
    services: Optional[Services] = None,
    *,
    warmup: bool = False,
) -> Flask:
    """Build the Flask app.

    ``services`` may be replaced with fakes in tests; ``warmup=True`` starts
    background threads that load the Whisper and Piper models right away.
    """
    cfg = config or Config.from_env()
    svc = services or build_services(cfg)
    issues = cfg.startup_issues()

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_audio_bytes + 512 * 1024
    app.config["JSON_AS_ASCII"] = False
    app.extensions["voxai"] = SimpleNamespace(
        config=cfg,
        services=svc,
        started_at=time.time(),
        issues=issues,
        version=VERSION,
    )

    from routes import chat_bp, health_bp, sessions_bp, voice_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(voice_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(sessions_bp)

    @app.get("/")
    def index():
        return render_template("index.html", version=VERSION, config=cfg.public())

    # ------------------------------------------------------------------ errors
    @app.errorhandler(ApiError)
    def _handle_api_error(exc: ApiError):
        return jsonify(exc.to_dict()), exc.status_code

    @app.errorhandler(HTTPException)
    def _handle_http_error(exc: HTTPException):
        if request.path.startswith("/api/"):
            code = (exc.name or "error").lower().replace(" ", "_")
            return jsonify({"error": exc.description, "code": code}), exc.code
        return exc.get_response(request.environ)

    @app.errorhandler(Exception)
    def _handle_unexpected(exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.path)
        body = {"error": "Internal server error.", "code": "internal_error"}
        if app.debug:
            body["detail"] = f"{type(exc).__name__}: {exc}"
        return jsonify(body), 500

    @app.after_request
    def _harden(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        if request.path.startswith("/api/") and not request.path.startswith("/api/audio/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    # ------------------------------------------------------------------ startup
    errors = [issue for issue in issues if issue["level"] == "error"]
    if cfg.strict_startup and errors:
        lines = "\n".join(f"  - {issue['message']}\n    fix: {issue['fix']}" for issue in errors)
        raise SystemExit(f"STRICT_STARTUP=1 and required models are missing:\n{lines}")

    if warmup:
        start_warmup(svc)

    return app


def start_warmup(svc: Services) -> None:
    """Load the Whisper and Piper models in background threads."""

    def _warm(component: str, loader) -> None:
        started = time.perf_counter()
        try:
            loader()
            logger.info("Warm-up finished: %s loaded in %.1fs", component, time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - warm-up failure is reported by /api/health
            logger.warning("Warm-up failed for %s: %s", component, exc)

    threading.Thread(target=_warm, args=("faster-whisper", svc.stt.load), name="warmup-stt", daemon=True).start()
    threading.Thread(target=_warm, args=("piper voice", svc.tts.load), name="warmup-tts", daemon=True).start()


def print_startup_report(app: Flask) -> None:
    """Print an actionable startup banner (model issues, fixes, next steps)."""
    context = app.extensions["voxai"]
    cfg = context.config
    print(f"\n  VoxAI {context.version} - local voice assistant")
    print(f"  http://{cfg.host}:{cfg.port}  (LLM: {cfg.ollama_model} @ {cfg.ollama_host})")
    if context.issues:
        print("\n  Startup issues:")
        for issue in context.issues:
            marker = "ERROR" if issue["level"] == "error" else "warn "
            print(f"    [{marker}] {issue['message']}")
            print(f"            fix: {issue['fix']}")
    else:
        print("  All required model files found.")
    print()


def run_app() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    application = create_app(warmup=True)
    print_startup_report(application)
    cfg = application.extensions["voxai"].config
    logger.info("Starting Flask (threaded=True) on http://%s:%s", cfg.host, cfg.port)
    # threaded=True keeps /api/health and other requests responsive while a
    # chat/transcription request is in flight.
    application.run(host=cfg.host, port=cfg.port, debug=False, threaded=True, use_reloader=False)


#: Default WSGI object for `flask --app app run` / gunicorn-style servers.
app = create_app()


if __name__ == "__main__":
    run_app()
