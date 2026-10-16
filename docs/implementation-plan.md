# VoxAI — implementation plan

Original plan written before implementation. It predates the current Gemini
integration; the current runtime uses local Faster-Whisper and Piper with
Google Gemini for chat.

## Goal

A browser voice assistant with local speech stages and an API-backed LLM:
**STT (faster-whisper)** → **LLM (Google Gemini API)** → **TTS (Piper)**, with
authoritative session history on the Flask backend.

## Order of work

1. **Environment** — clean venv, pinned-but-flexible `requirements.txt`, record
   the actual installed versions before writing code against them.
2. **Config** (`config.py`) — every setting from an env var with a safe default,
   plus `validate_startup()` that produces *actionable* messages (exact
   commands to run) when a required model file is missing.
3. **Services layer** (`services/`) — one module per AI stage, each with:
   lazy/one-time loading, thread-safe init, measurable latency, and its own
   typed exception carrying an HTTP status + human message.
   - `speech_to_text.py` — faster-whisper wrapper (bytes in → text + language),
     auto language detection (en/hi), silence/short/undecodable audio handled.
   - `llm_service.py` — official `google-genai` client, backend-only
     `GEMINI_API_KEY`, configurable model/timeout, history bounded by
     `MAX_HISTORY_TURNS`, output bounded by `MAX_RESPONSE_TOKENS`, concise
     spoken-style answers and safe errors for credentials/quota/model/timeout.
   - `text_to_speech.py` — Piper `PiperVoice` loaded **once** (module singleton,
     warm-up in a background thread), WAV written into a dedicated temp dir,
     TTL sweep for old files, optional Hindi voice when detected language is
     Hindi, failure never loses the text response.
   - `voice_pipeline.py` — orchestrates the three stages, measures each stage
     with `time.perf_counter()`, keeps timestamps labelled *measured* vs
     *estimated*, and never blocks unrelated Flask requests (only the model-load
     locks are held during init).
   - `session_store.py` — backend-authoritative history keyed by session id,
     strict isolation, TTL, bounded history. Client-supplied history is never
     read.
4. **Routes** (`routes/`) — health, voice (transcribe/voice/audio), chat,
   sessions; every body validated (size, type, required fields) with meaningful
   4xx codes and typed errors mapped to 4xx/5xx JSON.
5. **Frontend** (`templates/index.html`, `static/css/style.css`,
   `static/js/app.js`) — dark, accessible, MediaRecorder capture, status state
   machine (Ready → Listening → Transcribing → Thinking → Speaking / Error),    bubbles, `<audio>` playback with autoplay, session controls,
   settings.
6. **Tests** — `tests/` with mocked services for unit tests (no model download
   required) plus `-m integration` tests that hit Gemini and the real local
   speech services.
7. **Verification** — run unit tests, start the server, hit every endpoint with
   `curl`, run the integration suite, exercise the UI in a browser, then write
   README + docs (architecture diagram, demo checklist, multi-turn example,
   limitations) with **only measured** numbers.

## Key decisions

- Flask runs with `threaded=True` so `/api/health` stays responsive while a
  chat request is in flight (verified with a concurrency check).
- `POST /api/voice` returns plain JSON by default (contract) and accepts an
  optional `stream=progress` flag that emits honest stage events before the
  final JSON so the UI status labels are real, not faked timers.
- Generated WAVs live in `var/audio/` and are served only through
  `GET /api/audio/<filename>` with an allow-list regex + `realpath` containment
  check; filenames are UUID-derived so they cannot collide with user input.
