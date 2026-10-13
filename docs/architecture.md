# VoxAI — Architecture

Everything in VoxAI runs on the machine it is used on. There is no hosted
inference: the browser records, the Flask backend transcribes, reasons and
speaks using local models, and the browser plays the result back.

## High-level flow

```
┌────────────────────────────  Your browser  ────────────────────────────┐
│  templates/index.html  ·  static/css/style.css  ·  static/js/app.js   │
│                                                                        │
│  MediaRecorder ──► recorded WebM/Opus blob (start / stop controls)     │
│       │                                                                │
│       │  POST /api/voice  (multipart: audio, session_id, timings)      │
│       ▼                                                                │
└───────┼────────────────────────────────────────────────────────────────┘
        │  localhost HTTP (Flask, threaded=True)
        ▼
┌────────────────────────────  Flask backend  ───────────────────────────┐
│  app.py (application factory, error mapping, hardening headers)        │
│  routes/   health · voice · chat · sessions   (validation + status)    │
│  services/voice_pipeline.py  — orchestrates the three stages below     │
│                                                                        │
│   ① STT   services/speech_to_text.py                                  │
│           faster-whisper (base, CPU int8) — auto language detect      │
│               en / hi supported                                       │
│   ② LLM   services/llm_service.py                                     │
│           Ollama HTTP → qwen3:4b, reasoning excluded from the answer   │
│           history bounded by MAX_HISTORY_TURNS,                        │
│           answer bounded by MAX_RESPONSE_TOKENS                        │
│   ③ TTS   services/text_to_speech.py                                  │
│           Piper (en_US-lessac-medium, hi_IN-rohan-medium),             │
│           voice loaded once, WAV written to var/audio/                 │
│                                                                        │
│  services/session_store.py — authoritative history keyed by session id │
└───────┼────────────────────────────────────────────────────────────────┘
        │
        │  JSON (transcript, answer, audio_url, measured metrics)
        │  +  WAV bytes via  GET /api/audio/<filename>
        ▼
┌────────────────────────────  Your browser  ────────────────────────────┐
│  <audio controls> plays the generated WAV (autoplay optional)          │
│  bubbles, live status pill, latency panel                              │
└────────────────────────────────────────────────────────────────────────┘
```

In one line, as the assignment asks:

> **Browser → Faster-Whisper (STT) → Ollama / Qwen3 (LLM) → Piper (TTS) → Browser audio**

## Request lifecycle (POST /api/voice)

1. **Validate** — body size (`MAX_AUDIO_BYTES`, 413 above limit), audio type
   allow-list (415 otherwise), session id format, numeric parameters.
2. **Stage ① STT** — the recording is decoded and transcribed locally;
   language is auto-detected. Silence or undecodable audio → `422 no_speech`
   with a human-readable message; nothing is committed to the session.
3. **Stage ② LLM** — the transcript is appended to *backend-held* history
   (the client can never supply history) and sent to Ollama with the voice
   system prompt. Reasoning is kept out of the user-facing answer: the
   `think` flag stays at the model default so Ollama puts any chain-of-thought
   into `message.thinking` (never read by VoxAI) while `content` carries only
   the answer. This was chosen because on the qwen3:4b build used here
   `think=False` actually makes the model write its reasoning *into*
   `content`; tagged `</think>` leaks are still stripped as a second line of
   defence (`strip_thinking`).
4. **Stage ③ TTS** — the answer is synthesised to a WAV inside `var/audio/`.
   If Piper fails, the response still returns with `audio_url: null` and a
   `tts_error` — the text answer is never lost.
5. **Commit + respond** — only after a successful pipeline are the user and
   assistant messages appended to the session. The response carries per-stage
   timings.

`POST /api/voice?stream=progress` emits Server-Sent Events (`progress`
with `stage: stt|llm|tts`, then `done` or `error`) so the UI's status labels
are *real stage transitions*, not fake timers. The pipeline itself is
**buffered**: the whole recording is uploaded and processed before the answer
begins — the app never claims to stream.

## Sessions

* Session ids are `secrets.token_hex(16)` (32 hex chars), validated by regex
  before touching any store.
* History lives **only** on the backend (`SessionStore`, in memory, TTL
  `SESSION_TTL_SECONDS`). The browser holds just the id.
* Histories are strictly isolated per id and bounded to
  `MAX_HISTORY_TURNS` recent turns (trimmed on a turn boundary).
* `POST /api/session/new` creates one; `DELETE /api/session/<session_id>`
  empties one; forged `history` fields in requests are ignored.

## Latency reporting

Every stage is timed with `time.perf_counter()` on the server and labelled:

| Field | Source label | Meaning |
|---|---|---|
| `recording_ms` | `measured_client` | how long the user recorded |
| `stt_ms` / `llm_ms` / `tts_ms` | `measured_server` | per-stage wall time |
| `model_load_ms` | `measured_server` | one-off warm-up on first call |
| `backend_total_ms` | `measured_server` | whole request inside Flask |
| `total_ms` | `measured_client_plus_server` | recording + backend |

Nothing is estimated; the UI's latency panel marks each row with its source
and states `streaming: no`.

## Security notes

* `GET /api/audio/<filename>` accepts only `[0-9a-f]{32}.wav`-style names,
  then verifies the resolved path stays inside `var/audio/` (allow-list +
  containment check) — traversal attempts return 404.
* Uploads are bounded in size and MIME type; request bodies are validated and
  map to meaningful 4xx/5xx JSON errors (`error`, `code`, `hint`).
* API responses send `Cache-Control: no-store`; `X-Content-Type-Options:
  nosniff` is set globally.
* No keys leave the machine; the app needs no network beyond localhost.

## Concurrency

Flask runs `threaded=True`, so `/api/health` and static pages stay responsive
while a model request is in flight. Only the per-service model-load locks are
held during first-use warm-up; inference itself does not block the server's
other request threads beyond normal CPU contention.
