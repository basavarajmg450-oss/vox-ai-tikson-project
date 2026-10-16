# VoxAI — Architecture

Speech recognition and speech synthesis run locally. The Flask backend sends
chat prompts and bounded session history to Google's Gemini API; the browser
never receives the Gemini API key.

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
│           google-genai → runtime model discovery (models.list)        │
│           preferred GEMINI_MODEL honoured when available, otherwise   │
│           best discovered text-generation (Flash) model; cached       │
│           bounded history/output; key stays on the backend            │
│   ③ TTS   services/text_to_speech.py                                  │
│           Piper (en_US-ryan-medium, hi_IN-rohan-medium),               │
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
│  bubbles, live status pill, mic controls                              │
└────────────────────────────────────────────────────────────────────────┘
```

In one line, as the assignment asks:

> **Browser → local Faster-Whisper (STT) → Gemini API (LLM) → local Piper (TTS) → Browser audio**

## Request lifecycle (POST /api/voice)

1. **Validate** — body size (`MAX_AUDIO_BYTES`, 413 above limit), audio type
   allow-list (415 otherwise), session id format, numeric parameters.
2. **Stage ① STT** — the recording is decoded and transcribed locally;
   language is auto-detected. Silence or undecodable audio → `422 no_speech`
   with a human-readable message; nothing is committed to the session.
3. **Stage ② LLM** — the transcript is combined with *backend-held* recent
   history (the client can never supply history) and sent to Gemini with the
   voice system prompt, configured timeout, temperature and output-token cap.
   Any tagged reasoning text is stripped before the answer is displayed or
   spoken.
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

Nothing is estimated. These timings are returned in the API response's
`latency` object (and written by `scripts/benchmark.py`); the chat UI keeps
them out of sight, so no metrics are rendered next to conversation messages.

## Security notes

* `GET /api/audio/<filename>` accepts only `[0-9a-f]{32}.wav`-style names,
  then verifies the resolved path stays inside `var/audio/` (allow-list +
  containment check) — traversal attempts return 404.
* Uploads are bounded in size and MIME type; request bodies are validated and
  map to meaningful 4xx/5xx JSON errors (`error`, `code`, `hint`).
* API responses send `Cache-Control: no-store`; `X-Content-Type-Options:
  nosniff` is set globally.
* `GEMINI_API_KEY` is read only by the Flask backend, omitted from health
  responses and never embedded in frontend code. Prompts and recent turns are
  sent to Google's Gemini API over HTTPS.
* `/api/health` checks whether a key is configured but deliberately makes no
  Gemini API request. Model availability is therefore reported as
  `availability: "not_checked"` until model discovery has really run on a chat
  request; afterwards it reports `available` / `unavailable` / `error`
  honestly (`discovery`: `not_attempted` | `ok` | `failed`).

## Gemini model discovery and selection

Model selection lives entirely in `services/llm_service.py`:

1. **Discover** — the first chat request calls `client.models.list()` once
   (through the single configured `genai.Client`) and collects the models
   available to `GEMINI_API_KEY`.
2. **Filter** — only models whose metadata advertises `generateContent` are
   eligible; embedding-, image-, video- and audio-only models are excluded,
   as are preview/experimental/deprecated models when a stable alternative
   exists (they remain a last resort, or are used when pinned explicitly via
   `GEMINI_MODEL`). Models with missing capability metadata rank last.
3. **Select** — the configured `GEMINI_MODEL` is an *optional preference*: it
   wins when discovery finds it and it supports text generation. Otherwise
   the best stable model is chosen automatically (stable Flash first,
   alternating full Flash with Flash-Lite so a demand spike on one family
   cannot exhaust the bounded retries before the other is tried, then other
   Gemini models, newest first) — `gemini-2.5-flash` is never assumed to be
   accessible.
4. **Cache** — the ordered candidate list and the selected model are cached;
   `models.list()` is not repeated per message. Discovery is guarded by a
   lock, so concurrent requests never list or switch models repeatedly.
5. **Fall back** — if a generation request fails with model-not-found /
   unsupported-model, the model is remembered as gone for the process, the
   cache is invalidated (next message re-discovers without it) and the request
   tries the next suitable candidate. Transient overload answers (HTTP 5xx,
   "high demand") advance to the next candidate too, but keep the cache —
   model rejections (unavailable or quota-limited candidates) and overload
   rejections are each bounded by `GEMINI_MODEL_ATTEMPTS` (default 3), never
   endless. Authentication errors (401/403/400-key) stop immediately. A 429
   on one candidate — e.g. a per-model free-tier limit — is skipped within
   the same bounded budget so a model with remaining quota can answer; when
   quota is what blocks the answer, the request fails with 429
   `gemini_quota_exceeded` — reported as quota, never as "the model is
   unavailable".
6. **Report** — `/api/health` exposes `configured`, `discovery`,
   `availability`, `selected_model`, `reachable` and a sanitised
   `discovery_error`; the API key itself is never included anywhere.

## Concurrency

Flask runs `threaded=True`, so `/api/health` and static pages stay responsive
while a Gemini request is in flight. Local Whisper/Piper model-load locks are
held only during first-use warm-up; Gemini model discovery/selection is
guarded by its own lock (discover once, share the cached selection).
