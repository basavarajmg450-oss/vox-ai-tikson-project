# VoxAI — local voice, Gemini-powered answers

Speak a question in the browser → hear the recognized text → get an answer from a
Gemini Flash model → hear it spoken back. **Faster-Whisper and Piper run locally;
chat prompts are sent from the Flask backend to Google's Gemini API.**

```
Browser (MediaRecorder) → Faster-Whisper (local STT) → Gemini API (LLM) → Piper (local TTS) → Browser audio
```

![status](https://img.shields.io/badge/STT-faster--whisper-blue) ![status](https://img.shields.io/badge/LLM-Gemini%20Flash-blue) ![status](https://img.shields.io/badge/TTS-Piper-blue)

---

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Setup (exact PowerShell commands)](#setup-exact-powershell-commands)
- [Configuration](#configuration)
- [API endpoints](#api-endpoints)
- [Tests — exact commands and results](#tests--exact-commands-and-results)
- [Measured latency](#measured-latency)
- [Example multi-turn conversation (memory)](#example-multi-turn-conversation-memory)
- [Demo recording checklist (2–3 min)](#demo-recording-checklist-23-min)
- [Models, licenses and downloads](#models-licenses-and-downloads)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Future improvements](#future-improvements)

---

## Features

**Voice pipeline**
- Record in the browser with `MediaRecorder` (start/stop controls, mic button
  with recording animation, live timer).
- Transcribe with **faster-whisper** (`base`, CPU `int8`), auto language
  detection — English and Hindi supported; silence / empty / invalid audio
  produce clear errors, not crashes.
- Answer with **Google Gemini** using the official `google-genai` Python SDK:
  - **automatic model discovery**: the models available to your API key are
    queried at runtime (`models.list()`) and the best suitable text-generation
    model is selected — no manual `GEMINI_MODEL` editing when a model is
    unavailable,
  - `GEMINI_MODEL` remains an optional *preferred* model (honoured only when
    discovery finds it),
  - a voice-assistant system prompt (concise, spoken-style answers),
  - recent-turn history bounded by `MAX_HISTORY_TURNS` (server-side only),
  - answer length bounded by `MAX_RESPONSE_TOKENS`,
  - configurable temperature (settings slider),
  - API key and model configured only on the Flask backend,
  - clear messages for missing/invalid credentials, quota, model discovery
    failures, empty response and timeout.
- Speak the answer with **Piper** — one fixed, single-speaker voice per
  language: `en_US-ryan-medium` (English) and `hi_IN-rohan-medium`
  (Hindi), both **male voices**, so the speaker identity and gender stay
  consistent across responses and languages. Voices are loaded **once** and
  reused; WAV output, temp files swept by TTL. Language detection only picks
  the language model — never a different speaker — and a missing voice fails
  with a clear error instead of silently switching. If TTS fails, the text
  answer still shows with a visible warning.

**Live web search for current information**
- Optional, configurable web search using **DuckDuckGo** (no paid hosted AI inference API required).
- **Intelligent heuristics**: queries regarding current events, latest news, recent releases, weather, prices, or recent dates automatically trigger retrieval; ordinary questions use the local LLM directly without internet overhead.
- **Grounded answers & source links**: retrieved search snippets and source URLs are passed to the model prompt, and clickable source links are displayed in the chat interface.
- **Safeguards**: HTML stripping, length truncation, and removal of suspicious prompt injection directives (e.g. `ignore previous instructions`, `system: override`).
- **Honest error handling**: if search fails or the internet is unreachable, VoxAI tells the user that current web information could not be retrieved and does not falsely claim to have searched the web.

**Sessions**
- History is authoritative on the Flask backend, keyed by an unguessable
  32-hex session id; strictly isolated; client-sent `history` is ignored.
- New conversation / clear history actions; TTL expiry.

**Interface** (vanilla HTML/CSS/JS, dark theme)
- VoxAI branding, accessible contrast, subtle animations only.
- Status pill with real stage transitions: **Ready → Listening → Processing →
  Thinking → Speaking → Error** (fed by genuine SSE stage events, not timers).
- User/assistant bubbles, transcript + response text, `<audio>` playback
  controls with optional autoplay, settings section,
  permission/error messages, health chip.

**Honesty by design**
- The pipeline is **buffered**, not streaming: the whole recording is uploaded
  and processed before the answer starts — the UI says so, and the API says
  `"streaming": false`.
- Every latency number is measured with `time.perf_counter()` and labelled
  with its source (`measured_server`, `measured_client`,
  `measured_client_plus_server`). Nothing is estimated. The timings stay in
  the API response (`latency` object) and in `scripts/benchmark.py` reports
  for analysis — the chat UI deliberately does not display them.
- `/api/health` reports whether Gemini configuration is present without making
  an API request; model availability stays `not_checked` until model discovery
  has actually run on a real chat request, then reports `available`,
  `unavailable` or `error` honestly.
- The Gemini API key is never sent to the browser or included in API responses,
  logs, health responses or error messages.

---

## Architecture

Full write-up with the diagram and request lifecycle: **[docs/architecture.md](docs/architecture.md)**.

```
┌────────────────────────────  Your browser  ────────────────────────────┐
│  index.html · style.css · app.js                                       │
│  MediaRecorder ──► WebM/Opus blob   (Start / Stop / mic button)        │
│       │  POST /api/voice  (multipart: audio + session_id + timings)    │
└───────┼────────────────────────────────────────────────────────────────┘
        ▼
┌────────────────────────────  Flask backend  ───────────────────────────┐
│  routes/: health · voice · chat · sessions     (validation + errors)   │
│  services/voice_pipeline.py  orchestrates:                              │
│    ① STT  faster-whisper (CPU int8, auto language: en/hi)              │
│    ② LLM  Google Gemini API (recent history and output bounded)         │
│    ③ TTS  Piper → WAV in var/audio/ (loaded once)                      │
│  services/session_store.py  authoritative, isolated, TTL-bounded       │
└───────┼────────────────────────────────────────────────────────────────┘
        ▼  JSON (transcript, answer, audio_url, measured metrics)
           +  WAV bytes via GET /api/audio/<filename>
┌────────────────────────────  Your browser  ────────────────────────────┐
│  <audio controls> plays the WAV · bubbles · status · mic controls      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Requirements

**Software**
- Windows 11 (commands below are PowerShell; Linux/macOS work with `.venv/bin/…`)
- Python 3.11+ (3.11-compatible codebase)
- A Google AI Studio API key with access to a Gemini Flash model
- A modern browser with `MediaRecorder` + `getUserMedia` (Chrome/Edge/Firefox);
  microphone access over `http://localhost` (browsers allow it without HTTPS)

**Hardware (measured on the machine this repo was built on)**
- CPU: Intel Core 5-class (integrated graphics only — **no CUDA GPU needed**)
- RAM: 4 GB recommended for local Whisper/Piper models
- Disk: roughly 300 MB for Whisper base and the two Piper voices; Gemini runs remotely
- No GPU required for the local speech stages.

---

## Setup (exact PowerShell commands)

```powershell
# 1) Python environment + dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2) Create .env only if it does not already exist, then edit it.
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
#    Create a key at https://aistudio.google.com/app/apikey and set
#    GEMINI_API_KEY in .env (never put it in frontend code).

# 3) Whisper + Piper assets (once) — downloads into ./models
python scripts/prefetch_assets.py

# 4) Run the app
python app.py
#    → open http://localhost:5000

# 5) Run the tests
python -m pytest -m "not integration" -q                 # unit tests (mocked AI)
$env:VOXAI_INTEGRATION=1; python -m pytest -m integration -v   # Gemini + local Whisper/Piper
```

Without activating the venv, use `.\.venv\Scripts\python.exe` in place of `python`.
Start from the project directory. `.env` is ignored by Git; do not commit the API
key. For local development, the exact start command after setup is
`.\.venv\Scripts\python.exe app.py`, then open `http://localhost:5000`.

Whisper and Piper load lazily (and are warmed in background threads when started
with `python app.py`). Gemini latency depends on the network, model, and API
quota; measure it through the API's `latency` payload or `scripts/benchmark.py`
rather than assuming a speed improvement.

### Getting and configuring a Gemini API key

1. Sign in to [Google AI Studio](https://aistudio.google.com/app/apikey) and
   create an API key for a project that has Gemini API access.
2. Copy `.env.example` to `.env` if one does not exist, then set
   `GEMINI_API_KEY`. Set `GEMINI_MODEL` only if you want to pin a specific
   model — otherwise VoxAI discovers the best model your key can use
   automatically (see [Automatic Gemini model discovery](#automatic-gemini-model-discovery)).
3. Keep `.env` private. The browser only calls Flask endpoints; the key is read
   by the server and is never returned by `/api/health`.

---

## Configuration

All settings are environment variables (`.env` is loaded automatically). See
[.env.example](.env.example) for the full annotated list.

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(empty)* | Google AI Studio key; required for chat, kept server-side |
| `GEMINI_MODEL` | `gemini-2.5-flash` | *Optional preferred* Gemini model; used only when discovery finds it — leave empty to always auto-select |
| `GEMINI_MODEL_ATTEMPTS` | `3` | Bounded retries: rejected models and transient overload answers each try at most this many candidates (1–10) |
| `GEMINI_TIMEOUT_S` | `60` | One-request timeout → clear 504 message |
| `WHISPER_MODEL` | `base` | STT size: tiny/base/small/medium/large-v3 |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | `cpu` / `int8` | Inference device |
| `PIPER_MODEL_PATH` | `./models/voices/en_US-ryan-medium.onnx` | English voice (male) |
| `PIPER_MODEL_PATH_HI` | `./models/voices/hi_IN-rohan-medium.onnx` | Hindi voice (male, same gender as the English voice) |
| `WHISPER_BEAM_SIZE` | `1` | STT decoding width: 1 = greedy (~2× faster on CPU), 2–10 = beam search |
| `GEMINI_THINKING_BUDGET` | `0` | Hidden reasoning tokens: 0 = off (fast short answers), -1 = provider default |
| `MAX_HISTORY_TURNS` | `8` | Recent turns sent to the model |
| `MAX_RESPONSE_TOKENS` | `200` | Bound on the visible answer |
| `LLM_TEMPERATURE` | `0.6` | Sampling temperature |
| `SESSION_TTL_SECONDS` | `86400` | Idle session lifetime |
| `MAX_AUDIO_BYTES` | `15728640` | Upload limit (413 above it) |
| `AUDIO_DIR` | `./var/audio` | Generated WAVs (served only from here) |
| `STRICT_STARTUP` | `0` | `1` = abort boot if a model file is missing |
| `WEB_SEARCH_ENABLED` | `1` | `1` = enable live web search for questions needing current info; `0` = disable |
| `WEB_SEARCH_PROVIDER` | `duckduckgo` | Free search provider (no API key required) |
| `WEB_SEARCH_TIMEOUT_S` | `8.0` | Search request timeout in seconds |
| `WEB_SEARCH_MAX_RESULTS` | `3` | Maximum snippets passed to the LLM context (1–10) |

### Web search for current information

Web search integration lives in `services/web_search.py` and is fully configurable:

- **Heuristic query classification**: VoxAI inspects incoming queries using keyword patterns (`latest`, `current`, `news`, `today`, `weather`, `who won`, `2026`, etc., in English and Hindi). Ordinary offline questions (math, code explanation, storytelling, definitions) bypass web search completely and are answered directly by the local LLM.
- **Provider**: DuckDuckGo HTML endpoint with automatic library fallback. No paid AI search API or credit card is required.
- **Prompt injection safeguards**: search snippets are sanitised to remove HTML tags, bounded in length (max 400 characters per snippet), and scrubbed of prompt-injection attempts (such as `ignore previous instructions` or `system:` overrides).
- **Grounded context & source citations**: retrieved facts and URLs are injected into the LLM system prompt. The LLM produces a concise, conversational answer, and clickable source links are displayed in the chat interface.
- **Graceful degradation**: if the search service or internet connection fails, VoxAI clearly and honestly tells the user that current web information could not be retrieved. It never falsely claims to have searched the web.

### Automatic Gemini model discovery

Model selection lives in `services/llm_service.py` and runs on the backend
only:

1. **Discover** — the first chat request calls `client.models.list()` once
   through the single configured `genai.Client` to see which models your
   `GEMINI_API_KEY` can actually use. Nothing about accessibility is assumed:
   `gemini-2.5-flash` is only *preferred*, never required.
2. **Filter** — only models advertising `generateContent` are eligible.
   Embedding-, image-, video- and audio-only models are excluded, and so are
   preview/experimental/deprecated models when reliable metadata says so.
3. **Select** — the configured `GEMINI_MODEL` wins when it is discovered and
   supports text generation; otherwise the best stable model is chosen
   automatically (stable Gemini Flash first — good for low-latency
   conversation — alternating full Flash with Flash-Lite so a demand spike or
   per-model limit on one family cannot block the other, then other Gemini
   models, newest first).
4. **Cache** — the ordered candidates and the selected model are cached, so
   `models.list()` is *not* repeated for every message; a lock prevents
   concurrent requests from racing the discovery/switch.
5. **Fall back** — if a real generation request fails with model-not-found /
   unsupported-model, that model is remembered as gone for the process, the
   cache is invalidated (the next message re-discovers without it) and the
   next suitable candidate is tried. Transient overload answers (HTTP 5xx,
   “high demand”) also advance to the next candidate but keep the cache —
   neither ever switches providers. Model rejections (unavailable *or
   quota-limited* candidates) and overload rejections are each bounded by
   `GEMINI_MODEL_ATTEMPTS` (default 3) with `GEMINI_TIMEOUT_S` timeouts;
   retries are never endless. Authentication errors stop immediately. A 429
   on one candidate — e.g. a per-model free-tier limit such as “20
   requests/day on gemini-3.8-flash” — is skipped within the same bounded
   budget so a model with remaining quota can still answer; when quota is
   what actually blocks the reply, the request fails with an accurate
   `gemini_quota_exceeded` (429) — never disguised as “model unavailable”.

**To override**, set `GEMINI_MODEL` in `.env` to any model your key can use
(for example a specific Flash or a preview model you explicitly want) and
restart Flask; leave it empty for pure automatic selection. `/api/health`
reports the result as `gemini.discovery` (`not_attempted` / `ok` /
`failed`), `gemini.availability` (`not_checked` / `available` /
`unavailable` / `error`) and `gemini.selected_model`.

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Server + dependency status (Gemini configured + discovery/selection status; Whisper/Piper status); no Gemini API request |
| `POST` | `/api/transcribe` | Audio → transcript + detected language |
| `POST` | `/api/chat` | Text → LLM answer (history kept server-side) |
| `POST` | `/api/voice` | Full pipeline → transcript, answer, `audio_url`, metrics (optional `stream=progress` SSE) |
| `POST` | `/api/session/new` | Create an isolated session |
| `GET` | `/api/session/<session_id>` | Session info |
| `GET` | `/api/session/<session_id>/history` | This session's server-side history |
| `DELETE` | `/api/session/<session_id>` | Clear a session's history |
| `GET` | `/api/audio/<filename>` | Serve a generated WAV (sanitised names only) |

Validation: upload size (413), audio MIME allow-list (415), empty/missing body (400),
bad parameters (400 with `code` + `hint`), silence (422 `no_speech`), missing
Gemini credentials (503), invalid credentials (502), quota/rate limit (429),
model discovery found no usable model (503 `gemini_no_model_available`),
discovered models rejected by the API (503 `gemini_model_unavailable`),
model discovery failure (502 `gemini_discovery_failed`), timeout (504). Path
traversal on `/api/audio`
returns 404 (allow-list regex + resolved-path containment check).

Example:

```powershell
curl -F "audio=@recording.webm" http://localhost:5000/api/transcribe
curl -X POST -H "Content-Type: application/json" `
     -d '{"message":"What is 2+2?","session_id":""}' http://localhost:5000/api/chat
```

---

## Tests — exact commands and results

```powershell
# Unit tests (mocked AI services — no models needed):
python -m pytest -m "not integration" -q
#   163 passed, 11 deselected

# Integration tests (real Gemini + Whisper + Piper; requires .env and assets):
$env:VOXAI_INTEGRATION=1; python -m pytest -m integration -v
#   11 passed — needs the real key and downloaded local assets (last verified run).
```

Re-run the unit command after changes. Integration tests contact the configured
Gemini API and require downloaded Whisper/Piper assets.

Unit suite covers: health/dependency reporting and degradation, session
creation/isolation/bounded history/clearing, forged-history rejection, the full
voice pipeline (metrics, TTS-failure fallback, silence handling), SSE stage
events, request validation (400/413/415/422), audio path-traversal attempts,
Gemini model discovery, and **web search capabilities** (keyword heuristic
classification, prompt-injection sanitisation, HTML tag stripping, context
formatting, search failure handling, clickable source links, and ordinary offline
question bypass).

Integration suite (`-m integration`) runs against the **real** services:
health ready, **real model discovery** (`models.list()` against your key), real
LLM answer, **multi-turn memory recall**, Piper WAV validity
(readable RIFF), Whisper round-trip on Piper audio, the full
`/api/voice` end-to-end, and **voice-consistency checks** (the same configured
voice for every English response, for every Hindi response, and across
alternating English/Hindi turns).

Benchmark (measured latencies) is reproducible with:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py --turns 3
#   writes docs/benchmark-results.md and docs/benchmark-results.json
```

---

## Measured latency

No Gemini latency benchmark has been run for this integration, so no speedup is
claimed. Run `.\.venv\Scripts\python.exe scripts\benchmark.py --turns 3` with
the API key and local speech assets configured to measure actual Gemini,
Whisper, Piper, and end-to-end timings. Existing
[benchmark results](docs/benchmark-results.md) are from the previous Ollama
implementation and are historical only, not representative of Gemini.

---

## Example multi-turn conversation (memory)

Real exchange produced by the benchmark script (verbatim model output — the
follow-up never names the city, so the answer can only come from history):

```
You:  What is the capital of France and roughly how many people live there?
VoxAI: The capital of France is Paris. It has roughly 2.1 million residents.

You:  And how many people live in that city?
VoxAI: Paris has approximately 2.1 million people.
```

Also verified by an automated integration test:

```
You:  My name is Trikon. Remember it.     (session X)
You:  What is my name?                    (same session X)
VoxAI: answer contains "Trikon"            ← history survived server-side
```

And sessions are isolated: a second session created in the same test knows
nothing about the first (`tests/test_sessions.py::test_sessions_are_strictly_isolated`).

---

## Demo recording checklist (2–3 min)

Use this script when recording a demo:

1. **0:00–0:15** — Show the local Whisper/Piper assets and `/api/health`
   JSON: `status: ok`, `gemini.configured: true`, `whisper.loaded: true`.
2. **0:15–0:30** — Open `http://localhost:5000`. Point at the dark VoxAI UI:
   status pill “Ready”, health chip “backend online”.
3. **0:30–0:55** — Press **Start** (mic permission dialog if first time), say
   *“What's the capital of France?”*, press **Stop**. Show status moving
   Listening → Transcribing → Thinking → Speaking, the recognized transcript
   bubble, the answer bubble, and audio playing.
4. **0:55–1:25** — Ask the follow-up *“And how many people live there?”* —
   answer must reference Paris (memory demo).
5. **1:25–1:45** — Ask a Hindi question (e.g. *“भारत की राजधानी क्या है?”*) —
   transcript language tag `hi` — the same female voice continues in Hindi.
6. **1:45–2:05** — Settings panel: lower *Max answer length*, ask again →
   shorter reply. Toggle autoplay off → audio only on click.
7. **2:05–2:25** — **New conversation**, ask the same question again → it no
   longer remembers “Trikon”/Paris context (server history reset). **Clear
   history** → notice banner.
8. **2:25–2:45** — Error paths: deny mic permission → clear message; remove
   the backend API key → health reports Gemini as not configured.
9. **2:45–3:00** — Run `pytest -m "not integration"` on camera (all green).

---

## Models, licenses and downloads

| Component | Model / voice | License | Download |
|---|---|---|---|
| LLM | Gemini Flash (configured by `GEMINI_MODEL`) | Google Gemini API terms | [Google AI Studio](https://aistudio.google.com/) |
| STT | `Systran/faster-whisper-base` (CTranslate2) | MIT (conversion of `openai/whisper-base`) | auto via `scripts/prefetch_assets.py` or first request |
| TTS (en) | Piper `en_US-ryan-medium` (male) | see voice card | `scripts/prefetch_assets.py` |
| TTS (hi) | Piper `hi_IN-rohan-medium` (male, matches the English voice) | see voice card | `scripts/prefetch_assets.py` |

Sources: [Google AI Studio](https://aistudio.google.com/),
[faster-whisper-base](https://huggingface.co/Systran/faster-whisper-base),
[ryan voice card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/ryan/medium/MODEL_CARD),
[rohan voice card](https://huggingface.co/rhasspy/piper-voices/blob/main/hi/hi_IN/rohan/medium/MODEL_CARD),
[piper-voices repo](https://huggingface.co/rhasspy/piper-voices).

Model weights are **never committed** — `.gitignore` excludes `models/`,
`*.onnx`, `*.bin`, `*.gguf`, `var/` (recordings) and `.env`.

Manual alternative to the prefetch script:

```powershell
# Whisper
python -c "from faster_whisper.utils import download_model; download_model('base', local_dir='./models/whisper/base')"
# Piper voices (examples used by this repo)
# models/voices/en_US-ryan-medium.onnx(+.json) and hi_IN-rohan-medium.onnx(+.json)
# from https://huggingface.co/rhasspy/piper-voices
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Health shows Gemini not configured | Set `GEMINI_API_KEY` in the project `.env` and restart Flask. |
| Gemini authentication error (401/403) | Create/check the key in [Google AI Studio](https://aistudio.google.com/app/apikey); confirm Gemini API access for its project. Replace revoked/invalid keys and do not paste them into frontend code. |
| Gemini quota/rate-limit error (429) | VoxAI skips a quota-limited model within its bounded retries and keeps answering via other models; it reports 429 only when every tried model is quota-blocked — then check usage and quota reset times in Google AI Studio, or pin a model with remaining quota via `GEMINI_MODEL`. |
| Gemini temporarily overloaded (503 `gemini_overloaded`) | Transient Google-side demand spike. VoxAI already tried the next discovered models within `GEMINI_MODEL_ATTEMPTS`; wait a few seconds and send the message again. |
| Gemini model discovery finds no usable model (503) | VoxAI picks a model automatically; if discovery finds none, check that the key's project has Gemini API access and no restrictive key limitations in [Google AI Studio](https://aistudio.google.com/app/apikey), or set `GEMINI_MODEL` to a model the key can use and restart Flask. |
| “Microphone permission denied” | Allow the mic via the padlock icon in the address bar, then press Start again. |
| Gemini request times out (504) | Check network connectivity; increase `GEMINI_TIMEOUT_S` or reduce `MAX_RESPONSE_TOKENS`. |
| Gemini answers are slow | Check the API's measured `latency` payload (or `scripts/benchmark.py`), network, selected model, and Google AI Studio status; no specific speedup is guaranteed. |
| 422 `no_speech` | You recorded silence; speak closer to the mic for ~2 s. |
| 415 / 413 on upload | Unsupported type or clip too long; record shorter audio (limit `MAX_AUDIO_BYTES`). |
| No answer audio | TTS failed — the bubble shows a warning with the reason; check `PIPER_MODEL_PATH` / `PIPER_MODEL_PATH_HI` and re-run `scripts/prefetch_assets.py` (a missing Hindi voice raises `tts_hindi_voice_missing` instead of switching voices). |
| Windows activation blocked | `Set-ExecutionPolicy -Scope Process RemoteSigned` then re-activate, or skip activation and use `.\.venv\Scripts\python.exe …` directly. |
| Port already in use | Another process owns :5000 — stop it or change `PORT` in `.env`. |
| Whisper/Piper files missing | Startup lists the exact missing path and the command to fetch it (also in `/api/health` → `startup_issues`). |

---

## Known limitations

1. **Gemini requires network and API quota** — chat text and recent turns are
   sent from Flask to Google; response time depends on network conditions,
   model availability, and quota. Only STT and TTS remain local.
2. **Buffered, not streaming** — nothing is played until the whole recording
   is transcribed, answered and synthesised (stated in the UI and API; the SSE
   progress events only report *real* stage transitions).
3. **In-memory sessions** — history lives in the Flask process; restarting the
   server clears it (TTL `SESSION_TTL_SECONDS` also applies). A persistent
   store (SQLite) is the natural next step.
4. **The API key must be protected** — keep it in the ignored backend `.env`;
   rotate it if it is exposed.
5. **Hindi quality depends on the `base` Whisper size** — auto-detection works,
   but `small`/`medium` improve accented speech (`WHISPER_MODEL=small`).
6. **Single-process dev server** — Flask `threaded=True` is fine locally; use
   gunicorn/waitress for multiple users.
7. **No echo cancellation / VAD trimming on upload** — long silence inside a
   recording is transcribed, not trimmed (Whisper's internal VAD helps).
8. **Browser support** — needs `MediaRecorder` (Chrome/Edge/Firefox; Safari
   works but emits MP4 that Whisper may reject).

## Future improvements

- **Streaming playback**: sentence-level TTS so the voice starts while the LLM
  is still generating (real streaming, honestly labelled).
- **WebRTC input**: client-side noise suppression + VAD so only speech is sent.
- **Persistent, multi-device sessions** in SQLite with per-session titles.
- **Tool use**: local calendar/notes tools behind the same session state.
- **Gemini model selection** in the settings panel with measured per-model
  latency stats.
- **GPU acceleration** config passthrough (`WHISPER_DEVICE=cuda`,
  faster-whisper CUDA support) with automatic capability detection.
- **Streaming STT feedback**: live interim transcript while recording (browser
  speech API as an optional aid, local Whisper remains authoritative).
- **Speaker diarisation & wake word** (“Hey VoxAI”) running fully locally.
