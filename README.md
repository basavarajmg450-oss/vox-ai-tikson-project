# VoxAI — a fully local voice assistant

Speak a question in the browser → hear the recognized text → get an answer from a
local LLM → hear it spoken back — all on your own machine. **No hosted APIs, no
API keys, no network inference.**

```
Browser (MediaRecorder) → Faster-Whisper (STT) → Ollama / Qwen3 (LLM) → Piper (TTS) → Browser audio
```

![status](https://img.shields.io/badge/STT-faster--whisper-blue) ![status](https://img.shields.io/badge/LLM-qwen3%3A4b%20via%20Ollama-blue) ![status](https://img.shields.io/badge/TTS-Piper-blue)

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

**Voice pipeline (all local)**
- Record in the browser with `MediaRecorder` (start/stop controls, mic button
  with recording animation, live timer).
- Transcribe with **faster-whisper** (`base`, CPU `int8`), auto language
  detection — English and Hindi supported; silence / empty / invalid audio
  produce clear errors, not crashes.
- Answer with **Qwen3-4B through Ollama** with:
  - a voice-assistant system prompt (concise, spoken-style answers),
  - recent-turn history bounded by `MAX_HISTORY_TURNS` (server-side only),
  - answer length bounded by `MAX_RESPONSE_TOKENS`,
  - configurable temperature (settings slider),
  - **reasoning kept out of the answer**: thinking goes to Ollama's separate
    `message.thinking` field which VoxAI discards, and any tagged leak is
    stripped (`strip_thinking`).
  - clear messages for *Ollama stopped*, *model missing*, *timeout*.
- Speak the answer with **Piper** (`en_US-lessac-medium`, plus
  `hi_IN-rohan-medium` used automatically for Hindi) — voice loaded **once**
  and reused, WAV output, temp files swept by TTL. If TTS fails, the text
  answer still shows with a visible warning.

**Sessions**
- History is authoritative on the Flask backend, keyed by an unguessable
  32-hex session id; strictly isolated; client-sent `history` is ignored.
- New conversation / clear history actions; TTL expiry.

**Interface** (vanilla HTML/CSS/JS, dark theme)
- VoxAI branding, accessible contrast, subtle animations only.
- Status pill with real stage transitions: **Ready → Listening → Transcribing →
  Thinking → Speaking → Error** (fed by genuine SSE stage events, not timers).
- User/assistant bubbles, transcript + response text, `<audio>` playback
  controls with optional autoplay, latency panel, settings section,
  permission/error messages, health chip.

**Honesty by design**
- The pipeline is **buffered**, not streaming: the whole recording is uploaded
  and processed before the answer starts — the UI says so, and the API says
  `"streaming": false`.
- Every latency number is measured with `time.perf_counter()` and labelled
  with its source (`measured_server`, `measured_client`,
  `measured_client_plus_server`). Nothing is estimated.

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
│    ② LLM  Ollama → qwen3:4b (history bounded, reasoning excluded)      │
│    ③ TTS  Piper → WAV in var/audio/ (loaded once)                      │
│  services/session_store.py  authoritative, isolated, TTL-bounded       │
└───────┼────────────────────────────────────────────────────────────────┘
        ▼  JSON (transcript, answer, audio_url, measured metrics)
           +  WAV bytes via GET /api/audio/<filename>
┌────────────────────────────  Your browser  ────────────────────────────┐
│  <audio controls> plays the WAV · bubbles · status · latency panel     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Requirements

**Software**
- Windows 11 (commands below are PowerShell; Linux/macOS work with `.venv/bin/…`)
- Python 3.11+ (3.11-compatible codebase)
- [Ollama](https://ollama.com/download) with the server running (`ollama serve`)
- A modern browser with `MediaRecorder` + `getUserMedia` (Chrome/Edge/Firefox);
  microphone access over `http://localhost` (browsers allow it without HTTPS)

**Hardware (measured on the machine this repo was built on)**
- CPU: Intel Core 5-class (integrated graphics only — **no CUDA GPU needed**)
- RAM: 16 GB recommended (Ollama ≈ 3–4 GB with qwen3:4b, Whisper base ≈ 0.5 GB)
- Disk: ≈ 1.5 GB for models (qwen3:4b ≈ 2.3 GB download, whisper base ≈ 145 MB,
  two Piper voices ≈ 130 MB)
- No GPU required; a GPU only makes the LLM stage faster.

---

## Setup (exact PowerShell commands)

```powershell
# 1) Python environment + dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2) Ollama model (once). Then keep the server running in another terminal:
ollama pull qwen3:4b
ollama serve            # leave running

# 3) Whisper + Piper assets (once) — downloads into ./models
python scripts/prefetch_assets.py

# 4) Optional configuration
copy .env.example .env

# 5) Run the app
python app.py
#    → open http://localhost:5000

# 6) Run the tests
python -m pytest -m "not integration" -q                 # unit tests (mocked AI)
$env:VOXAI_INTEGRATION=1; python -m pytest -m integration -v   # real models
```

Without activating the venv, use `.\.venv\Scripts\python.exe` in place of `python`.

**First request is slow on purpose:** models load lazily (first transcription ≈ 4 s,
first TTS ≈ 6 s, first chat ≈ 30 s). Warm-up threads preload them at startup, and
`/api/health` reports the load state.

---

## Configuration

All settings are environment variables (`.env` is loaded automatically). See
[.env.example](.env.example) for the full annotated list.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:4b` | Chat model (verified present before serving) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `OLLAMA_TIMEOUT_S` | `180` | Model timeout → clear 504 message |
| `WHISPER_MODEL` | `base` | STT size: tiny/base/small/medium/large-v3 |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | `cpu` / `int8` | Inference device |
| `PIPER_MODEL_PATH` | `./models/voices/en_US-lessac-medium.onnx` | English voice |
| `PIPER_MODEL_PATH_HI` | `./models/voices/hi_IN-rohan-medium.onnx` | Hindi voice |
| `MAX_HISTORY_TURNS` | `8` | Recent turns sent to the model |
| `MAX_RESPONSE_TOKENS` | `200` | Bound on the visible answer |
| `LLM_TEMPERATURE` | `0.6` | Sampling temperature |
| `SESSION_TTL_SECONDS` | `86400` | Idle session lifetime |
| `MAX_AUDIO_BYTES` | `15728640` | Upload limit (413 above it) |
| `AUDIO_DIR` | `./var/audio` | Generated WAVs (served only from here) |
| `STRICT_STARTUP` | `0` | `1` = abort boot if a model file is missing |

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Server + dependency status (Ollama reachable, model present, Whisper/Piper loaded) |
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
model (503), Ollama down (503), timeout (504). Path traversal on `/api/audio`
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
.\.venv\Scripts\python.exe -m pytest -m "not integration" -q
#   54 passed, 6 deselected in 0.62s

# Integration tests (real Ollama + Whisper + Piper; requires `ollama serve`
# and downloaded assets):
$env:VOXAI_INTEGRATION=1; .venv\Scripts\python.exe -m pytest -m integration -v
#   6 passed, 54 deselected in 209.42s
```

*(Results from runs on this machine — re-run to reproduce; timings vary.)*

Unit suite covers: health/dependency reporting and degradation, session
creation/isolation/bounded history/clearing, forged-history rejection, the full
voice pipeline (metrics, TTS-failure fallback, silence handling), SSE stage
events, request validation (400/413/415/422), and audio path-traversal attempts.

Integration suite (`-m integration`) runs against the **real** services:
health ready, real LLM answer, **multi-turn memory recall**, Piper WAV validity
(readable RIFF), Whisper round-trip on Piper audio, and the full
`/api/voice` end-to-end.

Benchmark (measured latencies) is reproducible with:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py --turns 3
#   writes docs/benchmark-results.md and docs/benchmark-results.json
```

---

## Measured latency

From `scripts/benchmark.py --turns 3` on this machine (CPU-only, 3 measured
turns per stage; full output in [docs/benchmark-results.md](docs/benchmark-results.md)):

| Stage | Measured |
|---|---|
| Health check | 2 104 ms (Ollama ping; cached for 3 s in-app) |
| TTS first call (incl. model load) | 5 562 ms |
| **TTS synthesis** | min 173 / **median 174** / max 183 ms |
| STT first call (incl. model load) | 4 124 ms |
| **STT transcription (~3 s clip)** | min 1 433 / **median 1 519** / max 1 563 ms |
| LLM first call (incl. model load) | 31 540 ms |
| **LLM chat round-trip** | min 35 332 / **median 38 670** / max 55 790 ms |
| **Full `/api/voice` wall-clock** | min 31 273 / **median 31 832** / max 34 946 ms |

Notes:
- All values are `time.perf_counter()` measurements — recording time excluded,
  nothing estimated.
- The LLM dominates: Qwen3-4B reasons before answering (its thinking goes to a
  discarded field), and `num_predict` includes a reasoning headroom
  (`MAX_RESPONSE_TOKENS + 1024`) so the answer is never starved to empty.
  A smaller model (`qwen3:1.7b`) or a GPU cuts this dramatically.
- Pipeline is buffered — `streaming: false` is reported in every response.

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

Use this script when recording the assignment demo:

1. **0:00–0:15** — Show `models/`, `ollama list` (qwen3:4b), `/api/health`
   JSON: `status: ok`, `ollama.reachable: true`, `whisper.loaded: true`.
2. **0:15–0:30** — Open `http://localhost:5000`. Point at the dark VoxAI UI:
   status pill “Ready”, health chip “backend online”.
3. **0:30–0:55** — Press **Start** (mic permission dialog if first time), say
   *“What's the capital of France?”*, press **Stop**. Show status moving
   Listening → Transcribing → Thinking → Speaking, the recognized transcript
   bubble, the answer bubble, and audio playing.
4. **0:55–1:25** — Ask the follow-up *“And how many people live there?”* —
   answer must reference Paris (memory demo). Show the latency panel rows
   (measured labels).
5. **1:25–1:45** — Ask a Hindi question (e.g. *“भारत की राजधानी क्या है?”*) —
   transcript language tag `hi`, Hindi voice plays.
6. **1:45–2:05** — Settings panel: lower *Max answer length*, ask again →
   shorter reply. Toggle autoplay off → audio only on click.
7. **2:05–2:25** — **New conversation**, ask the same question again → it no
   longer remembers “Trikon”/Paris context (server history reset). **Clear
   history** → notice banner.
8. **2:25–2:45** — Error paths: deny mic permission → clear message; stop
   Ollama → health chip goes red with “start ollama serve” hint; restart it →
   recovers.
9. **2:45–3:00** — Run `pytest -m "not integration"` on camera (all green).

---

## Models, licenses and downloads

| Component | Model / voice | License | Download |
|---|---|---|---|
| LLM | `qwen3:4b` (Ollama) | Apache 2.0 | `ollama pull qwen3:4b` |
| STT | `Systran/faster-whisper-base` (CTranslate2) | MIT (conversion of `openai/whisper-base`) | auto via `scripts/prefetch_assets.py` or first request |
| TTS (en) | Piper `en_US-lessac-medium` | Blizzard 2013 Lessac dataset license (see voice card) | `scripts/prefetch_assets.py` |
| TTS (hi) | Piper `hi_IN-rohan-medium` | IndicTTS license (see voice card) | `scripts/prefetch_assets.py` |

Sources: [Ollama qwen3:4b](https://ollama.com/library/qwen3:4b),
[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B),
[faster-whisper-base](https://huggingface.co/Systran/faster-whisper-base),
[lessac voice card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/lessac/medium/MODEL_CARD),
[rohan voice card](https://huggingface.co/rhasspy/piper-voices/blob/main/hi/hi_IN/rohan/medium/MODEL_CARD),
[piper-voices repo](https://huggingface.co/rhasspy/piper-voices).

Model weights are **never committed** — `.gitignore` excludes `models/`,
`*.onnx`, `*.bin`, `*.gguf`, `var/` (recordings) and `.env`.

Manual alternative to the prefetch script:

```powershell
# Whisper
python -c "from faster_whisper.utils import download_model; download_model('base', local_dir='./models/whisper/base')"
# Piper voices (examples used by this repo)
# models/voices/en_US-lessac-medium.onnx(+.json) and hi_IN-rohan-medium.onnx(+.json)
# from https://huggingface.co/rhasspy/piper-voices
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Health chip “backend unreachable” | Start Ollama: `ollama serve`. Flask and Ollama must run at the same time. |
| `/api/health` shows `model_missing` | `ollama pull qwen3:4b` |
| “Microphone permission denied” | Allow the mic via the padlock icon in the address bar, then press Start again. |
| First answer takes ~30–60 s | Normal on CPU: model load + reasoning. Check the latency panel; try `qwen3:1.7b` for speed. |
| 504 “model did not answer” | Raise `OLLAMA_TIMEOUT_S`, lower `MAX_RESPONSE_TOKENS`, or use a smaller model. |
| 503 “Ollama is not reachable” | Same as unreachable chip — `ollama serve`. |
| 422 `no_speech` | You recorded silence; speak closer to the mic for ~2 s. |
| 415 / 413 on upload | Unsupported type or clip too long; record shorter audio (limit `MAX_AUDIO_BYTES`). |
| No answer audio | TTS failed — the bubble shows a warning with the reason; check `PIPER_MODEL_PATH` and re-run `scripts/prefetch_assets.py`. |
| Windows activation blocked | `Set-ExecutionPolicy -Scope Process RemoteSigned` then re-activate, or skip activation and use `.\.venv\Scripts\python.exe …` directly. |
| Port already in use | Another process owns :5000 — stop it or change `PORT` in `.env`. |
| Whisper/Piper files missing | Startup lists the exact missing path and the command to fetch it (also in `/api/health` → `startup_issues`). |

---

## Known limitations

1. **Latency is CPU-bound** — median full-pipeline ≈ 32 s and chat ≈ 39 s on
   this machine (measured; see above). Qwen3-4B reasons before answering and
   the thinking tokens count against `num_predict`, hence the headroom that
   makes answers reliable but slower. Mitigations: `qwen3:1.7b`, a GPU build of
   Ollama, or a faster machine.
2. **Buffered, not streaming** — nothing is played until the whole recording
   is transcribed, answered and synthesised (stated in the UI and API; the SSE
   progress events only report *real* stage transitions).
3. **In-memory sessions** — history lives in the Flask process; restarting the
   server clears it (TTL `SESSION_TTL_SECONDS` also applies). A persistent
   store (SQLite) is the natural next step.
4. **Reasoning is discarded, not zero-cost** — Qwen3's thinking still consumes
   time/tokens; VoxAI only guarantees it never reaches the user.
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
- **Smaller/faster models** selectable from the settings panel
  (`qwen3:1.7b`, `qwen3:4b-instruct` variants) with per-model latency stats.
- **GPU acceleration** config passthrough (`WHISPER_DEVICE=cuda`,
  Ollama on CUDA) with automatic capability detection.
- **Streaming STT feedback**: live interim transcript while recording (browser
  speech API as an optional aid, local Whisper remains authoritative).
- **Speaker diarisation & wake word** (“Hey VoxAI”) running fully locally.
