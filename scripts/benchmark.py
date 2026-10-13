"""Measure real end-to-end latency of the local VoxAI pipeline.

Run (PowerShell, with `ollama serve` running and assets downloaded):

    .venv\\Scripts\\python.exe scripts\\benchmark.py --turns 3

It warms each model once (reporting the load cost separately), then measures
per-stage latencies over several turns and prints a markdown table. Results
are written to docs/benchmark-results.md (and .json).

Every number printed is measured with time.perf_counter() - nothing is
estimated. No network inference is used: STT/LLM/TTS all run locally.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_TEXT = "What is the capital of France and roughly how many people live there?"
FOLLOWUP_TEXT = "And how many people live in that city?"


def ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


def wav_bytes(path: Path) -> bytes:
    return Path(path).read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser(description="VoxAI local latency benchmark")
    parser.add_argument("--turns", type=int, default=3, help="measured turns per stage (default 3)")
    parser.add_argument("--out", default=str(ROOT / "docs" / "benchmark-results.md"))
    args = parser.parse_args()

    from app import create_app

    print("== VoxAI benchmark: building app (real services) ==")
    t0 = time.perf_counter()
    app = create_app(warmup=False)
    svc = app.extensions["voxai"].services
    cfg = app.extensions["voxai"].config
    client = app.test_client()
    print(f"app built in {ms(t0)} ms (models not loaded yet)")

    results: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": cfg.ollama_model,
        "whisper_model": cfg.whisper_model,
        "device": cfg.whisper_device,
        "compute_type": cfg.whisper_compute_type,
        "turns": args.turns,
        "stages": {},
    }

    # ---------------------------------------------------------------- health
    t0 = time.perf_counter()
    health = client.get("/api/health").get_json()
    results["stages"]["health_check_ms"] = ms(t0)
    print(f"health: {health['status']} | ollama reachable={health['ollama']['reachable']} "
          f"model_present={health['ollama']['model_present']}")
    if health["status"] != "ok":
        print("!! backend not ready - run `ollama serve` and scripts/prefetch_assets.py first")
        return 1

    # ---------------------------------------------------------------- TTS
    print("\n== TTS (Piper) ==")
    t0 = time.perf_counter()
    first = svc.tts.synthesize(SAMPLE_TEXT)
    tts_load_ms = ms(t0)
    print(f"first call (includes model load): {tts_load_ms} ms")
    tts_runs = []
    for _ in range(args.turns):
        t0 = time.perf_counter()
        out = svc.tts.synthesize(SAMPLE_TEXT)
        tts_runs.append(ms(t0))
    results["stages"]["tts_load_ms"] = tts_load_ms
    results["stages"]["tts_ms"] = tts_runs
    print(f"synthesis x{args.turns}: {tts_runs}")

    # ---------------------------------------------------------------- STT
    print("\n== STT (faster-whisper) on Piper-generated audio ==")
    audio_path = Path(svc.tts.audio_dir) / out["filename"]
    payload = wav_bytes(audio_path)
    t0 = time.perf_counter()
    svc.stt.transcribe(payload, content_type="audio/wav")
    stt_load_ms = ms(t0)
    print(f"first call (includes model load): {stt_load_ms} ms")
    stt_runs = []
    for _ in range(args.turns):
        t0 = time.perf_counter()
        tr = svc.stt.transcribe(payload, content_type="audio/wav")
        stt_runs.append(ms(t0))
    results["stages"]["stt_load_ms"] = stt_load_ms
    results["stages"]["stt_ms"] = stt_runs
    print(f"transcribe x{args.turns}: {stt_runs}")
    print(f"transcript: {tr['text']!r} (language={tr['language']})")

    # ---------------------------------------------------------------- LLM
    print(f"\n== LLM ({cfg.ollama_model}) ==")
    sid = client.post("/api/session/new").get_json()["session_id"]
    t0 = time.perf_counter()
    client.post("/api/chat", json={"message": SAMPLE_TEXT, "session_id": sid})
    llm_load_ms = ms(t0)
    print(f"first call (includes model load): {llm_load_ms} ms")
    llm_runs = []
    for i in range(args.turns):
        text = SAMPLE_TEXT if i % 2 == 0 else FOLLOWUP_TEXT
        t0 = time.perf_counter()
        resp = client.post("/api/chat", json={"message": text, "session_id": sid})
        llm_runs.append(ms(t0))
        body = resp.get_json()
        print(f"  turn {i + 1}: {llm_runs[-1]} ms | {body['response'][:90]!r}")
    results["stages"]["llm_first_ms"] = llm_load_ms
    results["stages"]["llm_ms"] = llm_runs
    results["stages"]["llm_reported_ms"] = body["latency"]["llm_ms"]

    # ---------------------------------------------------------- full voice
    print("\n== Full pipeline POST /api/voice (STT -> LLM -> TTS) ==")
    voice_runs = []
    stage_rows = []
    for i in range(args.turns):
        data = {
            "audio": (io.BytesIO(payload), "bench.wav", "audio/wav"),
            "recording_ms": "3000",
        }
        t0 = time.perf_counter()
        resp = client.post("/api/voice", data=data, content_type="multipart/form-data")
        wall = ms(t0)
        body = resp.get_json()
        if resp.status_code != 200:
            print(f"  !! voice failed: {resp.status_code} {body}")
            return 1
        lat = body["latency"]
        voice_runs.append(wall)
        stage_rows.append({
            "wall_ms": wall,
            "stt_ms": lat["stt_ms"],
            "llm_ms": lat["llm_ms"],
            "tts_ms": lat["tts_ms"],
            "backend_total_ms": lat["backend_total_ms"],
        })
        print(f"  turn {i + 1}: wall={wall} ms | stt={lat['stt_ms']} "
              f"llm={lat['llm_ms']} tts={lat['tts_ms']} backend={lat['backend_total_ms']}")
    results["stages"]["voice_wall_ms"] = voice_runs
    results["stages"]["voice_detail"] = stage_rows

    # ---------------------------------------------------------------- table
    def fmt(vals):
        if not vals:
            return "—"
        med = statistics.median(vals)
        return f"min {min(vals):.0f} / median {med:.0f} / max {max(vals):.0f} ms"

    lines = [
        "# VoxAI measured latency",
        "",
        f"Generated {results['generated_at']} · model `{results['model']}` · "
        f"whisper `{results['whisper_model']}` ({results['device']}/{results['compute_type']}) · "
        f"CPU-only, {args.turns} measured turns per stage.",
        "",
        "| Stage | Measured |",
        "|---|---|",
        f"| Health check | {results['stages']['health_check_ms']} ms |",
        f"| TTS first call (incl. load) | {results['stages']['tts_load_ms']} ms |",
        f"| TTS synthesis | {fmt(tts_runs)} |",
        f"| STT first call (incl. load) | {results['stages']['stt_load_ms']} ms |",
        f"| STT transcription | {fmt(stt_runs)} |",
        f"| LLM first call (incl. load) | {results['stages']['llm_first_ms']} ms |",
        f"| LLM chat round-trip | {fmt(llm_runs)} |",
        f"| Full `/api/voice` wall-clock | {fmt(voice_runs)} |",
        "",
        "All values measured with `time.perf_counter()`; the pipeline is buffered "
        "(not streaming) and recording time is excluded from the rows above.",
        "",
        "Reproduce: `.venv\\\\Scripts\\\\python.exe scripts\\\\benchmark.py --turns 3`",
    ]
    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_md.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {out_md} and {out_md.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
