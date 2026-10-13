# VoxAI measured latency

Generated 2026-10-13T04:50:58.606924+00:00 · model `qwen3:4b` · whisper `base` (cpu/int8) · CPU-only, 3 measured turns per stage.

| Stage | Measured |
|---|---|
| Health check | 2104.0 ms |
| TTS first call (incl. load) | 5561.7 ms |
| TTS synthesis | min 173 / median 174 / max 183 ms |
| STT first call (incl. load) | 4124.2 ms |
| STT transcription | min 1433 / median 1519 / max 1563 ms |
| LLM first call (incl. load) | 31540.2 ms |
| LLM chat round-trip | min 35332 / median 38670 / max 55790 ms |
| Full `/api/voice` wall-clock | min 31273 / median 31832 / max 34946 ms |

All values measured with `time.perf_counter()`; the pipeline is buffered (not streaming) and recording time is excluded from the rows above.

Reproduce: `.venv\\Scripts\\python.exe scripts\\benchmark.py --turns 3`