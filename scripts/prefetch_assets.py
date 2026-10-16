#!/usr/bin/env python
"""Download every model file VoxAI needs, once.

    .venv\\Scripts\\python.exe scripts\\prefetch_assets.py

Downloads:
  * the faster-whisper model (default: base, override with WHISPER_MODEL)
  * the English Piper voice  (en_US-ryan-medium, male speaker)
  * the Hindi Piper voice    (hi_IN-rohan-medium, male speaker, so the
                              voice keeps one consistent gender across
                              English and Hindi responses)

Nothing is downloaded twice; existing files are skipped.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402

VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
#: (relative voice dir, file stem)
PIPER_VOICES = [
    ("en/en_US/ryan/medium", "en_US-ryan-medium"),
    ("hi/hi_IN/rohan/medium", "hi_IN-rohan-medium"),
]


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {dest.name} <- {url}")

    def hook(blocks: int, block_size: int, total: int) -> None:
        done = blocks * block_size
        if total > 0:
            pct = min(100.0, done * 100.0 / total)
            sys.stdout.write(f"\r        {done / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct:3.0f}%)")
        else:
            sys.stdout.write(f"\r        {done / 1e6:6.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, str(tmp), reporthook=hook)
    print()
    tmp.replace(dest)


def fetch_whisper(cfg: Config) -> None:
    print(f"\n== faster-whisper model: {cfg.whisper_model} ==")
    cache_dir = cfg.whisper_cache_dir()
    if "/" in cfg.whisper_model or Path(cfg.whisper_model).is_absolute():
        print(f"  [info] WHISPER_MODEL looks like a local path ({cfg.whisper_model}); skipping download.")
        return
    if cache_dir.exists() and any(cache_dir.rglob("model.bin")):
        print(f"  [skip] already cached at {cache_dir}")
        return
    from faster_whisper.utils import download_model

    path = download_model(
        cfg.whisper_model,
        cache_dir=str(cfg.whisper_download_root),
    )
    print(f"  [ ok ] downloaded to {path}")


def fetch_piper(cfg: Config) -> None:
    print("\n== Piper voices ==")
    targets = [(PIPER_VOICES[0], cfg.piper_model_path)]          # English (required)
    if cfg.piper_model_path_hi:
        targets.append((PIPER_VOICES[1], cfg.piper_model_path_hi))  # Hindi (optional)

    for (voice_dir, stem), model_path in targets:
        print(f"  -- {stem}")
        onnx = Path(model_path)
        download_file(f"{VOICES_BASE}/{voice_dir}/{stem}.onnx", onnx)
        download_file(f"{VOICES_BASE}/{voice_dir}/{stem}.onnx.json", Path(str(onnx) + ".json"))
        config_path = Path(str(onnx) + ".json")
        if config_path.exists():
            try:
                meta = json.loads(config_path.read_text(encoding="utf-8"))
                print(f"     language={meta.get('language', {}).get('code')} "
                      f"quality={meta.get('quality')} sample_rate={meta.get('audio', {}).get('sample_rate')}")
            except Exception:  # noqa: BLE001
                pass


def main() -> int:
    cfg = Config.from_env()
    from config import BASE_DIR

    print("VoxAI asset prefetch")
    print(f"  project root: {BASE_DIR}")
    fetch_whisper(cfg)
    fetch_piper(cfg)

    issues = cfg.startup_issues()
    remaining = [i for i in issues if i["level"] == "error"]
    warnings = [i for i in issues if i["level"] == "warning"]
    print("\n== Summary ==")
    for issue in warnings:
        print(f"  warn: {issue['message']}")
    for issue in remaining:
        print(f"  ERROR: {issue['message']}")
    if remaining:
        print("\nSome required files are still missing - see messages above.")
        return 1
    print("  All required model files are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
