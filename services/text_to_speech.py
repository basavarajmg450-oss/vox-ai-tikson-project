"""Text-to-speech via Piper (local ONNX voice models).

The Piper voice is loaded at most once per model file and reused for every
request; synthesis writes a fresh WAV into a dedicated directory that is served
only through ``GET /api/audio/<filename>``.  Old files are swept with a TTL.
"""

from __future__ import annotations

import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, Optional

from config import Config
from services.errors import ApiError


class TTSService:
    """Lazy, thread-safe wrapper around ``piper.PiperVoice``."""

    name = "piper"

    def __init__(self, config: Config) -> None:
        self.config = config
        self._voices: Dict[str, Any] = {}          # model path -> PiperVoice
        self._synth_lock = threading.Lock()        # piper/espeak keep global state
        self._load_lock = threading.Lock()
        self._state: str = "not_loaded"            # not_loaded | loaded | error
        self._last_error: Optional[str] = None
        self.audio_dir: Path = Path(config.audio_dir)
        #: seconds spent loading a voice during the most recent load() call
        self.last_load_ms: float = 0.0
        self.last_load_path: Optional[str] = None

    # -- state -------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return bool(self._voices)

    @property
    def state(self) -> str:
        if self._state == "error":
            return "error"
        return "loaded" if self.loaded else "not_loaded"

    def status(self) -> Dict[str, Any]:
        en_path = Path(self.config.piper_model_path)
        hi_path = Path(self.config.piper_model_path_hi) if self.config.piper_model_path_hi else None
        return {
            "name": self.name,
            "model_path": str(en_path),
            "model_present": en_path.exists(),
            "config_present": Path(str(en_path) + ".json").exists() if en_path.suffix == ".onnx" else True,
            "hindi_model_path": str(hi_path) if hi_path else None,
            "hindi_present": bool(hi_path and hi_path.exists()),
            "length_scale": self.config.piper_length_scale,
            "loaded": self.loaded,
            "voices_loaded": sorted(self._voices),
            "state": self.state,
            "last_error": self._last_error,
        }

    # -- loading -------------------------------------------------------------
    def load(self, model_path: Optional[Path] = None) -> Any:
        """Load (once) and return the Piper voice for ``model_path``."""
        path = Path(model_path or self.config.piper_model_path)
        key = str(path)
        with self._load_lock:
            voice = self._voices.get(key)
            if voice is not None:
                self.last_load_ms = 0.0
                self.last_load_path = key
                return voice
            if not path.exists():
                raise ApiError(
                    f"Piper voice model not found: {path}",
                    503,
                    "tts_model_missing",
                    hint=(
                        "Download the voice files once with: "
                        ".venv\\Scripts\\python.exe scripts\\prefetch_assets.py "
                        "(or set PIPER_MODEL_PATH in .env)."
                    ),
                )
            try:
                from piper import PiperVoice
            except Exception as exc:  # noqa: BLE001
                self._state = "error"
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise ApiError(
                    "The piper-tts package is not installed correctly.",
                    503,
                    "tts_package_missing",
                    hint="Reinstall dependencies: .venv\\Scripts\\python.exe -m pip install -r requirements.txt",
                ) from exc
            started = time.perf_counter()
            self.last_load_ms = 0.0
            try:
                voice = PiperVoice.load(str(path))
            except Exception as exc:  # noqa: BLE001
                self._state = "error"
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise ApiError(
                    f"Could not load the Piper voice '{path.name}' ({self._last_error}).",
                    503,
                    "tts_load_failed",
                    hint="Re-download the voice and its .onnx.json config with scripts/prefetch_assets.py",
                ) from exc
            self._voices[key] = voice
            self.last_load_ms = time.perf_counter() - started
            self.last_load_path = key
            self._state = "loaded"
            self._last_error = None
            return voice

    # -- voice selection ------------------------------------------------------
    def voice_for_language(self, language: Optional[str]) -> Path:
        """Pick the default voice, or the Hindi voice when available."""
        default = Path(self.config.piper_model_path)
        if language and str(language).startswith("hi"):
            hi = Path(self.config.piper_model_path_hi) if self.config.piper_model_path_hi else None
            if hi and hi.exists():
                return hi
        return default

    # -- synthesis --------------------------------------------------------------
    def synthesize(self, text: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Speak ``text`` and return WAV metadata for ``/api/audio/``.

        Returns ``{filename, duration_seconds, tts_ms, tts_load_ms, voice}``.
        Raises :class:`~services.errors.ApiError` (400/503) on failure; callers
        keep the text response even when this fails.
        """
        text = (text or "").strip()
        if not text:
            raise ApiError("Nothing to speak: the response text is empty.", 400, "empty_text")

        model_path = self.voice_for_language(language)
        voice = self.load(model_path)
        load_ms = self.last_load_ms * 1000.0

        try:
            from piper import SynthesisConfig
            synth_config = SynthesisConfig(length_scale=self.config.piper_length_scale)
        except Exception:  # pragma: no cover - older/newer piper without SynthesisConfig
            synth_config = None

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.wav"
        out_path = self.audio_dir / filename

        started = time.perf_counter()
        try:
            # The synth lock keeps espeak-ng's global state safe under
            # concurrent requests; it is NOT held during model loading.
            with self._synth_lock:
                with wave.open(str(out_path), "wb") as wav_file:
                    if synth_config is None:
                        voice.synthesize_wav(text, wav_file)
                    else:
                        voice.synthesize_wav(text, wav_file, syn_config=synth_config)
        except ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as an actionable 503
            self._state = "error"
            self._last_error = f"{type(exc).__name__}: {exc}"
            out_path.unlink(missing_ok=True)
            raise ApiError(
                f"Piper could not synthesize this response ({self._last_error}).",
                503,
                "tts_failed",
                hint="Check PIPER_MODEL_PATH and that the .onnx.json config sits next to the .onnx file.",
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if not out_path.exists() or out_path.stat().st_size <= 44:
            out_path.unlink(missing_ok=True)
            raise ApiError(
                "Piper produced an empty WAV file.",
                503,
                "tts_empty_output",
                hint="Check the Piper voice configuration.",
            )

        with wave.open(str(out_path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate() or 1
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
        duration = frames / float(rate)

        self.sweep()
        return {
            "filename": filename,
            "duration_seconds": round(duration, 2),
            "sample_rate": rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
            "bytes": out_path.stat().st_size,
            "tts_ms": round(elapsed_ms + load_ms, 1),
            "tts_load_ms": round(load_ms, 1),
            "voice": model_path.name,
        }

    # -- housekeeping -----------------------------------------------------------
    def sweep(self) -> int:
        """Delete generated WAVs older than AUDIO_TTL_SECONDS. Returns count removed."""
        now = time.time()
        removed = 0
        try:
            for path in self.audio_dir.glob("*.wav"):
                try:
                    if now - path.stat().st_mtime > self.config.audio_ttl_seconds:
                        path.unlink()
                        removed += 1
                except OSError:
                    continue
        except OSError:
            pass
        return removed

    def resolve_audio(self, filename: str) -> Path:
        """Map a safe filename to a real file, else raise a 404.

        Defence in depth: an allow-list regex, an extension check and a
        resolved-path containment check make path traversal impossible.
        """
        import re

        if not filename or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}\.wav", filename):
            raise ApiError("Unknown audio file.", 404, "audio_not_found")
        root = self.audio_dir.resolve()
        target = (root / filename).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise ApiError("Unknown audio file.", 404, "audio_not_found") from None
        if not target.is_file():
            raise ApiError("Unknown audio file.", 404, "audio_not_found")
        return target
