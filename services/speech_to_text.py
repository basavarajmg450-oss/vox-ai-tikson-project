"""Speech-to-text via faster-whisper (runs locally on CPU or GPU).

The Whisper model is loaded at most once per process and reused for every
request; only the brief model-load step is guarded by a lock, transcription
itself runs without holding any application lock.
"""

from __future__ import annotations

import io
import threading
import time
from typing import Any, Dict, Optional

from config import Config
from services.errors import ApiError

#: Anything shorter than this cannot contain usable speech.
MIN_AUDIO_SECONDS = 0.35


class SpeechToTextService:
    """Lazy, thread-safe wrapper around ``faster_whisper.WhisperModel``."""

    name = "faster-whisper"

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model: Optional[Any] = None
        self._load_lock = threading.Lock()
        self._load_seconds: float = 0.0
        self._state: str = "not_loaded"   # not_loaded | loading | loaded | error
        self._last_error: Optional[str] = None

    # -- state -------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def state(self) -> str:
        return self._state

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.config.whisper_model,
            "device": self.config.whisper_device,
            "compute_type": self.config.whisper_compute_type,
            "language": self.config.whisper_language or "auto",
            "vad": self.config.whisper_vad,
            "loaded": self.loaded,
            "state": self._state,
            "load_seconds": round(self._load_seconds, 3),
            "last_error": self._last_error,
        }

    # -- loading ------------------------------------------------------------
    def load(self) -> Any:
        """Load the model once. Returns the model instance.

        The elapsed time of this call is kept in :attr:`last_load_seconds`;
        it is ``0.0`` when the model was already warm.
        """
        with self._load_lock:
            if self._model is not None:
                self.last_load_seconds = 0.0
                return self._model
            self._state = "loading"
            started = time.perf_counter()
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.config.whisper_model,
                    device=self.config.whisper_device,
                    compute_type=self.config.whisper_compute_type,
                    download_root=str(self.config.whisper_download_root),
                )
            except Exception as exc:  # noqa: BLE001 - surfaced as an actionable 503
                self._state = "error"
                self._last_error = f"{type(exc).__name__}: {exc}"
                self.last_load_seconds = 0.0
                raise ApiError(
                    f"Could not load the Whisper model "
                    f"'{self.config.whisper_model}' ({self._last_error}).",
                    503,
                    "stt_load_failed",
                    hint=(
                        "Check WHISPER_MODEL / WHISPER_DEVICE in .env, and download the "
                        "weights once with: .venv\\Scripts\\python.exe scripts\\prefetch_assets.py"
                    ),
                ) from exc
            self._load_seconds = time.perf_counter() - started
            self.last_load_seconds = self._load_seconds
            self._state = "loaded"
            self._last_error = None
            return self._model

    last_load_seconds: float = 0.0

    # -- transcription --------------------------------------------------------
    def transcribe(self, audio_bytes: bytes, content_type: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe a browser recording.

        Returns a dict with ``text``, ``language``, ``language_probability``,
        ``audio_duration_seconds``, ``stt_ms`` and ``stt_load_ms``.
        Raises :class:`~services.errors.ApiError` (400/422/503) with a clear
        message for empty, too short, undecodable or unloaded audio.
        """
        if not audio_bytes:
            raise ApiError(
                "The uploaded recording is empty (0 bytes).",
                400,
                "empty_audio",
                hint="Record at least half a second of speech and try again.",
            )
        if len(audio_bytes) < 256:
            raise ApiError(
                "The uploaded recording is too short to contain speech.",
                422,
                "audio_too_short",
                hint="Hold the microphone button for at least one second.",
            )

        model = self.load()          # may raise ApiError(503); sets last_load_seconds
        load_ms = self.last_load_seconds * 1000.0

        options: Dict[str, Any] = {
            "language": self.config.whisper_language or None,
            "task": "transcribe",
            "beam_size": 5,
            "vad_filter": self.config.whisper_vad,
        }
        if self.config.whisper_vad:
            options["vad_parameters"] = {"min_silence_duration_ms": 400}

        started = time.perf_counter()
        try:
            segments, info = model.transcribe(io.BytesIO(audio_bytes), **options)
            parts = []
            for segment in segments:
                text = (segment.text or "").strip()
                if text:
                    parts.append(text)
        except ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 - decoder/model errors -> actionable 422
            detail = f"{type(exc).__name__}: {exc}"
            raise ApiError(
                "Could not decode this recording. "
                "Please re-record; supported formats are WebM, WAV, MP3, M4A and OGG.",
                422,
                "audio_decode_failed",
                hint=detail,
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        text = " ".join(parts).strip()
        duration = float(getattr(info, "duration", 0.0) or 0.0)

        if duration and duration < MIN_AUDIO_SECONDS:
            raise ApiError(
                f"The recording is only {duration:.2f}s long.",
                422,
                "audio_too_short",
                hint="Hold the microphone button for at least one second.",
            )
        if not text:
            raise ApiError(
                "No speech detected in the recording.",
                422,
                "no_speech",
                hint="The microphone may not have picked anything up - try again closer to the mic.",
            )

        language_probability = getattr(info, "language_probability", None)
        return {
            "text": text,
            "language": getattr(info, "language", None),
            "language_probability": (
                round(float(language_probability), 3)
                if language_probability is not None else None
            ),
            "audio_duration_seconds": round(duration, 2),
            "stt_ms": round(elapsed_ms + load_ms, 1),
            "stt_load_ms": round(load_ms, 1),
            "content_type": content_type,
        }
