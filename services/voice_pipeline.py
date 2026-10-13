"""End-to-end voice pipeline: STT -> LLM -> TTS with measured latency.

The pipeline never trusts client-supplied history: conversation state is read
from and written to the :class:`~services.session_store.SessionStore` keyed by
session id.  Every stage is timed with ``time.perf_counter()`` and reported as
a *measured* value; nothing is estimated and nothing is streamed - the whole
recording is uploaded and processed before the answer begins.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from config import Config
from services.errors import ApiError, bad_request
from services.llm_service import LLMService
from services.session_store import SessionStore
from services.speech_to_text import SpeechToTextService
from services.text_to_speech import TTSService

#: Callback used to emit honest progress events (``stage``: stt/llm/tts/done).
ProgressFn = Callable[[Dict[str, Any]], None]

PIPELINE_MODE = "buffered"
PIPELINE_NOTE = (
    "The full recording is uploaded and processed before the answer begins; "
    "this is a buffered pipeline, not streaming."
)


class VoicePipeline:
    """Coordinates the three local AI services for one request."""

    def __init__(
        self,
        config: Config,
        sessions: SessionStore,
        stt: SpeechToTextService,
        llm: LLMService,
        tts: TTSService,
    ) -> None:
        self.config = config
        self.sessions = sessions
        self.stt = stt
        self.llm = llm
        self.tts = tts

    # ------------------------------------------------------------------ input validation
    def validate_message(self, message: Any) -> str:
        if not isinstance(message, str) or not message.strip():
            raise bad_request(
                "'message' is required and must be a non-empty string.",
                code="empty_message",
                hint='Send JSON like {"message": "hello", "session_id": "..."}',
            )
        text = message.strip()
        if len(text) > self.config.max_message_chars:
            raise ApiError(
                f"'message' is too long ({len(text)} characters; "
                f"limit {self.config.max_message_chars}).",
                413,
                "message_too_long",
            )
        return text

    def validate_temperature(self, value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            temperature = float(value)
        except (TypeError, ValueError):
            raise bad_request("'temperature' must be a number between 0 and 2.", code="invalid_temperature")
        if not 0.0 <= temperature <= 2.0:
            raise bad_request("'temperature' must be between 0 and 2.", code="invalid_temperature")
        return temperature

    def validate_max_tokens(self, value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            tokens = int(value)
        except (TypeError, ValueError):
            raise bad_request("'max_tokens' must be an integer.", code="invalid_max_tokens")
        if not 1 <= tokens <= 4096:
            raise bad_request("'max_tokens' must be between 1 and 4096.", code="invalid_max_tokens")
        return tokens

    # ------------------------------------------------------------------ stages
    def transcribe(self, audio_bytes: bytes, content_type: Optional[str] = None) -> Dict[str, Any]:
        """STT only: returns transcript, detected language and timing."""
        result = self.stt.transcribe(audio_bytes, content_type=content_type)
        return {
            "transcript": result["text"],
            "language": result["language"],
            "language_probability": result["language_probability"],
            "audio_duration_seconds": result["audio_duration_seconds"],
            "latency": {
                "recording_ms": None,
                "stt_ms": result["stt_ms"],
                "model_load_ms": result.get("stt_load_ms", 0.0),
                "backend_total_ms": result["stt_ms"],
                "sources": {
                    "stt_ms": "measured_server",
                    "model_load_ms": "measured_server",
                    "backend_total_ms": "measured_server",
                },
                "mode": PIPELINE_MODE,
                "note": PIPELINE_NOTE,
            },
        }

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        temperature: Optional[Any] = None,
        max_tokens: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """LLM turn only (typed text or a transcript from ``/api/transcribe``)."""
        started = time.perf_counter()
        text = self.validate_message(message)
        temperature = self.validate_temperature(temperature)
        max_tokens = self.validate_max_tokens(max_tokens)

        session = self.sessions.resolve(session_id)
        history = self.sessions.history(session.session_id)
        messages = list(history) + [{"role": "user", "content": text}]

        answer, timing = self.llm.generate(
            messages,
            system_prompt=self.config.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Only commit the turn once the model answered successfully.
        turns = self.sessions.append(session.session_id, "user", text)
        turns = self.sessions.append(session.session_id, "assistant", answer)

        backend_total_ms = (time.perf_counter() - started) * 1000.0
        return {
            "session_id": session.session_id,
            "response": answer,
            "history_turns": turns,
            "language": None,
            "latency": {
                "recording_ms": None,
                "llm_ms": timing["llm_ms"],
                "backend_total_ms": round(backend_total_ms, 1),
                "sources": {
                    "llm_ms": "measured_server",
                    "backend_total_ms": "measured_server",
                },
                "mode": PIPELINE_MODE,
                "note": PIPELINE_NOTE,
            },
            "usage": {
                "prompt_tokens": timing.get("prompt_tokens"),
                "completion_tokens": timing.get("completion_tokens"),
                "model": timing.get("model"),
                "temperature": timing.get("temperature"),
                "max_tokens": timing.get("max_tokens"),
            },
        }

    def voice(
        self,
        audio_bytes: bytes,
        content_type: Optional[str] = None,
        session_id: Optional[str] = None,
        recording_ms: Optional[float] = None,
        temperature: Optional[Any] = None,
        max_tokens: Optional[Any] = None,
        progress: Optional[ProgressFn] = None,
    ) -> Dict[str, Any]:
        """Full pipeline: audio in -> transcript, answer, WAV URL, metrics."""
        started = time.perf_counter()
        temperature = self.validate_temperature(temperature)
        max_tokens = self.validate_max_tokens(max_tokens)
        if recording_ms is not None and not (0.0 <= float(recording_ms) <= 600_000):
            raise bad_request("'recording_ms' must be between 0 and 600000.", code="invalid_recording_ms")

        def emit(stage: str, extra: Optional[Dict[str, Any]] = None) -> None:
            if progress is not None:
                payload = {"stage": stage}
                if extra:
                    payload.update(extra)
                progress(payload)

        # ---- Stage 1: speech to text -----------------------------------
        emit("stt", {"message": "Transcribing audio locally"})
        stt_result = self.stt.transcribe(audio_bytes, content_type=content_type)
        transcript = stt_result["text"]
        language = stt_result["language"]

        # ---- Stage 2: LLM ------------------------------------------------
        emit("llm", {"transcript": transcript, "language": language})
        session = self.sessions.resolve(session_id)
        history = self.sessions.history(session.session_id)
        messages = list(history) + [{"role": "user", "content": transcript}]
        answer, timing = self.llm.generate(
            messages,
            system_prompt=self.config.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # ---- Stage 3: TTS (failure never loses the answer) ---------------
        emit("tts", {"message": "Speaking the response"})
        audio: Optional[Dict[str, Any]] = None
        tts_error: Optional[Dict[str, Any]] = None
        tts_started = time.perf_counter()
        try:
            audio = self.tts.synthesize(answer, language=language)
        except ApiError as exc:
            tts_error = {"code": exc.code, "message": exc.message, "hint": exc.hint}
        tts_block_ms = (time.perf_counter() - tts_started) * 1000.0

        # Commit the turn only after the full pipeline answered.
        turns = self.sessions.append(session.session_id, "user", transcript)
        turns = self.sessions.append(session.session_id, "assistant", answer)

        backend_total_ms = (time.perf_counter() - started) * 1000.0

        tts_ms = audio["tts_ms"] if audio else None
        model_load_ms = stt_result.get("stt_load_ms", 0.0) + (audio.get("tts_load_ms", 0.0) if audio else 0.0)

        latency: Dict[str, Any] = {
            "recording_ms": round(float(recording_ms), 1) if recording_ms is not None else None,
            "stt_ms": stt_result["stt_ms"],
            "llm_ms": timing["llm_ms"],
            "tts_ms": tts_ms,
            "tts_status": "measured" if audio else "failed",
            "tts_block_ms": round(tts_block_ms, 1),
            "model_load_ms": round(model_load_ms, 1),
            "backend_total_ms": round(backend_total_ms, 1),
            "streaming": False,
            "mode": PIPELINE_MODE,
            "note": PIPELINE_NOTE,
        }
        sources = {
            "stt_ms": "measured_server",
            "llm_ms": "measured_server",
            "tts_block_ms": "measured_server",
            "backend_total_ms": "measured_server",
            "model_load_ms": "measured_server",
        }
        if recording_ms is not None:
            sources["recording_ms"] = "measured_client"
            latency["total_ms"] = round(float(recording_ms) + backend_total_ms, 1)
            sources["total_ms"] = "measured_client_plus_server"
        else:
            latency["total_ms"] = round(backend_total_ms, 1)
            sources["total_ms"] = "measured_server"
        if audio:
            sources["tts_ms"] = "measured_server"
        latency["sources"] = sources

        emit("done")
        return {
            "session_id": session.session_id,
            "transcript": transcript,
            "language": language,
            "language_probability": stt_result["language_probability"],
            "response": answer,
            "audio_url": f"/api/audio/{audio['filename']}" if audio else None,
            "audio_duration_seconds": audio["duration_seconds"] if audio else None,
            "tts_voice": audio.get("voice") if audio else None,
            "tts_error": tts_error,
            "audio_duration_input_seconds": stt_result["audio_duration_seconds"],
            "history_turns": turns,
            "latency": latency,
            "usage": {
                "prompt_tokens": timing.get("prompt_tokens"),
                "completion_tokens": timing.get("completion_tokens"),
                "model": timing.get("model"),
                "temperature": timing.get("temperature"),
                "max_tokens": timing.get("max_tokens"),
            },
        }
