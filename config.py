"""Environment-driven configuration for VoxAI.

Every tunable value comes from an environment variable (optionally loaded
from a ``.env`` file) and has a safe default.  Paths pointing at model files
are resolved relative to the project root so the app can be launched from any
working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:  # python-dotenv is a listed dependency; stay importable without it.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_SYSTEM_PROMPT = (
    "You are VoxAI, a helpful voice assistant with local speech recognition and synthesis.\n"
    "- Be friendly, accurate and concise: normally 2-4 short sentences.\n"
    "- Your answer is read aloud by a text-to-speech engine, so write plain prose: "
    "no markdown, no bullet points, no code blocks, no emojis, no URLs unless asked.\n"
    "- Always reply in the same language the user used (English or Hindi).\n"
    "- Keep earlier turns of this conversation in mind and answer follow-up "
    "questions using that context.\n"
    "- If you are unsure, say so briefly instead of guessing."
)

#: Content types accepted for uploaded recordings.  ``application/octet-stream``
#: is allowed because some browsers send it for MediaRecorder blobs; the file
#: name extension is then checked instead.
ALLOWED_AUDIO_TYPES = frozenset(
    {
        "audio/webm",
        "audio/mp4",
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/ogg",
        "audio/opus",
        "audio/flac",
        "audio/aac",
        "audio/3gpp",
        "audio/quicktime",
        "application/octet-stream",
    }
)

#: Extensions accepted when the content type is generic (octet-stream).
ALLOWED_AUDIO_EXTENSIONS = frozenset(
    {".wav", ".webm", ".mp3", ".m4a", ".mp4", ".ogg", ".oga", ".opus", ".flac", ".aac", ".3gp"}
)

#: Hard ceiling on an upload regardless of configuration.
HARD_MAX_AUDIO_BYTES = 50 * 1024 * 1024
#: Hard ceiling on a typed chat message regardless of configuration.
HARD_MAX_MESSAGE_CHARS = 8000

_KNOWN_WHISPER_SIZES = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3",
    "distil-large-v2", "distil-large-v3", "distil-medium.en", "distil-small.en",
}

_DOWNLOAD_FIX = "Run:  .venv\\Scripts\\python.exe scripts\\prefetch_assets.py"


def _str_env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip()


def _int_env(name: str, default: int, minimum: Optional[int] = None,
             maximum: Optional[int] = None) -> int:
    raw = _str_env(name, "")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name}={raw!r} must be an integer "
            f"(example: {name}={default})."
        ) from exc
    if value == 0 and default != 0:
        # Some hosts (and shells) export PORT=0 as an "unset" sentinel.
        return default
    if minimum is not None and value < minimum:
        raise ValueError(f"{name}={value} is below the minimum of {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name}={value} is above the maximum of {maximum}.")
    return value


def _float_env(name: str, default: float, minimum: Optional[float] = None,
               maximum: Optional[float] = None) -> float:
    raw = _str_env(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name}={raw!r} must be a number "
            f"(example: {name}={default})."
        ) from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name}={value} is below the minimum of {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name}={value} is above the maximum of {maximum}.")
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = _str_env(name, "").lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name}={raw!r} must be true/false.")


def resolve_path(raw: str, default: Path) -> Path:
    """Resolve a possibly relative path against the project root."""
    text = (raw or "").strip()
    if not text:
        return default
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


@dataclass
class Config:
    """All runtime settings for VoxAI."""

    # ---- LLM (Google Gemini) -----------------------------------------
    gemini_api_key: str = ""
    #: Optional *preferred* model.  It is only used when model discovery
    #: finds it among the models available to the API key; otherwise the
    #: best discovered text-generation model is selected automatically.
    #: An empty value means pure automatic selection.
    gemini_model: str = "gemini-2.5-flash"
    #: How many discovered candidates one request may try when a model is
    #: rejected as unavailable/unsupported (small, bounded - never endless).
    gemini_model_attempts: int = 3
    gemini_timeout_s: float = 60.0
    llm_temperature: float = 0.6
    max_response_tokens: int = 200
    max_history_turns: int = 8
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # ---- STT (faster-whisper) ----------------------------------------
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_download_root: Path = BASE_DIR / "models" / "whisper"
    whisper_language: str = ""          # "" means auto-detect
    whisper_vad: bool = True
    #: Greedy decoding (1) is roughly 2x faster than the old beam-5 default
    #: on CPU; raise it (2-10) if transcription accuracy ever regresses.
    whisper_beam_size: int = 1

    # ---- TTS (Piper) --------------------------------------------------
    #: Both configured voices are MALE speakers of the same medium quality,
    #: so the speaker gender stays consistent across English and Hindi.
    piper_model_path: Path = BASE_DIR / "models" / "voices" / "en_US-ryan-medium.onnx"
    #: Hindi voice (male speaker hi_IN-rohan-medium).
    piper_model_path_hi: Path = BASE_DIR / "models" / "voices" / "hi_IN-rohan-medium.onnx"
    piper_length_scale: float = 1.0

    # ---- LLM reasoning budget ------------------------------------------
    #: Thinking tokens Gemini may spend before answering.  0 (default) skips
    #: reasoning entirely - much faster for short spoken answers and it stops
    #: thinking from eating MAX_RESPONSE_TOKENS (which produced occasional
    #: empty answers).  -1 leaves the provider default untouched.
    gemini_thinking_budget: int = 0

    # ---- HTTP / runtime ------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 5000

    # ---- Sessions, files and limits ------------------------------------
    audio_dir: Path = BASE_DIR / "var" / "audio"
    audio_ttl_seconds: int = 1800
    max_audio_bytes: int = 15 * 1024 * 1024
    max_message_chars: int = 2000
    session_ttl_seconds: int = 86400
    max_sessions: int = 500
    strict_startup: bool = False
    run_integration: bool = False

    # ---- Web search -------------------------------------------------
    web_search_enabled: bool = True
    web_search_provider: str = "duckduckgo"
    web_search_timeout_s: float = 8.0
    web_search_max_results: int = 3

    # --------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Config":
        """Build a :class:`Config` from the process environment.

        Raises ``ValueError`` with an actionable message when a variable
        holds an unusable value.
        """
        return cls(
            gemini_api_key=_str_env("GEMINI_API_KEY", ""),
            # Optional preferred model; an explicitly empty value disables the
            # preference so a suitable model is discovered automatically.
            gemini_model=_str_env("GEMINI_MODEL", "gemini-2.5-flash"),
            gemini_model_attempts=_int_env("GEMINI_MODEL_ATTEMPTS", 3, minimum=1, maximum=10),
            gemini_timeout_s=_float_env("GEMINI_TIMEOUT_S", 60.0, minimum=1.0, maximum=3600.0),
            llm_temperature=_float_env("LLM_TEMPERATURE", 0.6, minimum=0.0, maximum=2.0),
            max_response_tokens=_int_env("MAX_RESPONSE_TOKENS", 200, minimum=1, maximum=4096),
            max_history_turns=_int_env("MAX_HISTORY_TURNS", 8, minimum=0, maximum=100),
            system_prompt=_str_env("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            whisper_model=_str_env("WHISPER_MODEL", "base"),
            whisper_device=_str_env("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=_str_env("WHISPER_COMPUTE_TYPE", "int8"),
            whisper_download_root=resolve_path(
                _str_env("WHISPER_DOWNLOAD_ROOT", "./models/whisper"),
                BASE_DIR / "models" / "whisper",
            ),
            whisper_language=_str_env("WHISPER_LANGUAGE", "").lower(),
            whisper_vad=_bool_env("WHISPER_VAD", True),
            whisper_beam_size=_int_env("WHISPER_BEAM_SIZE", 1, minimum=1, maximum=10),
            piper_model_path=resolve_path(
                _str_env("PIPER_MODEL_PATH", "./models/voices/en_US-ryan-medium.onnx"),
                BASE_DIR / "models" / "voices" / "en_US-ryan-medium.onnx",
            ),
            piper_model_path_hi=resolve_path(
                _str_env("PIPER_MODEL_PATH_HI", "./models/voices/hi_IN-rohan-medium.onnx"),
                BASE_DIR / "models" / "voices" / "hi_IN-rohan-medium.onnx",
            ),
            piper_length_scale=_float_env("PIPER_LENGTH_SCALE", 1.0, minimum=0.1, maximum=4.0),
            gemini_thinking_budget=_int_env("GEMINI_THINKING_BUDGET", 0, minimum=-1, maximum=24576),
            host=_str_env("HOST", "127.0.0.1"),
            port=_int_env("PORT", 5000, minimum=1, maximum=65535),
            audio_dir=resolve_path(_str_env("AUDIO_DIR", "./var/audio"), BASE_DIR / "var" / "audio"),
            audio_ttl_seconds=_int_env("AUDIO_TTL_SECONDS", 1800, minimum=10),
            max_audio_bytes=min(
                _int_env("MAX_AUDIO_BYTES", 15 * 1024 * 1024, minimum=1024),
                HARD_MAX_AUDIO_BYTES,
            ),
            max_message_chars=min(
                _int_env("MAX_MESSAGE_CHARS", 2000, minimum=1),
                HARD_MAX_MESSAGE_CHARS,
            ),
            session_ttl_seconds=_int_env("SESSION_TTL_SECONDS", 86400, minimum=60),
            max_sessions=_int_env("MAX_SESSIONS", 500, minimum=1),
            strict_startup=_bool_env("STRICT_STARTUP", False),
            run_integration=_bool_env("RUN_INTEGRATION", False),
            web_search_enabled=_bool_env("WEB_SEARCH_ENABLED", True),
            web_search_provider=_str_env("WEB_SEARCH_PROVIDER", "duckduckgo"),
            web_search_timeout_s=_float_env("WEB_SEARCH_TIMEOUT_S", 8.0, minimum=1.0, maximum=30.0),
            web_search_max_results=_int_env("WEB_SEARCH_MAX_RESULTS", 3, minimum=1, maximum=10),
        )

    # --------------------------------------------------------------------
    def whisper_cache_dir(self) -> Path:
        """Directory faster-whisper uses for a named model size."""
        if "/" in self.whisper_model or Path(self.whisper_model).is_absolute() or "\\" in self.whisper_model:
            return Path(self.whisper_model)
        return self.whisper_download_root / f"models--Systran--faster-whisper-{self.whisper_model}"

    def _voice_missing_issue(self, label: str, path: Path) -> Dict[str, str]:
        return {
            "level": "error",
            "component": "piper",
            "message": f"{label} voice model not found: {path}",
            "fix": _DOWNLOAD_FIX,
        }

    def startup_issues(self) -> List[Dict[str, str]]:
        """Validate required model files.

        Returns a list of actionable issue dicts (``level`` is ``error`` for a
        missing file that blocks a component, ``warning`` for something that
        will be fetched automatically on first use).
        """
        issues: List[Dict[str, str]] = []

        en_voice = self.piper_model_path
        if not en_voice.exists():
            issues.append(self._voice_missing_issue("English (default)", en_voice))
        elif not Path(str(en_voice) + ".json").exists():
            issues.append({
                "level": "error",
                "component": "piper",
                "message": (
                    f"Voice config missing (expected {en_voice}.json next to the .onnx file)."
                ),
                "fix": _DOWNLOAD_FIX,
            })

        if self.piper_model_path_hi and not self.piper_model_path_hi.exists():
            issues.append({
                "level": "warning",
                "component": "piper",
                "message": (
                    f"Hindi voice not found at {self.piper_model_path_hi}; Hindi "
                    "answers keep their text but speech fails with a clear error "
                    "instead of switching to a different voice."
                ),
                "fix": _DOWNLOAD_FIX,
            })

        cache_dir = self.whisper_cache_dir()
        if not cache_dir.exists():
            if self.whisper_model in _KNOWN_WHISPER_SIZES or "/" in self.whisper_model:
                issues.append({
                    "level": "warning",
                    "component": "whisper",
                    "message": (
                        f"Whisper model '{self.whisper_model}' is not cached yet "
                        f"(expected {cache_dir}). It will download on first use "
                        "and therefore needs internet access once."
                    ),
                    "fix": _DOWNLOAD_FIX,
                })
            else:
                issues.append({
                    "level": "error",
                    "component": "whisper",
                    "message": (
                        f"Unknown WHISPER_MODEL '{self.whisper_model}'. Use one of: "
                        f"{', '.join(sorted(_KNOWN_WHISPER_SIZES))} (or a local path)."
                    ),
                    "fix": "Set WHISPER_MODEL=base in .env",
                })

        if self.max_history_turns > 0 and self.max_response_tokens <= 0:
            issues.append({
                "level": "error",
                "component": "llm",
                "message": "MAX_RESPONSE_TOKENS must be greater than zero.",
                "fix": "Set MAX_RESPONSE_TOKENS=200 in .env",
            })

        return issues

    # --------------------------------------------------------------------
    def public(self) -> Dict[str, object]:
        """Settings safe to expose through ``GET /api/health``."""
        return {
            "gemini_model": self.gemini_model,
            "gemini_model_attempts": self.gemini_model_attempts,
            "gemini_timeout_s": self.gemini_timeout_s,
            "gemini_configured": bool(self.gemini_api_key),
            "whisper_model": self.whisper_model,
            "whisper_device": self.whisper_device,
            "whisper_compute_type": self.whisper_compute_type,
            "whisper_language": self.whisper_language or "auto",
            "whisper_beam_size": self.whisper_beam_size,
            "piper_model_path": str(self.piper_model_path),
            "piper_hindi_voice_available": self.piper_model_path_hi.exists(),
            "max_history_turns": self.max_history_turns,
            "max_response_tokens": self.max_response_tokens,
            "temperature": self.llm_temperature,
            "max_message_chars": self.max_message_chars,
            "max_audio_bytes": self.max_audio_bytes,
            "web_search_enabled": self.web_search_enabled,
            "web_search_provider": self.web_search_provider,
            "web_search_max_results": self.web_search_max_results,
        }
