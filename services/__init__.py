"""Service container: one object holding every backend capability.

``build_services()`` constructs the real services.  None of them load a model
in their constructor, so creating the container is cheap and safe at import
time; models are loaded lazily (or by the background warm-up started by
``run_app()``).
"""

from __future__ import annotations

from dataclasses import dataclass

from config import Config
from services.llm_service import LLMService
from services.session_store import SessionStore
from services.speech_to_text import SpeechToTextService
from services.text_to_speech import TTSService
from services.voice_pipeline import VoicePipeline


@dataclass
class Services:
    config: Config
    sessions: SessionStore
    stt: SpeechToTextService
    llm: LLMService
    tts: TTSService
    pipeline: VoicePipeline


def build_services(config: Config) -> Services:
    """Create the real (locally running) AI services."""
    sessions = SessionStore(config)
    stt = SpeechToTextService(config)
    llm = LLMService(config)
    tts = TTSService(config)
    pipeline = VoicePipeline(config, sessions=sessions, stt=stt, llm=llm, tts=tts)
    return Services(config=config, sessions=sessions, stt=stt, llm=llm, tts=tts, pipeline=pipeline)
