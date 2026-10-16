"""Service container: one object holding every backend capability.

``build_services()`` constructs the real services.  None of them load a model
in their constructor, so creating the container is cheap and safe at import
time; models are loaded lazily (or by the background warm-up started by
``run_app()``).
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

from config import Config
from services.llm_service import LLMService
from services.session_store import SessionStore
from services.speech_to_text import SpeechToTextService
from services.text_to_speech import TTSService
from services.voice_pipeline import VoicePipeline
from services.web_search import WebSearchService


@dataclass
class Services:
    config: Config
    sessions: SessionStore
    stt: SpeechToTextService
    llm: LLMService
    tts: TTSService
    pipeline: VoicePipeline
    web_search: Optional[WebSearchService] = None


def build_services(config: Config) -> Services:
    """Create local speech services and the configured Gemini API service."""
    sessions = SessionStore(config)
    stt = SpeechToTextService(config)
    llm = LLMService(config)
    tts = TTSService(config)
    web_search = WebSearchService(config) if config.web_search_enabled else None
    pipeline = VoicePipeline(
        config, sessions=sessions, stt=stt, llm=llm, tts=tts, web_search=web_search
    )
    return Services(
        config=config,
        sessions=sessions,
        stt=stt,
        llm=llm,
        tts=tts,
        pipeline=pipeline,
        web_search=web_search,
    )
