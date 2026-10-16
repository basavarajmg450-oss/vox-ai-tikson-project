"""Gemini service tests using an in-memory SDK client; no API key is needed."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

from config import Config
from services.errors import ApiError
from services.llm_service import LLMService

#: A text-generation-capable model the fake SDK reports as available.
DEFAULT_MODEL_ENTRY = {
    "name": "models/gemini-2.5-flash",
    "supported_actions": ["generateContent", "countTokens"],
}


class FakeModels:
    def __init__(self, response=None, error=None, model_list=None, list_error=None):
        self.response = response
        self.error = error
        self.model_list = (
            list(model_list) if model_list is not None else [DEFAULT_MODEL_ENTRY]
        )
        self.list_error = list_error
        self.calls = []
        self.list_calls = []

    def list(self, **kwargs):
        """Mock of ``client.models.list()`` (model discovery)."""
        self.list_calls.append(kwargs)
        if self.list_error:
            raise self.list_error
        return list(self.model_list)

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, models):
        self.models = models


def make_service(*, api_key="test-only-key", max_history_turns=8, client=None):
    config = Config(
        gemini_api_key=api_key,
        gemini_model="gemini-2.5-flash",
        max_history_turns=max_history_turns,
    )
    models = FakeModels(
        response=SimpleNamespace(
            text="A concise answer.",
            model_version="gemini-2.5-flash",
            usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=5),
        )
    )
    service = LLMService(config, client=client or FakeClient(models))
    return service, models


def test_success_passes_model_output_cap_and_returns_usage():
    service, models = make_service()

    answer, timing = service.generate(
        [{"role": "user", "content": "Hello"}],
        system_prompt="Be concise.",
        temperature=0.3,
        max_tokens=80,
    )

    assert answer == "A concise answer."
    # Discovery ran once, then the discovered preferred model answered:
    assert len(models.list_calls) == 1
    call = models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
    assert call["config"].system_instruction == "Be concise."
    assert call["config"].max_output_tokens == 80
    assert timing["prompt_tokens"] == 12
    assert timing["completion_tokens"] == 5
    assert timing["model"] == "gemini-2.5-flash"
    assert timing["llm_ms"] >= 0
    assert service.selected_model == "gemini-2.5-flash"


def test_missing_api_key_returns_actionable_error_without_calling_gemini():
    models = FakeModels()
    service, _ = make_service(api_key="", client=FakeClient(models))

    with pytest.raises(ApiError) as exc:
        service.generate([{"role": "user", "content": "Hello"}])

    assert exc.value.code == "gemini_missing_credentials"
    assert "GEMINI_API_KEY" in exc.value.message
    assert models.calls == []
    assert models.list_calls == []


def test_environment_configuration_reads_gemini_settings_without_exposing_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("GEMINI_TIMEOUT_S", "45")

    config = Config.from_env()

    assert config.gemini_api_key == "test-only-key"
    assert config.gemini_model == "gemini-2.5-flash-lite"
    assert config.gemini_timeout_s == 45
    assert "test-only-key" not in repr(config.public())


def test_sdk_client_uses_configured_timeout_and_disables_retries(monkeypatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("services.llm_service.genai.Client", fake_client)
    service = LLMService(
        Config(gemini_api_key="test-only-key", gemini_timeout_s=12.5)
    )

    assert service._client is not None
    assert service.status()["availability"] == "not_checked"
    assert captured["http_options"].timeout == 12_500
    assert captured["http_options"].retry_options.attempts == 1


def test_quota_error_is_safe_and_actionable():
    service, models = make_service()
    models.error = errors.APIError(429, {"error": {"message": "quota exceeded"}})

    with pytest.raises(ApiError) as exc:
        service.generate([{"role": "user", "content": "Hello"}])

    assert exc.value.status_code == 429
    assert exc.value.code == "gemini_quota_exceeded"
    assert "quota" in exc.value.message.lower()
    assert "test-only-key" not in exc.value.message


def test_invalid_credentials_do_not_expose_sdk_error_text():
    service, models = make_service()
    models.error = errors.APIError(401, {"error": {"message": "invalid key test-only-key"}})

    with pytest.raises(ApiError) as exc:
        service.generate([{"role": "user", "content": "Hello"}])

    assert exc.value.code == "gemini_auth_error"
    assert "test-only-key" not in exc.value.message
    assert "test-only-key" not in (exc.value.hint or "")


def test_google_invalid_api_key_400_is_classified_as_authentication_error():
    service, models = make_service()
    models.error = errors.APIError(
        400, {"error": {"message": "API key not valid: test-only-key"}}
    )

    with pytest.raises(ApiError) as exc:
        service.generate([{"role": "user", "content": "Hello"}])

    assert exc.value.code == "gemini_auth_error"
    assert "test-only-key" not in exc.value.message


def test_unavailable_model_returns_configuration_hint():
    service, models = make_service()
    models.error = errors.APIError(404, {"error": {"message": "model not found"}})

    with pytest.raises(ApiError) as exc:
        service.generate([{"role": "user", "content": "Hello"}])

    assert exc.value.code == "gemini_model_unavailable"
    assert "GEMINI_MODEL" in exc.value.hint
    # The message no longer blames a single hard-coded model, and the key
    # never leaks into it:
    assert "test-only-key" not in exc.value.message
    # Discovery cache was invalidated so the next request re-lists:
    assert service._candidates is None
    assert len(models.list_calls) == 1


def test_timeout_maps_to_gateway_timeout_without_transport_details():
    service, models = make_service()
    models.error = httpx.ReadTimeout("request included test-only-key")

    with pytest.raises(ApiError) as exc:
        service.generate([{"role": "user", "content": "Hello"}])

    assert exc.value.status_code == 504
    assert exc.value.code == "llm_timeout"
    assert "test-only-key" not in exc.value.message


def test_network_failure_maps_to_safe_transport_error():
    service, models = make_service()
    models.error = httpx.ConnectError("connection includes test-only-key")

    with pytest.raises(ApiError) as exc:
        service.generate([{"role": "user", "content": "Hello"}])

    assert exc.value.code == "llm_transport_error"
    assert "test-only-key" not in exc.value.message


def test_empty_model_response_is_rejected():
    service, models = make_service()
    models.response = SimpleNamespace(text="", usage_metadata=None)

    with pytest.raises(ApiError) as exc:
        service.generate([{"role": "user", "content": "Hello"}])

    assert exc.value.code == "llm_empty_response"


def test_response_without_text_parts_is_rejected():
    service, models = make_service()

    class EmptyResponse:
        @property
        def text(self):
            raise ValueError("no text parts")

    models.response = EmptyResponse()

    with pytest.raises(ApiError) as exc:
        service.generate([{"role": "user", "content": "Hello"}])

    assert exc.value.code == "llm_empty_response"


def test_history_is_recent_and_assistant_roles_use_gemini_model_role():
    service, models = make_service(max_history_turns=2)
    history = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "recent user 1"},
        {"role": "assistant", "content": "recent assistant 1"},
        {"role": "user", "content": "recent user 2"},
        {"role": "assistant", "content": "recent assistant 2"},
    ]

    service.generate(history)

    contents = models.calls[0]["contents"]
    assert [message["parts"][0]["text"] for message in contents] == [
        "recent user 1",
        "recent assistant 1",
        "recent user 2",
        "recent assistant 2",
    ]
    assert [message["role"] for message in contents] == ["user", "model", "user", "model"]


def test_status_is_configuration_only_and_never_exposes_credentials():
    service, _ = make_service()

    status = service.status()

    assert status["configured"] is True
    assert status["availability"] == "not_checked"
    assert status["discovery"] == "not_attempted"
    assert status["selected_model"] is None
    assert status["reachable"] is None
    assert "test-only-key" not in repr(status)
