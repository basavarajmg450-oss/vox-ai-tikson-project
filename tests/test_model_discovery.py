"""Unit tests for automatic Gemini model discovery, selection and fallback.

Everything here runs against an in-memory mock of the ``google-genai`` client
(`FakeGenAI` in ``conftest.py``): no network access, no API key and no paid
request is ever needed.  The real-API check lives in ``test_integration.py``
behind ``VOXAI_INTEGRATION=1``.
"""

from __future__ import annotations

import threading

import httpx
import pytest
from google.genai import errors

from config import Config
from services.errors import ApiError
from services.llm_service import LLMService, normalize_model_name, select_candidates
from tests.conftest import FakeGenAI, gemini_model_entry

PREFERRED = "gemini-2.5-flash"


def make_service(
    entries=None,
    preferred=PREFERRED,
    api_key="test-only-key",
    attempts=3,
    client=None,
    **fake_kwargs,
):
    """Build an LLMService wired to a fake google-genai client."""
    config = Config(
        gemini_api_key=api_key,
        gemini_model=preferred,
        gemini_model_attempts=attempts,
        max_history_turns=8,
    )
    fake = client if client is not None else FakeGenAI(model_entries=entries, **fake_kwargs)
    service = LLMService(config, client=fake)
    return service, fake


def hello():
    return [{"role": "user", "content": "Hello"}]


# ---------------------------------------------------------------- selection
def test_preferred_model_is_selected_when_available_and_text_capable():
    entries = [gemini_model_entry("gemini-2.5-flash"), gemini_model_entry("gemini-2.0-flash")]
    service, fake = make_service(entries)

    answer, _timing = service.generate(hello())

    assert answer == "A concise answer."
    assert fake.generate_calls[0]["model"] == "gemini-2.5-flash"
    assert service.selected_model == "gemini-2.5-flash"
    status = service.status()
    assert status["availability"] == "available"
    assert status["selected_model"] == "gemini-2.5-flash"
    assert status["model"] == "gemini-2.5-flash"
    assert status["preferred_model"] == PREFERRED


def test_preferred_model_wins_over_a_newer_stable_flash():
    entries = [gemini_model_entry("gemini-2.5-flash"), gemini_model_entry("gemini-2.0-flash-lite")]
    service, fake = make_service(entries, preferred="gemini-2.0-flash-lite")

    service.generate(hello())

    assert fake.generate_calls[0]["model"] == "gemini-2.0-flash-lite"


def test_automatic_selection_when_the_preferred_model_is_unavailable():
    entries = [
        gemini_model_entry("gemini-1.5-flash"),
        gemini_model_entry("gemini-2.5-flash-lite"),
        gemini_model_entry("gemini-2.5-flash"),
    ]
    service, fake = make_service(entries, preferred="gemini-3.0-omni")

    service.generate(hello())

    used = fake.generate_calls[0]["model"]
    assert used == "gemini-2.5-flash"          # stable Flash, newest first
    assert used != "gemini-3.0-omni"
    assert service.status()["model"] == "gemini-2.5-flash"


def test_models_without_text_generation_are_never_selected():
    entries = [
        gemini_model_entry("text-embedding-004", actions=["embedContent", "countTokens"]),
        gemini_model_entry("imagen-3.0-generate-002", actions=["predict"]),
        gemini_model_entry("veo-3.0-generate-001", actions=["predict"]),
        gemini_model_entry("gemini-2.0-flash", actions=["generateContent"]),
    ]
    service, fake = make_service(entries)

    service.generate(hello())

    assert fake.generate_calls[0]["model"] == "gemini-2.0-flash"
    assert service.discover_models() == ["gemini-2.0-flash"]
    assert service.status()["availability"] == "available"


def test_select_candidates_filters_embedding_and_image_models():
    entries = [
        gemini_model_entry("text-embedding-004", actions=["embedContent"]),
        gemini_model_entry("gemini-2.5-flash-image", actions=["generateContent"]),
        gemini_model_entry("gemini-2.0-flash"),
    ]
    assert select_candidates(entries) == ["gemini-2.0-flash"]


def test_select_candidates_skips_preview_models_when_a_stable_exists():
    entries = [
        gemini_model_entry("gemini-2.5-flash-preview"),
        gemini_model_entry("gemini-2.0-flash"),
    ]
    assert select_candidates(entries) == ["gemini-2.0-flash"]


def test_select_candidates_excludes_preview_models_from_automatic_selection():
    # Even when nothing else is available, preview/experimental models are
    # never chosen automatically - the operator must pin one explicitly.
    entries = [gemini_model_entry("gemini-2.5-flash-preview")]
    assert select_candidates(entries) == []
    assert select_candidates(entries, preferred="gemini-2.5-flash-preview") == [
        "gemini-2.5-flash-preview"
    ]


def test_select_candidates_excludes_models_marked_deprecated_in_metadata():
    entries = [
        gemini_model_entry(
            "gemini-1.0-pro-002",
            description="Deprecated. The previous generation of Gemini models.",
        ),
        gemini_model_entry("gemini-2.0-flash"),
    ]
    assert select_candidates(entries) == ["gemini-2.0-flash"]


def test_select_candidates_ranks_unverified_metadata_after_verified_models():
    entries = [
        gemini_model_entry("mystery-model", actions=None),
        gemini_model_entry("gemini-2.0-flash"),
    ]
    assert select_candidates(entries) == ["gemini-2.0-flash", "mystery-model"]


def test_select_candidates_ranks_stable_flash_before_pro_and_unknown():
    entries = [
        gemini_model_entry("some-helper-model"),
        gemini_model_entry("gemini-2.5-pro"),
        gemini_model_entry("gemini-2.0-flash-lite"),
        gemini_model_entry("gemini-2.5-flash"),
    ]
    assert select_candidates(entries) == [
        "gemini-2.5-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-pro",
        "some-helper-model",
    ]


def test_select_candidates_honours_an_explicit_preview_preference():
    entries = [
        gemini_model_entry("gemini-2.5-flash-preview"),
        gemini_model_entry("gemini-2.0-flash"),
    ]
    ordered = select_candidates(entries, preferred="gemini-2.5-flash-preview")
    assert ordered[0] == "gemini-2.5-flash-preview"
    assert ordered[1] == "gemini-2.0-flash"


def test_select_candidates_normalises_the_models_prefix_and_deduplicates():
    entries = [
        {"name": "models/gemini-2.5-flash", "supported_actions": ["generateContent"]},
        {"name": "gemini-2.5-flash", "supported_actions": ["generateContent"]},
    ]
    assert select_candidates(entries) == ["gemini-2.5-flash"]
    assert normalize_model_name("models/gemini-2.5-flash") == "gemini-2.5-flash"


def test_select_candidates_understands_the_older_supported_generation_methods_field():
    # Older google-genai versions / raw REST payloads use this spelling.
    entries = [
        {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
    ]
    assert select_candidates(entries) == ["gemini-2.5-flash"]


def test_select_candidates_alternates_full_flash_and_lite_per_family():
    # A demand spike on the newest full-Flash models must not exhaust the
    # bounded attempt budget before a Flash-Lite candidate is tried.
    entries = [
        gemini_model_entry("gemini-3.8-flash"),
        gemini_model_entry("gemini-3.7-flash"),
        gemini_model_entry("gemini-3.5-flash-lite"),
        gemini_model_entry("gemini-3.1-flash-lite"),
    ]
    assert select_candidates(entries) == [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.7-flash",
        "gemini-3.1-flash-lite",
    ]


def test_select_candidates_excludes_music_transcription_and_image_models():
    # Reliable metadata (name/description) for Lyria (music), Transcribe
    # (speech-to-text) and Nano Banana (image) models in a voice pipeline.
    entries = [
        gemini_model_entry("lyria-3.5", description="Music Generation model"),
        gemini_model_entry("gemini-3.5-transcribe"),
        gemini_model_entry("gemini-nano-banana-2.1"),
        gemini_model_entry("gemini-2.0-flash"),
    ]
    assert select_candidates(entries) == ["gemini-2.0-flash"]


# ------------------------------------------------------------- empty / config
def test_empty_model_list_returns_an_actionable_error_without_crashing():
    service, fake = make_service(entries=[])

    with pytest.raises(ApiError) as exc:
        service.generate(hello())

    assert exc.value.status_code == 503
    assert exc.value.code == "gemini_no_model_available"
    assert "GEMINI" in exc.value.hint
    assert fake.generate_calls == []
    status = service.status()
    assert status["discovery"] == "ok"
    assert status["availability"] == "unavailable"

    # The (empty) result is cached - no model list hammering on every message.
    with pytest.raises(ApiError):
        service.generate(hello())
    assert fake.list_calls == 1


def test_missing_api_key_never_runs_discovery_and_reports_configuration():
    service, fake = make_service(entries=[gemini_model_entry(PREFERRED)], api_key="")

    with pytest.raises(ApiError) as exc:
        service.generate(hello())

    assert exc.value.code == "gemini_missing_credentials"
    assert "GEMINI_API_KEY" in exc.value.message
    assert fake.list_calls == 0
    assert fake.generate_calls == []
    status = service.status()
    assert status["configured"] is False
    assert status["availability"] == "not_checked"
    assert "GEMINI_API_KEY" in status["hint"]


def test_environment_supports_an_empty_preferred_model_and_attempt_limit(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    monkeypatch.setenv("GEMINI_MODEL", "")
    monkeypatch.setenv("GEMINI_MODEL_ATTEMPTS", "5")

    config = Config.from_env()

    assert config.gemini_model == ""          # pure automatic selection
    assert config.gemini_model_attempts == 5


# ------------------------------------------------------------ auth / quota
def test_authentication_failure_during_discovery_is_not_blamed_on_the_model():
    fake = FakeGenAI(
        list_error=errors.APIError(
            401, {"error": {"message": "API key not valid: test-only-key"}}
        )
    )
    service, _ = make_service(client=fake)

    with pytest.raises(ApiError) as exc:
        service.generate(hello())

    assert exc.value.code == "gemini_auth_error"
    assert "unavailable" not in exc.value.message.lower()
    assert "test-only-key" not in exc.value.message
    assert "test-only-key" not in (exc.value.hint or "")
    assert fake.generate_calls == []
    status = service.status()
    assert status["discovery"] == "failed"
    assert status["availability"] == "error"
    assert "GEMINI_API_KEY" in status["hint"]
    assert "test-only-key" not in repr(status)


def test_quota_failure_during_discovery_reports_quota_and_never_generates():
    fake = FakeGenAI(
        list_error=errors.APIError(429, {"error": {"message": "quota exceeded"}})
    )
    service, _ = make_service(client=fake)

    with pytest.raises(ApiError) as exc:
        service.generate(hello())

    assert exc.value.status_code == 429
    assert exc.value.code == "gemini_quota_exceeded"
    assert "quota" in exc.value.message.lower()
    assert fake.generate_calls == []
    assert service.status()["availability"] == "error"


def test_quota_failure_is_reported_as_quota_after_bounded_attempts():
    # Quota must always be reported AS quota - never as an unavailable model -
    # but a model-scoped daily limit must not abort the whole request while
    # other candidates still have quota: tries are bounded, not endless.
    entries = [gemini_model_entry("gemini-2.5-flash"), gemini_model_entry("gemini-2.0-flash")]
    fake = FakeGenAI(
        model_entries=entries,
        generate_error=errors.APIError(429, {"error": {"message": "quota exceeded"}}),
    )
    service, _ = make_service(client=fake)

    with pytest.raises(ApiError) as exc:
        service.generate(hello())

    assert exc.value.status_code == 429
    assert exc.value.code == "gemini_quota_exceeded"
    assert "quota" in exc.value.message.lower()
    assert 1 <= len(fake.generate_calls) <= 3   # small bounded retry limit
    assert fake.list_calls == 1                 # quota never churns discovery
    assert service.status()["availability"] == "available"


def test_model_scoped_quota_falls_through_to_a_model_with_quota():
    # The real Google free-tier shape: "Quota exceeded ... limit: 20, model:
    # gemini-3.8-flash" on one model while another answers immediately.
    per_model_quota = errors.APIError(
        429,
        {"error": {"message": "Quota exceeded for quota metric "
                              "'generate_content_free_tier_requests', limit: 20, "
                              "model: gemini-3.8-flash. Please retry in 5h."}},
    )
    entries = [gemini_model_entry("gemini-3.8-flash"), gemini_model_entry("gemini-3.5-flash-lite")]
    fake = FakeGenAI(model_entries=entries, fail_models={"gemini-3.8-flash": per_model_quota})
    service, _ = make_service(client=fake)

    answer, timing = service.generate(hello())

    assert answer == "A concise answer."
    assert timing["model"] == "gemini-3.5-flash-lite"
    assert [call["model"] for call in fake.generate_calls] == [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
    ]
    assert fake.list_calls == 1            # quota does not invalidate discovery
    # The quota-limited model is NOT remembered as dead: its quota resets on
    # its own schedule, so the next request may try it again (still bounded).


def test_mixed_quota_and_overload_reports_the_recoverable_overload():
    entries = [gemini_model_entry("gemini-a-flash"), gemini_model_entry("gemini-b-flash")]
    fake = FakeGenAI(
        model_entries=entries,
        fail_models={
            "gemini-a-flash": errors.APIError(429, {"error": {"message": "quota exceeded"}}),
            "gemini-b-flash": _overloaded(),
        },
    )
    service, _ = make_service(client=fake)

    with pytest.raises(ApiError) as exc:
        service.generate(hello())

    # A transient overload is the more actionable report (quota resets slowly,
    # overload shortly) - both are still honest, neither is "model unavailable".
    assert exc.value.status_code == 503
    assert exc.value.code == "gemini_overloaded"


def test_authentication_failure_during_generation_stops_immediately():
    entries = [gemini_model_entry("gemini-2.5-flash"), gemini_model_entry("gemini-2.0-flash")]
    fake = FakeGenAI(
        model_entries=entries,
        generate_error=errors.APIError(
            401, {"error": {"message": "invalid key test-only-key"}}
        ),
    )
    service, _ = make_service(client=fake)

    with pytest.raises(ApiError) as exc:
        service.generate(hello())

    assert exc.value.code == "gemini_auth_error"
    assert len(fake.generate_calls) == 1   # never "tries another model"
    status = service.status()
    assert status["availability"] == "error"
    assert "test-only-key" not in repr(status)


# ------------------------------------------------- fallback + cache invalidation
def test_model_not_found_falls_back_to_the_next_candidate_and_refreshes_cache():
    entries = [gemini_model_entry("gemini-2.5-flash"), gemini_model_entry("gemini-2.0-flash")]
    not_found = errors.APIError(404, {"error": {"message": "model not found"}})
    fake = FakeGenAI(model_entries=entries, fail_models={"gemini-2.5-flash": not_found})
    service, _ = make_service(client=fake)

    answer, timing = service.generate(hello())

    assert answer == "A concise answer."
    assert [call["model"] for call in fake.generate_calls] == [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]
    assert service.selected_model == "gemini-2.0-flash"
    assert timing["model"] == "gemini-2.0-flash"
    assert fake.list_calls == 1

    # The model-not-found error invalidated the cache: the next message
    # re-discovers the model list instead of trusting the stale selection.
    service.generate(hello())
    assert fake.list_calls == 2
    assert fake.generate_calls[-1]["model"] == "gemini-2.0-flash"


def test_fallback_respects_the_configurable_attempt_limit():
    entries = [
        gemini_model_entry("gemini-2.5-flash"),
        gemini_model_entry("gemini-2.0-flash"),
        gemini_model_entry("gemini-1.5-flash"),
    ]
    not_found = errors.APIError(404, {"error": {"message": "model not found"}})
    fake = FakeGenAI(
        model_entries=entries,
        fail_models={
            "gemini-2.5-flash": not_found,
            "gemini-2.0-flash": not_found,
            "gemini-1.5-flash": not_found,
        },
    )
    service, _ = make_service(client=fake, attempts=2)

    with pytest.raises(ApiError) as exc:
        service.generate(hello())

    assert exc.value.status_code == 503
    assert exc.value.code == "gemini_model_unavailable"
    assert len(fake.generate_calls) == 2            # small bounded retry limit
    assert exc.value.payload["models_tried"] == ["gemini-2.5-flash", "gemini-2.0-flash"]
    assert "GEMINI_MODEL" in exc.value.hint
    assert "test-only-key" not in exc.value.message


def test_unsupported_model_400_also_triggers_fallback():
    entries = [gemini_model_entry("gemini-2.5-flash"), gemini_model_entry("gemini-2.0-flash")]
    unsupported = errors.APIError(
        400, {"error": {"message": "model gemini-2.5-flash is not supported"}}
    )
    fake = FakeGenAI(model_entries=entries, fail_models={"gemini-2.5-flash": unsupported})
    service, _ = make_service(client=fake)

    service.generate(hello())

    assert fake.generate_calls[-1]["model"] == "gemini-2.0-flash"
    assert service.selected_model == "gemini-2.0-flash"


# ------------------------------------------------- transient overload (5xx)
def _overloaded():
    """The real 503 Google returns during demand spikes."""
    return errors.APIError(
        503,
        {"error": {"message": "This model is currently experiencing high demand. "
                              "Please try again later."}},
    )


def test_overload_advances_to_the_next_candidate_without_churning_the_cache():
    entries = [gemini_model_entry("gemini-2.5-flash"), gemini_model_entry("gemini-2.0-flash")]
    fake = FakeGenAI(model_entries=entries, fail_models={"gemini-2.5-flash": _overloaded()})
    service, _ = make_service(client=fake)

    answer, timing = service.generate(hello())

    assert answer == "A concise answer."
    assert [call["model"] for call in fake.generate_calls] == [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]
    assert timing["model"] == "gemini-2.0-flash"
    assert fake.list_calls == 1            # overload does not invalidate discovery
    assert service.status()["availability"] == "available"


def test_overloaded_full_flash_falls_back_to_flash_lite_within_the_bound():
    # The real-world shape: every full-Flash model answers 503 "high demand"
    # while Flash-Lite answers instantly - the bounded scan must reach it.
    entries = [
        gemini_model_entry("gemini-3.8-flash"),
        gemini_model_entry("gemini-3.5-flash-lite"),
        gemini_model_entry("gemini-3.7-flash"),
    ]
    fake = FakeGenAI(model_entries=entries, fail_models={"gemini-3.8-flash": _overloaded()})
    service, _ = make_service(client=fake, preferred="")

    answer, timing = service.generate(hello())

    assert answer == "A concise answer."
    assert timing["model"] == "gemini-3.5-flash-lite"
    assert [call["model"] for call in fake.generate_calls] == [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
    ]
    assert len(fake.generate_calls) <= 2    # within the default bound (3)


def test_all_candidates_overloaded_reports_overload_not_model_unavailable():
    entries = [gemini_model_entry(f"gemini-2.{n}-flash") for n in range(4)]
    fake = FakeGenAI(model_entries=entries, generate_error=_overloaded())
    service, _ = make_service(client=fake, attempts=2)

    with pytest.raises(ApiError) as exc:
        service.generate(hello())

    assert exc.value.status_code == 503
    assert exc.value.code == "gemini_overloaded"
    assert "overload" in exc.value.message.lower()
    assert "try again" in exc.value.message.lower()
    assert "test-only-key" not in exc.value.message
    assert len(fake.generate_calls) == 2   # bounded, never endless
    assert fake.list_calls == 1            # the model list itself is still valid
    assert service.status()["availability"] == "available"

    # The cache was kept: the next message reuses it instead of re-listing.
    with pytest.raises(ApiError):
        service.generate(hello())
    assert fake.list_calls == 1


def test_discovery_overload_is_reported_as_transient_not_as_a_model_problem():
    fake = FakeGenAI(
        model_entries=[gemini_model_entry(PREFERRED)], list_error=_overloaded()
    )
    service, _ = make_service(client=fake)

    with pytest.raises(ApiError) as exc:
        service.generate(hello())

    assert exc.value.status_code == 503
    assert exc.value.code == "gemini_overloaded"
    assert "unavailable" not in exc.value.message.lower()
    assert fake.generate_calls == []
    status = service.status()
    assert status["availability"] == "error"
    assert "high demand" in (status["hint"] or "").lower()


def test_models_rejected_as_gone_are_skipped_on_the_next_discovery():
    entries = [gemini_model_entry("gemini-2.5-flash"), gemini_model_entry("gemini-2.0-flash")]
    not_found = errors.APIError(404, {"error": {"message": "model not found"}})
    fake = FakeGenAI(model_entries=entries, fail_models={"gemini-2.5-flash": not_found})
    service, _ = make_service(client=fake)

    service.generate(hello())              # 404 on the preferred, then fallback OK
    calls_after_first = len(fake.generate_calls)

    answer, _ = service.generate(hello())  # re-discovers (the 404 invalidated the cache)
    assert answer == "A concise answer."
    assert fake.list_calls == 2
    # The dead preferred model no longer consumes an attempt on the next turn:
    assert len(fake.generate_calls) == calls_after_first + 1
    assert fake.generate_calls[-1]["model"] == "gemini-2.0-flash"
    # ...and it is excluded from the public candidate list as well:
    assert service.discover_models() == ["gemini-2.0-flash"]


# ------------------------------------------------------- request failures
def test_generation_timeout_maps_to_504_without_switching_models():
    entries = [
        gemini_model_entry("gemini-2.5-flash"),
        gemini_model_entry("gemini-2.0-flash"),
    ]
    fake = FakeGenAI(model_entries=entries, generate_error=httpx.ReadTimeout("slow"))
    service, _ = make_service(client=fake)

    with pytest.raises(ApiError) as exc:
        service.generate(hello())

    assert exc.value.status_code == 504
    assert exc.value.code == "llm_timeout"
    assert len(fake.generate_calls) == 1
    assert service.status()["reachable"] is False


def test_transport_failure_maps_to_502_without_switching_models():
    entries = [
        gemini_model_entry("gemini-2.5-flash"),
        gemini_model_entry("gemini-2.0-flash"),
    ]
    fake = FakeGenAI(model_entries=entries, generate_error=httpx.ConnectError("no route"))
    service, _ = make_service(client=fake)

    with pytest.raises(ApiError) as exc:
        service.generate(hello())

    assert exc.value.status_code == 502
    assert exc.value.code == "llm_transport_error"
    assert len(fake.generate_calls) == 1


def test_discovery_timeout_maps_to_504():
    fake = FakeGenAI(list_error=httpx.ReadTimeout("listing models is slow"))
    service, _ = make_service(client=fake)

    with pytest.raises(ApiError) as exc:
        service.generate(hello())

    assert exc.value.status_code == 504
    assert exc.value.code == "llm_timeout"
    assert service.status()["availability"] == "error"


def test_discovery_failure_is_retried_on_the_next_request_not_cached_forever():
    fake = FakeGenAI(
        model_entries=[gemini_model_entry(PREFERRED)],
        list_error=httpx.ConnectError("offline"),
    )
    service, _ = make_service(client=fake)

    with pytest.raises(ApiError) as exc:
        service.generate(hello())
    assert exc.value.code == "llm_transport_error"
    assert service.status()["discovery"] == "failed"

    fake.list_error = None  # network came back
    answer, _ = service.generate(hello())
    assert answer == "A concise answer."
    assert fake.list_calls == 2
    assert service.status()["availability"] == "available"


def test_empty_or_garbled_model_response_is_rejected():
    entries = [gemini_model_entry(PREFERRED)]
    fake = FakeGenAI(
        model_entries=entries,
        response=type("R", (), {"text": "", "usage_metadata": None})(),
    )
    service, _ = make_service(client=fake)

    with pytest.raises(ApiError) as exc:
        service.generate(hello())
    assert exc.value.code == "llm_empty_response"


# ------------------------------------------------- caching and concurrency
def test_model_list_is_cached_across_messages():
    entries = [gemini_model_entry(PREFERRED)]
    service, fake = make_service(entries)

    for _ in range(4):
        service.generate(hello())

    assert fake.list_calls == 1                 # listed once, not per message
    assert len(fake.generate_calls) == 4


def test_invalidate_model_cache_forces_a_fresh_discovery():
    entries = [gemini_model_entry(PREFERRED)]
    service, fake = make_service(entries)

    service.generate(hello())
    assert fake.list_calls == 1
    service.invalidate_model_cache()
    assert service.status()["availability"] == "not_checked"  # honest until re-verified
    service.generate(hello())
    assert fake.list_calls == 2


def test_concurrent_requests_discover_once_and_share_the_selected_model():
    entries = [gemini_model_entry("gemini-2.5-flash"), gemini_model_entry("gemini-2.0-flash")]
    fake = FakeGenAI(model_entries=entries, list_delay=0.05)
    service, _ = make_service(client=fake)

    results: list = []
    failures: list = []
    barrier = threading.Barrier(6)

    def worker():
        try:
            barrier.wait(timeout=10)
            results.append(service.generate(hello()))
        except Exception as exc:  # noqa: BLE001 - collected for assertion
            failures.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert not failures, failures
    assert len(results) == 6
    assert fake.list_calls == 1                 # discovery happened exactly once
    assert len(fake.generate_calls) == 6
    assert {call["model"] for call in fake.generate_calls} == {"gemini-2.5-flash"}
    assert service.status()["availability"] == "available"


def test_discover_models_public_api_is_cached_and_refreshable():
    entries = [gemini_model_entry(PREFERRED)]
    service, fake = make_service(entries)

    assert service.discover_models() == [PREFERRED]
    assert service.discover_models() == [PREFERRED]
    assert fake.list_calls == 1
    assert service.discover_models(refresh=True) == [PREFERRED]
    assert fake.list_calls == 2
