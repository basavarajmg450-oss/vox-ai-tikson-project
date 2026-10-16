"""Chat completions through Google's Gemini API.

The API key is read only by this backend service.  The model is *discovered*
at runtime: the first request calls ``client.models.list()`` for the models
available to the configured key, keeps the ones that support text generation
(``generateContent``), prefers the configured ``GEMINI_MODEL`` when it is among
them, and caches the ordered candidate list so the model list is not fetched
for every message.  A model-not-found or unsupported-model error invalidates
that cache, remembers the rejected model for this process (so a dead preferred
model is not retried on every message) and the request falls back to the next
suitable discovered candidate.  Transient overload answers (HTTP 5xx, "high
demand") also advance to the next candidate but never churn the cache or
change provider.  Model rejections (unavailable or quota-limited candidates)
and overload rejections are each bounded by ``GEMINI_MODEL_ATTEMPTS`` -
retries are never endless.  Authentication errors stop immediately; quota is
reported as quota (429 ``gemini_quota_exceeded``, never as an unavailable
model) whenever it is what blocks the answer.  Requests use a bounded recent
conversation, a finite timeout and no automatic SDK retries.  Only the
concise answer text is returned to the pipeline.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from google import genai
from google.genai import errors, types

from config import Config
from services.errors import ApiError

_TAG_ZW = r"[​‌﻿]*"
_THINK_TAG = rf"<{_TAG_ZW}/?{_TAG_ZW}{_TAG_ZW}think{_TAG_ZW}>"
_THINK_BLOCK_RE = re.compile(rf"\s*{_THINK_TAG}[\s\S]*?{_THINK_TAG}\s*", re.IGNORECASE)
_THINK_UNCLOSED_RE = re.compile(rf"\s*{_THINK_TAG}[\s\S]*$", re.IGNORECASE)

#: The capability a conversational text model must advertise.
GENERATE_ACTION = "generateContent"

#: Google API keys look like this; scrubbed from any text that may surface.
_API_KEY_RE = re.compile(r"AIza[0-9A-Za-z_\-]{16,}")

#: Names that mark a model as unusable for a *voice* pipeline even when the
#: capability metadata is ambiguous (embedding output, image/video/audio
#: generation, speech synthesis, music generation, transcription, ...).
_UNSUITABLE_NAME_RE = re.compile(
    r"embedding|imagen|veo|image|video|audio|\btts\b|speech|lyria|transcribe|banana",
    re.IGNORECASE,
)

#: Preview/experimental/deprecated names are never chosen automatically when
#: a stable alternative exists (they stay eligible only as a last resort, or
#: when the operator pinned them explicitly through ``GEMINI_MODEL``).
_EXPERIMENTAL_NAME_RE = re.compile(
    r"-preview|-experimental|-nightly|-unstable|-alpha\b|-beta\b|-exp\b",
    re.IGNORECASE,
)
_DEPRECATED_DESC_RE = re.compile(
    r"\bdeprecated\b|no longer (?:available|supported)|being retired",
    re.IGNORECASE,
)

_VERSION_RE = re.compile(r"(\d+)\.(\d+)")

#: Actionable hints shown by ``status()`` for each discovery failure code.
_DISCOVERY_HINTS = {
    "gemini_auth_error": (
        "Check or replace GEMINI_API_KEY in .env and verify the key is "
        "enabled for the Gemini API."
    ),
    "gemini_quota_exceeded": (
        "Check your Google AI Studio project, usage limits and quota reset time."
    ),
    "llm_timeout": "Try again or increase GEMINI_TIMEOUT_S in .env.",
    "gemini_overloaded": (
        "Gemini is under high demand right now; wait a moment and send "
        "another message to retry."
    ),
    "llm_transport_error": (
        "Check the backend's network connection to the Gemini API."
    ),
    "gemini_discovery_failed": (
        "Model discovery failed - check GEMINI_API_KEY and Google AI Studio "
        "service status, then send another message to retry."
    ),
    "default": (
        "Model discovery has not completed yet; send a message to verify "
        "which Gemini models this API key can use."
    ),
}


def strip_thinking(text: str) -> str:
    """Remove tagged reasoning traces before text is displayed or spoken."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_UNCLOSED_RE.sub("", cleaned)
    return cleaned.strip()


def cap_answer(text: str, max_tokens: int) -> str:
    """Bound visible answer length and cut at a sentence boundary when possible."""
    limit = max(64, int(max_tokens) * 4)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in (". ", "! ", "? ", "\u0964 ", "\n"):
        idx = cut.rfind(sep)
        if idx > limit // 2:
            return cut[: idx + 1].strip()
    return cut.rsplit(" ", 1)[0].strip() + "."


# ---------------------------------------------------------------------------
# Model discovery helpers (pure functions - unit tested without the SDK).
def _entry_get(entry: Any, *keys: str) -> Any:
    """Read a field from a ``Model`` object *or* a plain dict (mocks/tests)."""
    for key in keys:
        if isinstance(entry, dict):
            value = entry.get(key)
        else:
            value = getattr(entry, key, None)
        if value is not None:
            return value
    return None


def normalize_model_name(name: Any) -> str:
    """Strip the ``models/`` resource prefix the list API returns."""
    text = str(name or "").strip()
    if text.startswith("models/"):
        text = text[len("models/"):]
    return text


def model_supports_text_generation(entry: Any) -> Optional[bool]:
    """Does the model's metadata prove it supports ``generateContent``?

    Returns ``True``/``False`` when capability metadata is present and
    ``None`` when it is missing entirely (cannot be verified either way).
    Handles both SDK field spellings (``supported_actions`` in current
    ``google-genai``, ``supported_generation_methods`` in older ones and raw
    REST payloads).
    """
    actions = _entry_get(
        entry,
        "supported_actions",
        "supported_generation_methods",
        "supportedGenerationMethods",
    )
    if actions is None:
        return None
    try:
        values = [str(action) for action in actions]
    except TypeError:
        return None
    if not values:
        return None
    return GENERATE_ACTION in values


def _entry_description(entry: Any) -> str:
    return str(_entry_get(entry, "description") or "")


def _is_experimental(name: str, entry: Any) -> bool:
    if _EXPERIMENTAL_NAME_RE.search(name):
        return True
    return bool(_DEPRECATED_DESC_RE.search(_entry_description(entry)))


def _model_version(name: str) -> Tuple[int, int]:
    match = _VERSION_RE.search(name)
    if not match:
        return (-1, -1)
    return (int(match.group(1)), int(match.group(2)))


def _automatic_rank(name: str, *, experimental: bool, verified: bool) -> Tuple:
    """Sort key: verified before unverified, stable before preview/deprecated,
    then Flash (non-lite first), then other Gemini models, newest first."""
    lower = name.lower()
    if "flash" in lower:
        tier = 0 if "lite" not in lower else 1
    elif "gemini" in lower:
        tier = 2
    else:
        tier = 3
    major, minor = _model_version(lower)
    return (
        0 if verified else 1,
        1 if experimental else 0,
        tier,
        -major,
        -minor,
        name,
    )


def select_candidates(entries: Iterable[Any], preferred: str = "") -> List[str]:
    """Ordered candidate models for a conversational text pipeline.

    Order: the explicitly preferred model (when discovered and suitable) >
    verified stable models (stable Flash first - alternating full Flash with
    Flash-Lite so a demand spike on one family cannot exhaust the bounded
    attempt budget before the other family is tried - then other Gemini
    models, newest first) > models whose capability metadata is missing.
    Models that do not advertise ``generateContent``, embedding/image/video/
    audio/music/transcription-only models and preview/experimental/deprecated
    models are never returned automatically (a preview model may still be
    pinned explicitly through ``GEMINI_MODEL``).
    """
    wanted = normalize_model_name(preferred).lower()
    preferred_hit: Optional[str] = None
    pool: List[Tuple[Tuple, str]] = []
    seen = set()

    for entry in entries:
        name = normalize_model_name(_entry_get(entry, "name"))
        if not name or name in seen:
            continue
        seen.add(name)
        if _UNSUITABLE_NAME_RE.search(name):
            continue
        supports = model_supports_text_generation(entry)
        if supports is False:
            continue
        if wanted and name.lower() == wanted:
            # Explicit operator preference: honoured as long as the model is
            # discovered and is not a text-generation-impossible one.
            preferred_hit = name
            continue
        if _is_experimental(name, entry):
            # Preview/experimental/deprecated models are excluded from
            # automatic selection; pin one via GEMINI_MODEL if needed.
            continue
        verified = supports is True
        pool.append((_automatic_rank(name, experimental=False, verified=verified), name))

    pool.sort(key=lambda item: item[0])
    ordered: List[str] = []
    if preferred_hit:
        ordered.append(preferred_hit)

    # Round-robin across the tiers (full Flash, Flash-Lite, other Gemini,
    # other) inside each verified group: a demand spike or a model-local
    # quota on the newest full-Flash models must not consume the whole bounded
    # attempt budget before a Flash-Lite (or other) candidate is even tried.
    def _round_robin(rows: List[Tuple[Tuple, str]]) -> List[str]:
        buckets: Dict[int, List[str]] = {}
        for rank, name in rows:
            buckets.setdefault(rank[2], []).append(name)
        tiers = sorted(buckets)
        out: List[str] = []
        index = 0
        while True:
            added = False
            for tier in tiers:
                bucket = buckets[tier]
                if index < len(bucket):
                    out.append(bucket[index])
                    added = True
            if not added:
                return out
            index += 1

    verified_rows = [row for row in pool if row[0][0] == 0]
    unverified_rows = [row for row in pool if row[0][0] != 0]
    for name in _round_robin(verified_rows) + _round_robin(unverified_rows):
        if name not in ordered:
            ordered.append(name)
    return ordered


# ---------------------------------------------------------------------------
# Gemini API error classification (shared by discovery and generation).
def _api_status(exc: Exception) -> int:
    code = getattr(exc, "code", None)
    try:
        return int(code) if code is not None else 0
    except (TypeError, ValueError):
        return 0


def _api_message(exc: Exception) -> str:
    return (getattr(exc, "message", "") or "").lower()


def _is_quota_error(exc: Exception) -> bool:
    status = _api_status(exc)
    if status == 429:
        return True
    if status == 403:
        msg = _api_message(exc)
        return any(word in msg for word in ("quota", "rate limit", "rate_limit", "resource_exhausted"))
    return False


def _is_auth_error(exc: Exception) -> bool:
    status = _api_status(exc)
    if status in (401, 403):
        return True
    if status == 400:
        msg = _api_message(exc)
        return any(
            word in msg
            for word in ("api key", "api_key", "credential", "unauthenticated")
        )
    return False


def _is_model_not_found_error(exc: Exception) -> bool:
    """Model/unsupported errors that justify trying the next candidate."""
    status = _api_status(exc)
    if status == 404:
        return True
    if status != 400:
        return False
    msg = _api_message(exc)
    if "model" not in msg:
        return False
    return any(
        word in msg
        for word in (
            "not found",
            "unsupported",
            "not supported",
            "does not exist",
            "unavailable",
            "disabled",
            "no longer",
        )
    )


def _is_overload_error(exc: Exception) -> bool:
    """Transient overload (5xx / "high demand"): not a model, credential or
    quota problem - the request may advance to the next candidate within the
    bounded retry budget."""
    if _api_status(exc) in (500, 502, 503):
        return True
    msg = _api_message(exc)
    return any(
        word in msg
        for word in (
            "overloaded",
            "high demand",
            "temporarily unavailable",
            "try again later",
        )
    )


def _is_bad_request_error(exc: Exception) -> bool:
    """HTTP 400 that is *not* an auth, model-not-found or quota rejection.

    These are request-body problems (e.g. an unsupported ``thinking_config``
    on a model that does not honour it).  They are never retried on the same
    candidate and never invalidate the model cache - the caller simply moves
    to the next candidate in the bounded budget.
    """
    if _api_status(exc) != 400:
        return False
    if _is_auth_error(exc) or _is_model_not_found_error(exc) or _is_quota_error(exc):
        return False
    return True


class LLMService:
    """Gemini text generation with backend-only credentials and runtime
    model discovery."""

    name = "gemini"

    def __init__(self, config: Config, client: Optional[Any] = None) -> None:
        self.config = config
        self._client = client
        if self._client is None and config.gemini_api_key:
            self._client = genai.Client(
                api_key=config.gemini_api_key,
                http_options=types.HttpOptions(
                    timeout=int(config.gemini_timeout_s * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        # Discovery/selection state.  Writes happen under ``_model_lock`` so
        # concurrent requests never list models repeatedly or race the switch.
        self._model_lock = threading.Lock()
        self._candidates: Optional[List[str]] = None  # None = not discovered yet
        self._selected_model: Optional[str] = None
        # Models the API rejected as gone (404/unsupported) during this
        # process: excluded from later candidate lists so a dead preferred
        # model is not retried (and does not consume an attempt) per message.
        self._failed_models: set = set()
        self._discovery_state = "not_attempted"  # not_attempted | ok | failed
        self._discovery_error: Optional[str] = None   # sanitised, never the key
        self._discovery_error_code: Optional[str] = None
        self._reachable: Optional[bool] = None

    # -- helpers -----------------------------------------------------------
    def _sanitize(self, text: Any) -> str:
        """Strip anything resembling the API key from externally visible text."""
        out = str(text or "")
        key = (self.config.gemini_api_key or "").strip()
        if key:
            out = out.replace(key, "[redacted]")
        return _API_KEY_RE.sub("[redacted]", out)

    def _mark_discovery_failed(self, code: str, message: str) -> None:
        self._discovery_state = "failed"
        self._discovery_error_code = code
        self._discovery_error = self._sanitize(message)

    # -- public model-selection API ---------------------------------------
    @property
    def selected_model(self) -> Optional[str]:
        """Model chosen by discovery (or the last one that answered)."""
        return self._selected_model

    def invalidate_model_cache(self) -> None:
        """Force the next request to re-run ``models.list()``."""
        with self._model_lock:
            self._candidates = None

    def discover_models(self, refresh: bool = False) -> List[str]:
        """Ordered text-generation models available to this API key.

        Cached after the first successful call; ``refresh=True`` forces a new
        ``models.list()`` request.  Raises :class:`ApiError` when discovery
        fails (auth, quota, network, ...).
        """
        if refresh:
            self.invalidate_model_cache()
        return self._ordered_candidates()

    def status(self) -> Dict[str, Any]:
        """Report configuration and *already known* discovery status.

        Never makes an API request: until model discovery has actually run
        and succeeded, ``availability`` stays ``not_checked``.
        """
        configured = bool(self.config.gemini_api_key)
        if self._discovery_state == "failed":
            availability = "error"
        elif self._discovery_state == "ok":
            if self._candidates is None:
                # Cache invalidated after a model-not-found error: nothing is
                # verified again until the next request re-discovers.
                availability = "not_checked"
            elif not self._usable_candidates():
                # Either discovery found nothing or every discovered model has
                # since been rejected by the API - never claim availability.
                availability = "unavailable"
            else:
                availability = "available"
        else:
            availability = "not_checked"

        if not configured:
            hint = "Set GEMINI_API_KEY in the backend .env file."
        elif availability == "error":
            hint = _DISCOVERY_HINTS.get(
                self._discovery_error_code or "", _DISCOVERY_HINTS["gemini_discovery_failed"]
            )
        elif availability == "unavailable":
            hint = (
                "Model discovery ran but no usable text-generation model was "
                "returned for this API key. Check the key's restrictions or "
                "set GEMINI_MODEL to a model enabled for the key."
            )
        else:
            hint = None

        return {
            "name": self.name,
            "model": self._selected_model or self.config.gemini_model,
            "configured": configured,
            "availability": availability,
            "reachable": self._reachable,
            "hint": hint,
            "preferred_model": self.config.gemini_model,
            "selected_model": self._selected_model,
            "discovery": self._discovery_state,
            "discovery_error": self._discovery_error,
        }

    # -- discovery ---------------------------------------------------------
    def _remember_failed_model(self, name: str) -> None:
        """Remember a model the API rejected as gone so later requests skip it."""
        with self._model_lock:
            self._failed_models.add(name)

    def _usable_candidates(self) -> List[str]:
        """Cached candidates minus models already rejected as unavailable."""
        with self._model_lock:
            dead = set(self._failed_models)
            cached = self._candidates
        if cached is None:
            return []
        return [name for name in cached if name not in dead]

    def _ordered_candidates(self) -> List[str]:
        with self._model_lock:
            cached = self._candidates
        if cached is not None:
            return self._usable_candidates()
        with self._model_lock:
            if self._candidates is None:
                entries = self._list_models()          # may raise ApiError
                candidates = select_candidates(entries, self.config.gemini_model)
                dead = set(self._failed_models)
                candidates = [name for name in candidates if name not in dead]
                self._candidates = candidates
                self._discovery_state = "ok"
                self._discovery_error = None
                self._discovery_error_code = None
                if candidates:
                    self._selected_model = candidates[0]
        return self._usable_candidates()

    def _list_models(self) -> List[Any]:
        """Run ``models.list()`` once; classify failures as ApiErrors."""
        if self._client is None:
            error = ApiError(
                "Gemini model discovery cannot run because the client is not initialised.",
                502,
                "gemini_discovery_failed",
                hint="Restart VoxAI with GEMINI_API_KEY set in .env.",
            )
            self._mark_discovery_failed(error.code, error.message)
            raise error
        try:
            pager = self._client.models.list()
            entries = list(pager) if pager is not None else []
        except Exception as exc:  # noqa: BLE001 - mapped to a typed ApiError
            raise self._discovery_error_for(exc) from exc
        return entries

    def _discovery_error_for(self, exc: Exception) -> ApiError:
        """Map a ``models.list()`` failure to a safe, actionable ApiError."""
        if isinstance(exc, httpx.TimeoutException):
            self._reachable = False
            error = ApiError(
                f"Gemini did not respond within {self.config.gemini_timeout_s:.0f} seconds "
                "while listing the models available to this API key.",
                504,
                "llm_timeout",
                hint="Try again or increase GEMINI_TIMEOUT_S in .env.",
            )
        elif isinstance(exc, httpx.HTTPError):
            self._reachable = False
            error = ApiError(
                "Could not connect to the Gemini API to discover the models available "
                "to this API key.",
                502,
                "llm_transport_error",
                hint="Check the network connection and try again.",
            )
        elif isinstance(exc, errors.APIError):
            # The API answered: authentication/quota are reported as such and
            # never described as "the model is unavailable".
            self._reachable = True
            if _is_quota_error(exc):
                error = ApiError(
                    "Gemini API quota or rate limit reached while listing available models.",
                    429,
                    "gemini_quota_exceeded",
                    hint="Check your Google AI Studio project, usage limits and quota reset time.",
                )
            elif _is_auth_error(exc):
                error = ApiError(
                    "Gemini rejected the API credentials or they do not have access.",
                    502,
                    "gemini_auth_error",
                    hint=(
                        "Check or replace GEMINI_API_KEY in .env and verify the key is "
                        "enabled for the Gemini API."
                    ),
                )
            elif _is_overload_error(exc):
                # Transient "high demand" / 5xx: not a model or key problem.
                error = ApiError(
                    "Gemini is temporarily overloaded while listing the models "
                    "for this API key.",
                    503,
                    "gemini_overloaded",
                    hint=(
                        "Gemini is under high demand right now; wait a moment and "
                        "send another message to retry."
                    ),
                )
            else:
                error = ApiError(
                    "Gemini model discovery was rejected by the API.",
                    502,
                    "gemini_discovery_failed",
                    hint="Check GEMINI_API_KEY and Google AI Studio service status, then retry.",
                )
        else:
            error = ApiError(
                f"Model discovery failed unexpectedly ({type(exc).__name__}).",
                502,
                "gemini_discovery_failed",
                hint="Check GEMINI_API_KEY and Google AI Studio service status, then retry.",
            )
        self._mark_discovery_failed(error.code, error.message)
        return error

    # -- generation --------------------------------------------------------
    def _config_with_thinking(
        self, base_config: Dict[str, Any], budget: int
    ) -> Any:
        """Return a GenerateContentConfig with an explicit thinking budget.

        ``budget >= 0`` caps hidden reasoning; some models reject an explicit
        budget, so the caller must be prepared to fall back to the plain
        config (see the ``_is_bad_request_error`` branch in ``generate``).
        Falls back to a plain config when the installed SDK has no
        ``ThinkingConfig``.
        """
        try:
            return types.GenerateContentConfig(
                **base_config,
                thinking_config=types.ThinkingConfig(thinking_budget=budget),
            )
        except Exception:  # noqa: BLE001 - older SDK without ThinkingConfig
            return types.GenerateContentConfig(**base_config)

    def generate(
        self,
        history: Sequence[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Run one request and return ``(answer_text, measured_usage)``.

        Discovery, filtering and caching happen inside ``_ordered_candidates``
        (once per process unless invalidated).  Model rejections
        (unavailable or quota-limited candidates) and transient overload
        answers (5xx "high demand") each advance to the next discovered
        candidate, each bounded by ``GEMINI_MODEL_ATTEMPTS`` so retries are
        never endless.  Authentication failures stop immediately; when quota
        is what blocks the answer, the request fails as
        ``gemini_quota_exceeded`` (429) - never as an unavailable model.
        """
        if not self.config.gemini_api_key:
            raise ApiError(
                "Gemini is not configured because GEMINI_API_KEY is missing.",
                503,
                "gemini_missing_credentials",
                hint="Add a valid GEMINI_API_KEY to the backend .env file and restart VoxAI.",
            )

        temperature = self.config.llm_temperature if temperature is None else float(temperature)
        max_tokens = self.config.max_response_tokens if max_tokens is None else int(max_tokens)
        history_limit = self.config.max_history_turns * 2
        recent_history = list(history[-history_limit:]) if history_limit else []
        contents: List[Dict[str, Any]] = [
            {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            }
            for message in recent_history
        ]
        base_config = {
            "system_instruction": system_prompt or self.config.system_prompt,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        config = types.GenerateContentConfig(**base_config)
        # Cap (or disable) Gemini's hidden reasoning.  For short spoken answers
        # thinking adds latency and can consume MAX_RESPONSE_TOKENS entirely,
        # which surfaced as occasional empty answers.  -1 keeps provider default.
        # The budget is attached *per model* below: "lite" models have mandatory
        # thinking and reject thinking_budget=0 with a 400, so they always get
        # the provider default instead.
        thinking_budget = int(getattr(self.config, "gemini_thinking_budget", 0))

        # Include first-call discovery so ``llm_ms`` honestly reflects the
        # wall time the caller actually waited.
        started = time.perf_counter()
        candidates = self._ordered_candidates()
        if not candidates:
            raise ApiError(
                "Gemini model discovery found no model that supports text generation "
                "for this API key.",
                503,
                "gemini_no_model_available",
                hint=(
                    "Check that GEMINI_API_KEY is a valid Google AI Studio key with Gemini "
                    "API access, or set GEMINI_MODEL to a model enabled for the key."
                ),
            )

        attempts = max(1, int(self.config.gemini_model_attempts))
        tried: List[str] = []
        response: Any = None
        used_model: Optional[str] = None
        model_rejections = 0      # candidates rejected as unavailable/quota
        overload_rejections = 0   # transient 5xx "high demand" answers
        overload_seen = False
        quota_seen = False
        bad_request_seen = False  # a candidate rejected the request body (400)

        for model in candidates:
            if model_rejections >= attempts or overload_rejections >= attempts:
                break
            tried.append(model)
            # Build this model's config.  "lite" models do not honour an
            # explicit zero thinking budget (they reject it with HTTP 400), so
            # they always use the provider default; other models get the
            # configured budget when it is set.
            if thinking_budget >= 0 and "lite" not in model.lower():
                model_config = self._config_with_thinking(base_config, thinking_budget)
            else:
                model_config = config
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=model_config,
                )
                used_model = model
                break
            except httpx.TimeoutException as exc:
                self._reachable = False
                raise ApiError(
                    f"Gemini did not respond within {self.config.gemini_timeout_s:.0f} seconds.",
                    504,
                    "llm_timeout",
                    hint="Try again or increase GEMINI_TIMEOUT_S in .env.",
                ) from exc
            except httpx.HTTPError as exc:
                self._reachable = False
                raise ApiError(
                    "Could not connect to the Gemini API.",
                    502,
                    "llm_transport_error",
                    hint="Check the network connection and try again.",
                ) from exc
            except errors.APIError as exc:
                self._reachable = True  # the API answered, even if it said no
                if _is_quota_error(exc):
                    # A single candidate's quota/rate limit is exhausted
                    # (per-model free-tier limits reset on their own
                    # schedule).  Skip it within the same bounded budget so a
                    # model *with* quota can still answer; when quota blocks
                    # every allowed attempt it is reported as quota - never
                    # as an unavailable model and never by endless churn.
                    quota_seen = True
                    model_rejections += 1
                    continue
                if _is_auth_error(exc):
                    error = ApiError(
                        "Gemini rejected the API credentials or they do not have access.",
                        502,
                        "gemini_auth_error",
                        hint=(
                            "Check or replace GEMINI_API_KEY in .env and verify the key is "
                            "enabled for the Gemini API."
                        ),
                    )
                    # The key no longer works: stop claiming any model is available.
                    self._mark_discovery_failed(error.code, error.message)
                    raise error from exc
                if _is_model_not_found_error(exc):
                    # Unsupported/unavailable model: remember it for this
                    # process (so it is not retried on every message), drop the
                    # cached selection so the next request re-discovers, and
                    # try the next candidate within the attempt bound.
                    model_rejections += 1
                    self._remember_failed_model(model)
                    self.invalidate_model_cache()
                    continue
                if _is_overload_error(exc):
                    # Transient overload: advance to the next candidate within
                    # its own bounded budget; do not churn the model cache and
                    # never report this as "the model is unavailable".
                    overload_seen = True
                    overload_rejections += 1
                    continue
                if _is_bad_request_error(exc):
                    # A candidate rejected the request body itself - most often
                    # an unsupported thinking_config on a model that does not
                    # honour it.  Do not retry this candidate, do not churn the
                    # cache, and move to the next model in the bounded budget so
                    # a single bad pairing cannot fail the whole request.
                    bad_request_seen = True
                    model_rejections += 1
                    continue
                raise ApiError(
                    "The Gemini API could not complete the request.",
                    502,
                    "gemini_api_error",
                    hint="Check Google AI Studio service status and try again.",
                ) from exc

        if response is None or used_model is None:
            if overload_seen:
                # Transient overload was among the failures: report that
                # (recoverable shortly) and keep the discovered cache - the
                # model list itself is still valid.
                raise ApiError(
                    "Gemini is temporarily overloaded and did not answer within "
                    f"the bounded retry limit ({attempts}). Please try again shortly.",
                    503,
                    "gemini_overloaded",
                    hint=(
                        "Gemini is under high demand right now; wait a moment and "
                        "send the message again."
                    ),
                    payload={"models_tried": tried},
                )
            if quota_seen:
                # Quota is what actually blocked the answer.  The attempts
                # were bounded, so this is a report - not model churn.
                raise ApiError(
                    "Gemini's quota or rate limit was reached for every model "
                    f"tried in this request ({', '.join(tried)}).",
                    429,
                    "gemini_quota_exceeded",
                    hint=(
                        "Check Google AI Studio usage and quota reset times, wait "
                        "for the reset, or set GEMINI_MODEL to a model with "
                        "remaining quota."
                    ),
                    payload={"models_tried": tried},
                )
            # Every attempted candidate was rejected as unavailable/unsupported.
            self.invalidate_model_cache()
            raise ApiError(
                "Model discovery completed, but Gemini rejected the tried model(s) "
                f"as unavailable or unsupported: {', '.join(tried)}.",
                503,
                "gemini_model_unavailable",
                hint=(
                    "Set GEMINI_MODEL to a model enabled for your Google AI Studio API key "
                    "(a stable Gemini Flash model is recommended) and restart VoxAI."
                ),
                payload={"models_tried": tried},
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._selected_model = used_model
        if self._discovery_state != "ok":  # pragma: no cover - defensive
            self._discovery_state = "ok"

        try:
            raw_text = getattr(response, "text", None) or ""
        except (AttributeError, ValueError):
            raw_text = ""
        text = cap_answer(strip_thinking(raw_text).strip(), max_tokens)
        if not text:
            raise ApiError(
                "Gemini returned an empty answer.",
                502,
                "llm_empty_response",
                hint="Try rephrasing the message.",
            )

        usage = getattr(response, "usage_metadata", None)
        return text, {
            "llm_ms": round(elapsed_ms, 1),
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "completion_tokens": getattr(usage, "candidates_token_count", None),
            "model": getattr(response, "model_version", None) or used_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
