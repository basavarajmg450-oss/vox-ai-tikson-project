"""LLM access through a local Ollama server.

Key behaviours:
* Reasoning is kept out of the user-facing answer: the ``think`` flag is left
  at the model default, so Ollama returns any chain-of-thought in the
  separate ``message.thinking`` field (discarded here) while ``content``
  holds only the answer.  (Empirically, ``think=False`` on the qwen3:4b
  build used here makes the model write its reasoning straight into
  ``content`` - see docs/architecture.md.)  Tagged ``<think>`` leaks are
  stripped from the answer as a second line of defence.
* Output length: the *visible* answer is bounded by
  ``MAX_RESPONSE_TOKENS`` (enforced by :func:`cap_answer`); the server-side
  ``num_predict`` includes reasoning headroom - see :data:`THINK_HEADROOM_TOKENS`.
* Sampling temperature is configurable and validated.
* Ollama-down, model-missing and timeout conditions raise an
  :class:`~services.errors.ApiError` with an actionable hint.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import ollama
from ollama import ResponseError

from config import Config
from services.errors import ApiError

#: `` / `` blocks (Qwen3 reasoning), including an unclosed
#: trailing block, are removed from the model output.  Some tokenizers emit
#: zero-width characters inside the tags, so the tag pattern tolerates them.
_TAG_ZW = r"[​‌﻿]*"
_THINK_TAG = rf"<{_TAG_ZW}/?{_TAG_ZW}{_TAG_ZW}think{_TAG_ZW}>"
_THINK_BLOCK_RE = re.compile(rf"\s*{_THINK_TAG}[\s\S]*?{_THINK_TAG}\s*", re.IGNORECASE)
_THINK_UNCLOSED_RE = re.compile(rf"\s*{_THINK_TAG}[\s\S]*$", re.IGNORECASE)


#: Qwen3 reasons before answering, and Ollama counts those (discarded)
#: reasoning tokens against ``num_predict``.  The cap sent to the server
#: therefore includes headroom so the answer is never starved to empty;
#: ``cap_answer`` re-imposes the real MAX_RESPONSE_TOKENS bound locally.
THINK_HEADROOM_TOKENS = 1024


def strip_thinking(text: str) -> str:
    """Remove reasoning traces so only the answer remains."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_UNCLOSED_RE.sub("", cleaned)
    cleaned = cleaned.replace("/no_think", "").strip()
    return cleaned.strip()


def cap_answer(text: str, max_tokens: int) -> str:
    """Hard-bound the visible answer to ~max_tokens (4 chars/token).

    Cuts on a sentence boundary where possible so the text - and the speech
    synthesised from it - never ends mid-word.  The Hindi danda (``\u0964``)
    counts as a sentence end too.
    """
    limit = max(64, int(max_tokens) * 4)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in (". ", "! ", "? ", "\u0964 ", "\n"):
        idx = cut.rfind(sep)
        if idx > limit // 2:
            return cut[: idx + 1].strip()
    return cut.rsplit(" ", 1)[0].strip() + "."


def normalize_model_name(name: str) -> Tuple[str, str]:
    """Split ``qwen3:4b`` / ``qwen3:4b:latest`` into (base, tag)."""
    parts = (name or "").strip().lower().split(":")
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], ":".join(parts[1:])


def model_matches(available: str, wanted: str) -> bool:
    """True when an available model tag satisfies the requested model."""
    avail_base, avail_tag = normalize_model_name(available)
    want_base, want_tag = normalize_model_name(wanted)
    if avail_base != want_base:
        return False
    if not want_tag:
        return True
    return avail_tag in {want_tag, "latest"} or avail_tag == want_tag


class LLMService:
    """Chat completion against a local Ollama server."""

    name = "ollama"

    def __init__(self, config: Config, client: Optional[Any] = None) -> None:
        self.config = config
        # One shared client; the timeout covers a single /api/chat call.
        self._client = client or ollama.Client(
            host=config.ollama_host,
            timeout=config.ollama_timeout_s,
        )
        self._state: str = "not_checked"
        self._last_error: Optional[str] = None

    @property
    def state(self) -> str:
        return self._state

    # -- health -----------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        """Check reachability and whether the configured model is present."""
        started = time.perf_counter()
        try:
            listing = self._client.list()
        except Exception as exc:  # noqa: BLE001 - any client error means "unreachable"
            self._state = "unreachable"
            self._last_error = f"{type(exc).__name__}: {exc}"
            return {
                "name": self.name,
                "host": self.config.ollama_host,
                "model": self.config.ollama_model,
                "reachable": False,
                "model_present": False,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "error": self._last_error,
                "hint": (
                    f"Start the local server with:  ollama serve   "
                    f"(expected at {self.config.ollama_host})"
                ),
            }

        names = [getattr(m, "model", "") or "" for m in getattr(listing, "models", [])]
        present = any(model_matches(name, self.config.ollama_model) for name in names)
        self._state = "ready" if present else "model_missing"
        self._last_error = None
        return {
            "name": self.name,
            "host": self.config.ollama_host,
            "model": self.config.ollama_model,
            "reachable": True,
            "model_present": present,
            "available_models": names,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "error": None,
            "hint": None if present else (
                f"Download the model with:  ollama pull {self.config.ollama_model}"
            ),
        }

    # -- generation ----------------------------------------------------------
    def generate(
        self,
        history: Sequence[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Run one chat turn. Returns ``(answer_text, timing)``.

        ``history`` must already be bounded (see SessionStore); the system
        prompt is prepended here.  Only the answer text is returned -
        reasoning is suppressed with ``think=False`` and stripped afterwards.
        """
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt or self.config.system_prompt}]
        messages.extend({"role": m["role"], "content": m["content"]} for m in history)

        temperature = self.config.llm_temperature if temperature is None else float(temperature)
        max_tokens = self.config.max_response_tokens if max_tokens is None else int(max_tokens)

        options = {
            "temperature": temperature,
            "num_predict": max_tokens + THINK_HEADROOM_TOKENS,
        }

        started = time.perf_counter()
        try:
            response = self._client.chat(
                model=self.config.ollama_model,
                messages=messages,
                stream=False,
                # `think` intentionally omitted (model default): with this
                # qwen3:4b build, `think=False` writes reasoning into
                # `content`, while the default keeps reasoning in
                # `message.thinking` - which we never read - leaving only
                # the user-facing answer in `content`.
                options=options,
            )
        except ResponseError as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            if exc.status_code == 404:
                raise ApiError(
                    f"The model '{self.config.ollama_model}' is not available on the "
                    f"local Ollama server.",
                    503,
                    "model_missing",
                    hint=f"Download it once with:  ollama pull {self.config.ollama_model}",
                ) from exc
            raise ApiError(
                f"Ollama returned an error ({exc.status_code}): {exc.error}",
                502,
                "llm_error",
                hint=f"Check the Ollama logs for {self.config.ollama_host}.",
            ) from exc
        except httpx.TimeoutException as exc:
            raise ApiError(
                f"The model did not answer within {self.config.ollama_timeout_s:.0f}s.",
                504,
                "llm_timeout",
                hint="Use a smaller model, reduce MAX_RESPONSE_TOKENS, or raise OLLAMA_TIMEOUT_S.",
            ) from exc
        except ConnectionError as exc:
            raise ApiError(
                f"Ollama is not reachable at {self.config.ollama_host}.",
                503,
                "ollama_unreachable",
                hint="Start it with:  ollama serve",
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(
                f"Could not talk to Ollama ({type(exc).__name__}: {exc}).",
                502,
                "llm_transport_error",
                hint=f"Is something else listening on {self.config.ollama_host}?",
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        message = getattr(response, "message", None)
        raw_content = (getattr(message, "content", None) or "")
        # Never expose reasoning: `message.thinking` (if any) is dropped by
        # simply not reading it; strip tagged blocks that leaked into content,
        # then enforce the MAX_RESPONSE_TOKENS bound on the visible answer.
        text = cap_answer(strip_thinking(raw_content).strip(), max_tokens)
        if not text:
            raise ApiError(
                "The model returned an empty answer.",
                502,
                "llm_empty_response",
                hint="Try rephrasing, or increase MAX_RESPONSE_TOKENS.",
            )

        timing = {
            "llm_ms": round(elapsed_ms, 1),
            "prompt_tokens": getattr(response, "prompt_eval_count", None),
            "completion_tokens": getattr(response, "eval_count", None),
            "model": self.config.ollama_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return text, timing
