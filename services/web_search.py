"""Web search service for VoxAI.

Provides free web search retrieval using DuckDuckGo without requiring
any paid API key or hosted AI search service.  Called only for questions
that appear to need current information; ordinary questions are answered
directly by the LLM.

Retrieval results are sanitised to guard against prompt injection and
malicious instructions embedded in retrieved web pages.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("voxai.web_search")

# ---------------------------------------------------------------------------
# Heuristic: does the user's question need live web information?
# ---------------------------------------------------------------------------
_CURRENT_INFO_RE = re.compile(
    r"""(?ix)                     # case-insensitive, verbose
    \b(?:
        latest | current(?:ly)? | recent(?:ly)? | newest | updated? |
        today | yesterday | tonight | this\s+(?:week|month|year) |
        right\s+now | just\s+(?:now|happened|released|launched|announced) |
        news | headlines | breaking |
        price\s+of | stock\s+price | market\s+cap |
        weather\s+(?:in|for|at|today|tomorrow|forecast) |
        who\s+won | score(?:s)?\b | results?\s+of |
        election | president\s+of | prime\s+minister\s+of |
        released? | launched? | announced? | upcoming |
        launch | launch(?:ed)? | rocket |
        202[3-9] | 203\d |
        what\s+(?:is|are)\s+the\s+(?:latest|current|new) |
        how\s+much\s+(?:does|is|are) |
        is\s+(?:it|he|she|they)\s+(?:still|now) |
        did\s+(?:it|he|she|they|spacex|nasa|google|apple|microsoft|[a-z0-9_]+)\s+(?:just|already|recently|launch) |
        # Hindi keywords (transliterated & Devanagari)
        aaj | taaza | khabar | samachar | mausam |
        आज | ताज़ा | ताजा | खबर | समाचार | मौसम | वर्तमान | नवीनतम
    )\b
    """,
)


def needs_web_search(message: str) -> bool:
    """Return *True* when the message looks like it needs current information."""
    if not message or len(message.strip()) < 4:
        return False
    return bool(_CURRENT_INFO_RE.search(message))


# ---------------------------------------------------------------------------
# Content sanitisation — strip HTML, truncate, remove prompt-injection attempts
# ---------------------------------------------------------------------------
_HTML_TAG_RE = re.compile(r"<[^>]{0,500}>")
_INJECTION_LINE_RE = re.compile(
    r"(?im)^\s*(?:"
    r"ignore\s+(?:all\s+)?(?:previous|above|prior)"
    r"|you\s+are\s+now"
    r"|system\s*:"
    r"|assistant\s*:"
    r"|new\s+instructions?"
    r"|forget\s+(?:everything|all|your)"
    r"|disregard"
    r"|do\s+not\s+follow"
    r"|override"
    r").*$",
)

_MAX_SNIPPET_CHARS = 400


def sanitize_snippet(text: str) -> str:
    """Strip HTML, truncate, and remove suspicious instruction-like lines."""
    if not text:
        return ""
    text = _HTML_TAG_RE.sub("", text)
    text = _INJECTION_LINE_RE.sub("", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > _MAX_SNIPPET_CHARS:
        text = text[:_MAX_SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
    return text


def _extract_final_url(raw_href: str) -> str:
    """Extract destination URL from DuckDuckGo redirect link."""
    href = (raw_href or "").strip()
    if not href:
        return ""
    if "uddg=" in href:
        try:
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            dest = parsed.get("uddg", [href])[0]
            return dest
        except Exception:
            return href
    if href.startswith("//"):
        return f"https:{href}"
    return href


# ---------------------------------------------------------------------------
# Web search provider service
# ---------------------------------------------------------------------------
class WebSearchService:
    """Search the public web via free providers (no API key required)."""

    #: How long a successful result set is reused for an identical query.
    _CACHE_TTL_S = 120.0
    #: Bounded cache so memory cannot grow without limit.
    _CACHE_MAX_ENTRIES = 64

    def __init__(self, config: Any) -> None:
        self.config = config
        self.provider: str = getattr(config, "web_search_provider", "duckduckgo").lower()
        self._timeout: float = float(getattr(config, "web_search_timeout_s", 8.0))
        self._max_results: int = int(getattr(config, "web_search_max_results", 3))
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()

    def status(self) -> Dict[str, Any]:
        """Status for health and diagnostic reporting."""
        return {
            "name": "web_search",
            "provider": self.provider,
            "timeout_s": self._timeout,
            "max_results": self._max_results,
            "enabled": getattr(self.config, "web_search_enabled", True),
        }

    def search(
        self, query: str, max_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run a web search and return sanitised snippets.

        Returns ``{"ok": True, "snippets": [...], "sources": [...], "search_ms": float}``
        on success, or ``{"ok": False, "error": str, "search_ms": float}`` on failure.
        """
        started = time.perf_counter()
        limit = max_results or self._max_results
        clean_query = (query or "").strip()
        if not clean_query:
            return {
                "ok": False,
                "error": "Search query is empty.",
                "search_ms": 0.0,
            }

        # Reuse a recent identical query so a repeated question costs no network.
        cache_key = f"{clean_query.lower()}\x00{limit}"
        with self._cache_lock:
            hit = self._cache.get(cache_key)
        if hit is not None and (time.time() - hit[0]) < self._CACHE_TTL_S:
            cached = dict(hit[1])
            cached["search_ms"] = round((time.perf_counter() - started) * 1000, 1)
            cached["cached"] = True
            return cached

        # 1. Primary method: DuckDuckGo HTML endpoint via httpx
        try:
            results = self._search_ddg_html(clean_query, limit=limit)
            elapsed = round((time.perf_counter() - started) * 1000, 1)
            logger.info("Web search for %r returned %d snippet(s) in %.0f ms", clean_query[:60], len(results), elapsed)
            payload = {
                "ok": True,
                "snippets": results,
                "sources": [{"title": r["title"] or r["url"], "url": r["url"]} for r in results if r.get("url")],
                "search_ms": elapsed,
            }
            # A fetch that returns no snippets is not a successful search.  If
            # we called it "ok" the assistant could be reported as having found
            # web results when it found none - report failure instead so the
            # user is told current information could not be retrieved.
            if not results:
                return {
                    "ok": False,
                    "error": "The search provider returned no results for this query.",
                    "search_ms": elapsed,
                }
            self._store(cache_key, payload)
            return payload
        except Exception as html_exc:
            logger.warning("DuckDuckGo HTML search failed: %s; trying fallback", html_exc)

        # 2. Fallback method: duckduckgo_search library if installed
        try:
            results = self._search_ddgs_library(clean_query, limit=limit)
            elapsed = round((time.perf_counter() - started) * 1000, 1)
            payload = {
                "ok": True,
                "snippets": results,
                "sources": [{"title": r["title"] or r["url"], "url": r["url"]} for r in results if r.get("url")],
                "search_ms": elapsed,
            }
            if not results:
                return {
                    "ok": False,
                    "error": "The search provider returned no results for this query.",
                    "search_ms": elapsed,
                }
            self._store(cache_key, payload)
            return payload
        except Exception as lib_exc:
            elapsed = round((time.perf_counter() - started) * 1000, 1)
            err_msg = f"Web search failed: {type(lib_exc).__name__}"
            logger.warning("All web search attempts failed for %r: %s", clean_query[:60], lib_exc)
            return {
                "ok": False,
                "error": err_msg,
                "search_ms": elapsed,
            }

    def _store(self, key: str, payload: Dict[str, Any]) -> None:
        """Store a successful result set, evicting the oldest when full.

        Only ``ok`` results are cached; failures are never cached so a
        temporary network problem does not stick for the TTL window.
        """
        if not payload.get("ok"):
            return
        snapshot = {
            "ok": True,
            "snippets": payload.get("snippets", []),
            "sources": payload.get("sources", []),
        }
        with self._cache_lock:
            if key not in self._cache and len(self._cache) >= self._CACHE_MAX_ENTRIES:
                # dict preserves insertion order: drop the oldest entry.
                self._cache.pop(next(iter(self._cache)), None)
            self._cache[key] = (time.time(), snapshot)

    def _search_ddg_html(self, query: str, limit: int) -> List[Dict[str, str]]:
        """Retrieve results from https://html.duckduckgo.com/html/."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
            resp = client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers=headers,
            )
            # Only a true 200 carries DuckDuckGo result markup.  It answers a
            # rate-limited/anti-bot challenge with HTTP 202 and an HTML page
            # that has no results - treating that as success would make the
            # assistant claim it searched the web when it retrieved nothing.
            if resp.status_code != 200:
                raise RuntimeError(f"DuckDuckGo returned HTTP {resp.status_code}")

        soup = BeautifulSoup(resp.text, "html.parser")
        results: List[Dict[str, str]] = []
        for res in soup.select(".result"):
            a = res.select_one(".result__title a")
            snippet_elem = res.select_one(".result__snippet")
            if not a and not snippet_elem:
                continue

            raw_title = a.get_text(strip=True) if a else ""
            raw_body = snippet_elem.get_text(strip=True) if snippet_elem else ""
            raw_url = a["href"] if (a and a.has_attr("href")) else ""

            title = sanitize_snippet(raw_title)
            body = sanitize_snippet(raw_body)
            url = _extract_final_url(raw_url)

            if not title and not body:
                continue

            results.append({"title": title, "body": body, "url": url})
            if len(results) >= limit:
                break

        return results

    def _search_ddgs_library(self, query: str, limit: int) -> List[Dict[str, str]]:
        """Fallback using duckduckgo_search library if available."""
        from duckduckgo_search import DDGS

        results: List[Dict[str, str]] = []
        with DDGS(timeout=int(self._timeout)) as ddgs:
            raw = list(ddgs.text(query, backend="html", max_results=limit))
            for item in raw[:limit]:
                title = sanitize_snippet(item.get("title", ""))
                body = sanitize_snippet(item.get("body", ""))
                url = (item.get("href") or item.get("url") or "").strip()
                if title or body:
                    results.append({"title": title, "body": body, "url": url})
        return results


# ---------------------------------------------------------------------------
# Format search results into an LLM context block
# ---------------------------------------------------------------------------
def build_search_context(
    snippets: List[Dict[str, str]],
    sources: List[Dict[str, str]],
) -> str:
    """Format search snippets into a context block for the LLM prompt."""
    if not snippets:
        return ""

    lines = [
        "Current web search results (use these facts to answer accurately; cite sources when relevant):",
        "",
    ]
    for i, s in enumerate(snippets, 1):
        title = s.get("title", "")
        body = s.get("body", "")
        url = s.get("url", "")
        lines.append(f"[{i}] {title}")
        if body:
            lines.append(f"    Snippet: {body}")
        if url:
            lines.append(f"    Source URL: {url}")
        lines.append("")

    lines.append(
        "Instructions: Provide a concise, spoken-style answer based on the search results above. "
        "Mention the key facts and where the information came from if relevant. "
        "Do not invent facts not found in the search results."
    )
    return "\n".join(lines)
