"""Backend-authoritative conversation sessions.

History lives here, keyed by an unguessable session id.  The browser only ever
receives its session id - never the history itself - and any ``history`` field
sent by a client is ignored by the API layer.  Sessions are strictly isolated
from each other.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import Config
from services.errors import ApiError, bad_request, not_found

#: Session ids are ``secrets.token_hex(16)``.  Anything else is rejected before
#: it reaches the store so a malformed id can never collide with internal keys.
SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")

VALID_ROLES = frozenset({"user", "assistant"})


@dataclass
class Session:
    session_id: str
    created_at: float
    last_used_at: float
    messages: List[Dict[str, str]] = field(default_factory=list)

    @property
    def turns(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "user")

    def to_dict(self, now: float, ttl: int) -> Dict[str, object]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "turns": self.turns,
            "messages": len(self.messages),
            "expires_in_seconds": max(0, int(ttl - (now - self.last_used_at))),
        }


class SessionStore:
    """Thread-safe, in-memory session store with TTL eviction."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def validate_id(session_id: str) -> str:
        """Return the id if it is well-formed, else raise a 400."""
        candidate = (session_id or "").strip().lower()
        if not SESSION_ID_RE.fullmatch(candidate):
            raise bad_request(
                "Invalid session id: expected a 32-character hex id.",
                code="invalid_session_id",
                hint="Create one with POST /api/session/new.",
            )
        return candidate

    def _prune_locked(self, now: float) -> None:
        ttl = self.config.session_ttl_seconds
        expired = [
            sid for sid, session in self._sessions.items()
            if now - session.last_used_at > ttl
        ]
        for sid in expired:
            del self._sessions[sid]
        if len(self._sessions) > self.config.max_sessions:
            # Evict the least recently used sessions to honour MAX_SESSIONS.
            ordered = sorted(self._sessions.values(), key=lambda s: s.last_used_at)
            excess = len(self._sessions) - self.config.max_sessions
            for session in ordered[:excess]:
                del self._sessions[session.session_id]

    # -- public API --------------------------------------------------------
    def create(self) -> Session:
        now = time.time()
        session = Session(
            session_id=secrets.token_hex(16),
            created_at=now,
            last_used_at=now,
        )
        with self._lock:
            self._prune_locked(now)
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str, *, create_if_missing: bool = False) -> Session:
        sid = self.validate_id(session_id)
        now = time.time()
        with self._lock:
            session = self._sessions.get(sid)
            if session is None:
                if create_if_missing:
                    session = Session(session_id=sid, created_at=now, last_used_at=now)
                    self._sessions[sid] = session
                else:
                    raise not_found(
                        "Unknown session id.",
                        code="unknown_session",
                        hint="Start a new conversation with POST /api/session/new.",
                    )
            session.last_used_at = now
        return session

    def resolve(self, session_id: Optional[str]) -> Session:
        """Return the session for ``session_id``, creating one when absent.

        A well-formed but unknown id is treated as a new session so a
        conversation survives an app restart / TTL eviction; the history stays
        empty, so isolation is preserved.
        """
        if not (session_id or "").strip():
            return self.create()
        sid = self.validate_id(session_id)
        now = time.time()
        with self._lock:
            session = self._sessions.get(sid)
            if session is None:
                session = Session(session_id=sid, created_at=now, last_used_at=now)
                self._sessions[sid] = session
            session.last_used_at = now
            return session

    def history(self, session_id: str) -> List[Dict[str, str]]:
        session = self.get(session_id)
        with self._lock:
            return [dict(m) for m in session.messages]

    def append(self, session_id: str, role: str, content: str) -> int:
        """Append one message and trim history to MAX_HISTORY_TURNS.

        Returns the number of turns kept after trimming.
        """
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}, got {role!r}")
        content = (content or "").strip()
        if not content:
            raise ValueError("content must not be empty")
        session = self.get(session_id)
        with self._lock:
            session.messages.append({"role": role, "content": content})
            self._trim_locked(session)
            return session.turns

    def _trim_locked(self, session: Session) -> None:
        max_messages = self.config.max_history_turns * 2
        if max_messages <= 0:
            session.messages.clear()
            return
        while len(session.messages) > max_messages:
            session.messages.pop(0)
        # Never let history start with an assistant message.
        while session.messages and session.messages[0]["role"] == "assistant":
            session.messages.pop(0)

    def clear(self, session_id: str) -> Dict[str, object]:
        """Drop a session's history and return what was removed."""
        session = self.get(session_id)
        with self._lock:
            removed = len(session.messages)
            turns = session.turns
            session.messages.clear()
            session.created_at = time.time()
            session.last_used_at = time.time()
        return {"removed_messages": removed, "removed_turns": turns}

    def info(self, session_id: str) -> Dict[str, object]:
        session = self.get(session_id)
        with self._lock:
            return session.to_dict(time.time(), self.config.session_ttl_seconds)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "total_messages": sum(len(s.messages) for s in self._sessions.values()),
            }
