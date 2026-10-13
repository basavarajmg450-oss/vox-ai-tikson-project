"""Typed errors shared by the service and route layers.

Every error carries an HTTP status, a stable machine-readable ``code`` and an
optional ``hint`` that tells the user exactly how to fix the problem.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ApiError(Exception):
    """An exception that maps directly to a JSON error response."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: str = "error",
        hint: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = int(status_code)
        self.code = code
        self.hint = hint
        self.payload: Dict[str, Any] = payload or {}

    def to_dict(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {"error": self.message, "code": self.code}
        if self.hint:
            body["hint"] = self.hint
        body.update(self.payload)
        return body

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"ApiError(code={self.code!r}, status={self.status_code}, message={self.message!r})"


def bad_request(message: str, code: str = "bad_request", hint: Optional[str] = None) -> ApiError:
    return ApiError(message, 400, code, hint)


def not_found(message: str, code: str = "not_found", hint: Optional[str] = None) -> ApiError:
    return ApiError(message, 404, code, hint)


def unprocessable(message: str, code: str = "unprocessable", hint: Optional[str] = None) -> ApiError:
    return ApiError(message, 422, code, hint)


def unavailable(message: str, code: str = "service_unavailable",
                 hint: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> ApiError:
    return ApiError(message, 503, code, hint, payload)
