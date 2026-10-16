"""Frontend checks: JS syntax, the pure VAD logic, and page wiring.

These run Node when it is installed and skip cleanly when it is not, so the
ordinary Python suite never depends on a browser or network.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _node(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [NODE, *args], cwd=ROOT, capture_output=True, text=True, timeout=60, check=True
    )


def test_app_js_parses():
    _node("--check", "static/js/app.js")


def test_vad_js_parses():
    _node("--check", "static/js/vad.js")


def test_vad_logic_suite_passes():
    result = _node("tests/js/test_vad.mjs")
    assert "All VAD logic tests passed." in result.stdout


def test_page_wires_vad_and_status_labels():
    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    assert "/static/js/vad.js" in html
    assert 'id="vad-hint"' in html

    js = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    # The four statuses the assignment asks for.
    for label in ("Listening", "Processing", "Thinking", "Speaking"):
        assert label in js, f"status pill label {label!r} missing"
    # Auto-stop and manual fallback both present.
    assert "stopRecording('auto')" in js
    assert "stopRecording('max')" in js


def test_latency_panel_removed_from_chat_ui():
    """The latency panel and every timing metric are gone from the UI,
    while the microphone controls, timer, chat log and session buttons stay."""
    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    assert "latency" not in html.lower(), "latency panel markup still in index.html"
    for kept in ('id="mic-btn"', 'id="record-timer"', 'id="new-btn"',
                 'id="clear-btn"', 'id="chat-log"', 'id="status-pill"'):
        assert kept in html, f"{kept} must remain in the page"

    js = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "latency" not in js.lower(), "latency rendering code still in app.js"
    assert "showLatency" not in js
    assert "wallMs" not in js
    # No internal model names rendered next to chat messages.
    assert "usage.model" not in js

    css = (ROOT / "static" / "css" / "style.css").read_text(encoding="utf-8")
    assert "latency" not in css.lower(), "latency CSS still in style.css"
