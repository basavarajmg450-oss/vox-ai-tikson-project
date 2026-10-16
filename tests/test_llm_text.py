"""Pure-function tests for the LLM text helpers (no model needed).

Tag characters are assembled at runtime (chr / concatenation) so the test
source itself contains no literal tag sequences.
"""

from __future__ import annotations

from services.llm_service import cap_answer, strip_thinking

LT = chr(60)  # '<'
GT = chr(62)  # '>'
ZW = "\u200b"  # zero-width space variant seen in some tokenizers

OPEN = LT + "think" + GT
CLOSE = LT + "/" + "think" + GT
OPEN_ZW = LT + ZW + "think" + ZW + GT
CLOSE_ZW = LT + ZW + "/" + ZW + "think" + ZW + GT


def test_strip_removes_closed_think_block():
    out = strip_thinking(OPEN + "secret reasoning" + CLOSE + " The capital is Paris.")
    assert "secret reasoning" not in out
    assert "The capital is Paris." in out


def test_strip_removes_zero_width_variant_tags():
    out = strip_thinking(OPEN_ZW + "hidden" + CLOSE_ZW + " Paris is the answer.")
    assert "hidden" not in out
    assert out == "Paris is the answer."


def test_strip_removes_unclosed_trailing_block():
    out = strip_thinking("Real answer first. " + OPEN + "and then reasoning that never closed")
    assert out == "Real answer first."
    assert "never closed" not in out


def test_strip_keeps_plain_text_untouched():
    text = "Paris is the capital of France."
    assert strip_thinking(text) == text
    assert strip_thinking("") == ""


def test_cap_answer_keeps_short_text():
    assert cap_answer("Short.", 200) == "Short."


def test_cap_answer_cuts_on_sentence_boundary():
    long_text = ("Paris is the capital of France. " * 40).strip()
    budget = max(64, 10 * 4)  # the 64-char floor protects tiny MAX_RESPONSE_TOKENS
    capped = cap_answer(long_text, 10)
    assert len(capped) <= budget + 1
    assert capped.endswith(".")


def test_cap_answer_never_returns_empty_for_long_input():
    long_text = "word " * 500
    assert cap_answer(long_text, 10)
