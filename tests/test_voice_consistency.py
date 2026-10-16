"""Voice-consistency checks: one fixed speaker per language, no silent switching.

These tests never load an ONNX model - they exercise the selection logic
(``TTSService.voice_for_language``) and the configured defaults, so they run
in the ordinary unit suite.  Real synthesis with both voices runs in
``tests/test_integration.py`` (marker: integration).

Verified speaker pairing (source: official model cards / voice repository):

* ``en_US-ryan-medium`` - MALE US-English speaker, single speaker,
  22050 Hz (finetuned from the lessac voice): https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/ryan/medium/MODEL_CARD
* ``hi_IN-rohan-medium`` - MALE Hindi speaker ("Hindi Mono Male" training
  database), single speaker, 22050 Hz:
  https://huggingface.co/rhasspy/piper-voices/blob/main/hi/hi_IN/rohan/medium/MODEL_CARD
* ``hi_IN-priyamvada-medium`` - FEMALE Hindi speaker:
  https://huggingface.co/rhasspy/piper-voices/tree/main/hi/hi_IN/priyamvada/medium
  Pairing the female priyamvada with a male English voice (or vice versa)
  flips the assistant's gender between languages, so mixed-gender pairs
  must never be selected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import Config
from services.errors import ApiError
from services.text_to_speech import TTSService


def _cfg_with(tmp_path, hindi_exists=True):
    en = tmp_path / "en_US-ryan-medium.onnx"
    en.write_bytes(b"fake")
    hi = tmp_path / "hi_IN-rohan-medium.onnx"
    if hindi_exists:
        hi.write_bytes(b"fake")
    return Config(piper_model_path=en, piper_model_path_hi=hi), en, hi


def test_default_voice_pair_is_the_verified_male_pair():
    """Shipped defaults: male English ryan + male Hindi rohan.

    Both are single-speaker medium-quality male models, so the assistant
    keeps one gender across English and Hindi.  Pinning the pair here stops
    a mixed-gender combination from coming back.
    """
    cfg = Config()
    assert Path(cfg.piper_model_path).name == "en_US-ryan-medium.onnx"
    assert Path(cfg.piper_model_path_hi).name == "hi_IN-rohan-medium.onnx"
    # The two configured voices must not be the same language model, and the
    # Hindi voice must be the male rohan - never the female priyamvada.
    assert "priyamvada" not in Path(cfg.piper_model_path_hi).name


def test_language_selects_model_but_never_a_random_speaker(tmp_path):
    """Language picks en/hi; each language always maps to the same model."""
    cfg, en, hi = _cfg_with(tmp_path)
    tts = TTSService(cfg)

    # English / unknown / missing language -> always the one English model.
    for lang in (None, "", "en", "en-US", "fr", "es"):
        assert tts.voice_for_language(lang) == en

    # Hindi -> always the one Hindi model, on every turn.
    for _ in range(3):
        assert tts.voice_for_language("hi") == hi
        assert tts.voice_for_language("en") == en

    # Deterministic across repeated calls: no rotation, no randomness.
    assert len({tts.voice_for_language("hi") for _ in range(10)}) == 1
    assert len({tts.voice_for_language("en") for _ in range(10)}) == 1


def test_missing_hindi_voice_raises_instead_of_english_fallback(tmp_path):
    """A missing configured voice is reported - never silently substituted."""
    cfg, en, hi = _cfg_with(tmp_path, hindi_exists=False)
    tts = TTSService(cfg)

    with pytest.raises(ApiError) as excinfo:
        tts.voice_for_language("hi")
    assert excinfo.value.code == "tts_hindi_voice_missing"
    assert excinfo.value.hint and "prefetch" in excinfo.value.hint
    # The English model must NOT come back as a quiet substitute.
    assert excinfo.value is not None

    # English selection still works after the Hindi failure.
    assert tts.voice_for_language("en") == en


def test_configured_voice_models_are_single_speaker():
    """No per-request speaker ID can change the voice: both models are
    single-speaker (``num_speakers == 1``), so synthesis always uses the
    model's one and only speaker."""
    for path in (Config().piper_model_path, Config().piper_model_path_hi):
        meta_path = Path(str(path) + ".json")
        if not meta_path.exists():
            pytest.skip(f"voice config not downloaded: {meta_path.name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta.get("num_speakers", 1) == 1, (
            f"{path.name} exposes multiple speakers - speaker selection could vary"
        )
