"""Unit tests for reply/translation language resolution helpers."""

from app.utils.language import (
    normalize_language_code,
    resolve_reply_language,
    resolve_translation_language,
    sanitize_translated_reply_text,
)


def test_normalize_language_code_supports_common_names_and_locales():
    assert normalize_language_code("Malayalam") == "ml"
    assert normalize_language_code("en-US") == "en"
    assert normalize_language_code("mandarin") == "zh"
    assert normalize_language_code("unknown-language") is None


def test_resolve_reply_language_prefers_explicit_override():
    assert resolve_reply_language("hello", detected_lang="ml", override="en") == "en"


def test_resolve_translation_language_prefers_request_then_profile():
    assert resolve_translation_language("ta", "Malayalam", reply_language="en") == "ta"
    assert resolve_translation_language(None, "Malayalam", reply_language="en") == "ml"


def test_resolve_translation_language_drops_same_language():
    assert resolve_translation_language("en", "Malayalam", reply_language="en") is None


def test_sanitize_translated_reply_text_keeps_valid_malayalam_script():
    assert sanitize_translated_reply_text("How are you?", "സുഖമാണോ?", "ml") == "സുഖമാണോ?"


def test_sanitize_translated_reply_text_drops_unchanged_english():
    assert sanitize_translated_reply_text("How are you?", "How are you?", "ml") is None


def test_sanitize_translated_reply_text_drops_wrong_indic_script():
    assert sanitize_translated_reply_text("How are you?", "నీవు ఎలా ఉన్నావు?", "ml") is None


def test_sanitize_translated_reply_text_keeps_latin_translation_for_latin_language():
    assert sanitize_translated_reply_text("How are you?", "Como estas?", "es") == "Como estas?"
