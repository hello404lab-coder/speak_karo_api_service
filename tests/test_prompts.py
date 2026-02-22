"""Unit tests for prompt parsing (parse_gemini_response)."""
import pytest
from app.core.prompts import parse_gemini_response


def test_parse_gemini_response_valid_json():
    """Valid JSON with reply_text, correction, score returns correct dict."""
    raw = '{"reply_text": "Hello! How are you?", "correction": "", "score": 85}'
    result = parse_gemini_response(raw)
    assert result["reply_text"] == "Hello! How are you?"
    assert result["correction"] == ""
    assert result["score"] == 85


def test_parse_gemini_response_valid_json_with_code_fence():
    """Valid JSON inside ```json fence is parsed."""
    raw = '```json\n{"reply_text": "Sure!", "correction": "you was", "score": 70}\n```'
    result = parse_gemini_response(raw)
    assert result["reply_text"] == "Sure!"
    assert result["correction"] == "you was"
    assert result["score"] == 70


def test_parse_gemini_response_plain_text_fallback():
    """Plain text (no JSON) is used as reply_text with correction empty and score 70."""
    raw = "That's a great question! How about we discuss your hobbies or what you like to do in your free time? Does that sound interesting?"
    result = parse_gemini_response(raw)
    assert result["reply_text"] == raw
    assert result["correction"] == ""
    assert result["score"] == 70


def test_parse_gemini_response_empty_yields_fallback():
    """Empty or whitespace-only response yields 'I couldn't process that.'."""
    assert parse_gemini_response("")["reply_text"] == "I couldn't process that."
    assert parse_gemini_response("   \n  ")["reply_text"] == "I couldn't process that."


def test_parse_gemini_response_malformed_json_non_empty_body():
    """Malformed JSON with non-empty body uses full body as reply_text."""
    raw = "This is not JSON at all, just a sentence."
    result = parse_gemini_response(raw)
    assert result["reply_text"] == "This is not JSON at all, just a sentence."
    assert result["correction"] == ""
    assert result["score"] == 70


def test_parse_gemini_response_reply_key_alias():
    """Supports 'reply' as alias for reply_text in JSON."""
    raw = '{"reply": "Alias works", "correction": "", "score": 80}'
    result = parse_gemini_response(raw)
    assert result["reply_text"] == "Alias works"
    assert result["score"] == 80


def test_parse_gemini_response_score_clamped():
    """Score is clamped to 0-100."""
    raw = '{"reply_text": "Hi", "correction": "", "score": 150}'
    result = parse_gemini_response(raw)
    assert result["score"] == 100

    raw_low = '{"reply_text": "Hi", "correction": "", "score": -10}'
    result_low = parse_gemini_response(raw_low)
    assert result_low["score"] == 0
