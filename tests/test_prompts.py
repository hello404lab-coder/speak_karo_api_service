"""Unit tests for prompt parsing, validation, and post-processing."""
import pytest
from app.core.prompts import (
    build_display_correction,
    extract_correction_candidate_from_reply,
    get_system_instruction,
    parse_gemini_response,
    postprocess_llm_reply,
    user_analysis_needs_repair,
)


def test_parse_gemini_response_valid_json():
    """Valid JSON with reply_text, correction, score returns correct dict."""
    raw = '{"reply_text": "Hello! How are you?", "translated_reply_text": "Hola! Como estas?", "correction": "", "score": 85}'
    result = parse_gemini_response(raw)
    assert result["reply_text"] == "Hello! How are you?"
    assert result["translated_reply_text"] == "Hola! Como estas?"
    assert result["correction"] == ""
    assert result["score"] == 85


def test_parse_gemini_response_valid_json_with_code_fence():
    """Valid JSON inside ```json fence is parsed."""
    raw = '```json\n{"reply_text": "Sure!", "correction": "you was", "score": 70}\n```'
    result = parse_gemini_response(raw)
    assert result["reply_text"] == "Sure!"
    assert result["translated_reply_text"] is None
    assert result["correction"] == "you was"
    assert result["score"] == 70


def test_parse_gemini_response_plain_text_fallback():
    """Plain text (no JSON) is used as reply_text with correction empty and score 70."""
    raw = "That's a great question! How about we discuss your hobbies or what you like to do in your free time? Does that sound interesting?"
    result = parse_gemini_response(raw)
    assert result["reply_text"] == raw
    assert result["translated_reply_text"] is None
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
    assert result["translated_reply_text"] is None
    assert result["correction"] == ""
    assert result["score"] == 70


def test_parse_gemini_response_reply_key_alias():
    """Supports 'reply' as alias for reply_text in JSON."""
    raw = '{"reply": "Alias works", "correction": "", "score": 80}'
    result = parse_gemini_response(raw)
    assert result["reply_text"] == "Alias works"
    assert result["translated_reply_text"] is None
    assert result["score"] == 80


def test_parse_gemini_response_score_clamped():
    """Score is clamped to 0-100."""
    raw = '{"reply_text": "Hi", "correction": "", "score": 150}'
    result = parse_gemini_response(raw)
    assert result["score"] == 100

    raw_low = '{"reply_text": "Hi", "correction": "", "score": -10}'
    result_low = parse_gemini_response(raw_low)
    assert result_low["score"] == 0


def test_parse_gemini_response_translated_reply_null():
    """Explicit null translation stays null."""
    raw = '{"reply_text": "Hi", "translated_reply_text": null, "correction": "", "score": 90}'
    result = parse_gemini_response(raw)
    assert result["reply_text"] == "Hi"
    assert result["translated_reply_text"] is None


def test_system_instruction_no_longer_asks_gemini_for_reply_translation():
    """Assistant translation should come from the dedicated translation service, not the tutor prompt."""
    prompt = get_system_instruction("en", "ml")
    assert '"translated_reply_text"' not in prompt


def test_build_display_correction_polishes_simple_sentence():
    """Fallback correction should be readable and non-empty."""
    assert build_display_correction("i was happy today") == "I was happy today."


def test_postprocess_llm_reply_fills_missing_correction():
    """Feedback should still contain a corrected/polished sentence when the model leaves it blank."""
    result = postprocess_llm_reply(
        {
            "reply_text": "That's nice. What happened?",
            "translated_reply_text": None,
            "correction": "",
            "explanation": "",
            "example": "",
            "score": 100,
        },
        user_message="i was happy today",
        reply_language="en",
        translation_language=None,
    )
    assert result["correction"] == "I was happy today."
    assert result["explanation"] == ""
    assert result["example"] == ""
    assert result["score"] == 100


def test_postprocess_llm_reply_drops_unchanged_translation():
    """Unchanged English must not be stored as a Malayalam translation."""
    result = postprocess_llm_reply(
        {
            "reply_text": "Hi there! What's on your mind?",
            "translated_reply_text": "Hi there! What's on your mind?",
            "correction": "I was happy today.",
            "explanation": "",
            "example": "",
            "score": 100,
        },
        user_message="I was happy today.",
        reply_language="en",
        translation_language="ml",
    )
    assert result["translated_reply_text"] is None


def test_postprocess_llm_reply_drops_wrong_script_translation():
    """Translation in the wrong Indic script must be dropped."""
    result = postprocess_llm_reply(
        {
            "reply_text": "That's great! What made you happy today?",
            "translated_reply_text": "అది బాగుంది! ఈ రోజు మీకు సంతోషం కలిగించినది ఏమిటి?",
            "correction": "I was happy today.",
            "explanation": "",
            "example": "",
            "score": 100,
        },
        user_message="I was happy today.",
        reply_language="en",
        translation_language="ml",
    )
    assert result["translated_reply_text"] is None


def test_user_analysis_needs_repair_when_same_text_has_low_score():
    """Low-score feedback cannot simply repeat the learner message."""
    assert user_analysis_needs_repair(
        {
            "reply_text": "Let's improve that.",
            "correction": "I'm not going nowhere.",
            "explanation": "",
            "example": "",
            "score": 70,
        },
        "I'm not going nowhere.",
    )


def test_user_analysis_needs_repair_when_correction_looks_like_assistant_reply():
    """Correction must not contain assistant coaching or follow-up questions."""
    assert user_analysis_needs_repair(
        {
            "reply_text": "We can switch to Malayalam. What would you like to discuss first?",
            "correction": "Let's switch to Malayalam. What would you like to discuss first?",
            "explanation": "The user explicitly requested to switch to Malayalam.",
            "example": "",
            "score": 100,
        },
        "Let's talk in Malayalam.",
    )


def test_user_analysis_needs_repair_when_explanation_is_truncated():
    """Cut-off explanations should be repaired instead of stored."""
    assert user_analysis_needs_repair(
        {
            "reply_text": "Let's refine that.",
            "correction": "Yesterday I went to the shop. Today I am going now.",
            "explanation": "We use the simple past",
            "example": "",
            "score": 70,
        },
        "Yesterday I am going to the shop. Today I was going now.",
    )


def test_user_analysis_does_not_need_repair_for_clean_high_score_sentence():
    """Acceptable learner sentences should pass validation without repair."""
    assert not user_analysis_needs_repair(
        {
            "reply_text": "Nice. What would you like to discuss?",
            "correction": "Okay, I get it.",
            "explanation": "",
            "example": "",
            "score": 95,
        },
        "Okay, I get it.",
    )


def test_extract_correction_candidate_from_reply_prefers_teaching_quote():
    """Legacy rows can recover the corrected sentence from quoted tutor guidance."""
    reply = (
        "It's good you're practicing! When you want to say you aren't going anywhere, "
        "it's best to use just one negative word. So, you could say, "
        "\"I'm not going anywhere,\" or \"I'm going nowhere.\""
    )
    assert extract_correction_candidate_from_reply(reply, "I'm not going nowhere.") == "I'm not going anywhere,"
