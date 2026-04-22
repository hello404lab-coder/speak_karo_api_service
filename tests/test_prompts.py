"""Unit tests for prompt parsing, validation, and post-processing."""
import pytest
from app.core.prompts import (
    build_display_correction,
    extract_correction_candidate_from_reply,
    has_meaningful_correction,
    get_system_instruction,
    parse_gemini_response,
    postprocess_llm_reply,
    user_analysis_needs_repair,
)


def test_parse_gemini_response_valid_json():
    """Valid JSON with reply_text, correction, score returns correct dict."""
    raw = '{"reply_text": "Hello! How are you?", "translated_reply_text": "Hola! Como estas?", "correction": null, "score": 95}'
    result = parse_gemini_response(raw)
    assert result["reply_text"] == "Hello! How are you?"
    assert result["translated_reply_text"] == "Hola! Como estas?"
    assert result["correction"] is None
    assert result["score"] == 95


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
    assert result["correction"] is None
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
    assert result["correction"] is None
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
    raw = '{"reply_text": "Hi", "correction": null, "score": 150}'
    result = parse_gemini_response(raw)
    assert result["score"] == 100

    raw_low = '{"reply_text": "Hi", "correction": null, "score": -10}'
    result_low = parse_gemini_response(raw_low)
    assert result_low["score"] == 0


def test_parse_gemini_response_translated_reply_null():
    """Explicit null translation stays null."""
    raw = '{"reply_text": "Hi", "translated_reply_text": null, "correction": null, "score": 90}'
    result = parse_gemini_response(raw)
    assert result["reply_text"] == "Hi"
    assert result["translated_reply_text"] is None


def test_system_instruction_requires_null_feedback_for_high_scores():
    """The tutor prompt should forbid surface-level corrections for good English."""
    prompt = get_system_instruction("en", None)
    assert 'If score is 90 or above, "correction", "explanation", and "example" MUST be null.' in prompt


def test_system_instruction_no_longer_asks_gemini_for_reply_translation():
    """Assistant translation should come from the dedicated translation service, not the tutor prompt."""
    prompt = get_system_instruction("en", "ml")
    assert '"translated_reply_text"' not in prompt


def test_build_display_correction_polishes_simple_sentence():
    """Fallback correction should be readable and non-empty."""
    assert build_display_correction("i was happy today") == "I was happy today."


def test_build_display_correction_adds_question_mark_for_questions():
    """Fallback correction should preserve basic question intent."""
    assert build_display_correction("how are you doing") == "How are you doing?"


def test_postprocess_llm_reply_preserves_null_feedback_for_good_english():
    """Good English should not get a fabricated correction anymore."""
    result = postprocess_llm_reply(
        {
            "reply_text": "That's nice. What happened next?",
            "translated_reply_text": None,
            "correction": None,
            "explanation": None,
            "example": None,
            "score": 100,
        },
        user_message="I'm doing fine.",
        reply_language="en",
        translation_language=None,
    )
    assert result["correction"] is None
    assert result["explanation"] is None
    assert result["example"] is None
    assert result["score"] == 100


def test_postprocess_llm_reply_builds_fallback_for_real_rewrite_need():
    """If a real mistake needs teaching and correction is missing, fallback text should still exist."""
    result = postprocess_llm_reply(
        {
            "reply_text": "Instead of saying that, it sounds more natural to say, \"Can we discuss democracy in India?\" What part interests you most?",
            "translated_reply_text": None,
            "correction": None,
            "explanation": "We say 'discuss democracy', not 'discuss about democracy'.",
            "example": None,
            "score": 82,
        },
        user_message="Can we discuss about democracy in India?",
        reply_language="en",
        translation_language=None,
    )
    assert result["correction"] == "Can we discuss about democracy in India?"
    assert result["score"] == 82


def test_postprocess_llm_reply_raises_surface_only_low_score():
    """If only punctuation/casing changed, the score should not stay artificially low."""
    result = postprocess_llm_reply(
        {
            "reply_text": "I'm doing well. How about you?",
            "translated_reply_text": None,
            "correction": "How are you doing.",
            "explanation": "",
            "example": "",
            "score": 70,
        },
        user_message="how are you doing",
        reply_language="en",
        translation_language=None,
    )
    assert result["correction"] == "How are you doing."
    assert result["score"] == 92


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
    """Acceptable learner sentences with null learner feedback should pass validation."""
    assert not user_analysis_needs_repair(
        {
            "reply_text": "Nice. What would you like to discuss?",
            "correction": None,
            "explanation": None,
            "example": None,
            "score": 95,
        },
        "Okay, I get it.",
    )


def test_user_analysis_needs_repair_when_high_score_still_has_correction():
    """High-score responses should not surface learner feedback."""
    assert user_analysis_needs_repair(
        {
            "reply_text": "That's good. What happened today?",
            "correction": "I'm doing fine.",
            "explanation": "",
            "example": "",
            "score": 94,
        },
        "I'm doing fine.",
    )


def test_user_analysis_needs_repair_for_low_value_discuss_about_correction():
    """Surface-only corrections should be repaired when the learner wording needs a real rewrite."""
    assert user_analysis_needs_repair(
        {
            "reply_text": "Sure, we can discuss it.",
            "correction": "Can we discuss about democracy in India?",
            "explanation": "",
            "example": "",
            "score": 95,
        },
        "Can we discuss about democracy in India?",
    )


def test_user_analysis_needs_repair_for_attending_olympics_phrase():
    """Awkward but understandable phrasing should trigger learner-analysis repair."""
    assert user_analysis_needs_repair(
        {
            "reply_text": "That is an interesting topic.",
            "correction": "What do you think about India attending Olympics?",
            "explanation": "",
            "example": "",
            "score": 95,
        },
        "What do you think about India attending Olympics?",
    )


def test_user_analysis_needs_repair_when_noticeable_mistake_is_not_taught_in_reply():
    """Low-score corrections should be reflected naturally in the spoken reply."""
    assert user_analysis_needs_repair(
        {
            "reply_text": "That's an interesting topic. What else do you think?",
            "correction": "Can we discuss democracy in India?",
            "explanation": "We say 'discuss democracy', not 'discuss about democracy'.",
            "example": None,
            "score": 82,
        },
        "Can we discuss about democracy in India?",
    )


def test_user_analysis_does_not_need_repair_when_spoken_reply_teaches_naturally():
    """A low-score response is acceptable when the spoken reply gently teaches the mistake."""
    assert not user_analysis_needs_repair(
        {
            "reply_text": "Sure. Instead of saying 'discuss about democracy,' it's more natural to say 'discuss democracy.' What part of it interests you?",
            "correction": "Can we discuss democracy in India?",
            "explanation": "We say 'discuss democracy', not 'discuss about democracy'.",
            "example": "We discussed climate policy yesterday.",
            "score": 82,
        },
        "Can we discuss about democracy in India?",
    )


def test_user_analysis_needs_repair_when_reply_text_sounds_like_metadata():
    """Spoken replies should not read like labels or structured feedback."""
    assert user_analysis_needs_repair(
        {
            "reply_text": "Correction: Can we discuss democracy in India? Explanation: remove 'about'. What do you think?",
            "correction": "Can we discuss democracy in India?",
            "explanation": "Use 'discuss' without 'about'.",
            "example": None,
            "score": 82,
        },
        "Can we discuss about democracy in India?",
    )


def test_has_meaningful_correction_false_when_same_sentence_without_teaching_value():
    """Exact echo corrections should be hidden when they add no value."""
    assert not has_meaningful_correction(
        "Hi there, am I audible?",
        "Hi there, am I audible?",
        "",
        "",
    )


def test_has_meaningful_correction_true_when_explanation_adds_value():
    """A same-sentence correction can still be meaningful if explanation/example teaches something."""
    assert has_meaningful_correction(
        "Yeah, I was ready.",
        "Yeah, I was ready.",
        "Use 'I'm ready' for the current moment.",
        "",
    )


def test_extract_correction_candidate_from_reply_prefers_teaching_quote():
    """Legacy rows can recover the corrected sentence from quoted tutor guidance."""
    reply = (
        "It's good you're practicing! When you want to say you aren't going anywhere, "
        "it's best to use just one negative word. So, you could say, "
        "\"I'm not going anywhere,\" or \"I'm going nowhere.\""
    )
    assert extract_correction_candidate_from_reply(reply, "I'm not going nowhere.") == "I'm not going anywhere,"
