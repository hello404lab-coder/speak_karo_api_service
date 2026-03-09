"""Prompt templates for LLM interactions."""
import json
import logging
import re
from typing import List, Dict, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LLMReplySchema(BaseModel):
    """Schema for Gemini LLM reply. Used for response_json_schema and parsing."""
    reply_text: str = Field(..., description="Response to the user (spoken by TTS)")
    correction: str = Field(default="", description="Correct English phrase; display only")
    score: int = Field(default=70, ge=0, le=100, description="Score 0-100")

# Base schema for JSON output (used in system instruction)
# Only reply_text is spoken (TTS/stream); correction is for display only.
JSON_FORMAT_INSTRUCTION = """
Respond ONLY in valid JSON format with these keys:
{
  "reply_text": "your response to the user (this is the ONLY part that will be spoken by TTS)",
  "correction": "correct English phrase or sentence (what they should say in English); empty string if no correction (display only, not spoken)",
  "score": 0-100 integer
}
"""

# Language code (ISO 639-1) -> display name for the prompt. Unknown codes fall back to "the same language they are speaking".
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "pt": "Portuguese",
}

SYSTEM_PROMPT_TEMPLATE = """You are a friendly but thorough English tutor on a voice call. Your PRIMARY job is to catch and correct the learner's English mistakes — grammar, vocabulary, word order, tense, articles, prepositions — while keeping the conversation flowing naturally.

LANGUAGE: Respond ONLY in English in 'reply_text'.
{json_format}

HOW TO CORRECT:
- When the learner makes a mistake, briefly acknowledge what they said, then naturally model the correct form in your reply so they hear it spoken aloud.
- Put the corrected sentence or phrase in 'correction' (shown on-screen, never spoken). Write ONLY the corrected version, no labels or explanations.
- If they made multiple mistakes, correct the most important one. Don't overwhelm.
- If their English was correct, leave 'correction' as an empty string and give them a higher score.

HOW TO TEACH:
- After correcting, move the conversation forward. Ask a follow-up question that nudges them to use the corrected structure again.
- Introduce slightly more advanced vocabulary or structures when they're doing well.
- If they seem stuck, offer a simpler way to say what they're trying to express.

SCORING:
- 90-100: No mistakes, natural phrasing.
- 70-89: Minor errors (article, preposition) but understandable.
- 50-69: Noticeable grammar or vocabulary issues.
- 30-49: Hard to understand, frequent errors.
- 0-29: Very limited communication.

RULES:
1. Be brief: 2-4 short sentences. Under 50 words in 'reply_text'.
2. Always end with a question to keep them talking.
3. Speak naturally and conversationally — this is a voice call, not a textbook.
4. Only put spoken content in 'reply_text'. 'correction' is for on-screen reading only, never spoken by TTS.
5. Do NOT use filler praise like "Great job!" or "Don't worry!" unless they specifically need encouragement."""


def get_system_instruction(response_language: str = "en", long_term_context: Optional[str] = None) -> str:
    """Build the system instruction. English-only for now."""
    base = SYSTEM_PROMPT_TEMPLATE.format(json_format=JSON_FORMAT_INSTRUCTION)
    if long_term_context and long_term_context.strip():
        return f"Known about this learner: {long_term_context.strip()}\n\n{base}"
    return base


def prepare_history(conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Format history for Gemini: list of {"role": "user"|"model", "parts": [content]}.
    Maps assistant -> model. No trimming here; LLM layer trims by token budget.
    """
    formatted = []
    for msg in conversation_history or []:
        role = "user" if msg.get("role") == "user" else "model"
        content = msg.get("content", "")
        formatted.append({"role": role, "parts": [content]})
    return formatted


def _extract_json_string_value(raw: str, key: str) -> str:
    """Extract the first JSON string value for key from raw text (e.g. "reply_text": "...").
    Handles escaped quotes inside the value. Returns empty string if not found.
    """
    pattern = rf'"{re.escape(key)}"\s*:\s*"'
    m = re.search(pattern, raw)
    if not m:
        return ""
    start = m.end()
    result = []
    i = start
    while i < len(raw):
        c = raw[i]
        if c == "\\" and i + 1 < len(raw):
            result.append(raw[i + 1])
            i += 2
            continue
        if c == '"':
            break
        result.append(c)
        i += 1
    return "".join(result)


def parse_gemini_response(response_text: str) -> Dict[str, any]:
    """
    Parse JSON from Gemini response. Handles optional markdown code fences.
    Returns dict with reply_text, correction, score.
    When full JSON parse fails (e.g. truncated stream), tries to extract fields from raw text
    so reply_text is never the raw JSON string.
    """
    clean = re.sub(r"```json\s?|\s?```", "", response_text.strip()).strip()
    try:
        data = json.loads(clean)
        reply_text = data.get("reply_text") or data.get("reply", "")
        correction = data.get("correction") or ""
        score = data.get("score", 70)
        if not isinstance(score, int):
            try:
                score = int(score) if score is not None else 70
            except (TypeError, ValueError):
                score = 70
        score = max(0, min(100, score))
        return {
            "reply_text": (reply_text or "").strip() or _extract_json_string_value(clean, "reply_text"),
            "correction": (correction or "").strip(),
            "score": score,
        }
    except Exception:
        reply_text = _extract_json_string_value(clean, "reply_text")
        correction = _extract_json_string_value(clean, "correction")
        score_str = re.search(r'"score"\s*:\s*(\d+)', clean)
        score = int(score_str.group(1)) if score_str else 70
        score = max(0, min(100, score))
        # Plain-text fallback: if model returned natural language instead of JSON, use it as reply_text
        if not reply_text.strip() and clean.strip() and not clean.strip().startswith("{"):
            reply_text = clean.strip()
            logger.info("Using full response as reply_text (JSON parse failed)")
        return {
            "reply_text": reply_text.strip() or "I couldn't process that.",
            "correction": correction.strip(),
            "score": score,
        }
