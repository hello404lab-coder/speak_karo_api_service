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
    explanation: str = Field(default="", description="Short explanation of the mistake (max 20 words)")
    example: str = Field(default="", description="One example sentence showing correct usage")
    score: int = Field(default=70, ge=0, le=100, description="Score 0-100")

# Base schema for JSON output (used in system instruction)
# Only reply_text is spoken (TTS/stream); correction, explanation, example are for display only.
JSON_FORMAT_INSTRUCTION = """
Respond ONLY in valid JSON format with these keys:
{
  "reply_text": "spoken response that continues conversation (this is the ONLY part spoken by TTS)",
  "correction": "correct sentence if mistake exists, otherwise empty string (display only, not spoken)",
  "explanation": "short explanation of the mistake (max 20 words); empty if no mistake (display only)",
  "example": "one example sentence showing correct usage; empty if no mistake (display only)",
  "score": integer from 0 to 100
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

SYSTEM_PROMPT_TEMPLATE = """You are an experienced English speaking tutor helping a learner practice English in a voice conversation.

Your goals are:
1. Help the learner speak more naturally.
2. Correct mistakes clearly.
3. Teach one useful improvement at a time.
4. Keep the conversation natural and engaging.

You must behave like a friendly tutor on a voice call.

LANGUAGE: Respond ONLY in English in 'reply_text'.

{json_format}

IMPORTANT RULES:

1. If the learner makes a mistake:
   - Identify the most important mistake.
   - Provide the corrected sentence.
   - Give a short explanation (max 20 words).
   - Provide one example sentence.

2. Never correct more than ONE mistake in a single response.

3. If the sentence is correct:
   - Leave "correction", "explanation", and "example" empty.
   - Give a higher score.

4. Your spoken reply must:
   - sound natural
   - be conversational
   - be under 50 words
   - end with a question to continue the conversation

5. Do not include explanations inside reply_text.

6. Be encouraging but avoid excessive praise.

7. Adjust vocabulary difficulty based on learner level.

SCORING RULES:

Start from score 100.

Subtract:
-10 for grammar mistake
-10 for vocabulary mistake
-5 for article or preposition mistake
-5 for word order mistake

Minimum score = 0."""


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
    Returns dict with reply_text, correction, explanation, example, score.
    When full JSON parse fails (e.g. truncated stream), tries to extract fields from raw text
    so reply_text is never the raw JSON string.
    """
    clean = re.sub(r"```json\s?|\s?```", "", response_text.strip()).strip()
    try:
        data = json.loads(clean)
        reply_text = data.get("reply_text") or data.get("reply", "")
        correction = data.get("correction") or ""
        explanation = data.get("explanation") or ""
        example = data.get("example") or ""
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
            "explanation": (explanation or "").strip(),
            "example": (example or "").strip(),
            "score": score,
        }
    except Exception:
        reply_text = _extract_json_string_value(clean, "reply_text")
        correction = _extract_json_string_value(clean, "correction")
        explanation = _extract_json_string_value(clean, "explanation")
        example = _extract_json_string_value(clean, "example")
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
            "explanation": explanation.strip(),
            "example": example.strip(),
            "score": score,
        }
