"""Prompt templates for LLM interactions."""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from app.utils.language import LANGUAGE_NAMES, sanitize_translated_reply_text

logger = logging.getLogger(__name__)


class LLMReplySchema(BaseModel):
    """Schema for Gemini LLM reply. Used for response_json_schema and parsing."""
    reply_text: str = Field(..., description="Response to the user (spoken by TTS)")
    correction: str = Field(default="", description="Best corrected or polished English version of the learner sentence; never empty")
    explanation: str = Field(default="", description="Short explanation of the mistake or improvement point (1-2 short sentences)")
    example: str = Field(default="", description="One example sentence showing the correct usage when helpful")
    score: int = Field(default=70, ge=0, le=100, description="Score 0-100")


class LearnerAnalysisSchema(BaseModel):
    """Schema for learner-only feedback repair."""
    correction: str = Field(default="", description="Best corrected or polished English version of the learner's sentence")
    explanation: str = Field(default="", description="Short explanation of the most important issue or improvement")
    example: str = Field(default="", description="One short example sentence when helpful")
    score: int = Field(default=70, ge=0, le=100, description="Score 0-100")

# Base schema for JSON output (used in system instruction)
# Only reply_text is spoken (TTS/stream); translation and learner analysis are for display only.
JSON_FORMAT_INSTRUCTION = """
Respond ONLY in valid JSON format with these keys:
{
  "reply_text": "spoken response that continues conversation (this is the ONLY part spoken by TTS)",
  "correction": "best corrected or polished English version of the learner sentence; never empty (display only, not spoken)",
  "explanation": "short explanation of the mistake or improvement point in 1-2 short sentences; empty if there is no meaningful teaching point (display only)",
  "example": "one example sentence showing correct usage; empty if no example is needed (display only)",
  "score": integer from 0 to 100
}
"""

SYSTEM_PROMPT_TEMPLATE = """You are an experienced English speaking tutor helping a learner practice English in a voice conversation.

Your goals are:
1. Help the learner speak more naturally.
2. Correct mistakes clearly.
3. Teach one useful improvement at a time.
4. Keep the conversation natural and engaging.

You must behave like a friendly tutor on a voice call.

{language_instruction}

{json_format}

IMPORTANT RULES:

1. Always provide "correction":
   - Return the best corrected or polished English version of the learner's sentence.
   - If the learner made a mistake, correct the most important mistake only.
   - If the learner sentence is already acceptable, return the same sentence or a lightly polished natural version.

2. If the learner makes a mistake:
   - Give a short explanation in 1-2 short sentences.
   - Provide one example sentence when it helps.

3. Never correct more than ONE mistake in a single response.

4. If the sentence is correct:
   - Keep "correction" non-empty by returning the same sentence or a lightly polished version.
   - Leave "explanation" and "example" empty unless there is a clear teaching point.
   - Give a higher score, usually 90 or above.

5. Your spoken reply must:
   - sound natural
   - be conversational
   - be under 50 words
   - end with a question to continue the conversation

6. Do not include explanations inside reply_text.

7. Be encouraging but avoid excessive praise.

8. Adjust vocabulary difficulty based on learner level.

9. "correction" is ONLY the learner's corrected English wording:
   - Never copy assistant coaching, encouragement, or follow-up questions into "correction".
   - Never write things like "I understand", "You can say", or "What would you like..." in "correction".
   - If the learner is asking how to say something in English, "correction" should be that English sentence only.

10. "explanation" is ONLY about the learner's wording:
   - Do not mention "the user", "the learner", or describe system behavior.
   - Do not say the sentence was correct if the score is low.

SCORING RULES:

Start from score 100.

Subtract:
-10 for grammar mistake
-10 for vocabulary mistake
-5 for article or preposition mistake
-5 for word order mistake

Minimum score = 0."""


def _reply_language_instruction(response_language: str) -> str:
    """How reply_text (TTS) should be written: English vs learner's Indic language with native script."""
    code = (response_language or "en").strip().lower()
    if code == "en":
        return "LANGUAGE: Respond ONLY in English in 'reply_text'."
    name = LANGUAGE_NAMES.get(code)
    if not name:
        name = "the learner's language"
    return (
        f"LANGUAGE: Write **reply_text** entirely in {name}. "
        f"Use the standard native writing system for {name} (e.g. Malayalam script for Malayalam, "
        "Devanagari for Hindi, Tamil script for Tamil) so text-to-speech sounds natural and correct. "
        'For "correction", "explanation", and "example": keep English phrases in English when you are '
        "showing the corrected English sentence or an example."
    )


def get_system_instruction(
    response_language: str = "en",
    translation_language: Optional[str] = None,
    long_term_context: Optional[str] = None,
) -> str:
    """Build the system instruction for reply_text language and learner analysis."""
    del translation_language
    base = SYSTEM_PROMPT_TEMPLATE.format(
        language_instruction=_reply_language_instruction(response_language),
        json_format=JSON_FORMAT_INSTRUCTION,
    )
    if long_term_context and long_term_context.strip():
        return f"Known about this learner: {long_term_context.strip()}\n\n{base}"
    return base


LEARNER_ANALYSIS_REPAIR_INSTRUCTION = """
You are fixing learner feedback for an English tutor app.

Return ONLY valid JSON with these keys:
{
  "correction": "best corrected or polished English version of the learner's intended sentence",
  "explanation": "1-2 short sentences about the most important mistake or improvement point, or empty string",
  "example": "one short example sentence when helpful, or empty string",
  "score": integer from 0 to 100
}

Rules:
- Analyze ONLY the learner's message.
- "correction" must be a concise English sentence or phrase the learner should say.
- Never include encouragement, tutor commentary, or a follow-up question in "correction".
- Never copy the assistant reply into "correction".
- If the learner message is in another language or transliterated, infer the intended English sentence and put only that in "correction".
- If the learner sentence is already acceptable, keep "correction" the same or lightly polished, leave "explanation" and "example" empty unless there is a meaningful teaching point, and use a score of 90-100.
- If there is a real mistake, make the correction actually fix it.
- "explanation" must talk only about the wording issue, not about the app or the user.
- Do not mention "the user" or "the learner".
"""


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
    Returns dict with reply_text, translated_reply_text, correction, explanation, example, score.
    When full JSON parse fails (e.g. truncated stream), tries to extract fields from raw text
    so reply_text is never the raw JSON string.
    """
    clean = re.sub(r"```json\s?|\s?```", "", response_text.strip()).strip()
    try:
        data = json.loads(clean)
        reply_text = data.get("reply_text") or data.get("reply", "")
        translated_reply_text = data.get("translated_reply_text")
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
            "translated_reply_text": (
                translated_reply_text.strip()
                if isinstance(translated_reply_text, str) and translated_reply_text.strip()
                else None
            ),
            "correction": (correction or "").strip(),
            "explanation": (explanation or "").strip(),
            "example": (example or "").strip(),
            "score": score,
        }
    except Exception:
        reply_text = _extract_json_string_value(clean, "reply_text")
        translated_reply_text = _extract_json_string_value(clean, "translated_reply_text")
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
            "translated_reply_text": translated_reply_text.strip() or None,
            "correction": correction.strip(),
            "explanation": explanation.strip(),
            "example": example.strip(),
            "score": score,
        }


def build_display_correction(user_message: str) -> str:
    """Build a readable fallback correction when the model leaves it blank."""
    text = re.sub(r"\s+", " ", (user_message or "").strip())
    if not text:
        return ""

    replacements = {
        "i'm": "I'm",
        "i've": "I've",
        "i'll": "I'll",
        "i'd": "I'd",
        "can't": "can't",
        "don't": "don't",
        "doesn't": "doesn't",
        "didn't": "didn't",
        "won't": "won't",
    }
    for source, target in replacements.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text, flags=re.IGNORECASE)
    text = re.sub(r"\bi\b", "I", text, flags=re.IGNORECASE)
    text = re.sub(r"\bai\b", "AI", text, flags=re.IGNORECASE)

    chars = list(text)
    for idx, ch in enumerate(chars):
        if ch.isalpha():
            chars[idx] = ch.upper()
            break
    text = "".join(chars)

    if text and text[-1] not in ".!?":
        text += "."
    return text


_ASSISTANTISH_CORRECTION_PATTERNS = [
    r"\bi understand\b",
    r"\byou can say\b",
    r"\byou could say\b",
    r"\bwhat would you like\b",
    r"\bwhat topic\b",
    r"\bhow about\b",
    r"\bwe can\b",
    r"\blet'?s\b",
    r"\bthat's\b",
    r"\byou(?:'|’)re asking\b",
    r"\bmaybe i(?:'|’)m not explaining\b",
]
_BROKEN_EXPLANATION_PATTERNS = [
    r"\bthe user\b",
    r"\bthe learner\b",
    r"\bexplicitly requested\b",
    r"\byou(?:'|’)re asking\b",
]
_TRUNCATED_ENDINGS = {
    "a", "an", "and", "are", "as", "at", "for", "from", "if", "in", "into",
    "is", "of", "on", "or", "the", "to", "use", "we", "with",
}


def _normalized_compare_text(text: str) -> str:
    """Normalize text for loose equality checks."""
    collapsed = re.sub(r"\s+", " ", (text or "").strip()).casefold()
    return re.sub(r"[\W_]+", "", collapsed)


def _looks_truncated_text(text: str) -> bool:
    """Detect explanations/corrections that appear cut off mid-thought."""
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return False
    if clean[-1] in ".!?":
        return False
    last_word_match = re.search(r"([A-Za-z]+)$", clean)
    if not last_word_match:
        return False
    return last_word_match.group(1).casefold() in _TRUNCATED_ENDINGS


def _looks_like_assistantish_correction(text: str) -> bool:
    """Detect when correction contains assistant reply phrasing instead of learner wording."""
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return False
    if "\n" in text:
        return True
    if clean.count("?") > 0:
        return True
    sentence_breaks = len(re.findall(r"[.!?]", clean))
    if sentence_breaks > 1:
        return True
    lowered = clean.casefold()
    return any(re.search(pattern, lowered) for pattern in _ASSISTANTISH_CORRECTION_PATTERNS)


def _looks_like_non_english_output(text: str) -> bool:
    """Detect corrections that still contain mostly non-Latin script."""
    if not text:
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    latin = sum(1 for ch in letters if ("a" <= ch.lower() <= "z"))
    return (latin / len(letters)) < 0.55


def user_analysis_needs_repair(ai_response: Dict[str, Any], user_message: str) -> bool:
    """Return True when learner feedback is clearly unreliable and should be repaired."""
    correction = (ai_response.get("correction") or "").strip()
    explanation = (ai_response.get("explanation") or "").strip()
    example = (ai_response.get("example") or "").strip()
    reply_text = (ai_response.get("reply_text") or "").strip()
    score = ai_response.get("score", 70)
    if not isinstance(score, int):
        try:
            score = int(score) if score is not None else 70
        except (TypeError, ValueError):
            score = 70

    if not correction:
        return True

    correction_cmp = _normalized_compare_text(correction)
    user_cmp = _normalized_compare_text(user_message)
    reply_cmp = _normalized_compare_text(reply_text)

    if reply_cmp and correction_cmp == reply_cmp:
        return True

    if _looks_like_assistantish_correction(correction):
        return True

    if _looks_like_non_english_output(correction):
        return True

    if correction_cmp and user_cmp and correction_cmp == user_cmp and (score < 90 or explanation or example):
        return True

    if explanation:
        lowered_explanation = explanation.casefold()
        if any(re.search(pattern, lowered_explanation) for pattern in _BROKEN_EXPLANATION_PATTERNS):
            return True
        if _looks_truncated_text(explanation):
            return True
        if "already correct" in lowered_explanation and score < 90:
            return True

    return False


def extract_correction_candidate_from_reply(reply_text: str, user_message: str = "") -> Optional[str]:
    """Try to recover a learner correction from assistant reply quotes for legacy bad rows."""
    clean = re.sub(r"\s+", " ", (reply_text or "").strip())
    if not clean:
        return None

    cue_patterns = [
        r"""(?:you can say|you could say|you may say|we(?:'|’)d say|a more natural way to say this(?: when you feel misunderstood)? is|you could also say)[^"“”]*["“]([^"”]{3,160})["”]""",
        r"""(?:for example|for yesterday)[^"“”]*["“]([^"”]{3,160})["”]""",
    ]
    for pattern in cue_patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate and _normalized_compare_text(candidate) != _normalized_compare_text(user_message):
                return candidate

    for candidate in re.findall(r"""["“]([^"”]{3,160})["”]""", clean):
        stripped = candidate.strip()
        if not stripped:
            continue
        if _normalized_compare_text(stripped) == _normalized_compare_text(user_message):
            continue
        if _looks_like_assistantish_correction(stripped):
            continue
        if _looks_like_non_english_output(stripped):
            continue
        return stripped
    return None


def postprocess_llm_reply(
    ai_response: Dict[str, Any],
    user_message: str,
    reply_language: str = "en",
    translation_language: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize LLM output so API consumers always receive usable feedback and sane translation."""
    reply_text = (ai_response.get("reply_text") or "").strip() or "I couldn't process that."

    score = ai_response.get("score", 70)
    if not isinstance(score, int):
        try:
            score = int(score) if score is not None else 70
        except (TypeError, ValueError):
            score = 70
    score = max(0, min(100, score))

    correction = (ai_response.get("correction") or "").strip()
    if not correction:
        correction = build_display_correction(user_message)

    explanation = (ai_response.get("explanation") or "").strip()
    example = (ai_response.get("example") or "").strip()
    translated_reply_text = sanitize_translated_reply_text(
        reply_text,
        ai_response.get("translated_reply_text"),
        translation_language,
    )

    return {
        "reply_text": reply_text,
        "translated_reply_text": translated_reply_text,
        "correction": correction,
        "explanation": explanation,
        "example": example,
        "score": score,
    }
