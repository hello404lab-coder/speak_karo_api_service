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
    correction: Optional[str] = Field(default=None, description="Corrected English version of the learner sentence, or null when no correction is needed")
    explanation: Optional[str] = Field(default=None, description="Short explanation of the mistake or improvement point, or null when no correction is needed")
    example: Optional[str] = Field(default=None, description="One example sentence showing the correct usage, or null when no correction is needed")
    score: int = Field(default=70, ge=0, le=100, description="Score 0-100")


class LearnerAnalysisSchema(BaseModel):
    """Schema for learner-only feedback repair."""
    correction: Optional[str] = Field(default=None, description="Corrected English version of the learner sentence, or null when no correction is needed")
    explanation: Optional[str] = Field(default=None, description="Short explanation of the most important issue or improvement, or null")
    example: Optional[str] = Field(default=None, description="One short example sentence when helpful, or null")
    score: int = Field(default=70, ge=0, le=100, description="Score 0-100")

# Base schema for JSON output (used in system instruction)
# Only reply_text is spoken (TTS/stream); translation and learner analysis are for display only.
JSON_FORMAT_INSTRUCTION = """
Respond ONLY in valid JSON format with these keys:
{
  "reply_text": "spoken tutor response; when there is a noticeable mistake, gently teach it here before continuing the conversation (this is the ONLY part spoken by TTS)",
  "correction": "corrected English version of the learner sentence, or null if no real correction is needed (display only)",
  "explanation": "short explanation of the main mistake or improvement point in 1-2 short sentences, or null if no correction is needed (display only)",
  "example": "one example sentence showing correct usage, or null if no example is needed (display only)",
  "score": integer from 0 to 100
}
"""

SYSTEM_PROMPT_TEMPLATE = """You are an expert, encouraging English tutor on a live voice call with a learner.

Your goals are:
1. Teach actively when the learner makes a noticeable mistake.
2. Do not micromanage or polish English that is already natural and correct.
3. Teach one useful improvement at a time.
4. Keep the conversation warm, conversational, and engaging.

You must behave like a friendly teacher on a real voice call.

{language_instruction}

{json_format}

IMPORTANT RULES:

1. Decide first whether the learner actually needs correction:
   - If the learner's English is grammatically correct and naturally understandable, do NOT correct or polish it.
   - In that case, set "correction", "explanation", and "example" to null.
   - If the learner made a real grammar, collocation, preposition, article, or naturalness mistake, correct only the single highest-value issue.
   - If the learner is understandable but awkward or clearly non-native, rewrite it into more natural English.
   - Do not correct tiny casing or punctuation issues unless there is no other teaching point.

2. If the learner makes a mistake:
   - Put the teaching naturally inside "reply_text".
   - Use a short sandwich structure: acknowledge, gently correct, continue the conversation.
   - Keep the correction brief and kind, then move on.
   - Also provide "correction", "explanation", and "example" for display.

3. Never correct more than ONE mistake in a single response.

4. If the sentence is correct or natural enough:
   - Do not rewrite it just to make it sound more polite, more formal, or different.
   - Phrases like "What about you?" and "I'm doing fine." are valid and should not be corrected.
   - Set "correction", "explanation", and "example" to null.
   - If score is 90 or above, "correction", "explanation", and "example" MUST be null.

5. Your spoken reply must:
   - sound natural and teacher-like
   - be conversational
   - be under 50 words
   - end with a question to continue the conversation
   - only include a spoken correction when there is a noticeable mistake
   - never speak a correction for casing, punctuation, or near-equivalent valid phrasing

6. The spoken correction must not sound like metadata:
   - Do not say things like "correction", "score", "example", or "grammar note".
   - Do not sound robotic or instructional.
   - Teach as if you are talking naturally on a call.

7. Be encouraging but avoid excessive praise.

8. Adjust vocabulary difficulty based on learner level.

9. "correction" is ONLY the learner's corrected English wording:
   - Never copy assistant coaching, encouragement, or follow-up questions into "correction".
   - Never write things like "I understand", "By the way", "You can say", or "What would you like..." in "correction".
   - If the learner is asking how to say something in English, "correction" should be that English sentence only.
   - If no correction is needed, return null.

10. "explanation" is ONLY about the learner's wording:
   - Do not mention "the user", "the learner", or describe system behavior.
   - Do not say the sentence was correct if the score is low.
   - If no correction is needed, return null.

11. Prioritize learner value:
   - If the learner says something understandable but awkward, improve it to more natural English and gently teach it in "reply_text".
   - Examples of valuable corrections include fixing collocations like "discuss about" -> "discuss" or making awkward phrasing sound natural.
   - Greetings and very short conversational fragments are fine to keep simple unless there is a clear issue.
   - When there is no clear issue, build confidence and continue the conversation instead of correcting.

SCORING RULES:

Start from score 100.

Use this scale consistently:
- 96-100: natural, fluent, or only tiny surface fixes
- 90-95: correct and natural enough; do not surface a correction
- 75-89: understandable but has a clear grammar, preposition, article, collocation, or naturalness issue
- 50-74: obvious mistakes that make it sound clearly wrong
- 0-49: very hard to understand

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
        f"LANGUAGE: Write **reply_text** primarily in {name}. "
        f"Use the standard native writing system for {name} (e.g. Malayalam script for Malayalam, "
        "Devanagari for Hindi, Tamil script for Tamil) so text-to-speech sounds natural and correct. "
        'If you give a spoken correction, keep the corrected English phrase itself in English, then continue naturally in '
        f"{name}. "
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
  "correction": "corrected English version of the learner's intended sentence, or null",
  "explanation": "1-2 short sentences about the most important mistake or improvement point, or null",
  "example": "one short example sentence when helpful, or null",
  "score": integer from 0 to 100
}

Rules:
- Analyze ONLY the learner's message.
- "correction" must be a concise English sentence or phrase the learner should say.
- Never include encouragement, tutor commentary, or a follow-up question in "correction".
- Never copy the assistant reply into "correction".
- If the learner message is in another language or transliterated, infer the intended English sentence and put only that in "correction".
- If the learner sentence is already acceptable and natural enough, return null for "correction", "explanation", and "example", and use a score of 90-100.
- If the learner sentence is understandable but awkward or non-native, rewrite it into more natural English.
- Do not limit yourself to capitalization or punctuation when there is a better grammar, article, preposition, collocation, or word-choice correction.
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
        correction = data.get("correction")
        explanation = data.get("explanation")
        example = data.get("example")
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
            "correction": correction.strip() if isinstance(correction, str) and correction.strip() else None,
            "explanation": explanation.strip() if isinstance(explanation, str) and explanation.strip() else None,
            "example": example.strip() if isinstance(example, str) and example.strip() else None,
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
            "correction": correction.strip() or None,
            "explanation": explanation.strip() or None,
            "example": example.strip() or None,
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
    for lang_name in LANGUAGE_NAMES.values():
        text = re.sub(rf"\b{re.escape(lang_name)}\b", lang_name, text, flags=re.IGNORECASE)

    chars = list(text)
    for idx, ch in enumerate(chars):
        if ch.isalpha():
            chars[idx] = ch.upper()
            break
    text = "".join(chars)

    if text and text[-1] not in ".!?":
        first_word_match = re.match(r"[A-Za-z']+", text.strip())
        first_word = first_word_match.group(0).casefold() if first_word_match else ""
        if first_word in {
            "who", "what", "when", "where", "why", "how",
            "can", "could", "would", "should", "do", "does", "did",
            "is", "are", "am", "was", "were", "will", "shall", "have",
            "has", "had", "may",
        }:
            text += "?"
        else:
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
_MEANINGFUL_REWRITE_PATTERNS = [
    r"\bdiscuss about\b",
    r"\bexplain about\b",
    r"\bdescribe about\b",
    r"\bregarding about\b",
    r"\breturn back\b",
    r"\battending (?:the )?olympics\b",
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


def _needs_meaningful_rewrite(user_message: str) -> bool:
    """Detect common learner patterns that deserve more than surface-level polishing."""
    clean = re.sub(r"\s+", " ", (user_message or "").strip()).casefold()
    if not clean:
        return False
    return any(re.search(pattern, clean) for pattern in _MEANINGFUL_REWRITE_PATTERNS)


def _needs_spoken_teaching(
    user_message: str,
    correction: str = "",
    score: int = 100,
) -> bool:
    """Return True when the spoken reply should gently teach the learner."""
    correction_cmp = _normalized_compare_text(correction)
    user_cmp = _normalized_compare_text(user_message)
    if score >= 90:
        return False
    if not correction_cmp:
        return _needs_meaningful_rewrite(user_message)
    return correction_cmp != user_cmp or _needs_meaningful_rewrite(user_message)


_SPOKEN_TEACHING_CUE_PATTERNS = [
    r"\binstead of saying\b",
    r"\bit sounds more natural to say\b",
    r"\byou can say\b",
    r"\byou could say\b",
    r"\bwe usually say\b",
    r"\ba more natural way\b",
    r"\bbetter to say\b",
    r"\bmore natural\b",
    r"\bby the way\b",
]
_METADATA_REPLY_PATTERNS = [
    r"\bcorrection\b",
    r"\bgrammar note\b",
    r"\bscore\b",
    r"\bexample sentence\b",
    r"\bexplanation\b",
]


def _reply_text_has_spoken_teaching(reply_text: str, correction: str = "") -> bool:
    """Detect whether reply_text naturally teaches the correction aloud."""
    clean = re.sub(r"\s+", " ", (reply_text or "").strip())
    if not clean:
        return False
    lowered = clean.casefold()
    if any(re.search(pattern, lowered) for pattern in _SPOKEN_TEACHING_CUE_PATTERNS):
        return True
    correction_cmp = _normalized_compare_text(correction)
    reply_cmp = _normalized_compare_text(clean)
    if correction_cmp and len(correction_cmp) >= 12 and correction_cmp in reply_cmp:
        return True
    return False


def _reply_text_overteaches(reply_text: str, correction: str = "", score: int = 100) -> bool:
    """Detect spoken replies that teach when they should simply continue the chat."""
    if score < 90:
        return False
    return _reply_text_has_spoken_teaching(reply_text, correction)


def _reply_text_sounds_like_metadata(reply_text: str) -> bool:
    """Detect robotic, metadata-style spoken replies."""
    clean = re.sub(r"\s+", " ", (reply_text or "").strip()).casefold()
    if not clean:
        return False
    return any(re.search(pattern, clean) for pattern in _METADATA_REPLY_PATTERNS)


def has_meaningful_correction(
    correction: str,
    user_message: str,
    explanation: str = "",
    example: str = "",
) -> bool:
    """Return True when learner feedback adds visible value beyond echoing the original text."""
    correction_cmp = _normalized_compare_text(correction)
    user_cmp = _normalized_compare_text(user_message)
    if not correction_cmp:
        return False
    if correction_cmp != user_cmp:
        return True
    if (explanation or "").strip() or (example or "").strip():
        return True
    return False


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

    needs_teaching = _needs_spoken_teaching(user_message, correction, score)
    has_feedback = bool(correction or explanation or example)

    if score >= 90 and has_feedback:
        return True

    if not correction and not needs_teaching:
        return False

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

    if correction_cmp and user_cmp and correction_cmp == user_cmp and _needs_meaningful_rewrite(user_message):
        return True

    if correction_cmp and user_cmp and correction_cmp == user_cmp and (score < 90 or explanation or example):
        return True

    if needs_teaching and not _reply_text_has_spoken_teaching(reply_text, correction):
        return True

    if _reply_text_overteaches(reply_text, correction, score):
        return True

    if _reply_text_sounds_like_metadata(reply_text):
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

    correction = (ai_response.get("correction") or "").strip() or None
    explanation = (ai_response.get("explanation") or "").strip() or None
    example = (ai_response.get("example") or "").strip() or None
    translated_reply_text = sanitize_translated_reply_text(
        reply_text,
        ai_response.get("translated_reply_text"),
        translation_language,
    )

    if score >= 90:
        correction = None
        explanation = None
        example = None
    elif not correction and (_needs_meaningful_rewrite(user_message) or explanation or example):
        correction = build_display_correction(user_message)

    normalized = {
        "reply_text": reply_text,
        "translated_reply_text": translated_reply_text,
        "correction": correction,
        "explanation": explanation,
        "example": example,
        "score": score,
    }

    correction_cmp = _normalized_compare_text(correction or "")
    user_cmp = _normalized_compare_text(user_message)
    if correction_cmp and user_cmp and correction_cmp == user_cmp and not explanation and not example and score < 90:
        normalized["score"] = 92

    return normalized
