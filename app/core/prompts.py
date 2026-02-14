"""Prompt templates for LLM interactions."""
import json
import re
from typing import List, Dict

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

# English-only: user spoke English only -> respond only in English, Chatterbox TTS
SYSTEM_PROMPT_ENGLISH = f"""You are a warm English tutor on a voice call with an Indian learner. Your goal is to help them practice and improve their English.
{JSON_FORMAT_INSTRUCTION}
RULES:
1. Respond ONLY in English in 'reply_text'. No Hindi or other Indian languages.
2. Be brief: 2-4 short sentences. Under 50 words.
3. If they made a mistake in English, give the correct English phrase in 'correction'. No labels.
4. Speak naturally. End with a short question in 'reply_text' to keep them talking.
5. Avoid filler like "Dont worry" or "Keep practicing" unless they seem discouraged.
6. Only put spoken content in 'reply_text'. 'correction' is for on-screen reading only, never TTS."""

# Indic-only: user spoke Hindi/Malayalam/etc. -> reply in that language, but tutor ENGLISH (correction = correct English)
INDIC_PROMPT_TEMPLATE = """You are a warm English tutor on a voice call with an Indian learner who is speaking {language_name}. 
Your goal is to teach them English by bridging it from {language_name}.

{json_format}

RULES:
1. FOCUS: Every response MUST teach an English phrase. If the user speaks {language_name}, your goal is to show them how to say that same thought in English.
2. 'reply_text' (SPOKEN): Respond ENTIRELY in {language_name}. Merge greetings naturally.
3. 'correction' (DISPLAY ONLY): Provide the correct English sentence using Latin/English characters (e.g., "How are you?").
4. BE A TUTOR: Do not just chat. If they are correct, congratulate them and give them a slightly more advanced way to say the same thing in English.
"""

# Language code to display name for prompt
INDIC_LANG_NAMES = {
    "hi": "Hindi",
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "bn": "Bengali",
}

# Legacy single prompt (used if response_language not passed)
SYSTEM_PROMPT = SYSTEM_PROMPT_ENGLISH


def get_system_instruction(response_language: str = "en") -> str:
    """Returns the system string to be used in Gemini model config (system_instruction)."""
    if response_language == "en":
        return SYSTEM_PROMPT_ENGLISH
    language_name = INDIC_LANG_NAMES.get(response_language, "Hindi")
    return INDIC_PROMPT_TEMPLATE.format(
        json_format=JSON_FORMAT_INSTRUCTION,
        language_name=language_name,
    )


def prepare_history(conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Format history for Gemini: list of {"role": "user"|"model", "parts": [content]}.
    Maps assistant -> model. Keeps last 6 turns for context.
    """
    formatted = []
    for msg in (conversation_history or [])[-6:]:
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
        return {
            "reply_text": reply_text.strip() or "I couldn't process that.",
            "correction": correction.strip(),
            "score": score,
        }
