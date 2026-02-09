"""Language and script detection for response and TTS routing."""
import re
from typing import Optional

# Unicode ranges for Indic scripts (used for script detection)
DEVANAGARI = r"[\u0900-\u097F]"  # Hindi, Marathi, etc.
MALAYALAM = r"[\u0D00-\u0D7F]"
TAMIL = r"[\u0B80-\u0BFF]"
TELUGU = r"[\u0C00-\u0C7F]"
KANNADA = r"[\u0C80-\u0CFF]"
BENGALI = r"[\u0980-\u09FF]"

# Map script pattern to IndicF5 language code
SCRIPT_TO_LANG = [
    (DEVANAGARI, "hi"),   # Hindi (fallback for Devanagari)
    (MALAYALAM, "ml"),
    (TAMIL, "ta"),
    (TELUGU, "te"),
    (KANNADA, "kn"),
    (BENGALI, "bn"),
]

# STT detected_lang to response_language (for known Indic codes)
STT_INDIC_MAP = {
    "hi": "hi",
    "mr": "hi",   # Marathi -> use Hindi ref for now
    "ml": "ml",
    "ta": "ta",
    "te": "te",
    "kn": "kn",
    "bn": "bn",
    "gu": "hi",   # Gujarati -> Hindi ref fallback
    "pa": "hi",   # Punjabi -> Hindi ref fallback
}


def _has_indic_script(text: str) -> bool:
    """Return True if text contains any Indic script."""
    for pattern, _ in SCRIPT_TO_LANG:
        if re.search(pattern, text):
            return True
    return False


def _script_to_lang(text: str) -> Optional[str]:
    """Return Indic language code from first detected script in text."""
    for pattern, lang in SCRIPT_TO_LANG:
        if re.search(pattern, text):
            return lang
    return None


def get_response_language(text: str, detected_lang: Optional[str] = None) -> str:
    """
    Determine response_language for LLM and TTS routing.

    - English mode: detected_lang is "en" and text has no Indic script -> "en"
    - Indic mode: otherwise -> "hi" / "ml" / "ta" / etc. (fallback "hi" if unknown Indic)

    Args:
        text: Transcribed or user message text
        detected_lang: Language code from STT (e.g. info.language), or None for text-only

    Returns:
        "en" for English (Chatterbox), or "hi"/"ml"/"ta"/etc. for IndicF5
    """
    has_indic = _has_indic_script(text)
    script_lang = _script_to_lang(text)

    # If text contains Indic script, use that script's language
    if script_lang:
        return script_lang

    # If STT detected an Indic language, use it (or map to supported code)
    if detected_lang and detected_lang in STT_INDIC_MAP:
        return STT_INDIC_MAP[detected_lang]

    # If STT said English and no Indic script -> English
    if detected_lang == "en" or (detected_lang is None and not has_indic):
        return "en"

    # Unknown or other language: default to English so we use Chatterbox
    return "en"
