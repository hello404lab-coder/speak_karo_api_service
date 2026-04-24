"""Language and script detection for reply, TTS routing, and translation preferences."""
import re
from typing import Optional


LANGUAGE_NAMES = {
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

LANGUAGE_ALIASES = {
    "english": "en",
    "en-us": "en",
    "en-gb": "en",
    "hindi": "hi",
    "hindustani": "hi",
    "marathi": "mr",
    "malayalam": "ml",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "bengali": "bn",
    "bangla": "bn",
    "gujarati": "gu",
    "punjabi": "pa",
    "urdu": "ur",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "mandarin": "zh",
    "mandarin chinese": "zh",
    "chinese": "zh",
    "arabic": "ar",
    "portuguese": "pt",
    "japanese": "ja",
    "korean": "ko",
    "हिंदी": "hi",
    "മലയാളം": "ml",
    "தமிழ்": "ta",
    "తెలుగు": "te",
    "ಕನ್ನಡ": "kn",
    "বাংলা": "bn",
    "मराठी": "mr",
    "ગુજરાતી": "gu",
    "ਪੰਜਾਬੀ": "pa",
    "اردو": "ur",
    "العربية": "ar",
    "中文": "zh",
    "日本語": "ja",
    "한국어": "ko",
}

# Unicode ranges for supported scripts (used for script detection and translation validation)
DEVANAGARI = r"[\u0900-\u097F]"  # Hindi, Marathi, etc.
MALAYALAM = r"[\u0D00-\u0D7F]"
TAMIL = r"[\u0B80-\u0BFF]"
TELUGU = r"[\u0C00-\u0C7F]"
KANNADA = r"[\u0C80-\u0CFF]"
BENGALI = r"[\u0980-\u09FF]"
GUJARATI = r"[\u0A80-\u0AFF]"
GURMUKHI = r"[\u0A00-\u0A7F]"
ARABIC = r"[\u0600-\u06FF]"
HAN = r"[\u4E00-\u9FFF]"
HIRAGANA_KATAKANA = r"[\u3040-\u30FF]"
HANGUL = r"[\uAC00-\uD7AF]"

# Map script pattern to IndicF5 language code
SCRIPT_TO_LANG = [
    (DEVANAGARI, "hi"),   # Hindi (fallback for Devanagari)
    (MALAYALAM, "ml"),
    (TAMIL, "ta"),
    (TELUGU, "te"),
    (KANNADA, "kn"),
    (BENGALI, "bn"),
]

TARGET_SCRIPT_PATTERNS = {
    "hi": DEVANAGARI,
    "mr": DEVANAGARI,
    "ml": MALAYALAM,
    "ta": TAMIL,
    "te": TELUGU,
    "kn": KANNADA,
    "bn": BENGALI,
    "gu": GUJARATI,
    "pa": GURMUKHI,
    "ur": ARABIC,
    "ar": ARABIC,
    "zh": HAN,
    "ja": rf"(?:{HAN}|{HIRAGANA_KATAKANA})",
    "ko": HANGUL,
}

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


def normalize_language_code(value: Optional[str]) -> Optional[str]:
    """Normalize a free-text language name or locale into a short ISO-style language code."""
    raw = (value or "").strip()
    if not raw:
        return None

    normalized = raw.lower().replace("_", "-")
    primary = normalized.split("-", 1)[0]
    if primary in LANGUAGE_NAMES:
        return primary
    if normalized in LANGUAGE_NAMES:
        return normalized

    alias = LANGUAGE_ALIASES.get(normalized)
    if alias:
        return alias

    alias = LANGUAGE_ALIASES.get(raw.lower())
    if alias:
        return alias

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
    normalized_detected = normalize_language_code(detected_lang)
    if normalized_detected and normalized_detected in STT_INDIC_MAP:
        return STT_INDIC_MAP[normalized_detected]

    # If STT said English and no Indic script -> English
    if normalized_detected == "en" or (detected_lang is None and not has_indic):
        return "en"

    # Unknown or other language: default to English so we use Chatterbox
    return "en"


def resolve_response_language(
    text: str,
    detected_lang: Optional[str] = None,
    override: Optional[str] = None,
) -> str:
    """
    Like get_response_language, but if override is a non-empty code (e.g. from the client),
    use it for this turn (helps romanized Indic where script detection yields en).
    """
    override_code = normalize_language_code(override)
    if override_code:
        return override_code
    return get_response_language(text, detected_lang)


def resolve_reply_language(
    text: str,
    detected_lang: Optional[str] = None,
    override: Optional[str] = None,
) -> str:
    """Resolve the actual assistant reply/TTS language."""
    return resolve_response_language(text, detected_lang, override)


def resolve_translation_language(
    requested_language: Optional[str],
    fallback_native_language: Optional[str] = None,
    reply_language: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve the assistant translation target.

    Prefers the explicit request language, then the user's stored native language.
    Returns None when no supported target exists or when the target matches reply_language.
    """
    requested_code = normalize_language_code(requested_language)
    fallback_code = normalize_language_code(fallback_native_language)
    target = requested_code or fallback_code
    if not target:
        return None

    reply_code = normalize_language_code(reply_language)
    if reply_code and reply_code == target:
        return None

    return target


def sanitize_translated_reply_text(
    reply_text: str,
    translated_reply_text: Optional[str],
    translation_language: Optional[str],
) -> Optional[str]:
    """
    Drop obviously bad translations while staying in a single-call design.

    Returns cleaned translated text, or None when the text is blank, unchanged from reply_text,
    or clearly not written in the target language's expected script.
    """
    target = normalize_language_code(translation_language)
    clean = re.sub(r"\s+", " ", (translated_reply_text or "").strip())
    if not target or not clean:
        return None

    source_clean = re.sub(r"\s+", " ", (reply_text or "").strip())
    source_cmp = re.sub(r"[\W_]+", "", source_clean.casefold())
    clean_cmp = re.sub(r"[\W_]+", "", clean.casefold())
    if source_cmp and clean_cmp == source_cmp:
        return None

    target_pattern = TARGET_SCRIPT_PATTERNS.get(target)
    if not target_pattern:
        return clean

    if re.search(target_pattern, clean):
        return clean

    for lang, pattern in TARGET_SCRIPT_PATTERNS.items():
        if lang != target and re.search(pattern, clean):
            return None

    total_letters = sum(1 for ch in clean if ch.isalpha())
    latin_letters = sum(1 for ch in clean if ("a" <= ch.lower() <= "z"))
    if total_letters and (latin_letters / total_letters) >= 0.4:
        return None

    return None
