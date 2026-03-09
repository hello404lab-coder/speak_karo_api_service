"""OpenAI Whisper API STT backend: no local model; uses POST /v1/audio/transcriptions."""
import logging
import os
import tempfile
from typing import Optional

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

OPENAI_TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"
MODEL = "whisper-1"

# Map verbose_json language name (e.g. "english") to ISO 639-1 (e.g. "en")
# https://platform.openai.com/docs/api-reference/audio/verbose-json-object
LANGUAGE_NAME_TO_ISO639: dict[str, str] = {
    "english": "en",
    "chinese": "zh",
    "german": "de",
    "spanish": "es",
    "russian": "ru",
    "korean": "ko",
    "french": "fr",
    "japanese": "ja",
    "portuguese": "pt",
    "turkish": "tr",
    "polish": "pl",
    "arabic": "ar",
    "hindi": "hi",
    "bengali": "bn",
    "tamil": "ta",
    "telugu": "te",
    "marathi": "mr",
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "punjabi": "pa",
    "urdu": "ur",
    "vietnamese": "vi",
    "italian": "it",
    "dutch": "nl",
    "indonesian": "id",
    "thai": "th",
}


def _lang_to_iso639(lang: Optional[str]) -> str:
    """Convert API language (e.g. 'english') to ISO 639-1."""
    if not lang:
        return "en"
    key = (lang or "").strip().lower()
    return LANGUAGE_NAME_TO_ISO639.get(key, key[:2] if len(key) >= 2 else "en")


def transcribe(
    audio_file: bytes,
    filename: str = "audio.wav",
    language_hint: Optional[str] = None,
) -> tuple[str, str]:
    """
    Transcribe using OpenAI Whisper API. No local model loaded.
    Returns (text, detected_lang) with detected_lang as ISO 639-1.
    """
    if not settings.openai_api_key:
        raise ValueError(
            "OpenAI API key not configured. Set OPENAI_API_KEY when using Whisper API (STT_WHISPER_LOCAL_ENABLED=false)."
        )
    timeout = getattr(settings, "stt_timeout_seconds", 30)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            files = {"file": (filename or "audio.wav", f, "audio/wav")}
            data = {"model": MODEL, "response_format": "verbose_json"}
            if language_hint:
                data["language"] = language_hint
            headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
            logger.info("Transcribing with OpenAI Whisper API (model=%s)", MODEL)
            resp = requests.post(
                OPENAI_TRANSCRIPTIONS_URL,
                files=files,
                data=data,
                headers=headers,
                timeout=timeout,
            )
        resp.raise_for_status()
        out = resp.json()
        text = (out.get("text") or "").strip()
        if not text:
            raise ValueError("Could not transcribe audio. Please try speaking more clearly.")
        detected_lang = _lang_to_iso639(out.get("language"))
        logger.info("Transcription: text=%s detected_lang=%s (len=%d)", text, detected_lang, len(text))
        return (text, detected_lang)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            raise ValueError("OpenAI API key invalid or expired. Check OPENAI_API_KEY.") from e
        if e.response is not None and e.response.status_code == 429:
            raise ValueError("OpenAI rate limit exceeded. Please try again shortly.") from e
        try:
            err_body = e.response.json() if e.response else {}
            err_msg = err_body.get("error", {}).get("message", str(e))
        except Exception:
            err_msg = str(e)
        logger.error("OpenAI Whisper API error: %s", err_msg)
        raise ValueError("Could not process audio. Please try again.") from e
    except requests.exceptions.Timeout:
        raise ValueError("Transcription request timed out. Please try again.") from None
    except requests.exceptions.RequestException as e:
        logger.exception("OpenAI Whisper API request failed")
        raise ValueError("Could not process audio. Please try again.") from e
    finally:
        try:
            os.unlink(tmp_path)
        except Exception as e:
            logger.warning("Failed to delete temp file %s: %s", tmp_path, e)


def warmup() -> None:
    """No-op: no local model to load. Used by /init-models when mode is openai_whisper_api."""
    if not settings.openai_api_key:
        raise ValueError(
            "OpenAI API key not configured. Set OPENAI_API_KEY when using Whisper API (STT_WHISPER_LOCAL_ENABLED=false)."
        )
    logger.info("OpenAI Whisper API backend ready (no local model)")
