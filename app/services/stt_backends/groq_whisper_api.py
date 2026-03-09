"""Groq Speech-to-Text backend: no local model; uses Groq whisper-large-v3-turbo (or whisper-large-v3).
https://console.groq.com/docs/speech-to-text
"""
import logging
import os
import tempfile
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# Groq models: whisper-large-v3-turbo (faster, cheaper), whisper-large-v3 (higher accuracy)


def transcribe(
    audio_file: bytes,
    filename: str = "audio.wav",
    language_hint: Optional[str] = None,
) -> tuple[str, str]:
    """
    Transcribe using Groq Whisper API (whisper-large-v3-turbo or whisper-large-v3).
    Returns (text, detected_lang) with detected_lang as ISO 639-1. Raw: no language hint passed by default.
    """
    if not getattr(settings, "groq_api_key", None):
        raise ValueError(
            "Groq API key not configured. Set GROQ_API_KEY when using Groq Whisper API (STT_WHISPER_LOCAL_ENABLED=false)."
        )
    model = getattr(settings, "stt_groq_model", "whisper-large-v3-turbo") or "whisper-large-v3-turbo"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file)
        tmp_path = tmp.name
    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)
        with open(tmp_path, "rb") as f:
            kwargs = {
                "file": f,
                "model": model,
                "response_format": "verbose_json",
                "temperature": 0.0,
            }
            if language_hint:
                kwargs["language"] = language_hint
            logger.info("Transcribing with Groq Whisper API (model=%s)", model)
            transcription = client.audio.transcriptions.create(**kwargs)
        text = (getattr(transcription, "text", None) or "").strip()
        if not text:
            raise ValueError("Could not transcribe audio. Please try speaking more clearly.")
        detected_lang = getattr(transcription, "language", None) or "en"
        if isinstance(detected_lang, str) and len(detected_lang) > 2:
            detected_lang = detected_lang[:2].lower()
        logger.info("Transcription: text=%s detected_lang=%s (len=%d)", text, detected_lang, len(text))
        return (text, detected_lang)
    except ImportError as e:
        logger.error("groq package not installed: %s", e)
        raise ValueError(
            "Groq SDK not installed. Install with: pip install groq"
        ) from e
    except Exception as e:
        err_msg = str(e).lower()
        if "401" in err_msg or "invalid" in err_msg or "authenticate" in err_msg:
            raise ValueError("Groq API key invalid or expired. Check GROQ_API_KEY.") from e
        if "429" in err_msg or "rate" in err_msg:
            raise ValueError("Groq rate limit exceeded. Please try again shortly.") from e
        logger.exception("Groq Whisper API error: %s", e)
        raise ValueError("Could not process audio. Please try again.") from e
    finally:
        try:
            os.unlink(tmp_path)
        except Exception as e:
            logger.warning("Failed to delete temp file %s: %s", tmp_path, e)


def warmup() -> None:
    """No-op: no local model to load. Used by /init-models when mode is groq_whisper_api."""
    if not getattr(settings, "groq_api_key", None):
        raise ValueError(
            "Groq API key not configured. Set GROQ_API_KEY when using Whisper API (STT_WHISPER_LOCAL_ENABLED=false)."
        )
    logger.info("Groq Whisper API backend ready (model=%s)", getattr(settings, "stt_groq_model", "whisper-large-v3-turbo"))
