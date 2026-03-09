"""Speech-to-Text service: dispatches to backends (local Whisper or OpenAI Whisper API)."""
import logging
from typing import Optional

from app.core.config import settings
from app.utils.audio import validate_wav_file

logger = logging.getLogger(__name__)

ALLOWED_STT_MODES = ("faster_whisper_medium", "faster_whisper_large", "openai_whisper_large_v3", "openai_whisper_api", "groq_whisper_api")
MODE_OPENAI_WHISPER_API = "openai_whisper_api"
MODE_GROQ_WHISPER_API = "groq_whisper_api"


def transcribe_audio(
    audio_file: bytes,
    filename: str = "audio.wav",
    mode: Optional[str] = None,
) -> tuple[str, str]:
    """
    Transcribe audio file to text using the configured or requested STT backend.
    When STT_WHISPER_LOCAL_ENABLED=false, only Groq Whisper API is used (no local model loaded).

    Args:
        audio_file: Audio file bytes
        filename: Original filename (for format detection)
        mode: One of faster_whisper_medium, faster_whisper_large, openai_whisper_large_v3, openai_whisper_api, groq_whisper_api; if None, uses settings

    Returns:
        Tuple of (transcribed_text, detected_lang). detected_lang is the ISO 639-1 code (e.g. "en", "hi").

    Raises:
        ValueError: If audio file is invalid or transcription fails
    """
    try:
        validate_wav_file(audio_file)
    except ValueError as e:
        logger.error(f"Audio validation failed: {e}")
        raise ValueError("Invalid audio file. Please upload a valid WAV file.")

    # When local Whisper is disabled, always use Groq Whisper API (no local model loaded)
    if not getattr(settings, "stt_whisper_local_enabled", True):
        from app.services.stt_backends import groq_whisper_api
        return groq_whisper_api.transcribe(
            audio_file, filename, language_hint="en"  # raw: auto-detect, transcribe in spoken language
        )

    effective_mode = mode if mode in ALLOWED_STT_MODES else settings.stt_mode
    # Don't allow openai_whisper_api as a mode when local is enabled (it's only used when local is disabled)
    if effective_mode == MODE_OPENAI_WHISPER_API or effective_mode == MODE_GROQ_WHISPER_API:
        effective_mode = settings.stt_mode

    try:
        if effective_mode == "faster_whisper_medium":
            from app.services.stt_backends import faster_whisper
            return faster_whisper.transcribe(audio_file, filename, variant="medium")
        if effective_mode == "faster_whisper_large":
            from app.services.stt_backends import faster_whisper
            return faster_whisper.transcribe(audio_file, filename, variant="large")
        if effective_mode == "openai_whisper_large_v3":
            from app.services.stt_backends import transformers_whisper
            return transformers_whisper.transcribe(
                audio_file, filename, language_hint="en"  # raw: auto-detect, transcribe in spoken language
            )
        # Fallback to large if unknown
        from app.services.stt_backends import faster_whisper
        return faster_whisper.transcribe(audio_file, filename, variant="large")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"STT transcription error ({effective_mode}): {e}", exc_info=True)
        raise ValueError("Could not process audio. Please try again.")


def init_stt_models() -> dict:
    """
    Load the STT model for the current config (used by /init-models warmup).
    When STT_WHISPER_LOCAL_ENABLED=false, no local model is loaded; only Groq Whisper API is used.
    Returns {"status": "loaded"|"disabled", "mode": str} or {"status": "failed", "error": str}.
    """
    if not getattr(settings, "stt_whisper_local_enabled", True):
        try:
            from app.services.stt_backends import groq_whisper_api
            groq_whisper_api.warmup()
            return {"status": "loaded", "mode": MODE_GROQ_WHISPER_API}
        except Exception as e:
            logger.exception("STT Groq Whisper API warmup failed")
            return {"status": "failed", "error": str(e)}

    try:
        if settings.stt_mode == "faster_whisper_medium":
            from app.services.stt_backends import faster_whisper
            faster_whisper.warmup("medium")
            return {"status": "loaded", "mode": settings.stt_mode}
        if settings.stt_mode == "faster_whisper_large":
            from app.services.stt_backends import faster_whisper
            faster_whisper.warmup("large")
            return {"status": "loaded", "mode": settings.stt_mode}
        if settings.stt_mode == "openai_whisper_large_v3":
            from app.services.stt_backends import transformers_whisper
            transformers_whisper.warmup()
            return {"status": "loaded", "mode": settings.stt_mode}
        # Default: large
        from app.services.stt_backends import faster_whisper
        faster_whisper.warmup("large")
        return {"status": "loaded", "mode": settings.stt_mode}
    except Exception as e:
        logger.exception("STT init failed")
        return {"status": "failed", "error": str(e)}
