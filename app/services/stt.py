"""Speech-to-Text service: dispatches to backends (faster_whisper or openai/whisper-large-v3 via Transformers)."""
import logging
from typing import Optional

from app.core.config import settings
from app.utils.audio import validate_wav_file

logger = logging.getLogger(__name__)

ALLOWED_STT_MODES = ("faster_whisper_medium", "faster_whisper_large", "openai_whisper_large_v3")


def transcribe_audio(
    audio_file: bytes,
    filename: str = "audio.wav",
    mode: Optional[str] = None,
) -> tuple[str, str]:
    """
    Transcribe audio file to text using the configured or requested STT backend.

    Args:
        audio_file: Audio file bytes
        filename: Original filename (for format detection)
        mode: One of faster_whisper_medium, faster_whisper_large, openai_whisper_large_v3; if None, uses settings.stt_mode

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

    effective_mode = mode if mode in ALLOWED_STT_MODES else settings.stt_mode

    try:
        if effective_mode == "faster_whisper_medium":
            from app.services.stt_backends import faster_whisper
            return faster_whisper.transcribe(audio_file, filename, variant="medium")
        if effective_mode == "faster_whisper_large":
            from app.services.stt_backends import faster_whisper
            return faster_whisper.transcribe(audio_file, filename, variant="large")
        if effective_mode == "openai_whisper_large_v3":
            from app.services.stt_backends import transformers_whisper
            return transformers_whisper.transcribe(audio_file, filename)
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
    Load the STT model for the current stt_mode (used by /init-models warmup).
    Returns {"status": "loaded", "mode": str} or {"status": "failed", "error": str}.
    """
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
