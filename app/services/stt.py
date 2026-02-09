"""Speech-to-Text service: faster_whisper_medium or faster_whisper_large (both local)."""
import logging
import os
import tempfile
import threading
from typing import Optional
from app.core.config import settings
from app.utils.audio import validate_wav_file
from app.utils.device import get_infer_device

logger = logging.getLogger(__name__)

# Lazy-loaded faster-whisper models (loaded on first use); locks prevent double-load under concurrency
_faster_whisper_model = None
_faster_whisper_model_large_v3 = None
_lock_medium = threading.Lock()
_lock_large = threading.Lock()

# Inference lock: with Gunicorn workers=1 and multiple threads, only one STT inference runs at a
# time on the GPU to avoid OOM and ensure stability.
_inference_lock = threading.Lock()

# Systran/faster-whisper-large-v3 (CTranslate2 format, Hugging Face) for faster_whisper_large mode
# https://huggingface.co/Systran/faster-whisper-large-v3
WHISPER_LARGE_V3_HF = "Systran/faster-whisper-large-v3"


def _transcribe_faster_whisper(audio_file: bytes, filename: str) -> tuple[str, str]:
    """Transcribe using local faster-whisper (configurable model size). Returns (text, detected_lang)."""
    model = _get_faster_whisper_model()
    return _run_faster_whisper_transcribe(
        model, audio_file, settings.stt_faster_whisper_model_size
    )


def _transcribe_whisper_large_v3(audio_file: bytes, filename: str) -> tuple[str, str]:
    """Transcribe using Systran/faster-whisper-large-v3 (local). Returns (text, detected_lang)."""
    model = _get_faster_whisper_model_large_v3()
    return _run_faster_whisper_transcribe(model, audio_file, WHISPER_LARGE_V3_HF)


def _run_faster_whisper_transcribe(
    model, audio_file: bytes, model_label: str
) -> tuple[str, str]:
    """Common transcribe path for faster-whisper models. Holds _inference_lock for GPU safety."""
    with _inference_lock:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_file)
            temp_file_path = temp_file.name
        try:
            logger.info(f"Transcribing audio with faster-whisper (model: {model_label})")
            segments, info = model.transcribe(
                temp_file_path,
                language=None,
                beam_size=5,
            )
            logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            if not transcribed_text:
                raise ValueError("Could not transcribe audio. Please try speaking more clearly.")
            logger.info(f"Transcription successful: {len(transcribed_text)} characters")
            detected_lang = getattr(info, "language", "en") or "en"
            return (transcribed_text, detected_lang)
        finally:
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")


def _get_stt_device_and_compute():
    """Device and compute_type for STT: GPU in prod when CUDA available, else CPU int8."""
    device = get_infer_device()
    if device == "cuda":
        return "cuda", "float16"
    return "cpu", "int8"


def _get_faster_whisper_model():
    """Lazy load faster-whisper model (configurable size). Thread-safe."""
    global _faster_whisper_model
    with _lock_medium:
        if _faster_whisper_model is None:
            try:
                from faster_whisper import WhisperModel
                model_size = settings.stt_faster_whisper_model_size
                device, compute_type = _get_stt_device_and_compute()
                logger.info(f"Loading faster-whisper model: {model_size} (device={device}, compute_type={compute_type})")
                _faster_whisper_model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                )
                logger.info("faster-whisper model loaded successfully")
            except ImportError:
                logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
                raise ValueError("faster-whisper is not installed. Please install it: pip install faster-whisper")
    return _faster_whisper_model


def _get_faster_whisper_model_large_v3():
    """Lazy load Systran/faster-whisper-large-v3. Thread-safe."""
    global _faster_whisper_model_large_v3
    with _lock_large:
        if _faster_whisper_model_large_v3 is None:
            try:
                from faster_whisper import WhisperModel
                device, compute_type = _get_stt_device_and_compute()
                logger.info(f"Loading faster-whisper model: {WHISPER_LARGE_V3_HF} (device={device}, compute_type={compute_type})")
                _faster_whisper_model_large_v3 = WhisperModel(
                    WHISPER_LARGE_V3_HF,
                    device=device,
                    compute_type=compute_type,
                )
                logger.info("faster-whisper (Systran/faster-whisper-large-v3) model loaded successfully")
            except ImportError:
                logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
                raise ValueError("faster-whisper is not installed. Please install it: pip install faster-whisper")
    return _faster_whisper_model_large_v3


def transcribe_audio(
    audio_file: bytes,
    filename: str = "audio.wav",
    mode: Optional[str] = None,
) -> tuple[str, str]:
    """
    Transcribe audio file to text using faster-whisper (medium or large, both local).

    Args:
        audio_file: Audio file bytes
        filename: Original filename (for format detection)
        mode: "faster_whisper_medium" or "faster_whisper_large"; if None, uses settings.stt_mode

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

    effective_mode = mode if mode in ("faster_whisper_medium", "faster_whisper_large") else settings.stt_mode

    try:
        if effective_mode == "faster_whisper_medium":
            return _transcribe_faster_whisper(audio_file, filename)
        else:
            return _transcribe_whisper_large_v3(audio_file, filename)
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
            _get_faster_whisper_model()
            return {"status": "loaded", "mode": settings.stt_mode}
        _get_faster_whisper_model_large_v3()
        return {"status": "loaded", "mode": settings.stt_mode}
    except Exception as e:
        logger.exception("STT init failed")
        return {"status": "failed", "error": str(e)}
