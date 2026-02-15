"""Faster-whisper STT backend: medium or large (Systran/faster-whisper-large-v3)."""
import logging
import os
import tempfile
import threading
from typing import Literal

from app.core.config import settings
from app.utils.device import get_infer_device

logger = logging.getLogger(__name__)

_faster_whisper_model = None
_faster_whisper_model_large_v3 = None
_lock_medium = threading.Lock()
_lock_large = threading.Lock()
_inference_lock = threading.Lock()

WHISPER_LARGE_V3_HF = "Systran/faster-whisper-large-v3"


def _get_stt_device_and_compute() -> tuple[str, str]:
    """Device and compute_type for STT: GPU in prod when CUDA available, else CPU int8."""
    device = get_infer_device()
    if device == "cuda":
        return "cuda", "float16"
    return "cpu", "int8"


def _get_model(variant: Literal["medium", "large"]):
    """Lazy load faster-whisper model. Thread-safe."""
    global _faster_whisper_model, _faster_whisper_model_large_v3
    if variant == "medium":
        with _lock_medium:
            if _faster_whisper_model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError:
                    logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
                    raise ValueError("faster-whisper is not installed. Please install it: pip install faster-whisper")
                model_size = settings.stt_faster_whisper_model_size
                device, compute_type = _get_stt_device_and_compute()
                logger.info(
                    f"Loading faster-whisper model: {model_size} (device={device}, compute_type={compute_type})"
                )
                _faster_whisper_model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                )
                logger.info("faster-whisper model loaded successfully")
            return _faster_whisper_model
    else:
        with _lock_large:
            if _faster_whisper_model_large_v3 is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError:
                    logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
                    raise ValueError("faster-whisper is not installed. Please install it: pip install faster-whisper")
                device, compute_type = _get_stt_device_and_compute()
                logger.info(
                    f"Loading faster-whisper model: {WHISPER_LARGE_V3_HF} (device={device}, compute_type={compute_type})"
                )
                _faster_whisper_model_large_v3 = WhisperModel(
                    WHISPER_LARGE_V3_HF,
                    device=device,
                    compute_type=compute_type,
                )
                logger.info("faster-whisper (Systran/faster-whisper-large-v3) model loaded successfully")
            return _faster_whisper_model_large_v3


def _run_transcribe(model, audio_file: bytes, model_label: str) -> tuple[str, str]:
    """Common transcribe path. Holds _inference_lock for GPU safety."""
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
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            if not transcribed_text:
                raise ValueError("Could not transcribe audio. Please try speaking more clearly.")
            detected_lang = getattr(info, "language", "en") or "en"
            logger.info(
                "Detected language: %s (probability: %.2f)",
                info.language, info.language_probability,
            )
            logger.info(
                "Transcription: text=%s detected_lang=%s (len=%d)",
                transcribed_text, detected_lang, len(transcribed_text),
            )
            return (transcribed_text, detected_lang)
        finally:
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")


def transcribe(
    audio_file: bytes,
    filename: str,
    variant: Literal["medium", "large"] = "large",
) -> tuple[str, str]:
    """Transcribe using faster-whisper. Returns (text, detected_lang)."""
    model = _get_model(variant)
    label = settings.stt_faster_whisper_model_size if variant == "medium" else WHISPER_LARGE_V3_HF
    return _run_transcribe(model, audio_file, label)


def warmup(variant: Literal["medium", "large"]) -> None:
    """Load the model for the given variant (used by /init-models)."""
    _get_model(variant)
