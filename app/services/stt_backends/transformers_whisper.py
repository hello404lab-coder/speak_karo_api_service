"""Hugging Face Transformers STT backend: openai/whisper-large-v3."""
import logging
import os
import tempfile
import threading

from app.utils.device import get_infer_device

logger = logging.getLogger(__name__)

MODEL_ID = "openai/whisper-large-v3"

_pipeline = None
_lock = threading.Lock()
_inference_lock = threading.Lock()


def _get_pipeline():
    """Lazy load Whisper ASR pipeline. Thread-safe."""
    global _pipeline
    with _lock:
        if _pipeline is None:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            device = get_infer_device()
            device_str = "cuda:0" if device == "cuda" else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

            logger.info(
                f"Loading Transformers Whisper model: {MODEL_ID} (device={device_str}, dtype={torch_dtype})"
            )
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                MODEL_ID,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(device_str)
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            _pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device_str,
                chunk_length_s=30,
                batch_size=1,
            )
            logger.info(f"Transformers Whisper ({MODEL_ID}) loaded successfully")
    return _pipeline


def transcribe(audio_file: bytes, filename: str) -> tuple[str, str]:
    """Transcribe using openai/whisper-large-v3. Returns (text, detected_lang)."""
    pipe = _get_pipeline()
    with _inference_lock:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_file)
            temp_path = temp_file.name
        try:
            logger.info(f"Transcribing audio with Transformers Whisper (model: {MODEL_ID})")
            result = pipe(temp_path)
            text = (result.get("text") or "").strip()
            if not text:
                raise ValueError("Could not transcribe audio. Please try speaking more clearly.")
            detected_lang = result.get("language", "en") or "en"
            logger.info(
                "Transcription: text=%s detected_lang=%s (len=%d)",
                text, detected_lang, len(text),
            )
            return (text, detected_lang)
        finally:
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")


def warmup() -> None:
    """Load the pipeline (used by /init-models)."""
    _get_pipeline()
