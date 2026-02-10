"""Text-to-Speech service with cloud/local storage."""
import asyncio
import hashlib
import logging
import os
import re
import struct
import threading
from io import BytesIO
from pathlib import Path
from typing import Iterator, Optional
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from app.core.config import settings
from app.services.cache import get, set
from app.utils.device import get_infer_device

logger = logging.getLogger(__name__)

# Lazy-loaded Chatterbox-Turbo model (English, loaded on first use); lock prevents double-load
_turbo_model = None
_turbo_device = None
_lock_turbo = threading.Lock()

# Lazy-loaded IndicF5 model and vocoder (loaded on first use); lock prevents double-load
_indicf5_model = None
_indicf5_vocoder = None
_indicf5_device = None
_indicf5_available = True  # Set False if load fails so we fallback to Turbo
_lock_indicf5 = threading.Lock()

# Inference lock: one TTS inference at a time on the GPU to avoid OOM with workers=1 and multiple threads.
_inference_lock = threading.Lock()

# IndicF5 ref audio filenames and ref text per language (must match the ref WAV content)
INDICF5_REF_FILENAMES = {
    "hi": "MAR_F_HAPPY_00001.wav",   # Devanagari (Marathi) for Hindi
    "ml": "MAL_F_HAPPY_00001.wav",
    "ta": "TAM_F_HAPPY_00001.wav",
    "te": "TAM_F_HAPPY_00001.wav",   # Fallback to Tamil if no Telugu ref
    "kn": "TAM_F_HAPPY_00001.wav",
    "bn": "TAM_F_HAPPY_00001.wav",
}
INDICF5_REF_TEXTS = {
    "hi": "आपकी बात समझ में आई। हम इंग्लिश प्रैक्टिस करेंगे।",
    "ml": "കുറച്ചു നേരമായി ഞാൻ നിന്നെ കാത്തിരിക്കുന്നു, എവിടെയായിരുന്നു നീ?",
    "ta": "உங்களுடைய ஹோம்வொர்க் எங்கே? இன்னும் முடிக்கவில்லையா? பரவாயில்லை, இப்போதே ட்ரை பண்ணுங்க, நான் ஹெல்ப் பண்றேன்.",
    "te": "ఉంగళుడుగారి హోంవర్క్ ఎక్కడ? ఇంకా ముగించలేదా?",
    "kn": "ನಿಮ್ಮ ಹೋಮ್‌ವರ್ಕ್ ಎಲ್ಲಿ? ಇನ್ನೂ ಮುಗಿಸಿಲ್ಲವೇ?",
    "bn": "আপনার হোমওয়ার্ক কোথায়? এখনও শেষ করেননি?",
}

# Ensure audio storage directory exists
if not os.path.exists(settings.audio_storage_path):
    os.makedirs(settings.audio_storage_path, exist_ok=True)


def _generate_cache_key(text: str, response_language: str = "en") -> str:
    """Generate cache key from text and response language (so en vs Indic don't collide)."""
    return f"tts:{response_language}:{hashlib.md5(text.encode()).hexdigest()}"


def _store_audio_local(audio_bytes: bytes, filename: str) -> str:
    """Store audio file locally and return URL."""
    filepath = os.path.join(settings.audio_storage_path, filename)
    
    with open(filepath, "wb") as f:
        f.write(audio_bytes)
    
    # Return URL for local serving
    return f"{settings.audio_base_url}/{filename}"


def _generate_presigned_url(s3_key: str) -> Optional[str]:
    """Generate a presigned GET URL for an S3 object. Returns None if S3 not configured or on error."""
    if not all([
        settings.aws_access_key_id,
        settings.aws_secret_access_key,
        settings.s3_bucket_name
    ]):
        return None
    try:
        import boto3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region or 'us-east-1'
        )
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': settings.s3_bucket_name, 'Key': s3_key},
            ExpiresIn=settings.s3_presigned_expiry_seconds
        )
        return url
    except ImportError:
        logger.warning("boto3 not installed")
        return None
    except Exception as e:
        logger.error(f"Presigned URL generation failed: {e}")
        return None


def _store_audio_cloud(audio_bytes: bytes, filename: str) -> Optional[str]:
    """Store audio file in S3 and return the S3 key (e.g. 'audio/filename.mp3'), or None."""
    if not all([
        settings.aws_access_key_id,
        settings.aws_secret_access_key,
        settings.s3_bucket_name
    ]):
        return None
    
    s3_key = f"ai/audio/{filename}"
    try:
        import boto3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region or 'us-east-1'
        )
        s3_client.put_object(
            Bucket=settings.s3_bucket_name,
            Key=s3_key,
            Body=audio_bytes,
            ContentType="audio/mpeg"
        )
        return s3_key
    except ImportError:
        logger.warning("boto3 not installed, falling back to local storage")
        return None
    except Exception as e:
        logger.error(f"Cloud storage error: {e}, falling back to local")
        return None


def _parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    """
    Parses bits per sample and rate from an audio MIME type string.
    
    Args:
        mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000" or "audio/L16;codec=pcm;rate=24000").
    
    Returns:
        A dictionary with "bits_per_sample" and "rate" keys.
    """
    bits_per_sample = 16
    rate = 24000
    
    # Check main type for L16, L24, etc.
    if mime_type.startswith("audio/L"):
        try:
            # Extract L16, L24, etc. from "audio/L16" or "audio/L16;..."
            main_part = mime_type.split(";")[0]  # Get "audio/L16"
            bits_str = main_part.split("L", 1)[1]  # Get "16"
            bits_per_sample = int(bits_str)
        except (ValueError, IndexError):
            pass  # Keep default
    
    # Extract rate from parameters
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass  # Keep rate as default
    
    logger.debug(f"Parsed MIME type '{mime_type}': bits_per_sample={bits_per_sample}, rate={rate}")
    return {"bits_per_sample": bits_per_sample, "rate": rate}


def _convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """
    Generates a WAV file header for the given audio data and parameters.
    
    Args:
        audio_data: The raw audio data as a bytes object.
        mime_type: Mime type of the audio data.
    
    Returns:
        A bytes object representing the WAV file with header.
    """
    parameters = _parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size
    
    # WAV file format header
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",          # ChunkID
        chunk_size,       # ChunkSize (total file size - 8 bytes)
        b"WAVE",          # Format
        b"fmt ",          # Subchunk1ID
        16,               # Subchunk1Size (16 for PCM)
        1,                # AudioFormat (1 for PCM)
        num_channels,     # NumChannels
        sample_rate,      # SampleRate
        byte_rate,        # ByteRate
        block_align,      # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",          # Subchunk2ID
        data_size         # Subchunk2Size (size of audio data)
    )
    return header + audio_data


def _convert_wav_to_mp3(wav_bytes: bytes) -> bytes:
    """
    Convert WAV audio bytes to MP3 format.
    
    Args:
        wav_bytes: WAV audio data
    
    Returns:
        MP3 audio bytes
    """
    try:
        # Create AudioSegment from WAV data
        wav_buffer = BytesIO(wav_bytes)
        wav_buffer.seek(0)  # Ensure we're at the start
        audio = AudioSegment.from_wav(wav_buffer)
        
        # Check if audio is valid
        if len(audio) == 0:
            logger.error("Audio segment is empty after WAV conversion")
            raise ValueError("Generated audio is empty")
        
        duration_ms = len(audio)
        duration_sec = duration_ms / 1000.0
        logger.info(f"Audio duration: {duration_ms}ms ({duration_sec:.2f}s), frame rate: {audio.frame_rate}Hz, channels: {audio.channels}")
        
        if duration_sec < 0.1:  # Less than 100ms is suspicious
            logger.warning(f"Audio duration is very short: {duration_sec:.2f}s")
        
        # Export to MP3 format
        mp3_buffer = BytesIO()
        audio.export(mp3_buffer, format="mp3", bitrate="128k")
        mp3_buffer.seek(0)
        mp3_data = mp3_buffer.read()
        
        if len(mp3_data) < 100:  # MP3 files should be at least a few hundred bytes
            logger.error(f"MP3 conversion produced suspiciously small file: {len(mp3_data)} bytes for {duration_sec:.2f}s audio")
            raise ValueError("MP3 conversion failed - output too small")
        
        logger.info(f"MP3 conversion successful: {len(mp3_data)} bytes for {duration_sec:.2f}s audio")
        return mp3_data
    except FileNotFoundError as e:
        logger.error(f"ffmpeg not found. Please install ffmpeg: {e}")
        raise ValueError("Audio conversion requires ffmpeg. Please install ffmpeg and try again.")
    except Exception as e:
        logger.error(f"Error converting WAV to MP3: {e}", exc_info=True)
        raise ValueError("Failed to convert audio format. Please try again.")


def _get_turbo_model():
    """Lazy load Chatterbox-Turbo (or ChatterboxTTS fallback) for English TTS. Thread-safe. Uses get_infer_device() so STT and TTS share the same GPU in prod."""
    global _turbo_model, _turbo_device
    with _lock_turbo:
        if _turbo_model is None:
            device = get_infer_device()
            _turbo_device = device

            # Both Turbo and ChatterboxTTS use perth.PerthImplicitWatermarker(); in some envs it is None (resemble-ai/chatterbox#198). Patch once before loading either.
            try:
                import perth
                if perth.PerthImplicitWatermarker is None and getattr(perth, "DummyWatermarker", None) is not None:
                    perth.PerthImplicitWatermarker = perth.DummyWatermarker
                    logger.debug("Patched perth.PerthImplicitWatermarker to DummyWatermarker")
            except Exception:
                pass

            # Prefer ChatterboxTurboTTS (chatterbox.tts_turbo); PyPI package may only have chatterbox.tts
            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                logger.info(f"Loading Chatterbox-Turbo model (device: {device})")
                _turbo_model = ChatterboxTurboTTS.from_pretrained(device=device)
                logger.info("Chatterbox-Turbo model loaded successfully")
            except ImportError:
                try:
                    from chatterbox.tts import ChatterboxTTS
                    logger.info(f"Loading ChatterboxTTS (fallback, device: {device})")
                    _turbo_model = ChatterboxTTS.from_pretrained(device=device)
                    logger.info("ChatterboxTTS model loaded successfully (use Turbo from source for lower latency)")
                except ImportError as e:
                    logger.error(f"Chatterbox dependencies not installed: {e}")
                    raise ValueError(
                        "Chatterbox-Turbo (chatterbox-tts) not installed. "
                        "Install with: pip install chatterbox-tts"
                    ) from e
            except Exception as e:
                logger.error(f"Failed to load Chatterbox model: {e}", exc_info=True)
                raise ValueError(f"Could not load Chatterbox-Turbo model: {e}") from e
    return _turbo_model, _turbo_device


def _get_indicf5_ref_audio_dir() -> Optional[str]:
    """Return IndicF5 ref audio directory; try default project path if not set."""
    dir_path = settings.tts_indicf5_ref_audio_dir
    if dir_path and os.path.isdir(dir_path):
        return dir_path
    # Default: project root / IndicF5 / prompts (when running from backend)
    backend_dir = Path(__file__).resolve().parents[2]
    default = backend_dir.parent / "IndicF5" / "prompts"
    if default.exists():
        return str(default)
    return None


def _get_indicf5_model():
    """Lazy load IndicF5 model and vocoder. On failure set _indicf5_available=False. Thread-safe."""
    global _indicf5_model, _indicf5_vocoder, _indicf5_device, _indicf5_available
    with _lock_indicf5:
        if not _indicf5_available:
            return None, None, None

        if _indicf5_model is not None:
            return _indicf5_model, _indicf5_vocoder, _indicf5_device

        if not _get_indicf5_ref_audio_dir():
            logger.info("IndicF5 disabled: tts_indicf5_ref_audio_dir not set")
            _indicf5_available = False
            return None, None, None

        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from f5_tts.model import DiT
            from f5_tts.infer.utils_infer import load_model, load_vocoder

            device = get_infer_device()
            _indicf5_device = device
            logger.info(f"Loading IndicF5 model (device: {device})")

            repo_id = "ai4bharat/IndicF5"
            vocab_path = hf_hub_download(repo_id, filename="checkpoints/vocab.txt")
            ckpt_path = hf_hub_download(repo_id, filename="model.safetensors")

            _indicf5_vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)
            _indicf5_model = load_model(
                DiT,
                dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                mel_spec_type="vocos",
                vocab_file=vocab_path,
                device=device,
            )
            state_dict = load_file(ckpt_path, device=device)
            state_dict = {
                k.replace("ema_model._orig_mod.", ""): v
                for k, v in state_dict.items()
                if k.startswith("ema_model.")
            }
            _indicf5_model.load_state_dict(state_dict)
            _indicf5_model.eval()
            logger.info("IndicF5 model loaded successfully")
            return _indicf5_model, _indicf5_vocoder, _indicf5_device
        except Exception as e:
            logger.warning(f"IndicF5 load failed: {e}. Indic languages will use Turbo fallback.")
            _indicf5_available = False
            return None, None, None


def _get_indicf5_ref(indic_lang: str) -> Optional[tuple[str, str]]:
    """Return (ref_audio_path, ref_text) for the given Indic language, or None if not configured."""
    ref_audio_dir = _get_indicf5_ref_audio_dir()
    if not ref_audio_dir:
        return None
    ref_filename = INDICF5_REF_FILENAMES.get(indic_lang)
    ref_text = INDICF5_REF_TEXTS.get(indic_lang)
    if not ref_filename or not ref_text:
        return None
    ref_audio_path = os.path.join(ref_audio_dir, ref_filename)
    if not os.path.exists(ref_audio_path):
        logger.warning(f"IndicF5 ref audio not found: {ref_audio_path}")
        return None
    return (ref_audio_path, ref_text)


def _tts_with_indicf5(text: str, indic_lang: str) -> bytes:
    """
    Generate audio using IndicF5 for the given Indic language.

    Args:
        text: Text to speak (in that language)
        indic_lang: Language code (hi, ml, ta, etc.)

    Returns:
        WAV bytes (caller converts to MP3 for storage when needed).

    Raises:
        ValueError: If ref not found or generation fails
    """
    ref = _get_indicf5_ref(indic_lang)
    if not ref:
        raise ValueError(f"No IndicF5 ref configured for language: {indic_lang}")

    model, vocoder, device = _get_indicf5_model()
    if model is None or vocoder is None:
        raise ValueError("IndicF5 model not available. Check config and logs.")

    ref_audio_path, ref_text = ref
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text, infer_process

    if len(text) > 2000:
        text = text[:2000]

    ref_audio, ref_text_processed = preprocess_ref_audio_text(ref_audio_path, ref_text, device=device)
    audio, sample_rate, _ = infer_process(
        ref_audio,
        ref_text_processed,
        text,
        model,
        vocoder,
        mel_spec_type="vocos",
        device=device,
        speed=getattr(settings, "tts_indicf5_speed", 0.9),
    )

    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    if isinstance(audio, np.ndarray) and audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()

    wav_buffer = BytesIO()
    sf.write(wav_buffer, audio, sample_rate if sample_rate else 24000, format="WAV")
    wav_buffer.seek(0)
    wav_bytes = wav_buffer.getvalue()
    # Direct WAV return for max performance (streaming path uses this; no MP3 conversion).
    # return _convert_wav_to_mp3(wav_bytes)  # uncomment for MP3 output in this path
    return wav_bytes


def _resolve_audio_prompt_path() -> Optional[str]:
    """Resolve tts_audio_prompt_path to an absolute path. Chatterbox-Turbo requires a reference clip."""
    if not settings.tts_audio_prompt_path:
        return None
    path = settings.tts_audio_prompt_path
    if os.path.isabs(path) and os.path.exists(path):
        return path
    # Try audio_storage_path, then backend/audio_storage
    candidate = os.path.join(settings.audio_storage_path, path)
    if os.path.exists(candidate):
        return candidate
    backend_dir = Path(__file__).resolve().parents[2]
    candidate = backend_dir / "audio_storage" / path
    if candidate.exists():
        return str(candidate)
    return None


def _tts_with_turbo(text: str) -> bytes:
    """
    Generate audio using Chatterbox-Turbo (ResembleAI/chatterbox-turbo). English only.

    Requires a reference audio clip for voice cloning (tts_audio_prompt_path).

    Returns:
        Audio bytes (WAV format; caller converts to MP3 for storage when needed).
    """
    try:
        import torch

        model, device = _get_turbo_model()

        logger.info(f"Generating audio with Chatterbox-Turbo (device: {device})")
        logger.info(f"Text to convert: '{text[:100]}...' (total length: {len(text)} characters)")

        if len(text) > 2000:
            logger.warning(f"Text is {len(text)} characters, truncating to 2000")
            text = text[:2000]

        audio_prompt_path = _resolve_audio_prompt_path()
        if not audio_prompt_path:
            raise ValueError(
                "Chatterbox-Turbo requires a reference clip for voice cloning. "
                "Set tts_audio_prompt_path in config (e.g. a 10s WAV file)."
            )
        logger.info(f"Using audio prompt for voice cloning: {audio_prompt_path}")

        with torch.no_grad():
            # Chatterbox-Turbo: generate(text, audio_prompt_path=...), returns tensor, model.sr
            wav_tensor = model.generate(text, audio_prompt_path=audio_prompt_path)

        wav_array = wav_tensor.cpu().numpy()
        if wav_array.ndim > 1:
            wav_array = wav_array.squeeze()

        sample_rate = model.sr
        logger.info(f"Generated audio: {len(wav_array)} samples at {sample_rate}Hz")

        if wav_array.dtype != np.float32:
            wav_array = wav_array.astype(np.float32)
        max_val = np.abs(wav_array).max()
        if max_val > 1.0:
            wav_array = wav_array / max_val

        wav_buffer = BytesIO()
        sf.write(wav_buffer, wav_array, sample_rate, format="WAV")
        wav_buffer.seek(0)
        wav_bytes = wav_buffer.getvalue()

        # Direct WAV return for max performance (streaming path uses this; no MP3 conversion).
        # return _convert_wav_to_mp3(wav_bytes)  # uncomment for MP3 output in this path
        return wav_bytes

    except ImportError as e:
        logger.error(f"Chatterbox-Turbo dependencies not installed: {e}")
        raise ValueError(
            "Chatterbox-Turbo not installed. Install with: pip install chatterbox-tts"
        )
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Chatterbox-Turbo error: {e}", exc_info=True)
        raise ValueError("Could not generate audio. Please try again.")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS. Keeps trailing punctuation with sentence."""
    if not text or not text.strip():
        return []
    # Split on sentence boundaries (after . ! ?) followed by space; keep " ... " as part of previous
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _generate_tts_bytes(text: str, response_language: str = "en") -> bytes:
    """
    Generate TTS audio bytes for the given text without storing or using URL cache.
    Used for streaming: returns raw WAV bytes (no MP3 conversion for maximum performance).
    """
    return generate_tts_bytes(text, response_language)


def generate_tts_bytes(text: str, response_language: str = "en") -> bytes:
    """
    Public entrypoint: generate TTS audio bytes for one sentence (WAV).
    Used by the chat/stream pipeline for sentence-level TTS.
    """
    use_indicf5 = response_language != "en"
    if use_indicf5:
        ref = _get_indicf5_ref(response_language)
        model, _, _ = _get_indicf5_model()
        if ref and model is not None:
            try:
                with _inference_lock:
                    return _tts_with_indicf5(text, response_language)
            except ValueError:
                raise
            except Exception as e:
                logger.warning(f"IndicF5 generation failed: {e}. Falling back to Chatterbox-Turbo.")
        use_indicf5 = False
    if not use_indicf5:
        with _inference_lock:
            return _tts_with_turbo(text)
    raise ValueError("Something went wrong generating audio. Please try again.")


def text_to_speech_stream(text: str, response_language: str = "en") -> Iterator[bytes]:
    """
    Generate TTS audio in chunks (one WAV chunk per sentence) for streaming over SSE.
    Yields raw WAV bytes for each sentence (no MP3 conversion for max performance).
    Caller concatenates and converts to MP3 once for final storage if needed.
    """
    sentences = _split_sentences(text)
    if not sentences:
        # No sentence boundary: treat whole text as one chunk
        if text.strip():
            yield _generate_tts_bytes(text.strip(), response_language)
        return
    for sentence in sentences:
        if not sentence.strip():
            continue
        yield _generate_tts_bytes(sentence, response_language)


def feed_tts_stream_to_queue(
    queue: asyncio.Queue,
    text: str,
    response_language: str,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Producer: run text_to_speech_stream in the current thread and put each WAV chunk
    into the queue (thread-safe via loop.call_soon_threadsafe). Puts None as sentinel
    when done; on exception puts ("error", str(e)) so the consumer can stop.
    Intended to be run from a dedicated thread; the API consumer awaits queue.get().
    """
    try:
        for chunk in text_to_speech_stream(text, response_language):
            loop.call_soon_threadsafe(queue.put_nowait, chunk)
        loop.call_soon_threadsafe(queue.put_nowait, None)
    except Exception as e:
        logger.exception("TTS stream error in feed_tts_stream_to_queue")
        loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))


def store_audio_mp3(audio_bytes: bytes, filename: str) -> str:
    """
    Store MP3 bytes to local or S3 and return the playback URL.
    Used after concatenating streamed TTS chunks.
    """
    s3_key = _store_audio_cloud(audio_bytes, filename)
    if s3_key:
        presigned = _generate_presigned_url(s3_key)
        if presigned:
            return presigned
    return _store_audio_local(audio_bytes, filename)


def text_to_speech(text: str, response_language: str = "en") -> str:
    """
    Convert text to speech and return audio URL. Routes to Chatterbox-Turbo (English) or IndicF5 (Indic).

    Args:
        text: Text to convert
        response_language: "en" -> Chatterbox-Turbo; "hi"/"ml"/"ta"/etc. -> IndicF5 (fallback to Turbo if unavailable)

    Returns:
        URL to audio file

    Raises:
        ValueError: If audio generation fails
    """
    cache_key = _generate_cache_key(text, response_language)
    cached = get(cache_key)
    if cached:
        logger.info("Cache hit for TTS")
        if cached.startswith("s3:"):
            s3_key = cached[3:]
            presigned = _generate_presigned_url(s3_key)
            if presigned is None:
                logger.error("Presigned URL generation failed on cache hit")
                raise ValueError("Audio temporarily unavailable. Please try again.")
            return presigned
        return cached

    use_indicf5 = response_language != "en"
    if use_indicf5:
        ref = _get_indicf5_ref(response_language)
        model, _, _ = _get_indicf5_model()
        if ref and model is not None:
            logger.info(f"Using TTS provider: IndicF5 (language: {response_language})")
            try:
                with _inference_lock:
                    audio_bytes = _tts_with_indicf5(text, response_language)
                audio_bytes = _convert_wav_to_mp3(audio_bytes)
                filename = f"{response_language}_{hashlib.md5(text.encode()).hexdigest()}.mp3"
                s3_key = _store_audio_cloud(audio_bytes, filename)
                if s3_key:
                    presigned = _generate_presigned_url(s3_key)
                    if presigned:
                        set(cache_key, "s3:" + s3_key, settings.tts_cache_ttl)
                        return presigned
                audio_url = _store_audio_local(audio_bytes, filename)
                set(cache_key, audio_url, settings.tts_cache_ttl)
                return audio_url
            except ValueError:
                raise
            except Exception as e:
                logger.warning(f"IndicF5 generation failed: {e}. Falling back to Chatterbox-Turbo.")
        else:
            logger.warning(f"IndicF5 not configured for {response_language}. Using Chatterbox-Turbo.")
            use_indicf5 = False

    if not use_indicf5:
        logger.info("Using TTS provider: Chatterbox-Turbo")
        try:
            with _inference_lock:
                audio_bytes = _tts_with_turbo(text)
            audio_bytes = _convert_wav_to_mp3(audio_bytes)
            filename = f"{hashlib.md5(text.encode()).hexdigest()}.mp3"
            s3_key = _store_audio_cloud(audio_bytes, filename)
            if s3_key:
                presigned = _generate_presigned_url(s3_key)
                if presigned:
                    set(cache_key, "s3:" + s3_key, settings.tts_cache_ttl)
                    return presigned
            audio_url = _store_audio_local(audio_bytes, filename)
            set(cache_key, audio_url, settings.tts_cache_ttl)
            return audio_url
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Unexpected TTS error: {e}")
            raise ValueError("Something went wrong generating audio. Please try again.")

    raise ValueError("Something went wrong generating audio. Please try again.")


def init_tts_models() -> dict:
    """
    Load TTS models (Turbo and optionally IndicF5) for warmup.
    Returns {"turbo": "loaded"|"failed", "indicf5": "loaded"|"skipped"|"failed"}.
    """
    result: dict = {}
    try:
        _get_turbo_model()
        result["turbo"] = "loaded"
    except Exception as e:
        logger.exception("TTS Turbo init failed")
        result["turbo"] = "failed"
        result["turbo_error"] = str(e)
    model, _, _ = _get_indicf5_model()
    if model is not None:
        result["indicf5"] = "loaded"
    else:
        result["indicf5"] = "skipped"
    return result
