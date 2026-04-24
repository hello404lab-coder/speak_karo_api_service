"""Text-to-Speech service with cloud/local storage."""
import asyncio
import base64
import hashlib
import json
import logging
import os
import queue as queue_lib
import re
import struct
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator, Literal, Optional
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from app.core.config import settings
from app.services.cache import get, set
from app.utils.device import get_infer_device
from app.utils.language import _script_to_lang

logger = logging.getLogger(__name__)


STORAGE_REF_S3_PREFIX = "s3:"
STORAGE_REF_LOCAL_PREFIX = "local:"


@dataclass(frozen=True)
class StoredAudioRecord:
    """Durable storage reference plus the best available playback URL."""

    playback_url: str
    storage_ref: str


@dataclass
class ChirpPCMFilterState:
    """Streaming filter state so chunk-to-chunk cleanup stays continuous."""

    prev_x: float = 0.0
    prev_y: float = 0.0


def _get_indicf5_torch_device() -> str:
    """
    Torch device string for IndicF5 only. MPS lacks ComplexFloat and full FFT support for the F5 spectral path.
    """
    if getattr(settings, "tts_indicf5_force_cpu", False):
        return "cpu"
    infer = get_infer_device()
    if infer == "mps":
        return "cpu"
    return infer

# Lazy-loaded Chatterbox-Turbo model (English, loaded on first use); lock prevents double-load
_turbo_model = None
_turbo_device = None
_lock_turbo = threading.Lock()
_turbo_voice_prepared = False  # Voice cloning: prepare once, reuse for all generate() calls
_turbo_warmup_done = False  # One inference after load so compiled kernels are ready for first user

# Lazy-loaded IndicF5 model and vocoder (loaded on first use); lock prevents double-load
_indicf5_model = None
_indicf5_vocoder = None
_indicf5_device = None
_indicf5_available = True  # Set False if load fails so we fallback to Turbo
_lock_indicf5 = threading.Lock()

# Inference semaphore: up to N TTS inferences at once (configurable) to reduce latency for multi-sentence responses; set to 1 for strict serialization on low VRAM.
_inference_semaphore = threading.Semaphore(settings.tts_concurrent_inferences)

# Lazy-loaded Gemini client for TTS (when Chatterbox disabled). API-based, no GPU lock.
_gemini_tts_client = None

# Lazy-loaded Resemble HTTP client (TTS_CHATTERBOX_MODE=api)
_resemble_client = None
_lock_resemble = threading.Lock()

# Lazy-loaded Google Cloud Text-to-Speech client for Chirp 3 HD.
_chirp_client = None
_lock_chirp = threading.Lock()
_chirp_fallback_warning_logged = False

# Pooled httpx client for Smallest.ai Waves (Lightning TTS).
_smallest_client = None
_lock_smallest = threading.Lock()
_smallest_fallback_warning_logged = False

# Internal language code to BCP-47 for Gemini TTS (en-US, hi-IN, etc.)
LANG_TO_BCP47 = {
    "en": "en-US",
    "hi": "hi-IN",
    "ml": "ml-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "bn": "bn-BD",
}

CHIRP_LANG_TO_BCP47 = {
    "en": "en-IN",
    "hi": "hi-IN",
    "ml": "ml-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "bn": "bn-IN",
}

# Raw PCM event (used by Chirp and Smallest streaming for stitched replay)
CLOUD_STREAM_EVENT_RAW_PCM = "raw_pcm"
CHIRP_STREAM_EVENT_RAW_PCM = CLOUD_STREAM_EVENT_RAW_PCM
CHIRP_DEFAULT_SAMPLE_RATE_HZ = 24000
CHIRP_TARGET_PACKET_MS = 240
TTSBackend = Literal["turbo", "resemble_api", "indicf5", "gemini", "chirp3_hd", "smallest"]

# Smallest.ai Lightning (docs: e.g. en, hi, ta). Malayalam uses Chirp 3 HD when TTS_CLOUD_PROVIDER=smallest.
SMALLEST_SUPPORTED_LANGS = frozenset({"en", "hi", "ta"})
SMALLEST_LANG_TO_CODE = {
    "en": "en",
    "hi": "hi",
    "ta": "ta",
}

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


def _generate_cache_key(
    text: str,
    response_language: str = "en",
    *,
    gemini_indic: bool = False,
    provider_tag: Optional[str] = None,
) -> str:
    """Generate cache key from text and language. Indic Gemini uses a separate key so model changes invalidate cache."""
    h = hashlib.md5(text.encode()).hexdigest()
    lang = response_language or "en"
    provider_part = f":{provider_tag}" if provider_tag else ""
    if gemini_indic:
        mid = getattr(settings, "tts_gemini_model_indic", None) or "gemini-2.5-flash-preview-tts"
        tag = hashlib.md5(mid.encode()).hexdigest()[:8]
        return f"tts:{lang}{provider_part}:gi:{tag}:{h}"
    return f"tts:{lang}{provider_part}:{h}"


def _store_audio_local(audio_bytes: bytes, filename: str) -> str:
    """Store audio file locally and return URL."""
    filepath = os.path.join(settings.audio_storage_path, filename)
    
    with open(filepath, "wb") as f:
        f.write(audio_bytes)
    
    # Return URL for local serving
    return f"{settings.audio_base_url}/{filename}"


def _storage_ref_for_local(filename: str) -> str:
    """Encode a local audio file reference."""
    return f"{STORAGE_REF_LOCAL_PREFIX}{filename}"


def _storage_ref_for_s3(s3_key: str) -> str:
    """Encode an S3 audio file reference."""
    return f"{STORAGE_REF_S3_PREFIX}{s3_key}"


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
            region_name=settings.aws_region or 'ap-south-1'
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


def _store_audio_cloud(audio_bytes: bytes, filename: str, content_type: str = "audio/mpeg") -> Optional[str]:
    """Store audio file in S3 and return the S3 key (e.g. 'ai/audio/filename.mp3'), or None."""
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
            ContentType=content_type
        )
        return s3_key
    except ImportError:
        logger.warning("boto3 not installed, falling back to local storage")
        return None
    except Exception as e:
        logger.error(f"Cloud storage error: {e}, falling back to local")
        return None


def store_user_voice_wav(audio_bytes: bytes, filename: str) -> str:
    """
    Store user voice recording (WAV) in S3 or local and return the playback URL.
    Used for voice-chat and voice-chat/stream to persist the user's recording.
    On S3/store failure, falls back to local; if both fail, caller should handle (log and use None).
    """
    return store_user_voice_wav_record(audio_bytes, filename).playback_url


def store_user_voice_wav_record(audio_bytes: bytes, filename: str) -> StoredAudioRecord:
    """Store user voice recording and return both playback URL and a durable storage reference."""
    try:
        s3_key = _store_audio_cloud(audio_bytes, f"user_voice/{filename}", content_type="audio/wav")
        if s3_key:
            presigned = _generate_presigned_url(s3_key)
            if presigned:
                return StoredAudioRecord(
                    playback_url=presigned,
                    storage_ref=_storage_ref_for_s3(s3_key),
                )
    except Exception as e:
        logger.warning("User voice S3 upload failed: %s, falling back to local", e)

    playback_url = _store_audio_local(audio_bytes, filename)
    return StoredAudioRecord(
        playback_url=playback_url,
        storage_ref=_storage_ref_for_local(filename),
    )


def resolve_stored_audio_playback_url(storage_ref: Optional[str], fallback_url: Optional[str] = None) -> Optional[str]:
    """Resolve a durable storage reference into a playback URL."""
    if not storage_ref:
        return fallback_url

    if storage_ref.startswith(STORAGE_REF_S3_PREFIX):
        s3_key = storage_ref[len(STORAGE_REF_S3_PREFIX):]
        return _generate_presigned_url(s3_key) or fallback_url

    if storage_ref.startswith(STORAGE_REF_LOCAL_PREFIX):
        filename = storage_ref[len(STORAGE_REF_LOCAL_PREFIX):]
        return f"{settings.audio_base_url}/{filename}"

    return fallback_url


def _legacy_audio_cache_value_to_record(cached_value: str) -> Optional[StoredAudioRecord]:
    """Convert legacy cache payloads into a durable record when possible."""
    if not cached_value:
        return None

    if cached_value.startswith(STORAGE_REF_S3_PREFIX):
        playback_url = resolve_stored_audio_playback_url(cached_value)
        if playback_url:
            return StoredAudioRecord(playback_url=playback_url, storage_ref=cached_value)
        return None

    if cached_value.startswith(STORAGE_REF_LOCAL_PREFIX):
        playback_url = resolve_stored_audio_playback_url(cached_value)
        if playback_url:
            return StoredAudioRecord(playback_url=playback_url, storage_ref=cached_value)
        return None

    prefix = f"{settings.audio_base_url}/"
    if cached_value.startswith(prefix):
        filename = cached_value[len(prefix):]
        storage_ref = _storage_ref_for_local(filename)
        playback_url = resolve_stored_audio_playback_url(storage_ref, cached_value)
        if playback_url:
            return StoredAudioRecord(playback_url=playback_url, storage_ref=storage_ref)

    return None


def delete_stored_audio(storage_ref: Optional[str]) -> None:
    """Delete audio from durable storage. Failures are logged and ignored."""
    if not storage_ref:
        return

    if storage_ref.startswith(STORAGE_REF_LOCAL_PREFIX):
        filename = storage_ref[len(STORAGE_REF_LOCAL_PREFIX):]
        path = os.path.join(settings.audio_storage_path, filename)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.warning("Failed to delete local audio %s: %s", path, e)
        return

    if storage_ref.startswith(STORAGE_REF_S3_PREFIX):
        s3_key = storage_ref[len(STORAGE_REF_S3_PREFIX):]
        if not all([
            settings.aws_access_key_id,
            settings.aws_secret_access_key,
            settings.s3_bucket_name,
        ]):
            return
        try:
            import boto3

            s3_client = boto3.client(
                "s3",
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region or "ap-south-1",
            )
            s3_client.delete_object(Bucket=settings.s3_bucket_name, Key=s3_key)
        except ImportError:
            logger.warning("boto3 not installed")
        except Exception as e:
            logger.warning("Failed to delete S3 audio %s: %s", s3_key, e)


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


def chirp_pcm_to_wav(audio_data: bytes, sample_rate_hz: Optional[int] = None) -> bytes:
    """Wrap raw PCM bytes from Chirp streaming in a WAV container for client playback."""
    rate = sample_rate_hz or int(getattr(settings, "tts_chirp_sample_rate_hz", CHIRP_DEFAULT_SAMPLE_RATE_HZ))
    return _convert_to_wav(audio_data, f"audio/L16;rate={rate}")


def _postprocess_chirp_pcm_chunk(
    audio_data: bytes,
    sample_rate_hz: Optional[int] = None,
    state: Optional[ChirpPCMFilterState] = None,
) -> bytes:
    """
    Clean up Chirp PCM for streaming playback.
    Applies a gentle DC/high-pass cleanup plus a very short edge fade to reduce
    faint clicks/static-like artifacts at chunk boundaries without adding noticeable latency.
    """
    if not audio_data:
        return audio_data

    sample_rate = sample_rate_hz or int(getattr(settings, "tts_chirp_sample_rate_hz", CHIRP_DEFAULT_SAMPLE_RATE_HZ))
    pcm = np.frombuffer(audio_data, dtype="<i2")
    if pcm.size == 0:
        return audio_data

    x = pcm.astype(np.float32) / 32768.0

    # Remove DC offset / very low-frequency rumble with a light one-pole high-pass filter.
    if state is None:
        state = ChirpPCMFilterState()
    dt = 1.0 / float(sample_rate)
    rc = 1.0 / (2.0 * np.pi * 35.0)
    alpha = rc / (rc + dt)
    y = np.empty_like(x)
    prev_x = state.prev_x
    prev_y = state.prev_y
    for i in range(x.size):
        current_x = x[i]
        current_y = alpha * (prev_y + current_x - prev_x)
        y[i] = current_y
        prev_x = current_x
        prev_y = current_y
    state.prev_x = prev_x
    state.prev_y = prev_y

    # Center any tiny residual DC bias.
    mean = float(y.mean())
    if abs(mean) > 1e-5:
        y = y - mean

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.995:
        y = y / peak * 0.995

    out = np.clip(y * 32767.0, -32768.0, 32767.0).astype("<i2")
    return out.tobytes()


def _import_chirp_texttospeech():
    """Import Google Cloud Text-to-Speech lazily so the app still boots without Chirp deps."""
    try:
        from google.cloud import texttospeech
    except ImportError as e:
        raise ValueError(
            "Google Cloud Text-to-Speech is not installed. Install google-cloud-texttospeech to use Chirp 3 HD."
        ) from e
    return texttospeech


def chirp_runtime_status() -> dict[str, Optional[str]]:
    """
    Return whether Chirp 3 HD is usable in this process.
    Uses a lightweight ADC check so startup logs reflect real availability.
    """
    try:
        _import_chirp_texttospeech()
    except ValueError as e:
        return {"available": False, "reason": str(e)}

    try:
        from google.auth import default as google_auth_default
        google_auth_default(scopes=("https://www.googleapis.com/auth/cloud-platform",))
        return {"available": True, "reason": None}
    except Exception as e:
        return {"available": False, "reason": str(e)}


def _log_chirp_fallback_once(reason: Exception | str) -> None:
    """Avoid repeating the same Chirp fallback warning for every sentence/chunk."""
    global _chirp_fallback_warning_logged
    if _chirp_fallback_warning_logged:
        return
    _chirp_fallback_warning_logged = True
    logger.warning("Chirp 3 HD unavailable; falling back to Gemini TTS: %s", reason)


def _chirp_endpoint() -> Optional[str]:
    """Return an explicit endpoint when configured, otherwise a best-effort regional endpoint."""
    explicit = (getattr(settings, "tts_chirp_endpoint", None) or "").strip()
    if explicit:
        return explicit
    region = (getattr(settings, "tts_chirp_region", "global") or "global").strip()
    if not region or region == "global":
        return None
    return f"{region}-texttospeech.googleapis.com"


def _chirp_locale_for_lang(lang: str) -> str:
    """Resolve the preferred Chirp locale for an app language code."""
    locale = CHIRP_LANG_TO_BCP47.get(lang)
    if not locale:
        raise ValueError(f"Chirp 3 HD is not configured for language: {lang}")
    return locale


def _chirp_voice_name_for_lang(lang: str) -> tuple[str, str]:
    """Return (locale, voice_name) for Chirp 3 HD."""
    locale = _chirp_locale_for_lang(lang)
    configured = (getattr(settings, "tts_chirp_voice", "Charon") or "Charon").strip()
    if not configured:
        configured = "Charon"
    if configured.startswith(f"{locale}-") and "Chirp3-HD" in configured:
        return locale, configured
    if "Chirp3-HD" in configured and configured.count("-") >= 2:
        return locale, configured
    return locale, f"{locale}-Chirp3-HD-{configured}"


def _cloud_tts_provider_for_lang(lang: str, allow_fallback: bool = True) -> TTSBackend:
    """Resolve the configured cloud TTS provider for a language."""
    preferred = getattr(settings, "tts_cloud_provider", "gemini")
    if preferred == "smallest":
        if getattr(settings, "smallest_api_key", None) and lang in SMALLEST_SUPPORTED_LANGS:
            return "smallest"
        if allow_fallback:
            logger.warning(
                "Smallest.ai TTS not used for lang=%s (supported=%s, key set=%s); falling back to Chirp 3 HD or Gemini",
                lang,
                sorted(SMALLEST_SUPPORTED_LANGS),
                bool(getattr(settings, "smallest_api_key", None)),
            )
            try:
                _chirp_locale_for_lang(lang)
                return "chirp3_hd"
            except ValueError as e:
                logger.warning("Chirp 3 HD unsupported for %s after Smallest skip, using Gemini: %s", lang, e)
                return "gemini"
        raise ValueError(f"Smallest TTS is not configured for language: {lang}")
    if preferred == "chirp3_hd":
        try:
            _chirp_locale_for_lang(lang)
            return "chirp3_hd"
        except ValueError as e:
            if allow_fallback:
                logger.warning("Chirp 3 HD unsupported for %s, falling back to Gemini: %s", lang, e)
                return "gemini"
            raise
    return "gemini"


def resolve_tts_backend(text: str, response_language: str = "en") -> TTSBackend:
    """
    Resolve the runtime TTS backend for the given text/language without generating audio.
    Local backends keep priority; cloud routing uses the configured provider switch.
    """
    lang = _effective_response_language(text, response_language)
    chatterbox_enabled = getattr(settings, "tts_chatterbox_enabled", True)

    if lang != "en" and getattr(settings, "tts_indicf5_enabled", False):
        ref = _get_indicf5_ref(lang)
        model, _, _ = _get_indicf5_model()
        if ref and model is not None:
            return "indicf5"

    if lang == "en" and chatterbox_enabled:
        if getattr(settings, "tts_chatterbox_mode", "local") == "api":
            return "resemble_api"
        return "turbo"

    return _cloud_tts_provider_for_lang(lang)


def chirp_streaming_enabled_for_text(text: str, response_language: str = "en") -> bool:
    """True when this text would route to Chirp 3 HD and can use its bidirectional stream."""
    try:
        if resolve_tts_backend(text, response_language) != "chirp3_hd":
            return False
        _import_chirp_texttospeech()
        _get_chirp_client()
        return True
    except Exception:
        return False


def smallest_streaming_enabled_for_text(text: str, response_language: str = "en") -> bool:
    """True when this text would route to Smallest TTS and SSE streaming is enabled."""
    try:
        if not getattr(settings, "tts_smallest_streaming_enabled", True):
            return False
        if not getattr(settings, "smallest_api_key", None):
            return False
        if resolve_tts_backend(text, response_language) != "smallest":
            return False
        return True
    except Exception:
        return False


def _get_chirp_client():
    """Lazy-load a reusable Cloud TTS client for Chirp 3 HD."""
    global _chirp_client
    if _chirp_client is not None:
        return _chirp_client

    texttospeech = _import_chirp_texttospeech()
    with _lock_chirp:
        if _chirp_client is None:
            kwargs = {}
            endpoint = _chirp_endpoint()
            if endpoint:
                try:
                    from google.api_core.client_options import ClientOptions
                    kwargs["client_options"] = ClientOptions(api_endpoint=endpoint)
                except ImportError:
                    kwargs["client_options"] = {"api_endpoint": endpoint}
            _chirp_client = texttospeech.TextToSpeechClient(**kwargs)
            logger.info(
                "Chirp 3 HD client initialized (region=%s, endpoint=%s, voice=%s)",
                getattr(settings, "tts_chirp_region", "global"),
                endpoint or "default",
                getattr(settings, "tts_chirp_voice", "Charon"),
            )
    return _chirp_client


def _gemini_tts_model_for_lang(lang: str) -> str:
    """English uses tts_gemini_model; Indic uses tts_gemini_model_indic (Gemini 2.5 Flash TTS)."""
    if lang != "en":
        return getattr(settings, "tts_gemini_model_indic", None) or "gemini-2.5-flash-preview-tts"
    return getattr(settings, "tts_gemini_model", None) or "gemini-2.5-flash-lite-preview-tts"


def _get_gemini_tts_client():
    """Lazy load Gemini client for TTS (used when Chatterbox is disabled). Same SDK as LLM."""
    global _gemini_tts_client
    if _gemini_tts_client is None:
        if not settings.gemini_api_key:
            raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY for TTS when Chatterbox is disabled.")
        from google import genai
        timeout_ms = getattr(settings, "tts_timeout_seconds", 45) * 1000
        try:
            from google.genai.types import HttpOptions
            _gemini_tts_client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=HttpOptions(timeout=timeout_ms),
            )
        except (ImportError, AttributeError):
            _gemini_tts_client = genai.Client(api_key=settings.gemini_api_key)
        logger.info(
            "Gemini TTS client initialized (en=%s, indic=%s)",
            getattr(settings, "tts_gemini_model", "gemini-2.5-flash-lite-preview-tts"),
            getattr(settings, "tts_gemini_model_indic", "gemini-2.5-flash-preview-tts"),
        )
    return _gemini_tts_client


def _tts_with_gemini(text: str, response_language: str) -> bytes:
    """
    Generate TTS audio using Gemini TTS. English uses tts_gemini_model; Indic uses tts_gemini_model_indic.
    Returns WAV bytes.
    """
    from google.genai import types
    if not text or not text.strip():
        raise ValueError("Empty text for Gemini TTS")
    text = text.strip()
    if len(text) > 4000:
        text = text[:4000]
    client = _get_gemini_tts_client()
    lang_code = LANG_TO_BCP47.get(response_language, "en-US")
    voice_name = getattr(settings, "tts_gemini_voice", "Puck") or "Puck"
    model_name = _gemini_tts_model_for_lang(response_language)
    contents = f"Say the following: {text}"
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            language_code=lang_code,
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name),
            ),
        ),
    )
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )
    if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
        raise ValueError("No audio content in Gemini TTS response")
    part = response.candidates[0].content.parts[0]
    if not getattr(part, "inline_data", None) or not getattr(part.inline_data, "data", None):
        raise ValueError("Gemini TTS response missing inline_data")
    pcm_data = part.inline_data.data
    if not pcm_data:
        raise ValueError("Gemini TTS returned empty audio")
    return _convert_to_wav(pcm_data, "audio/L16;rate=24000")


def _chirp_streaming_audio_encoding(texttospeech):
    """Resolve the PCM enum for Chirp streaming audio."""
    pcm = None
    streaming_audio_encoding = getattr(
        getattr(texttospeech, "StreamingAudioConfig", None),
        "AudioEncoding",
        None,
    )
    if streaming_audio_encoding is not None:
        pcm = getattr(streaming_audio_encoding, "PCM", None)
    if pcm is None:
        pcm = getattr(getattr(texttospeech, "AudioEncoding", None), "PCM", None)
    if pcm is None:
        raise ValueError(
            "Installed Cloud TTS library does not expose a PCM audio encoding enum for streaming."
        )
    return pcm


def _build_chirp_streaming_config(texttospeech, response_language: str):
    """Build Chirp 3 HD streaming config for a language."""
    locale, voice_name = _chirp_voice_name_for_lang(response_language)
    sample_rate = int(getattr(settings, "tts_chirp_sample_rate_hz", CHIRP_DEFAULT_SAMPLE_RATE_HZ))
    speaking_rate = float(getattr(settings, "tts_chirp_speaking_rate", 1.0) or 1.0)
    return texttospeech.StreamingSynthesizeConfig(
        voice=texttospeech.VoiceSelectionParams(
            language_code=locale,
            name=voice_name,
        ),
        streaming_audio_config=texttospeech.StreamingAudioConfig(
            audio_encoding=_chirp_streaming_audio_encoding(texttospeech),
            sample_rate_hertz=sample_rate,
            speaking_rate=speaking_rate,
        ),
    )


def _tts_with_chirp(text: str, response_language: str) -> bytes:
    """
    Generate audio using Cloud TTS Chirp 3 HD and return WAV bytes.
    Uses unary synthesize_speech for non-streaming endpoints and storage paths.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text for Chirp 3 HD TTS")
    if len(text) > 5000:
        text = text[:5000]

    texttospeech = _import_chirp_texttospeech()
    client = _get_chirp_client()
    locale, voice_name = _chirp_voice_name_for_lang(response_language)
    timeout = float(getattr(settings, "tts_chirp_timeout_seconds", 30) or 30)

    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(language_code=locale, name=voice_name),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=int(getattr(settings, "tts_chirp_sample_rate_hz", CHIRP_DEFAULT_SAMPLE_RATE_HZ)),
            speaking_rate=float(getattr(settings, "tts_chirp_speaking_rate", 1.0) or 1.0),
        ),
        timeout=timeout,
    )
    audio_content = bytes(getattr(response, "audio_content", b"") or b"")
    if not audio_content:
        raise ValueError("Chirp 3 HD returned empty audio")
    if audio_content[:4] == b"RIFF":
        return audio_content
    return chirp_pcm_to_wav(audio_content)


def _log_smallest_fallback_once(reason: Exception | str) -> None:
    global _smallest_fallback_warning_logged
    if _smallest_fallback_warning_logged:
        return
    _smallest_fallback_warning_logged = True
    logger.warning("Smallest TTS failed; trying Chirp 3 HD then Gemini: %s", reason)


def _parse_smallest_voice_map() -> dict[str, str]:
    raw = (getattr(settings, "tts_smallest_voice_per_lang", None) or "").strip()
    if not raw:
        return {}
    out: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        k, v = k.strip().lower(), v.strip()
        if k and v:
            out[k] = v
    return out


def _smallest_language_code_for_lang(lang: str) -> str:
    return SMALLEST_LANG_TO_CODE.get(lang, "en")


def _smallest_voice_for_lang(lang: str) -> str:
    m = _parse_smallest_voice_map()
    if lang in m:
        return m[lang]
    v = (getattr(settings, "tts_smallest_voice", None) or "magnus") or "magnus"
    return v.strip() or "magnus"


def _smallest_model_segment() -> str:
    m = (getattr(settings, "tts_smallest_model", "lightning-v3.1") or "lightning-v3.1").strip()
    m = m.strip("/")
    if m.startswith("v1/"):
        m = m[3:].lstrip("/")
    return m or "lightning-v3.1"


def _get_smallest_client():
    """Lazy-load pooled httpx client for Smallest.ai Waves (thread-safe)."""
    global _smallest_client
    if _smallest_client is not None:
        return _smallest_client
    with _lock_smallest:
        if _smallest_client is None:
            if not getattr(settings, "smallest_api_key", None):
                raise ValueError("SMALLEST_API_KEY is not set. Set it to use Smallest TTS.")
            import httpx

            base = (getattr(settings, "tts_smallest_base_url", None) or "https://api.smallest.ai/waves/v1").rstrip(
                "/"
            )
            tmo = float(getattr(settings, "tts_smallest_timeout_seconds", 30) or 30)
            _smallest_client = httpx.Client(
                base_url=base + "/",
                headers={
                    "Authorization": f"Bearer {settings.smallest_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(connect=10.0, read=tmo, write=10.0, pool=5.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
            logger.info("Smallest TTS client initialized (model=%s)", _smallest_model_segment())
    return _smallest_client


def _tts_with_smallest(text: str, response_language: str) -> bytes:
    """
    Generate audio via Smallest.ai Lightning TTS unary /get_speech. Returns WAV bytes.
    See https://docs.smallest.ai
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text for Smallest TTS")
    if len(text) > 5000:
        text = text[:5000]
    client = _get_smallest_client()
    model = _smallest_model_segment()
    out_fmt = getattr(settings, "tts_smallest_output_format", "wav") or "wav"
    sample_rate = int(getattr(settings, "tts_smallest_sample_rate_hz", 24000) or 24000)
    speed = float(getattr(settings, "tts_smallest_speed", 1.0) or 1.0)
    lang = response_language
    if lang not in SMALLEST_SUPPORTED_LANGS:
        raise ValueError(f"Smallest TTS does not support language: {lang}")
    payload: dict = {
        "text": text,
        "voice_id": _smallest_voice_for_lang(lang),
        "sample_rate": sample_rate,
        "output_format": out_fmt,
        "language": _smallest_language_code_for_lang(lang),
        "speed": speed,
    }
    timeout = float(getattr(settings, "tts_smallest_timeout_seconds", 30) or 30)
    r = client.post(
        f"{model}/get_speech",
        json=payload,
        timeout=timeout,
    )
    if r.status_code != 200:
        preview = (r.text or "")[:500]
        raise ValueError(f"Smallest TTS HTTP {r.status_code}: {preview}")
    data = r.content
    if not data:
        raise ValueError("Smallest TTS returned empty audio")
    if out_fmt == "wav" and data[:4] == b"RIFF":
        return data
    if out_fmt in ("pcm", "mulaw"):
        if out_fmt == "pcm":
            return chirp_pcm_to_wav(data, sample_rate_hz=sample_rate)
        return data
    if out_fmt == "mp3":
        # Caller paths expect WAV from generate_tts_bytes; convert MP3 -> WAV for downstream
        return _mp3_bytes_to_wav(data)
    return data


def _mp3_bytes_to_wav(mp3_data: bytes) -> bytes:
    """Decode MP3 bytes to PCM WAV (used when Smallest output_format=mp3)."""
    try:
        buf = BytesIO(mp3_data)
        buf.seek(0)
        seg = AudioSegment.from_file(buf, format="mp3")
        out = BytesIO()
        seg.export(out, format="wav")
        return out.getvalue()
    except Exception as e:
        raise ValueError(f"Failed to convert Smallest MP3 to WAV: {e}") from e


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
    if not getattr(settings, "tts_chatterbox_enabled", True):
        raise ValueError("Chatterbox is disabled. Set TTS_CHATTERBOX_ENABLED=true to use Chatterbox-Turbo.")
    global _turbo_model, _turbo_device
    with _lock_turbo:
        if _turbo_model is None:
            device = get_infer_device()
            _turbo_device = device

            # Patch must run before any model instantiation so that watermark weights are not loaded when using the dummy.
            # Both Turbo and ChatterboxTTS use perth.PerthImplicitWatermarker(); in some envs it is None (resemble-ai/chatterbox#198). Patch once before loading either.
            try:
                import perth
                use_dummy = (
                    perth.PerthImplicitWatermarker is None
                    or getattr(settings, "tts_use_dummy_watermarker", False)
                )
                if use_dummy and getattr(perth, "DummyWatermarker", None) is not None:
                    perth.PerthImplicitWatermarker = perth.DummyWatermarker
                    logger.debug("Patched perth.PerthImplicitWatermarker to DummyWatermarker")
            except Exception:
                pass

            # Prefer SDPA over manual attention on CUDA (faster). Set global backend before loading so the model uses it when instantiated.
            if device == "cuda":
                try:
                    import torch
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_math_sdp(False)
                    logger.debug("SDPA backends set for Chatterbox load (flash=True, math=False)")
                except Exception as e:
                    logger.debug("Could not set SDPA backends: %s", e)

            # Prefer ChatterboxTurboTTS (chatterbox.tts_turbo); PyPI package may only have chatterbox.tts
            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                logger.info(f"Loading Chatterbox-Turbo model (device: {device})")
                use_bfloat16 = device == "cuda" and getattr(settings, "tts_turbo_use_bfloat16", True)
                _turbo_model = None
                if use_bfloat16:
                    try:
                        import torch as _torch
                        _turbo_model = ChatterboxTurboTTS.from_pretrained(device=device, torch_dtype=_torch.bfloat16)
                    except TypeError:
                        pass
                if _turbo_model is None:
                    try:
                        _turbo_model = ChatterboxTurboTTS.from_pretrained(device=device, attn_implementation="sdpa")
                    except TypeError:
                        _turbo_model = ChatterboxTurboTTS.from_pretrained(device=device)
                logger.info("Chatterbox-Turbo model loaded successfully")
                # Force SDPA: wrap transformer forward so output_attentions=False, output_hidden_states=False (avoids manual attention fallback). Depends on transformers version; for max throughput a fork that sets these at call site (e.g. rsxdalv/chatterbox#127) can be used.
                if device == "cuda" and getattr(settings, "tts_force_sdpa_attention", True) and hasattr(_turbo_model.t3, "tfmr"):
                    try:
                        _orig_forward = _turbo_model.t3.tfmr.forward
                        def _wrapped_forward(*args, **kwargs):
                            kwargs["output_attentions"] = False
                            kwargs["output_hidden_states"] = False
                            return _orig_forward(*args, **kwargs)
                        _turbo_model.t3.tfmr.forward = _wrapped_forward
                        logger.debug("Chatterbox-Turbo tfmr.forward wrapped for SDPA (output_attentions=False, output_hidden_states=False)")
                    except Exception as e:
                        logger.debug("Could not wrap Turbo tfmr.forward for SDPA: %s", e)
                # Post-load bfloat16 when not loaded with torch_dtype (e.g. library does not support it).
                if device == "cuda" and use_bfloat16:
                    try:
                        import torch as _torch
                        if _turbo_model.t3.tfmr.weight.dtype != _torch.bfloat16:
                            _turbo_model.t3 = _turbo_model.t3.to(_torch.bfloat16).eval()
                            _turbo_model.s3gen = _turbo_model.s3gen.to(_torch.bfloat16).eval()
                            _turbo_model.ve = _turbo_model.ve.to(_torch.bfloat16).eval()
                            logger.debug("Chatterbox-Turbo converted to bfloat16")
                    except Exception as e:
                        logger.warning("Could not convert Turbo to bfloat16: %s", e)
                if device == "cuda" and getattr(settings, "tts_turbo_compile_t3", True):
                    try:
                        import torch as _torch
                        _turbo_model.t3 = _torch.compile(_turbo_model.t3, mode="reduce-overhead", fullgraph=True)
                        logger.debug("Chatterbox-Turbo t3 compiled with torch.compile (reduce-overhead)")
                    except Exception as e:
                        logger.debug("Full t3 compile failed: %s; trying _step_compilation_target if present", e)
                        step_target = getattr(_turbo_model.t3, "_step_compilation_target", None)
                        if callable(step_target):
                            try:
                                import torch as _torch
                                _turbo_model.t3._step_compilation_target = _torch.compile(step_target, fullgraph=True, backend="cudagraphs")
                                logger.debug("Chatterbox-Turbo t3._step_compilation_target compiled with cudagraphs")
                            except Exception as e2:
                                logger.debug("Could not compile Turbo step target: %s", e2)
                max_cache_len = getattr(settings, "tts_turbo_max_cache_len", None)
                if max_cache_len is not None and hasattr(_turbo_model.t3, "max_cache_len"):
                    try:
                        _turbo_model.t3.max_cache_len = max_cache_len
                        logger.debug("Chatterbox-Turbo max_cache_len set to %s", max_cache_len)
                    except Exception as e:
                        logger.debug("Could not set Turbo max_cache_len: %s", e)
                # Wrap inference_turbo to pass max_gen_len from config or per-call override (reduces loop overhead).
                t3_obj = _turbo_model.t3
                _orig_inference_turbo = getattr(t3_obj, "inference_turbo", None)
                if callable(_orig_inference_turbo):
                    # Use unbound function so we can pass (t3_obj, *args, **kwargs) without double self.
                    _orig_inference_turbo = getattr(_orig_inference_turbo, "__func__", _orig_inference_turbo)
                    _max_gen_len_default = getattr(settings, "tts_turbo_max_gen_len", 400)

                    def _wrapped_inference_turbo(*args, **kwargs):
                        mg = getattr(t3_obj, "_max_gen_len_override", None)
                        kwargs["max_gen_len"] = mg if mg is not None else _max_gen_len_default
                        return _orig_inference_turbo(t3_obj, *args, **kwargs)

                    _turbo_model.t3.inference_turbo = _wrapped_inference_turbo
                    logger.debug("Chatterbox-Turbo t3.inference_turbo wrapped with max_gen_len from config")
                # Compile S3 decoder when enabled (CUDA).
                if device == "cuda" and getattr(settings, "tts_turbo_compile_s3gen", True) and hasattr(_turbo_model, "s3gen"):
                    try:
                        import torch as _torch
                        _turbo_model.s3gen = _torch.compile(_turbo_model.s3gen, mode="reduce-overhead")
                        logger.debug("Chatterbox-Turbo s3gen compiled with torch.compile (reduce-overhead)")
                    except Exception as e:
                        logger.debug("s3gen compile failed: %s; leaving uncompiled", e)
            except ImportError:
                try:
                    from chatterbox.tts import ChatterboxTTS
                    logger.info(f"Loading ChatterboxTTS (fallback, device: {device})")
                    try:
                        _turbo_model = ChatterboxTTS.from_pretrained(device=device, attn_implementation="sdpa")
                    except TypeError:
                        _turbo_model = ChatterboxTTS.from_pretrained(device=device)
                    logger.info("ChatterboxTTS model loaded successfully (use Turbo from source for lower latency)")
                    # SDPA wrap for fallback too (non-Turbo uses tfmr in inference loop).
                    if device == "cuda" and getattr(settings, "tts_force_sdpa_attention", True) and hasattr(_turbo_model.t3, "tfmr"):
                        try:
                            _orig_fwd = _turbo_model.t3.tfmr.forward
                            def _wrap_fwd(*args, **kwargs):
                                kwargs["output_attentions"] = False
                                kwargs["output_hidden_states"] = False
                                return _orig_fwd(*args, **kwargs)
                            _turbo_model.t3.tfmr.forward = _wrap_fwd
                            logger.debug("ChatterboxTTS tfmr.forward wrapped for SDPA")
                        except Exception as e:
                            logger.debug("Could not wrap ChatterboxTTS tfmr.forward: %s", e)
                    # Wrap inference to pass max_new_tokens from config or per-call override.
                    t3_obj = _turbo_model.t3
                    _orig_inference = getattr(t3_obj, "inference", None)
                    if callable(_orig_inference):
                        _orig_inference = getattr(_orig_inference, "__func__", _orig_inference)
                        _max_new_tokens_default = getattr(settings, "tts_turbo_max_gen_len", 400)

                        def _wrapped_inference(*args, **kwargs):
                            mn = getattr(t3_obj, "_max_gen_len_override", None)
                            kwargs["max_new_tokens"] = mn if mn is not None else _max_new_tokens_default
                            return _orig_inference(t3_obj, *args, **kwargs)

                        _turbo_model.t3.inference = _wrapped_inference
                        logger.debug("ChatterboxTTS t3.inference wrapped with max_new_tokens from config")
                    # Compile S3 decoder when enabled (CUDA).
                    if device == "cuda" and getattr(settings, "tts_turbo_compile_s3gen", True) and hasattr(_turbo_model, "s3gen"):
                        try:
                            import torch as _torch
                            _turbo_model.s3gen = _torch.compile(_turbo_model.s3gen, mode="reduce-overhead")
                            logger.debug("ChatterboxTTS s3gen compiled with torch.compile (reduce-overhead)")
                        except Exception as e:
                            logger.debug("s3gen compile failed: %s; leaving uncompiled", e)
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
    """Lazy load IndicF5 model and vocoder. On failure set _indicf5_available=False. When tts_indicf5_enabled=False, never loads. Thread-safe."""
    global _indicf5_model, _indicf5_vocoder, _indicf5_device, _indicf5_available
    with _lock_indicf5:
        if not getattr(settings, "tts_indicf5_enabled", False):
            _indicf5_available = False
            return None, None, None
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

            device = _get_indicf5_torch_device()
            _indicf5_device = device
            infer = get_infer_device()
            if device == "cpu" and infer == "mps":
                logger.info("Loading IndicF5 model (device: cpu; MPS unsupported for F5 FFT/complex ops)")
            elif device == "cpu" and getattr(settings, "tts_indicf5_force_cpu", False):
                logger.info("Loading IndicF5 model (device: cpu; TTS_INDICF5_FORCE_CPU=true)")
            else:
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
            logger.warning(
                "IndicF5 load failed: %s. Indic languages will use Gemini TTS (Indic) if GEMINI_API_KEY is set.",
                e,
            )
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


def _get_resemble_client():
    """Lazy-load pooled httpx client for Resemble streaming TTS (thread-safe)."""
    global _resemble_client
    with _lock_resemble:
        if _resemble_client is None:
            if not settings.resemble_api_key:
                raise ValueError(
                    "RESEMBLE_API_KEY not set. Set it when TTS_CHATTERBOX_MODE=api."
                )
            if not settings.resemble_voice_uuid:
                raise ValueError(
                    "RESEMBLE_VOICE_UUID not set. Set it when TTS_CHATTERBOX_MODE=api."
                )
            import httpx

            _resemble_client = httpx.Client(
                base_url="https://f.cluster.resemble.ai",
                headers={
                    "Authorization": f"Bearer {settings.resemble_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=float(settings.resemble_api_timeout),
                    write=10.0,
                    pool=5.0,
                ),
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                ),
            )
    return _resemble_client


def _parse_resemble_error_body(body: bytes, status_code: int) -> str:
    """Extract a user-safe message from Resemble JSON error responses."""
    if not body:
        return f"Resemble TTS request failed (HTTP {status_code})"
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
        if isinstance(data, dict):
            msg = data.get("message") or data.get("error") or data.get("detail")
            if msg:
                return str(msg)
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    preview = body[:500].decode("utf-8", errors="replace")
    return f"Resemble TTS request failed (HTTP {status_code}): {preview}"


def _tts_with_resemble_api(text: str) -> bytes:
    """
    Generate WAV via Resemble AI streaming synthesis (POST /stream).
    https://docs.resemble.ai/api-reference/text-to-speech/stream-synthesize
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text for Resemble TTS")
    if len(text) > 2000:
        logger.warning("Resemble TTS text length %s, truncating to 2000", len(text))
        text = text[:2000]

    client = _get_resemble_client()
    payload = {
        "voice_uuid": settings.resemble_voice_uuid,
        "data": text,
        "model": settings.resemble_api_model,
        "sample_rate": settings.resemble_sample_rate,
        "precision": settings.resemble_precision,
        "use_hd": settings.resemble_use_hd,
    }
    max_retries = max(0, int(getattr(settings, "resemble_api_max_retries", 2)))

    import httpx

    last_error_msg: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            with client.stream("POST", "/stream", json=payload) as response:
                if response.status_code != 200:
                    err_body = response.read()
                    msg = _parse_resemble_error_body(err_body, response.status_code)
                    if 400 <= response.status_code < 500:
                        raise ValueError(msg)
                    last_error_msg = msg
                    if attempt < max_retries:
                        delay = 0.5 * (2**attempt)
                        logger.warning(
                            "Resemble TTS HTTP %s, retry %s/%s after %.1fs",
                            response.status_code,
                            attempt + 1,
                            max_retries,
                            delay,
                        )
                        time.sleep(delay)
                        continue
                    raise ValueError(msg)

                buf = bytearray()
                for chunk in response.iter_bytes():
                    if chunk:
                        buf.extend(chunk)
                wav = bytes(buf)
                if not wav:
                    raise ValueError("Resemble TTS returned empty audio")
                logger.debug("Resemble TTS generated %s bytes WAV", len(wav))
                return wav

        except ValueError:
            raise
        except (httpx.TimeoutException, httpx.ConnectError, httpx.TransportError) as e:
            last_error_msg = str(e)
            if attempt < max_retries:
                delay = 0.5 * (2**attempt)
                logger.warning(
                    "Resemble TTS transport error, retry %s/%s after %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    e,
                )
                time.sleep(delay)
                continue
            logger.error("Resemble TTS failed after retries: %s", e, exc_info=True)
            raise ValueError("Resemble TTS temporarily unavailable. Please try again.") from e

    raise ValueError(last_error_msg or "Resemble TTS failed.")


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
        # Prepare voice conditionals once and reuse (Turbo supports prepare_conditionals + generate without path)
        global _turbo_voice_prepared
        with _lock_turbo:
            if getattr(model, "prepare_conditionals", None) and not _turbo_voice_prepared:
                exaggeration = getattr(settings, "tts_turbo_exaggeration", 0.7)
                try:
                    model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration, norm_loudness=True)
                except TypeError:
                    try:
                        model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
                    except TypeError:
                        model.prepare_conditionals(audio_prompt_path)
                _turbo_voice_prepared = True
                logger.info(
                    "Voice cloning reference loaded once: %s (reusing for subsequent requests)",
                    audio_prompt_path,
                )

        gen_kwargs = {
            "temperature": getattr(settings, "tts_turbo_temperature", 0.8),
            "top_p": getattr(settings, "tts_turbo_top_p", 0.95),
            "top_k": getattr(settings, "tts_turbo_top_k", 1000),
            "repetition_penalty": getattr(settings, "tts_turbo_repetition_penalty", 1.2),
        }
        # ChatterboxTTS (non-Turbo fallback) does not accept top_k; it uses temperature, top_p, repetition_penalty, cfg_weight, exaggeration, min_p.
        gen_kwargs_fallback = {k: v for k, v in gen_kwargs.items() if k != "top_k"}
        gen_kwargs_fallback["cfg_weight"] = getattr(settings, "tts_turbo_cfg_weight", 0.3)

        def _do_generate():
            if _turbo_voice_prepared:
                try:
                    return model.generate(text, **gen_kwargs)
                except TypeError:
                    try:
                        return model.generate(text, **gen_kwargs_fallback)
                    except TypeError:
                        return model.generate(text)
            else:
                try:
                    return model.generate(text, audio_prompt_path=audio_prompt_path, **gen_kwargs)
                except TypeError:
                    try:
                        return model.generate(text, audio_prompt_path=audio_prompt_path, **gen_kwargs_fallback)
                    except TypeError:
                        return model.generate(text, audio_prompt_path=audio_prompt_path)

        # generate_stream is not in upstream; requires a streaming-capable fork (e.g. rsxdalv/chatterbox).
        use_streaming = getattr(settings, "tts_turbo_use_streaming", False) and callable(getattr(model, "generate_stream", None))
        if use_streaming:
            chunk_size = getattr(settings, "tts_turbo_stream_chunk_size", 25)
            chunks = []
            try:
                for item in model.generate_stream(text, chunk_size=chunk_size):
                    if isinstance(item, tuple):
                        audio_chunk = item[0]
                    else:
                        audio_chunk = item
                    if hasattr(audio_chunk, "cpu"):
                        audio_chunk = audio_chunk.cpu().numpy()
                    if isinstance(audio_chunk, np.ndarray):
                        chunks.append(audio_chunk)
            except TypeError:
                use_streaming = False
            if use_streaming:
                wav_array = np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=np.float32)
                if wav_array.ndim > 1:
                    wav_array = wav_array.squeeze()
        if not use_streaming:
            # Dynamic cap: min(config ceiling, max(100, len(text)*3)) for lower latency.
            max_gen_len_ceiling = getattr(settings, "tts_turbo_max_gen_len", 400) or 400
            if hasattr(model, "t3"):
                model.t3._max_gen_len_override = min(max_gen_len_ceiling, max(100, len(text) * 3))
            try:
                no_grad_ctx = torch.inference_mode()
            except AttributeError:
                no_grad_ctx = torch.no_grad()
            with no_grad_ctx:
                wav_tensor = _do_generate()
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


def _effective_response_language(text: str, response_language: str) -> str:
    """
    Use response_language when set to an Indic code; when 'en' (or default),
    infer from text script so Indic text (e.g. Malayalam) uses IndicF5 even if client didn't send language.
    """
    if response_language != "en":
        return response_language
    inferred = _script_to_lang(text)
    return inferred if inferred else "en"


def _generate_tts_bytes_impl(text: str, response_language: str = "en") -> bytes:
    """
    Core TTS routing: IndicF5 when enabled, else Gemini Flash TTS for Indic; English Chatterbox or Gemini.
    """
    lang = _effective_response_language(text, response_language)
    backend = resolve_tts_backend(text, lang)

    if backend == "indicf5":
        try:
            with _inference_semaphore:
                return _tts_with_indicf5(text, lang)
        except ValueError:
            raise
        except Exception as e:
            logger.warning(
                "IndicF5 generation failed: %s. Falling back to configured cloud TTS.",
                e,
            )
            backend = _cloud_tts_provider_for_lang(lang)

    if backend == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "Gemini text-to-speech requires GEMINI_API_KEY when the configured cloud provider is Gemini."
            )
        return _tts_with_gemini(text, lang)

    if backend == "smallest":
        try:
            return _tts_with_smallest(text, lang)
        except ValueError:
            raise
        except Exception as e:
            _log_smallest_fallback_once(e)
            try:
                return _tts_with_chirp(text, lang)
            except Exception as e2:
                if settings.gemini_api_key:
                    _log_chirp_fallback_once(e2)
                    return _tts_with_gemini(text, lang)
                logger.error(
                    "Smallest and Chirp 3 HD generation failed with no Gemini fallback: %s",
                    e2,
                )
                raise ValueError("Smallest text-to-speech is temporarily unavailable.") from e2

    if backend == "chirp3_hd":
        try:
            return _tts_with_chirp(text, lang)
        except ValueError:
            raise
        except Exception as e:
            if settings.gemini_api_key:
                _log_chirp_fallback_once(e)
                return _tts_with_gemini(text, lang)
            logger.error("Chirp 3 HD generation failed with no Gemini fallback available: %s", e)
            raise ValueError("Chirp 3 HD text-to-speech is temporarily unavailable.") from e

    if backend == "resemble_api":
        return _tts_with_resemble_api(text)

    with _inference_semaphore:
        return _tts_with_turbo(text)


def generate_tts_bytes(text: str, response_language: str = "en") -> bytes:
    """
    Public entrypoint: generate TTS audio bytes for one sentence (WAV).
    Used by the chat/stream pipeline for sentence-level TTS.
    """
    return _generate_tts_bytes_impl(text, response_language)


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
    text: str,
    response_language: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Producer: run text_to_speech_stream in the current thread and put each WAV chunk
    into the queue (thread-safe via loop.call_soon_threadsafe). Tuple protocol:
    ("audio", chunk) per sentence; (None, None) as sentinel when done; ("error", str(e)) on exception.
    Intended to be run from a dedicated thread; the API consumer awaits queue.get().
    """
    try:
        for chunk in text_to_speech_stream(text, response_language):
            loop.call_soon_threadsafe(queue.put_nowait, ("audio", chunk))
        loop.call_soon_threadsafe(queue.put_nowait, (None, None))
    except Exception as e:
        logger.exception("TTS stream error in feed_tts_stream_to_queue")
        loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))


def split_text_for_chirp_stream(text: str, max_chars: int = 1024) -> list[str]:
    """Split text into clause/sentence-like fragments for Chirp streaming input."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []
    fragments: list[str] = []
    remaining = cleaned
    while remaining:
        if len(remaining) <= max_chars:
            fragments.append(remaining)
            break
        window = remaining[:max_chars]
        split_at = max(window.rfind("."), window.rfind("?"), window.rfind("!"), window.rfind(";"), window.rfind(":"))
        if split_at <= 0:
            split_at = max(window.rfind(","), window.rfind(" "))
        if split_at <= 0:
            split_at = max_chars
        fragment = remaining[:split_at].strip()
        if fragment:
            fragments.append(fragment)
        remaining = remaining[split_at:].strip()
    return fragments


def feed_chirp_stream_to_queue(
    fragments: "queue_lib.Queue[Optional[str]]",
    response_language: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    stop_event: threading.Event,
) -> None:
    """
    Bridge Chirp bidirectional streaming to the async SSE layer.
    Input comes from a thread-safe fragment queue; output emits WAV-wrapped audio chunks and the
    final raw PCM buffer for stitched replay audio.
    """
    start_time = time.perf_counter()
    first_chunk_at: Optional[float] = None
    chunk_count = 0
    raw_pcm_chunks: list[bytes] = []
    filter_state = ChirpPCMFilterState()
    sample_rate_hz = int(getattr(settings, "tts_chirp_sample_rate_hz", CHIRP_DEFAULT_SAMPLE_RATE_HZ))
    target_packet_bytes = max(4096, int(sample_rate_hz * 2 * (CHIRP_TARGET_PACKET_MS / 1000.0)))
    pending_emit = bytearray()
    try:
        texttospeech = _import_chirp_texttospeech()
        client = _get_chirp_client()
        timeout = float(getattr(settings, "tts_chirp_timeout_seconds", 30) or 30)

        def _request_iter():
            yield texttospeech.StreamingSynthesizeRequest(
                streaming_config=_build_chirp_streaming_config(texttospeech, response_language),
            )
            while not stop_event.is_set():
                fragment = fragments.get()
                if fragment is None:
                    return
                cleaned = (fragment or "").strip()
                if not cleaned:
                    continue
                yield texttospeech.StreamingSynthesizeRequest(
                    input=texttospeech.StreamingSynthesisInput(text=cleaned),
                )

        for response in client.streaming_synthesize(requests=_request_iter(), timeout=timeout):
            if stop_event.is_set():
                break
            audio_content = bytes(getattr(response, "audio_content", b"") or b"")
            if not audio_content:
                continue
            cleaned_audio = _postprocess_chirp_pcm_chunk(
                audio_content,
                sample_rate_hz=sample_rate_hz,
                state=filter_state,
            )
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
            raw_pcm_chunks.append(cleaned_audio)
            pending_emit.extend(cleaned_audio)
            while len(pending_emit) >= target_packet_bytes:
                packet = bytes(pending_emit[:target_packet_bytes])
                del pending_emit[:target_packet_bytes]
                chunk_count += 1
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ("audio", chirp_pcm_to_wav(packet, sample_rate_hz=sample_rate_hz)),
                )

        if pending_emit:
            chunk_count += 1
            loop.call_soon_threadsafe(
                queue.put_nowait,
                ("audio", chirp_pcm_to_wav(bytes(pending_emit), sample_rate_hz=sample_rate_hz)),
            )

        total_ms = (time.perf_counter() - start_time) * 1000.0
        ttfa_ms = ((first_chunk_at - start_time) * 1000.0) if first_chunk_at is not None else None
        logger.info(
            "Chirp stream completed (lang=%s, chunks=%s, ttfa_ms=%s, total_ms=%.1f)",
            response_language,
            chunk_count,
            f"{ttfa_ms:.1f}" if ttfa_ms is not None else "none",
            total_ms,
        )
        loop.call_soon_threadsafe(queue.put_nowait, (CLOUD_STREAM_EVENT_RAW_PCM, b"".join(raw_pcm_chunks)))
        loop.call_soon_threadsafe(queue.put_nowait, (None, None))
    except Exception as e:
        logger.exception("Chirp stream error")
        loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))


def split_text_for_smallest_stream(text: str, max_chars: int = 1500) -> list[str]:
    """Split text into fragments for Smallest SSE (one /stream request per fragment)."""
    return split_text_for_chirp_stream(text, max_chars=max_chars)


def _smallest_parse_sse_json_line(line: str) -> Optional[dict]:
    """
    Parse one Smallest.ai SSE line: 'data: {...}' (space optional) or a bare JSON object line.
    """
    s = line.strip()
    if not s:
        return None
    if s.startswith(":"):
        return None
    if s[:5].lower() == "data:":
        payload = s[5:].lstrip()
        if not payload or payload.strip() == "[DONE]":
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    if s.startswith("{"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None
    return None


def _smallest_stream_body_to_json_objects(raw: bytes) -> list[dict]:
    """
    Smallest /stream responses vary: single-line data:, multiline SSE blocks, NDJSON, or
    back-to-back JSON. Some API versions (v4) use event/chunk multiline data fields.
    Returns decoded JSON objects (or a single {"_riff_wav": bytes} for raw WAV body).
    """
    if not raw:
        return []
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return [{"_riff_wav": raw}]

    text = raw.decode("utf-8", errors="replace")
    out: list[dict] = []

    def _try_obj(s: str) -> None:
        s = s.strip()
        if not s.startswith("{"):
            return
        try:
            out.append(json.loads(s))
        except json.JSONDecodeError:
            return

    # 1) SSE: blank-line delimited event blocks, and per-line data: (each line is usually one JSON object)
    for block in re.split(r"\r\n\r\n|\n\n", text):
        block = (block or "").strip()
        if not block or block.startswith(":"):
            continue
        for line in block.splitlines():
            ls = line.strip()
            if not ls or ls.startswith(":"):
                continue
            if ls.startswith("{"):
                _try_obj(ls)
                continue
            if ls[:5].lower() == "data:":
                payload = ls[5:].lstrip()
                if payload and payload != "[DONE]":
                    _try_obj(payload)
    if out:
        return out
    out = []

    # 2) Line-based: data: or bare JSON
    for line in text.splitlines():
        ls = (line or "").strip()
        if not ls or ls.startswith(":"):
            continue
        if ls[:5].lower() == "data:":
            ls = ls[5:].lstrip()
        if not ls or ls == "[DONE]":
            continue
        _try_obj(ls)
    if out:
        return out
    out = []

    # 3) Contiguous JSON objects (no newlines between)
    dec = json.JSONDecoder()
    idx = 0
    t = text
    while idx < len(t):
        while idx < len(t) and t[idx] in " \t\r\n":
            idx += 1
        if idx >= len(t):
            break
        try:
            obj, end = dec.raw_decode(t, idx)
            out.append(obj)
            idx = end
        except json.JSONDecodeError:
            break
    return out


def _smallest_b64_from_chunk_object(obj: dict) -> Optional[str]:
    """Extract base64 audio from a stream chunk object (v3/v4 shape differences)."""
    st = obj.get("status") or obj.get("type")
    if st in ("error", "complete", "done"):
        return None
    if obj.get("error"):
        return None
    d = obj.get("data")
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        a = d.get("audio")
        if isinstance(a, str):
            return a
    a = obj.get("audio")
    if isinstance(a, str):
        return a
    return None


def feed_smallest_stream_to_queue(
    fragments: "queue_lib.Queue[Optional[str]]",
    response_language: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    stop_event: threading.Event,
) -> None:
    """
    Bridge Smallest.ai SSE /stream to the async SSE layer (Lightning TTS).
    See https://docs.smallest.ai — one HTTP streaming request per text fragment.
    """
    start_time = time.perf_counter()
    first_chunk_at: Optional[float] = None
    chunk_count = 0
    raw_pcm_chunks: list[bytes] = []
    sample_rate_hz = int(getattr(settings, "tts_smallest_sample_rate_hz", CHIRP_DEFAULT_SAMPLE_RATE_HZ) or 24000)
    target_packet_bytes = max(4096, int(sample_rate_hz * 2 * (CHIRP_TARGET_PACKET_MS / 1000.0)))
    pending_emit = bytearray()
    model = _smallest_model_segment()
    stream_timeout = float(getattr(settings, "tts_smallest_timeout_seconds", 30) or 30)
    lang = response_language
    voice_id = _smallest_voice_for_lang(lang)
    speed = float(getattr(settings, "tts_smallest_speed", 1.0) or 1.0)
    import httpx
    import wave

    try:
        client = _get_smallest_client()
        while not stop_event.is_set():
            fragment = fragments.get()
            if fragment is None:
                break
            cleaned = (fragment or "").strip()
            if not cleaned:
                continue
            filter_state = ChirpPCMFilterState()
            tmo = httpx.Timeout(connect=10.0, read=stream_timeout, write=10.0, pool=5.0)
            # Match official GET /stream examples: only text, voice_id, sample_rate, optional speed.
            # output_format+language in the body led to 200 with no parseable audio lines in production.
            stream_payload: dict = {
                "text": cleaned,
                "voice_id": voice_id,
                "sample_rate": sample_rate_hz,
            }
            if abs(speed - 1.0) > 1e-6:
                stream_payload["speed"] = speed
            with client.stream(
                "POST",
                f"{model}/stream",
                json=stream_payload,
                headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
                timeout=tmo,
            ) as response:
                if response.status_code != 200:
                    err_body = response.read()
                    err_preview = (err_body or b"")[:500].decode("utf-8", errors="replace")
                    raise ValueError(
                        f"Smallest TTS stream HTTP {response.status_code}: {err_preview!r}"
                    )
                # Buffer full body: Smallest may send multiline SSE, NDJSON, or no newlines; line-only parsers miss chunks.
                raw = b"".join(response.iter_bytes())
                if not raw:
                    logger.warning(
                        "Smallest TTS: empty body (content-type=%s)",
                        (response.headers.get("content-type") or ""),
                    )
                    continue

                def _emit_decoded_raw_pcm(raw_b: bytes) -> None:
                    nonlocal first_chunk_at, chunk_count, pending_emit, filter_state, raw_pcm_chunks
                    if not raw_b:
                        return
                    cleaned_a = _postprocess_chirp_pcm_chunk(
                        raw_b,
                        sample_rate_hz=sample_rate_hz,
                        state=filter_state,
                    )
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter()
                    raw_pcm_chunks.append(cleaned_a)
                    pending_emit.extend(cleaned_a)
                    while len(pending_emit) >= target_packet_bytes:
                        packet = bytes(pending_emit[:target_packet_bytes])
                        del pending_emit[:target_packet_bytes]
                        chunk_count += 1
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            ("audio", chirp_pcm_to_wav(packet, sample_rate_hz=sample_rate_hz)),
                        )

                objects = _smallest_stream_body_to_json_objects(raw)
                if not objects:
                    logger.warning(
                        "Smallest TTS: could not parse stream (len=%s, preview=%r)",
                        len(raw),
                        raw[:1500].decode("utf-8", errors="replace"),
                    )
                    continue

                for obj in objects:
                    if stop_event.is_set():
                        break
                    if obj.get("_riff_wav") is not None:
                        wbytes: bytes = obj["_riff_wav"]
                        if first_chunk_at is None:
                            first_chunk_at = time.perf_counter()
                        try:
                            bio = BytesIO(wbytes)
                            with wave.open(bio, "rb") as wfr:
                                raw_pcm_chunks.append(wfr.readframes(wfr.getnframes()))
                        except Exception:
                            pass
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            ("audio", wbytes if wbytes[:4] == b"RIFF" else chirp_pcm_to_wav(wbytes, sample_rate_hz=sample_rate_hz)),
                        )
                        chunk_count += 1
                        continue
                    st = obj.get("status") or obj.get("type")
                    if st == "error" or obj.get("error"):
                        logger.warning("Smallest TTS error event: %s", obj)
                        continue
                    if st in ("complete", "done"):
                        break
                    b64u = _smallest_b64_from_chunk_object(obj)
                    if not b64u:
                        continue
                    try:
                        raw_b = base64.b64decode(b64u)
                    except Exception:
                        continue
                    if not raw_b:
                        continue
                    _emit_decoded_raw_pcm(raw_b)
            if pending_emit:
                chunk_count += 1
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ("audio", chirp_pcm_to_wav(bytes(pending_emit), sample_rate_hz=sample_rate_hz)),
                )
                pending_emit.clear()
        total_ms = (time.perf_counter() - start_time) * 1000.0
        ttfa_ms = ((first_chunk_at - start_time) * 1000.0) if first_chunk_at is not None else None
        logger.info(
            "Smallest TTS stream completed (lang=%s, chunks=%s, ttfa_ms=%s, total_ms=%.1f)",
            response_language,
            chunk_count,
            f"{ttfa_ms:.1f}" if ttfa_ms is not None else "none",
            total_ms,
        )
        loop.call_soon_threadsafe(queue.put_nowait, (CLOUD_STREAM_EVENT_RAW_PCM, b"".join(raw_pcm_chunks)))
        loop.call_soon_threadsafe(queue.put_nowait, (None, None))
    except Exception as e:
        logger.exception("Smallest stream error")
        loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))


def store_pcm_audio_mp3(audio_bytes: bytes, filename: str, sample_rate_hz: Optional[int] = None) -> str:
    """Convert raw PCM audio to MP3, store it, and return the playback URL."""
    wav_bytes = chirp_pcm_to_wav(audio_bytes, sample_rate_hz=sample_rate_hz)
    return store_audio_mp3(_convert_wav_to_mp3(wav_bytes), filename)


def store_audio_mp3_record(audio_bytes: bytes, filename: str) -> StoredAudioRecord:
    """Store MP3 bytes and return a durable storage reference plus playback URL."""
    s3_key = _store_audio_cloud(audio_bytes, filename)
    if s3_key:
        storage_ref = _storage_ref_for_s3(s3_key)
        presigned = resolve_stored_audio_playback_url(storage_ref)
        if presigned:
            return StoredAudioRecord(playback_url=presigned, storage_ref=storage_ref)
    playback_url = _store_audio_local(audio_bytes, filename)
    return StoredAudioRecord(
        playback_url=playback_url,
        storage_ref=_storage_ref_for_local(filename),
    )


def store_audio_mp3(audio_bytes: bytes, filename: str) -> str:
    """
    Store MP3 bytes to local or S3 and return the playback URL.
    Used after concatenating streamed TTS chunks.
    """
    return store_audio_mp3_record(audio_bytes, filename).playback_url


def text_to_speech_record(text: str, response_language: str = "en") -> StoredAudioRecord:
    """Convert text to speech and return a playback URL with a durable storage reference."""
    lang = _effective_response_language(text, response_language)
    provider = resolve_tts_backend(text, lang)
    cache_key = _generate_cache_key(
        text,
        lang,
        gemini_indic=(provider == "gemini" and lang != "en"),
        provider_tag=provider,
    )
    cached = get(cache_key)
    if cached:
        logger.info("Cache hit for TTS (%s)", provider)
        cached_record = _legacy_audio_cache_value_to_record(cached)
        if cached_record:
            return cached_record

    logger.info("Using TTS provider: %s (language=%s)", provider, lang)
    try:
        audio_bytes = _generate_tts_bytes_impl(text, lang)
        audio_bytes = _convert_wav_to_mp3(audio_bytes)
        filename_prefix = f"{lang}_" if lang != "en" else ""
        filename = f"{filename_prefix}{hashlib.md5(text.encode()).hexdigest()}.mp3"
        record = store_audio_mp3_record(audio_bytes, filename)
        set(cache_key, record.storage_ref, settings.tts_cache_ttl)
        return record
    except ValueError:
        raise
    except Exception as e:
        logger.error("Unexpected TTS error from provider %s: %s", provider, e)
        raise ValueError("Something went wrong generating audio. Please try again.") from e


def text_to_speech(text: str, response_language: str = "en") -> str:
    """
    Convert text to speech and return audio URL.
    English: Chatterbox-Turbo (or Gemini if Chatterbox disabled).
    Indic: IndicF5 when enabled, else Gemini 2.5 Flash TTS (requires GEMINI_API_KEY).

    Args:
        text: Text to convert
        response_language: "en" or Indic code (hi, ml, ta, ...)

    Returns:
        URL to audio file

    Raises:
        ValueError: If audio generation fails
    """
    return text_to_speech_record(text, response_language).playback_url


def init_tts_models() -> dict:
    """
    Load TTS models (Turbo and optionally IndicF5) for warmup.
    When TTS_CHATTERBOX_ENABLED=false, Turbo is not loaded.
    When TTS_CHATTERBOX_MODE=api, local Turbo is not loaded; Resemble API is warmed with a short synthesis.
    When TTS_INDICF5_ENABLED=false, IndicF5 is not loaded.
    Runs one short TTS inference after loading Turbo so torch.compile/CUDA kernels are warmed.
    Returns {"turbo": ..., "indicf5": ...}.
    """
    global _turbo_warmup_done
    result: dict = {}
    if not getattr(settings, "tts_chatterbox_enabled", True):
        result["turbo"] = "disabled"
    elif getattr(settings, "tts_chatterbox_mode", "local") == "api":
        if not settings.resemble_api_key or not settings.resemble_voice_uuid:
            result["turbo"] = "failed"
            result["turbo_error"] = (
                "RESEMBLE_API_KEY and RESEMBLE_VOICE_UUID are required when TTS_CHATTERBOX_MODE=api"
            )
        else:
            try:
                _tts_with_resemble_api("Hi.")
                _turbo_warmup_done = True
                result["turbo"] = "api_mode"
                logger.info("Resemble TTS API warm-up completed")
            except Exception as e:
                logger.exception("Resemble TTS API init failed")
                result["turbo"] = "failed"
                result["turbo_error"] = str(e)
    else:
        try:
            _get_turbo_model()
            result["turbo"] = "loaded"
            if not _turbo_warmup_done:
                try:
                    generate_tts_bytes("Hello.", "en")
                    _turbo_warmup_done = True
                    logger.info("TTS warm-up inference completed (compiled kernels ready)")
                except Exception as warmup_e:
                    logger.warning("TTS warm-up inference failed: %s (first user request may be slower)", warmup_e)
        except Exception as e:
            logger.exception("TTS Turbo init failed")
            result["turbo"] = "failed"
            result["turbo_error"] = str(e)
    if not getattr(settings, "tts_indicf5_enabled", False):
        result["indicf5"] = "disabled"
    else:
        model, _, _ = _get_indicf5_model()
        if model is not None:
            result["indicf5"] = "loaded"
        else:
            result["indicf5"] = "skipped"

    if getattr(settings, "tts_cloud_provider", "gemini") == "chirp3_hd":
        chirp_status = chirp_runtime_status()
        result["chirp3_hd"] = "available" if chirp_status["available"] else "unavailable"
        if chirp_status["reason"]:
            result["chirp3_hd_reason"] = chirp_status["reason"]

    if getattr(settings, "tts_cloud_provider", "gemini") == "smallest":
        if not getattr(settings, "smallest_api_key", None):
            result["smallest"] = "unavailable"
            result["smallest_reason"] = "SMALLEST_API_KEY not set"
        else:
            try:
                _tts_with_smallest("Hi.", "en")
                result["smallest"] = "available"
            except Exception as e:
                result["smallest"] = "unavailable"
                result["smallest_reason"] = str(e)

    return result
