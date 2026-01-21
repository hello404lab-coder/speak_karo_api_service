"""Text-to-Speech service with cloud/local storage."""
import base64
import hashlib
import logging
import os
import struct
from io import BytesIO
from typing import Optional
from openai import OpenAI
from openai import APITimeoutError, APIError
from google import genai
from google.genai import types
from pydub import AudioSegment
from app.core.config import settings
from app.services.cache import get, set

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=settings.openai_api_key)

# Ensure audio storage directory exists
if not os.path.exists(settings.audio_storage_path):
    os.makedirs(settings.audio_storage_path, exist_ok=True)


def _generate_cache_key(text: str) -> str:
    """Generate cache key from text."""
    return f"tts:{hashlib.md5(text.encode()).hexdigest()}"


def _store_audio_local(audio_bytes: bytes, filename: str) -> str:
    """Store audio file locally and return URL."""
    filepath = os.path.join(settings.audio_storage_path, filename)
    
    with open(filepath, "wb") as f:
        f.write(audio_bytes)
    
    # Return URL for local serving
    return f"{settings.audio_base_url}/{filename}"


def _store_audio_cloud(audio_bytes: bytes, filename: str) -> Optional[str]:
    """Store audio file in cloud storage (S3) and return URL."""
    if not all([
        settings.aws_access_key_id,
        settings.aws_secret_access_key,
        settings.s3_bucket_name
    ]):
        return None
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region or 'us-east-1'
        )
        
        s3_client.put_object(
            Bucket=settings.s3_bucket_name,
            Key=f"audio/{filename}",
            Body=audio_bytes,
            ContentType="audio/mpeg"
        )
        
        # Generate public URL
        url = f"https://{settings.s3_bucket_name}.s3.{settings.aws_region or 'us-east-1'}.amazonaws.com/audio/{filename}"
        return url
        
    except ImportError:
        logger.warning("boto3 not installed, falling back to local storage")
        return None
    except Exception as e:
        logger.error(f"Cloud storage error: {e}, falling back to local")
        return None


def _tts_with_openai(text: str) -> bytes:
    """
    Generate audio using OpenAI TTS API.
    
    Args:
        text: Text to convert
    
    Returns:
        Audio bytes (MP3 format)
    
    Raises:
        ValueError: If audio generation fails
    """
    try:
        # Generate audio using OpenAI TTS
        response = client.audio.speech.create(
            model=settings.tts_model,
            voice=settings.tts_voice,
            input=text,
            speed=settings.tts_speed
        )
        
        return response.content
        
    except APITimeoutError:
        logger.error("OpenAI TTS API timeout")
        raise ValueError("Audio generation is taking too long. Please try again.")
    except APIError as e:
        logger.error(f"OpenAI TTS API error: {e}")
        raise ValueError("Could not generate audio. Please try again.")
    except Exception as e:
        logger.error(f"Unexpected OpenAI TTS error: {e}")
        raise ValueError("Something went wrong generating audio. Please try again.")


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


def _tts_with_gemini(text: str) -> bytes:
    """
    Generate audio using Gemini TTS API.
    
    Args:
        text: Text to convert
    
    Returns:
        Audio bytes (MP3 format, converted from WAV)
    
    Raises:
        ValueError: If audio generation fails
    """
    if not settings.gemini_api_key:
        raise ValueError("Gemini API key not configured. Please set GEMINI_API_KEY in environment.")
    
    try:
        # Initialize Gemini client
        gemini_client = genai.Client(api_key=settings.gemini_api_key)
        
        logger.info(f"Calling Gemini TTS API with model: {settings.tts_gemini_model}, voice: {settings.tts_gemini_voice}")
        
        # Check text length - Gemini TTS has limits (4KB per field, 8KB total)
        logger.info(f"Text to convert: '{text[:100]}...' (total length: {len(text)} characters)")
        
        if len(text.encode('utf-8')) > 4000:
            logger.warning(f"Text is {len(text.encode('utf-8'))} bytes, which exceeds Gemini TTS limit. Truncating...")
            text_bytes = text.encode('utf-8')
            text = text_bytes[:4000].decode('utf-8', errors='ignore')
            logger.info(f"Truncated text to {len(text)} characters")
        
        # === OFFICIAL GOOGLE AI DOCUMENTATION PATTERN ===
        # From: https://ai.google.dev/gemini-api/docs/speech-generation
        # 
        # Key points:
        # 1. Use response_modalities=["AUDIO"] (UPPERCASE)
        # 2. Pass text as simple string in contents parameter
        # 3. Use non-streaming generate_content() for reliability
        # 4. Audio is returned as base64 in inline_data.data
        # 5. Style directives can be added to text for tone control
        # 6. language_code controls accent (en-IN for Indian English)
        
        # Add style directive for tone, accent, and style control
        # Format: Style directive followed by the actual text
        style_directive = "Say in a happy, warm, and energetic tone with a natural Indian English accent, as an informative and friendly assistant:"
        styled_text = f"{style_directive}{text}"
        
        # Check if styled text exceeds limit (account for style directive)
        if len(styled_text.encode('utf-8')) > 4000:
            # If styled text is too long, truncate original text to fit
            max_text_bytes = 4000 - len(style_directive.encode('utf-8'))
            text_bytes = text.encode('utf-8')
            if len(text_bytes) > max_text_bytes:
                text = text_bytes[:max_text_bytes].decode('utf-8', errors='ignore')
                styled_text = f"{style_directive}{text}"
                logger.warning(f"Text truncated to fit style directive. New length: {len(text)} characters")
        
        logger.info(f"Generating audio with non-streaming API (official pattern)")
        logger.info(f"Style: Happy, Warm, Energetic | Accent: Indian English | Tone: Informative & Friendly")
        
        response = gemini_client.models.generate_content(
            model=settings.tts_gemini_model,
            contents=styled_text,  # Text with style directive
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],  # UPPERCASE as per official docs
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=settings.tts_gemini_voice
                        )
                    ),
                    language_code="en-IN"  # Indian English accent
                )
            )
        )
        
        # Extract audio data from response
        # Structure: response.candidates[0].content.parts[0].inline_data
        if (
            response.candidates is None
            or len(response.candidates) == 0
            or response.candidates[0].content is None
            or response.candidates[0].content.parts is None
            or len(response.candidates[0].content.parts) == 0
        ):
            raise ValueError("No audio data in Gemini TTS response")
        
        part = response.candidates[0].content.parts[0]
        
        # Check for inline_data (audio)
        if part.inline_data is None:
            raise ValueError(f"No inline_data in response part. Part type: {type(part)}")
        
        # Get the raw audio bytes and MIME type
        audio_data = part.inline_data.data
        mime_type = part.inline_data.mime_type or "audio/L16;rate=24000"
        
        logger.info(f"Received audio data: {len(audio_data)} bytes, MIME type: {mime_type}")
        
        # The audio_data from Gemini is base64 encoded - decode it
        if isinstance(audio_data, str):
            audio_bytes = base64.b64decode(audio_data)
            logger.info(f"Decoded base64 audio: {len(audio_bytes)} bytes")
        else:
            # Already bytes
            audio_bytes = audio_data
            logger.info(f"Audio data is already bytes: {len(audio_bytes)} bytes")
        
        # Verify byte alignment for PCM L16 (must be even - 16-bit = 2 bytes per sample)
        if len(audio_bytes) % 2 != 0:
            logger.warning(f"Odd byte length: {len(audio_bytes)} bytes. Padding with zero byte for PCM L16 alignment.")
            audio_bytes += b'\x00'
        
        # Calculate expected duration
        expected_duration = (len(audio_bytes) / 2) / 24000  # bytes / 2 (16-bit) / sample_rate
        logger.info(f"Audio: {len(audio_bytes)} bytes, expected duration: ~{expected_duration:.2f}s")
        
        # Warn if audio seems too short for the text length
        estimated_duration = len(text) * 0.1  # ~0.1 seconds per character estimate
        if expected_duration < estimated_duration * 0.1:  # If actual is less than 10% of estimated
            logger.warning(f"Short audio ({expected_duration:.2f}s) for text ({len(text)} chars, estimated ~{estimated_duration:.1f}s)")
            # Fall back to OpenAI TTS
            logger.warning("Gemini TTS produced short audio. Falling back to OpenAI TTS...")
            try:
                return _tts_with_openai(text)
            except Exception as e:
                logger.error(f"OpenAI TTS fallback also failed: {e}")
                logger.warning("Proceeding with short Gemini audio as last resort")
        
        # Convert to WAV format (add WAV header if needed)
        if not mime_type.startswith("audio/wav") and not mime_type.startswith("audio/x-wav"):
            # Need to add WAV header
            wav_bytes = _convert_to_wav(audio_bytes, mime_type)
            header_size = len(wav_bytes) - len(audio_bytes)
            logger.info(f"Converted to WAV: {len(wav_bytes)} bytes (header: {header_size} bytes, data: {len(audio_bytes)} bytes)")
            
            # Validate WAV file structure
            if len(wav_bytes) < 44:  # Minimum WAV file size
                raise ValueError("Generated WAV file is too small")
            if wav_bytes[:4] != b"RIFF":
                raise ValueError("Invalid WAV file - missing RIFF header")
            if wav_bytes[8:12] != b"WAVE":
                raise ValueError("Invalid WAV file - missing WAVE format")
            if wav_bytes[12:16] != b"fmt ":
                raise ValueError("Invalid WAV file - missing fmt chunk")
            if wav_bytes[36:40] != b"data":
                raise ValueError("Invalid WAV file - missing data chunk")
            
            # Log WAV file details for debugging
            import struct
            sample_rate = struct.unpack("<I", wav_bytes[24:28])[0]
            bits_per_sample = struct.unpack("<H", wav_bytes[34:36])[0]
            channels = struct.unpack("<H", wav_bytes[22:24])[0]
            data_size = struct.unpack("<I", wav_bytes[40:44])[0]
            logger.info(f"WAV file details: {channels} channel(s), {sample_rate}Hz, {bits_per_sample}-bit, {data_size} bytes data")
        else:
            # Already WAV format
            wav_bytes = audio_bytes
            logger.info(f"Using WAV format directly: {len(wav_bytes)} bytes")
        
        # Convert WAV to MP3
        mp3_bytes = _convert_wav_to_mp3(wav_bytes)
        
        logger.info(f"Converted to MP3: {len(mp3_bytes)} bytes")
        
        return mp3_bytes
        
    except ValueError:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Gemini TTS API error: {e}", exc_info=True)
        raise ValueError("Could not generate audio. Please try again.")


def text_to_speech(text: str) -> str:
    """
    Convert text to speech and return audio URL using configured TTS provider.
    
    Args:
        text: Text to convert
    
    Returns:
        URL to audio file
    
    Raises:
        ValueError: If audio generation fails
    """
    # Check cache
    cache_key = _generate_cache_key(text)
    cached_url = get(cache_key)
    if cached_url:
        logger.info("Cache hit for TTS")
        return cached_url
    
    # Route to appropriate provider
    provider = settings.tts_provider.lower()
    logger.info(f"Using TTS provider: {provider}")
    
    try:
        # Generate audio bytes based on provider
        if provider == "gemini":
            audio_bytes = _tts_with_gemini(text)
        elif provider == "openai":
            audio_bytes = _tts_with_openai(text)
        else:
            logger.error(f"Unknown TTS provider: {provider}")
            raise ValueError(f"Invalid TTS provider configuration: {provider}. Use 'openai' or 'gemini'.")
        
        # Generate filename
        filename = f"{hashlib.md5(text.encode()).hexdigest()}.mp3"
        
        # Try cloud storage first, fallback to local
        audio_url = _store_audio_cloud(audio_bytes, filename)
        if not audio_url:
            audio_url = _store_audio_local(audio_bytes, filename)
        
        # Cache URL
        set(cache_key, audio_url, settings.tts_cache_ttl)
        
        return audio_url
        
    except ValueError:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Unexpected TTS error: {e}")
        raise ValueError("Something went wrong generating audio. Please try again.")
