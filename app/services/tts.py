"""Text-to-Speech service with cloud/local storage."""
import hashlib
import logging
import os
from typing import Optional
from openai import OpenAI
from openai import APITimeoutError, APIError
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


def text_to_speech(text: str) -> str:
    """
    Convert text to speech and return audio URL.
    
    Args:
        text: Text to convert
    
    Returns:
        URL to audio file
    """
    # Check cache
    cache_key = _generate_cache_key(text)
    cached_url = get(cache_key)
    if cached_url:
        logger.info("Cache hit for TTS")
        return cached_url
    
    try:
        # Generate audio using OpenAI TTS
        response = client.audio.speech.create(
            model=settings.tts_model,
            voice=settings.tts_voice,
            input=text,
            speed=settings.tts_speed
        )
        
        audio_bytes = response.content
        
        # Generate filename
        filename = f"{hashlib.md5(text.encode()).hexdigest()}.mp3"
        
        # Try cloud storage first, fallback to local
        audio_url = _store_audio_cloud(audio_bytes, filename)
        if not audio_url:
            audio_url = _store_audio_local(audio_bytes, filename)
        
        # Cache URL
        set(cache_key, audio_url, settings.tts_cache_ttl)
        
        return audio_url
        
    except APITimeoutError:
        logger.error("TTS API timeout")
        raise ValueError("Audio generation is taking too long. Please try again.")
    except APIError as e:
        logger.error(f"TTS API error: {e}")
        raise ValueError("Could not generate audio. Please try again.")
    except Exception as e:
        logger.error(f"Unexpected TTS error: {e}")
        raise ValueError("Something went wrong generating audio. Please try again.")
