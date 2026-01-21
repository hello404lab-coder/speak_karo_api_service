"""Speech-to-Text service using Whisper API."""
import logging
from io import BytesIO
from openai import OpenAI
from openai import APITimeoutError, APIError
import requests
from app.core.config import settings
from app.utils.audio import validate_wav_file

logger = logging.getLogger(__name__)

# Initialize OpenAI client (Whisper is part of OpenAI API)
client = OpenAI(api_key=settings.openai_api_key)


def _transcribe_with_openai(audio_file: bytes, filename: str) -> str:
    """
    Transcribe audio using OpenAI Whisper API.
    
    Args:
        audio_file: Audio file bytes
        filename: Original filename (for format detection)
    
    Returns:
        Transcribed text
    
    Raises:
        ValueError: If transcription fails
    """
    try:
        # Create a temporary file-like object
        audio_buffer = BytesIO(audio_file)
        audio_buffer.name = filename
        
        # Call Whisper API
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_buffer,
            language="en"  # English only for this app
        )
        
        transcribed_text = transcript.text.strip()
        
        if not transcribed_text:
            raise ValueError("Could not transcribe audio. Please try speaking more clearly.")
        
        return transcribed_text
        
    except APITimeoutError:
        logger.error("STT API timeout")
        raise ValueError("Audio processing is taking too long. Please try again.")
    except APIError as e:
        logger.error(f"STT API error: {e}")
        raise ValueError("Could not process audio. Please try again.")
    except Exception as e:
        logger.error(f"Unexpected STT error: {e}")
        raise ValueError("Something went wrong processing your audio. Please try again.")


def _transcribe_with_qubrid(audio_file: bytes, filename: str) -> str:
    """
    Transcribe audio using Qubrid Whisper Large v3 API.
    
    Args:
        audio_file: Audio file bytes
        filename: Original filename (for format detection)
    
    Returns:
        Transcribed text
    
    Raises:
        ValueError: If transcription fails
    """
    if not settings.qubrid_api_key:
        raise ValueError("Qubrid API key not configured. Please set QUBRID_API_KEY in environment.")
    
    try:
        url = "https://platform.qubrid.com/api/v1/qubridai/audio/transcribe"
        headers = {
            "Authorization": f"Bearer {settings.qubrid_api_key}"
        }
        
        # Prepare file for multipart/form-data upload
        files = {
            "file": (filename, audio_file, "audio/wav")
        }
        
        data = {
            "model": "openai/whisper-large-v3",
            "language": "en"
        }
        
        logger.info(f"Calling Qubrid API: {url} with model: {data['model']}")
        
        # Make API request
        response = requests.post(
            url,
            headers=headers,
            files=files,
            data=data,
            timeout=30
        )
        
        logger.info(f"Qubrid API response status: {response.status_code}")
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Extract transcribed text from response
        # Qubrid API response format may vary, handle common formats
        if isinstance(result, dict):
            transcribed_text = result.get("text", "").strip()
            if not transcribed_text:
                # Try alternative response formats
                transcribed_text = result.get("transcription", "").strip()
        elif isinstance(result, str):
            transcribed_text = result.strip()
        else:
            transcribed_text = str(result).strip()
        
        if not transcribed_text:
            raise ValueError("Could not transcribe audio. Please try speaking more clearly.")
        
        return transcribed_text
        
    except requests.exceptions.Timeout:
        logger.error("Qubrid STT API timeout")
        raise ValueError("Audio processing is taking too long. Please try again.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Qubrid STT API error: {e}")
        raise ValueError("Could not process audio. Please try again.")
    except Exception as e:
        logger.error(f"Unexpected Qubrid STT error: {e}")
        raise ValueError("Something went wrong processing your audio. Please try again.")


def transcribe_audio(audio_file: bytes, filename: str = "audio.wav") -> str:
    """
    Transcribe audio file to text using configured STT provider.
    
    Args:
        audio_file: Audio file bytes
        filename: Original filename (for format detection)
    
    Returns:
        Transcribed text
    
    Raises:
        ValueError: If audio file is invalid or transcription fails
    """
    # Validate audio file
    try:
        validate_wav_file(audio_file)
    except ValueError as e:
        logger.error(f"Audio validation failed: {e}")
        raise ValueError("Invalid audio file. Please upload a valid WAV file.")
    
    # Route to appropriate provider
    provider = settings.stt_provider.lower()
    logger.info(f"Using STT provider: {provider}")
    
    if provider == "qubrid":
        if not settings.qubrid_api_key:
            logger.error("Qubrid API key not configured")
        else:
            logger.info("Qubrid API key found, using Qubrid for transcription")
        return _transcribe_with_qubrid(audio_file, filename)
    elif provider == "openai":
        logger.info("Using OpenAI for transcription")
        return _transcribe_with_openai(audio_file, filename)
    else:
        logger.error(f"Unknown STT provider: {provider}")
        raise ValueError(f"Invalid STT provider configuration: {provider}. Use 'openai' or 'qubrid'.")
