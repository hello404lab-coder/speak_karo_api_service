"""Audio validation utilities."""
import struct
from typing import Tuple


def validate_wav_file(audio_bytes: bytes, max_size_mb: int = 10, max_duration_seconds: int = 60) -> None:
    """
    Validate WAV file format, size, and duration.
    
    Args:
        audio_bytes: Audio file bytes
        max_size_mb: Maximum file size in MB
        max_duration_seconds: Maximum duration in seconds
    
    Raises:
        ValueError: If validation fails
    """
    # Check file size
    max_size_bytes = max_size_mb * 1024 * 1024
    if len(audio_bytes) > max_size_bytes:
        raise ValueError(f"Audio file too large. Maximum size is {max_size_mb}MB.")
    
    if len(audio_bytes) < 44:  # WAV header is at least 44 bytes
        raise ValueError("Invalid audio file. File too small.")
    
    # Check WAV header
    if audio_bytes[:4] != b'RIFF':
        raise ValueError("Invalid audio file. Not a valid WAV file.")
    
    if audio_bytes[8:12] != b'WAVE':
        raise ValueError("Invalid audio file. Not a valid WAV file.")
    
    # Try to extract duration (basic check)
    try:
        # Find fmt chunk
        fmt_pos = audio_bytes.find(b'fmt ')
        if fmt_pos == -1:
            raise ValueError("Invalid WAV file format.")
        
        # Extract sample rate and byte rate
        sample_rate = struct.unpack('<I', audio_bytes[fmt_pos + 12:fmt_pos + 16])[0]
        byte_rate = struct.unpack('<I', audio_bytes[fmt_pos + 16:fmt_pos + 20])[0]
        
        # Find data chunk
        data_pos = audio_bytes.find(b'data')
        if data_pos == -1:
            raise ValueError("Invalid WAV file format.")
        
        data_size = struct.unpack('<I', audio_bytes[data_pos + 4:data_pos + 8])[0]
        
        # Calculate duration
        if byte_rate > 0:
            duration = data_size / byte_rate
            if duration > max_duration_seconds:
                raise ValueError(f"Audio too long. Maximum duration is {max_duration_seconds} seconds.")
    except (struct.error, ValueError) as e:
        # If we can't parse, just check basic format
        if "Invalid" in str(e) or "too long" in str(e):
            raise
        # Otherwise, just log and continue (basic validation passed)
        pass
