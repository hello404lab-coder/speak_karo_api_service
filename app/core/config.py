"""Application configuration using Pydantic settings."""
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Literal, Optional


def _default_cache_enabled() -> bool:
    """In prod default to True when CACHE_ENABLED not set; in dev default False."""
    if os.getenv("CACHE_ENABLED") is not None:
        return os.getenv("CACHE_ENABLED", "").lower() in ("1", "true")
    return os.getenv("APP_ENV", "dev").lower() == "prod"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Environment: dev (CPU, relaxed) vs prod (GPU when available, strict)
    app_env: Literal["dev", "prod"] = Field(default="dev", description="APP_ENV: dev or prod")
    
    # API Keys
    gemini_api_key: Optional[str] = None  # Required for LLM; required in prod (validated at startup)
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/english_practice"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Cache (LLM and TTS). Prod defaults True when CACHE_ENABLED not set.
    cache_enabled: bool = Field(default_factory=_default_cache_enabled, description="CACHE_ENABLED")
    
    # Cloud Storage (Optional - S3)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None
    s3_bucket_name: Optional[str] = None
    s3_presigned_expiry_seconds: int = 3600  # Expiry for presigned GET URLs when S3 is used. Env: S3_PRESIGNED_EXPIRY_SECONDS
    
    # Application
    app_name: str = "AI English Practice Backend"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Audio Storage
    # IMPORTANT: Keep this outside `backend/` so `uvicorn --reload` doesn't restart
    # every time we write a new MP3 (which looks like the test page “reloads”).
    audio_storage_path: str = str(Path(__file__).resolve().parents[3] / "audio_storage")
    audio_base_url: str = "http://localhost:8000/audio"  # For local serving
    
    # LLM Settings
    llm_model: str = "gemini-2.5-flash"  # Gemini model for LLM (fast and efficient)
    llm_max_tokens: int = 400  # Increased for complete responses (Gemini 2.5 Flash supports up to 65,536)
    llm_temperature: float = 0.4
    
    # STT Settings
    stt_mode: Literal["faster_whisper_medium", "faster_whisper_large"] = "faster_whisper_large"  # Env: STT_MODE
    stt_faster_whisper_model_size: str = "medium"  # Used for faster_whisper_medium
    
    # TTS Settings - Chatterbox-Turbo (English, https://huggingface.co/ResembleAI/chatterbox-turbo)
    # Device is auto-detected: cuda > mps > cpu. Requires a reference clip for voice cloning.
    tts_audio_prompt_path: Optional[str] = "chirp3-hd-puck.wav"  # Path to ~10s reference WAV for voice cloning (required for Turbo)

    # TTS Settings - IndicF5 (for Indic languages: hi, ml, ta)
    # Base directory containing ref WAVs (e.g. IndicF5/prompts or backend/assets/indicf5_prompts)
    # Default: project root / IndicF5 / prompts if that path exists
    tts_indicf5_ref_audio_dir: Optional[str] = None  # Set to path for ref WAVs; None disables IndicF5
    tts_indicf5_speed: float = 0.9  # Speech speed (0.9 in IndicF5 main.py)

    # Cache TTLs
    llm_cache_ttl: int = 86400  # 24 hours
    tts_cache_ttl: int = 604800  # 7 days

    # Timeouts (seconds) for inference; sync calls are run in executor and wrapped with asyncio.wait_for
    llm_timeout_seconds: int = 60
    stt_timeout_seconds: int = 30
    tts_timeout_seconds: int = 45
    
    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra environment variables
    }


settings = Settings()
