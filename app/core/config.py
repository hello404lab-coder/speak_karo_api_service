"""Application configuration using Pydantic settings."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str = ""  # Required for LLM, STT, TTS - set via OPENAI_API_KEY env var
    whisper_api_key: Optional[str] = None  # Can use OpenAI for Whisper
    tts_api_key: Optional[str] = None  # Can use OpenAI for TTS
    qubrid_api_key: Optional[str] = None  # Optional - set via QUBRID_API_KEY env var
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/english_practice"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Cloud Storage (Optional - S3)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None
    s3_bucket_name: Optional[str] = None
    
    # Application
    app_name: str = "AI English Practice Backend"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Audio Storage
    audio_storage_path: str = "./audio_storage"  # Local fallback
    audio_base_url: str = "http://localhost:8000/audio"  # For local serving
    
    # LLM Settings
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 150
    llm_temperature: float = 0.4
    llm_timeout: int = 30
    
    # STT Settings
    stt_provider: str = "qubrid"  # Options: "openai" or "qubrid"
    
    # TTS Settings
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"
    tts_speed: float = 0.9  # Slower speaking rate
    
    # Cache TTLs
    llm_cache_ttl: int = 86400  # 24 hours
    tts_cache_ttl: int = 604800  # 7 days
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra environment variables
    }


settings = Settings()
