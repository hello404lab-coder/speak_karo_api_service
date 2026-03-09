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
    openai_api_key: Optional[str] = None  # Reserved; STT uses Groq when local Whisper disabled
    groq_api_key: Optional[str] = None  # For Groq Speech-to-Text when STT_WHISPER_LOCAL_ENABLED=false
    
    # Auth: JWT and OAuth
    jwt_secret: str = Field(default="change-me-in-production", description="JWT_SECRET: secret for signing tokens")
    jwt_access_token_expire_minutes: int = Field(default=15, description="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    jwt_refresh_token_expire_days: int = Field(default=30, description="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    google_client_id: Optional[str] = Field(default=None, description="GOOGLE_CLIENT_ID: for Google OAuth ID token verification")
    apple_client_id: Optional[str] = Field(default=None, description="APPLE_CLIENT_ID: for Apple OAuth ID token verification")
    
    # Database: PostgreSQL (postgresql://user:pass@host:5432/db) or SQLite (sqlite:///./data/english_practice.sqlite or sqlite:///:memory:)
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/english_practice",
        description="DATABASE_URL: PostgreSQL or SQLite connection URL",
    )
    
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
    llm_max_tokens: int = 200  # Increased for complete responses (Gemini 2.5 Flash supports up to 65,536)
    llm_temperature: float = 0.2
    # Context: max input tokens for system + history + current message (trimming drops oldest first)
    llm_context_token_budget: int = 16384
    # DB layer: max exchanges to load from conversation (actual context length controlled by token budget)
    llm_history_max_exchanges: int = 10
    
    # STT Settings
    stt_mode: Literal["faster_whisper_medium", "faster_whisper_large", "openai_whisper_large_v3"] = "openai_whisper_large_v3"  # Env: STT_MODE (used only when stt_whisper_local_enabled=True)
    stt_faster_whisper_model_size: str = "medium"  # Used for faster_whisper_medium
    # When False: local Whisper models are never loaded; transcription uses Groq Whisper API (requires GROQ_API_KEY)
    stt_whisper_local_enabled: bool = Field(default=False, description="STT_WHISPER_LOCAL_ENABLED: use local Whisper; if false, use Groq Whisper API")
    # Groq STT model when local disabled: whisper-large-v3-turbo (faster, cheaper) or whisper-large-v3 (higher accuracy)
    stt_groq_model: str = Field(default="whisper-large-v3-turbo", description="STT_GROQ_MODEL: Groq transcription model")
    # STT outputs raw transcription in the spoken language (no language hint passed; auto-detect).
    # Reserved for optional use: force transcription language (e.g. "en"). When set, could be passed to backends for non-raw mode.
    stt_force_language: Optional[str] = None  # Env: STT_FORCE_LANGUAGE
    
    # TTS Settings - Chatterbox-Turbo (English, https://huggingface.co/ResembleAI/chatterbox-turbo)
    # Device is auto-detected: cuda > mps > cpu. Requires a reference clip for voice cloning.
    tts_audio_prompt_path: Optional[str] = "chirp3-hd-sulafat.wav"  # Path to ~10s reference WAV for voice cloning (required for Turbo)

    # TTS Settings - IndicF5 (for Indic languages: hi, ml, ta)
    # When False: IndicF5 is never loaded; Indic TTS uses Gemini TTS or Chatterbox-Turbo fallback
    tts_indicf5_enabled: bool = Field(default=False, description="TTS_INDICF5_ENABLED: enable local IndicF5 for Indic languages")
    # Base directory containing ref WAVs (e.g. IndicF5/prompts or backend/assets/indicf5_prompts). Only used when tts_indicf5_enabled=True.
    tts_indicf5_ref_audio_dir: Optional[str] = None  # Set to path for ref WAVs
    tts_indicf5_speed: float = 0.9  # Speech speed (0.9 in IndicF5 main.py)

    # TTS Settings - Chatterbox toggle and Gemini TTS fallback
    # When False: Chatterbox is never loaded; English/Indic fallback use Gemini gemini-2.5-flash-lite-preview-tts
    tts_chatterbox_enabled: bool = Field(default=True, description="TTS_CHATTERBOX_ENABLED: enable local Chatterbox-Turbo (GPU)")
    tts_gemini_model: str = Field(default="gemini-2.5-flash-lite-preview-tts", description="TTS_GEMINI_MODEL: Gemini TTS model when Chatterbox disabled")
    tts_gemini_voice: str = Field(default="Puck", description="TTS_GEMINI_VOICE: prebuilt voice name for Gemini TTS")
    # Max concurrent TTS inferences (1 = strict serialization for low VRAM; 2+ = Semaphore for lower latency)
    tts_concurrent_inferences: int = Field(default=2, description="TTS_CONCURRENT_INFERENCES: max concurrent TTS inferences")
    # When True, force DummyWatermarker to skip loading watermark weights (patch must run before model instantiation)
    tts_use_dummy_watermarker: bool = Field(default=False, description="TTS_USE_DUMMY_WATERMARKER: force DummyWatermarker to skip loading watermark weights")

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
