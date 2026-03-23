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
    # When local: GPU Chatterbox-Turbo. When api: Resemble AI https://f.cluster.resemble.ai/stream (requires RESEMBLE_* keys)
    tts_chatterbox_mode: Literal["local", "api"] = Field(
        default="local",
        description="TTS_CHATTERBOX_MODE: local for GPU inference, api for Resemble AI cloud API",
    )
    resemble_api_key: Optional[str] = Field(
        default=None,
        description="RESEMBLE_API_KEY: API token from https://app.resemble.ai/account/api",
    )
    resemble_voice_uuid: Optional[str] = Field(
        default=None,
        description="RESEMBLE_VOICE_UUID: voice UUID for Resemble API synthesis",
    )
    resemble_api_model: str = Field(
        default="chatterbox-turbo",
        description="RESEMBLE_API_MODEL: model for Resemble API (chatterbox-turbo for lower latency)",
    )
    resemble_sample_rate: str = Field(
        default="44100",
        description="RESEMBLE_SAMPLE_RATE: audio sample rate for Resemble API",
    )
    resemble_precision: str = Field(
        default="PCM_16",
        description="RESEMBLE_PRECISION: audio precision for Resemble API (PCM_16, PCM_32, etc.)",
    )
    resemble_use_hd: bool = Field(
        default=False,
        description="RESEMBLE_USE_HD: enable HD synthesis (small latency trade-off)",
    )
    resemble_api_timeout: int = Field(
        default=30,
        description="RESEMBLE_API_TIMEOUT: HTTP timeout in seconds for Resemble API calls",
    )
    resemble_api_max_retries: int = Field(
        default=2,
        description="RESEMBLE_API_MAX_RETRIES: max retry attempts on transient failures",
    )
    tts_gemini_model: str = Field(default="gemini-2.5-flash-lite-preview-tts", description="TTS_GEMINI_MODEL: Gemini TTS model when Chatterbox disabled")
    tts_gemini_voice: str = Field(default="Puck", description="TTS_GEMINI_VOICE: prebuilt voice name for Gemini TTS")
    # Max concurrent TTS inferences (1 = strict serialization for low VRAM; 2+ = Semaphore for lower latency)
    tts_concurrent_inferences: int = Field(default=2, description="TTS_CONCURRENT_INFERENCES: max concurrent TTS inferences")
    # When True, force DummyWatermarker to skip loading watermark weights (patch must run before model instantiation)
    tts_use_dummy_watermarker: bool = Field(default=False, description="TTS_USE_DUMMY_WATERMARKER: force DummyWatermarker to skip loading watermark weights")

    # TTS Settings - Chatterbox-Turbo low-latency (optional; upstream + fork-friendly)
    tts_turbo_use_bfloat16: bool = Field(default=True, description="TTS_TURBO_USE_BFLOAT16: use bfloat16 for Turbo on CUDA (saves memory bandwidth)")
    tts_turbo_max_cache_len: Optional[int] = Field(default=550, description="TTS_TURBO_MAX_CACHE_LEN: KV cache length for Turbo (500-600 for latency; only used if model exposes it)")
    tts_turbo_temperature: float = Field(default=0.8, description="TTS_TURBO_TEMPERATURE: sampling temperature for Turbo generate()")
    tts_turbo_top_p: float = Field(default=0.95, description="TTS_TURBO_TOP_P: top-p for Turbo generate()")
    tts_turbo_top_k: int = Field(default=1000, description="TTS_TURBO_TOP_K: top-k for Turbo generate()")
    tts_turbo_repetition_penalty: float = Field(default=1.05, description="TTS_TURBO_REPETITION_PENALTY: repetition penalty for Turbo generate(); 1.05 reduces hesitations for low-latency")
    tts_turbo_exaggeration: float = Field(default=0.7, description="TTS_TURBO_EXAGGERATION: exaggeration for prepare_conditionals (0.7+ for faster pacing)")
    tts_turbo_use_streaming: bool = Field(default=True, description="TTS_TURBO_USE_STREAMING: use model.generate_stream when available (requires streaming-capable fork)")
    tts_turbo_stream_chunk_size: int = Field(default=25, description="TTS_TURBO_STREAM_CHUNK_SIZE: chunk size for generate_stream when used (smaller = lower TTFS)")
    tts_force_sdpa_attention: bool = Field(default=True, description="TTS_FORCE_SDPA_ATTENTION: force output_attentions=False on transformer forward to use SDPA (avoids manual attention fallback)")
    tts_turbo_cfg_weight: float = Field(default=0.3, description="TTS_TURBO_CFG_WEIGHT: CFG weight for generate (0.3 for faster pacing); used by non-Turbo ChatterboxTTS")
    tts_turbo_compile_t3: bool = Field(default=True, description="TTS_TURBO_COMPILE_T3: compile T3 with torch.compile for lower overhead; disable if it causes graph breaks")
    tts_turbo_max_gen_len: int = Field(default=400, description="TTS_TURBO_MAX_GEN_LEN: ceiling for T3 speech tokens per sentence; per-call uses min(400, max(100, len(text)*3)). Lower = less loop overhead.")
    tts_turbo_compile_s3gen: bool = Field(default=True, description="TTS_TURBO_COMPILE_S3GEN: compile S3 decoder with torch.compile; disable if it causes graph breaks")

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
