"""Application configuration using Pydantic settings."""
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Literal, Optional

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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
    admin_bootstrap_email: Optional[str] = Field(
        default=None,
        description="ADMIN_BOOTSTRAP_EMAIL: email for the bootstrap admin account",
    )
    admin_bootstrap_password: Optional[str] = Field(
        default=None,
        description="ADMIN_BOOTSTRAP_PASSWORD: password for the bootstrap admin account",
    )

    # Billing: Razorpay Subscriptions
    razorpay_key_id: Optional[str] = Field(default=None, description="RAZORPAY_KEY_ID")
    razorpay_key_secret: Optional[str] = Field(default=None, description="RAZORPAY_KEY_SECRET")
    razorpay_webhook_secret: Optional[str] = Field(default=None, description="RAZORPAY_WEBHOOK_SECRET")
    razorpay_webhook_secret_previous: Optional[str] = Field(
        default=None,
        description="RAZORPAY_WEBHOOK_SECRET_PREVIOUS: previous secret accepted for webhook retries during rotation",
    )
    razorpay_plan_id_vuvl_plus_test: Optional[str] = Field(
        default=None,
        description="RAZORPAY_PLAN_ID_VUVL_PLUS_TEST",
    )
    razorpay_plan_id_vuvl_plus_live: Optional[str] = Field(
        default=None,
        description="RAZORPAY_PLAN_ID_VUVL_PLUS_LIVE",
    )
    razorpay_plan_id_vuvl_pro_test: Optional[str] = Field(
        default=None,
        description="RAZORPAY_PLAN_ID_VUVL_PRO_TEST",
    )
    razorpay_plan_id_vuvl_pro_live: Optional[str] = Field(
        default=None,
        description="RAZORPAY_PLAN_ID_VUVL_PRO_LIVE",
    )
    razorpay_timeout_seconds: int = Field(default=15, description="RAZORPAY_TIMEOUT_SECONDS")
    razorpay_monthly_total_count: int = Field(
        default=1200,
        description="RAZORPAY_MONTHLY_TOTAL_COUNT: total recurring billing cycles to model long-running monthly plans",
    )
    razorpay_checkout_reuse_minutes: int = Field(
        default=30,
        description="RAZORPAY_CHECKOUT_REUSE_MINUTES: reuse a recently-created pending checkout subscription instead of creating duplicates",
    )
    
    # Database: PostgreSQL only (sync psycopg2 driver).
    # Format: postgresql+psycopg2://user:pass@host:5432/db
    database_url: str = Field(
        default="postgresql+psycopg2://user:password@localhost:5432/english_practice",
        description="DATABASE_URL: PostgreSQL sync connection URL (postgresql+psycopg2://...)",
    )
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Agora RTC (social voice matchmaking; tokens minted server-side)
    agora_app_id: Optional[str] = Field(default=None, description="AGORA_APP_ID")
    agora_app_certificate: Optional[str] = Field(default=None, description="AGORA_APP_CERTIFICATE")
    agora_token_ttl_seconds: int = Field(default=3600, description="AGORA_TOKEN_TTL_SECONDS: RTC token lifetime")

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
    # CORS: allow_origins=* with allow_credentials=True is invalid in browsers; use explicit origins.
    cors_allowed_origins: str = Field(
        default="https://luna.404lab.tech,https://vuvl.in,http://localhost:8080",
        description="CORS_ALLOW_ORIGINS: comma-separated browser origins. Required when APP_ENV=prod.",
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="CORS_ALLOW_CREDENTIALS: set false if the client does not use credentialed cross-origin requests",
    )
    
    # Audio Storage
    # IMPORTANT: Keep this outside `backend/` so `uvicorn --reload` doesn't restart
    # every time we write a new MP3 (which looks like the test page “reloads”).
    audio_storage_path: str = str(Path(__file__).resolve().parents[3] / "audio_storage")
    audio_base_url: str = "http://localhost:8000/audio"  # For local serving
    
    # LLM Settings
    llm_model: str = "gemini-2.5-flash-lite"  # Gemini model for LLM (fast and efficient) gemini-2.5-flash-lite, gemini-2.5-flash
    llm_max_tokens: int = 200  # Increased for complete responses (Gemini 2.5 Flash supports up to 65,536)
    llm_temperature: float = 0.2
    # Context: max input tokens for system + history + current message (trimming drops oldest first)
    llm_context_token_budget: int = 16384
    # DB layer: max exchanges to load from conversation (actual context length controlled by token budget)
    llm_history_max_exchanges: int = 10
    gemini_service_tier: Literal["standard", "flex", "priority"] = Field(
        default="priority",
        description="GEMINI_SERVICE_TIER: GenerateContent service_tier (priority = Google Priority inference; needs eligible billing tier)",
    )

    # Translation Settings - Google Cloud Translation Advanced v3
    translation_provider: Literal["google_cloud", "disabled"] = Field(
        default="google_cloud",
        description="TRANSLATION_PROVIDER: assistant reply translation backend",
    )
    translation_api_key: Optional[str] = Field(
        default=None,
        description="TRANSLATION_API_KEY: Google Cloud Translation API key for Basic v2 text translation",
    )
    translation_google_project_id: Optional[str] = Field(
        default=None,
        description="TRANSLATION_GOOGLE_PROJECT_ID: optional GCP project id override for Cloud Translation",
    )
    translation_google_location: str = Field(
        default="global",
        description="TRANSLATION_GOOGLE_LOCATION: Cloud Translation location (global unless using region-specific resources)",
    )
    translation_timeout_seconds: int = Field(
        default=10,
        description="TRANSLATION_TIMEOUT_SECONDS: HTTP timeout in seconds for Cloud Translation",
    )
    translation_cache_ttl: int = Field(
        default=604800,
        description="TRANSLATION_CACHE_TTL: cache TTL for assistant reply translations",
    )
    
    # STT Settings
    stt_mode: Literal["faster_whisper_medium", "faster_whisper_large", "openai_whisper_large_v3"] = "faster_whisper_large"  # Env: STT_MODE (used only when stt_whisper_local_enabled=True)
    stt_faster_whisper_model_size: str = "medium"  # Used for faster_whisper_medium
    # When False: local Whisper models are never loaded; transcription uses Groq Whisper API (requires GROQ_API_KEY)
    stt_whisper_local_enabled: bool = Field(default=False, description="STT_WHISPER_LOCAL_ENABLED: use local Whisper; if false, use Groq Whisper API")
    # Groq STT model when local disabled: whisper-large-v3-turbo (faster, cheaper) or whisper-large-v3 (higher accuracy)
    stt_groq_model: str = Field(default="whisper-large-v3", description="STT_GROQ_MODEL: Groq transcription model")
    # STT outputs raw transcription in the spoken language (no language hint passed; auto-detect).
    # Reserved for optional use: force transcription language (e.g. "en"). When set, could be passed to backends for non-raw mode.
    stt_force_language: Optional[str] = None  # Env: STT_FORCE_LANGUAGE
    
    # TTS Settings - Chatterbox-Turbo (English, https://huggingface.co/ResembleAI/chatterbox-turbo)
    # Device is auto-detected: cuda > mps > cpu. Requires a reference clip for voice cloning.
    tts_audio_prompt_path: Optional[str] = "chirp3-hd-sulafat.wav"  # Path to ~10s reference WAV for voice cloning (required for Turbo)

    # TTS Settings - IndicF5 (for Indic languages: hi, ml, ta)
    # When False: IndicF5 is never loaded; Indic TTS uses Gemini TTS or Chatterbox-Turbo fallback
    tts_indicf5_enabled: bool = Field(default=True, description="TTS_INDICF5_ENABLED: enable local IndicF5 for Indic languages")
    # Base directory containing ref WAVs (e.g. IndicF5/prompts or backend/assets/indicf5_prompts). Only used when tts_indicf5_enabled=True.
    tts_indicf5_ref_audio_dir: Optional[str] = 'IndicF5/prompts'  # Set to path for ref WAVs
    tts_indicf5_speed: float = 0.9  # Speech speed (0.9 in IndicF5 main.py)
    # When True, IndicF5 always uses CPU. When False, still uses CPU on MPS (no ComplexFloat); CUDA uses GPU.
    tts_indicf5_force_cpu: bool = Field(
        default=False,
        description="TTS_INDICF5_FORCE_CPU: force IndicF5 on CPU (e.g. to free VRAM on CUDA)",
    )

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

    tts_gemini_model: str = Field(default="gemini-2.5-flash-lite-preview-tts", description="TTS_GEMINI_MODEL: Gemini TTS model when Chatterbox disabled (English)")
    tts_gemini_model_indic: str = Field(
        default="gemini-2.5-flash-preview-tts",
        description="TTS_GEMINI_MODEL_INDIC: Gemini TTS model for Indic languages (when IndicF5 off or after IndicF5 failure)",
    )
    tts_gemini_voice: str = Field(default="Puck", description="TTS_GEMINI_VOICE: prebuilt voice name for Gemini TTS")
    tts_cloud_provider: Literal["gemini", "chirp3_hd", "smallest"] = Field(
        default="gemini",
        description="TTS_CLOUD_PROVIDER: cloud TTS backend when local providers are unavailable or disabled",
    )
    tts_chirp_region: str = Field(
        default="global",
        description="TTS_CHIRP_REGION: Cloud TTS Chirp region hint (global, us, eu, asia-southeast1, ...)",
    )
    tts_chirp_endpoint: Optional[str] = Field(
        default=None,
        description="TTS_CHIRP_ENDPOINT: optional explicit Cloud TTS API endpoint override",
    )
    tts_chirp_voice: str = Field(
        default="Sulafat",
        description="TTS_CHIRP_VOICE: Chirp 3 HD voice suffix or full voice name override",
    )
    tts_chirp_speaking_rate: float = Field(
        default=1.0,
        description="TTS_CHIRP_SPEAKING_RATE: speaking rate for Chirp 3 HD synthesis",
    )
    tts_chirp_sample_rate_hz: int = Field(
        default=44100,
        description="TTS_CHIRP_SAMPLE_RATE_HZ: sample rate for Chirp 3 HD audio",
    )
    tts_chirp_stream_encoding: Literal["pcm"] = Field(
        default="pcm",
        description="TTS_CHIRP_STREAM_ENCODING: streaming audio encoding for Chirp 3 HD (currently pcm only)",
    )
    tts_chirp_timeout_seconds: int = Field(
        default=30,
        description="TTS_CHIRP_TIMEOUT_SECONDS: timeout in seconds for Chirp 3 HD streaming and unary requests",
    )

    # Smallest.ai Waves (Lightning TTS) — https://docs.smallest.ai
    smallest_api_key: Optional[str] = Field(
        default=None,
        description="SMALLEST_API_KEY: Smallest.ai Waves API key",
    )
    tts_smallest_model: str = Field(
        default="lightning-v3.1",
        description="TTS_SMALLEST_MODEL: model id (path segment, e.g. lightning-v3.1)",
    )
    tts_smallest_voice: str = Field(
        default="magnus",
        description="TTS_SMALLEST_VOICE: default Smallest voice_id",
    )
    tts_smallest_voice_per_lang: Optional[str] = Field(
        default=None,
        description="TTS_SMALLEST_VOICE_PER_LANG: optional comma-separated lang:voice (e.g. ta:magnus,hi:xyz); Malayalam is not routed to Smallest",
    )
    tts_smallest_sample_rate_hz: int = Field(
        default=24000,
        description="TTS_SMALLEST_SAMPLE_RATE_HZ: sample rate for Smallest TTS (8000–44100 per API)",
    )
    tts_smallest_speed: float = Field(
        default=1.0,
        description="TTS_SMALLEST_SPEED: speech speed 0.5–2.0",
    )
    tts_smallest_output_format: Literal["wav", "mp3", "pcm", "mulaw"] = Field(
        default="wav",
        description="TTS_SMALLEST_OUTPUT_FORMAT: output format for unary /get_speech",
    )
    tts_smallest_streaming_enabled: bool = Field(
        default=True,
        description="TTS_SMALLEST_STREAMING_ENABLED: use SSE /stream in LLM voice pipeline when provider is Smallest",
    )
    tts_smallest_timeout_seconds: int = Field(
        default=30,
        description="TTS_SMALLEST_TIMEOUT_SECONDS: HTTP timeout for Smallest TTS",
    )
    tts_smallest_base_url: str = Field(
        default="https://api.smallest.ai/waves/v1",
        description="TTS_SMALLEST_BASE_URL: Smallest Waves API base URL",
    )

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
    voice_draft_ttl_hours: int = Field(
        default=24,
        description="VOICE_DRAFT_TTL_HOURS: pending voice transcript drafts expire after this many hours",
    )

    # Gemini Live (control plane only; client opens WebSocket to Google)
    gemini_live_model: str = Field(
        default="gemini-2.5-flash-native-audio-preview-12-2025",
        description="GEMINI_LIVE_MODEL: Live-capable Gemini model id",
    )
    gemini_live_prompt_version: str = Field(
        default="1",
        description="GEMINI_LIVE_PROMPT_VERSION: server-side prompt template version",
    )
    gemini_live_voice: Optional[str] = Field(
        default="Puck",
        description="GEMINI_LIVE_VOICE: prebuilt voice name for Live speech_config",
    )
    gemini_live_language_code: Optional[str] = Field(
        default="en-US",
        description="GEMINI_LIVE_LANGUAGE_CODE: BCP-47 hint for clients (not sent in Live speech_config; API unsupported)",
    )
    gemini_live_temperature: float = Field(
        default=0.4,
        description="GEMINI_LIVE_TEMPERATURE: Live generation temperature",
    )
    gemini_live_min_plan: Literal["free", "trial", "vuvl_plus", "vuvl_pro"] = Field(
        default="vuvl_pro",
        description="GEMINI_LIVE_MIN_PLAN: minimum subscription tier for Live (free|trial|vuvl_plus|vuvl_pro)",
    )
    gemini_live_max_session_duration_seconds: int = Field(
        default=3600,
        description="GEMINI_LIVE_MAX_SESSION_DURATION_SECONDS: cap for server-computed session length on end",
    )
    gemini_live_system_instruction_max_chars: int = Field(
        default=8000,
        description="GEMINI_LIVE_SYSTEM_INSTRUCTION_MAX_CHARS: max length for built system_instruction",
    )
    gemini_live_token_uses: int = Field(
        default=1,
        description="GEMINI_LIVE_TOKEN_USES: uses count for ephemeral Live auth token (0 = unlimited per SDK)",
    )
    gemini_live_token_new_session_seconds: int = Field(
        default=120,
        description="GEMINI_LIVE_TOKEN_NEW_SESSION_SECONDS: window in which new Live sessions may start with the token",
    )
    gemini_live_heartbeat_stale_seconds: int = Field(
        default=90,
        description="GEMINI_LIVE_HEARTBEAT_STALE_SECONDS: no heartbeat => session eligible for auto-end",
    )
    gemini_live_reaper_enabled: bool = Field(
        default=True,
        description="GEMINI_LIVE_REAPER_ENABLED: periodic job to auto-end stale active sessions",
    )
    gemini_live_reaper_interval_seconds: int = Field(
        default=60,
        description="GEMINI_LIVE_REAPER_INTERVAL_SECONDS: sleep between reaper runs in the app process",
    )
    gemini_live_token_requests_per_minute: int = Field(
        default=3,
        description="GEMINI_LIVE_TOKEN_REQUESTS_PER_MINUTE: max /live/token mints per user per rolling minute (Redis)",
    )
    gemini_live_token_rate_window_seconds: int = Field(
        default=60,
        description="GEMINI_LIVE_TOKEN_RATE_WINDOW_SECONDS: Redis TTL for token rate counter",
    )
    gemini_live_redis_required_for_token: bool = Field(
        default=True,
        description="GEMINI_LIVE_REDIS_REQUIRED_FOR_TOKEN: if true, prod requires Redis for /live/token rate limit",
    )
    
    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"

    @property
    def cors_origins(self) -> list[str]:
        """Allowed Origin values for CORSMiddleware: explicit env, else dev-friendly localhost defaults."""
        raw = (self.cors_allowed_origins or "").strip()
        if raw:
            return [o.rstrip("/") for o in (x.strip() for x in raw.split(",")) if o.strip()]
        if self.is_prod:
            return []
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]

    @property
    def razorpay_live_mode(self) -> bool:
        """Choose live vs test plan ids from APP_ENV."""
        return self.is_prod

    @property
    def razorpay_plan_id_map(self) -> dict[str, str | None]:
        """Return backend-supported Razorpay plan ids keyed by internal plan code."""
        if self.razorpay_live_mode:
            return {
                "vuvl_plus": self.razorpay_plan_id_vuvl_plus_live,
                "vuvl_pro": self.razorpay_plan_id_vuvl_pro_live,
            }
        return {
            "vuvl_plus": self.razorpay_plan_id_vuvl_plus_test,
            "vuvl_pro": self.razorpay_plan_id_vuvl_pro_test,
        }

    def razorpay_plan_id_for(self, plan_code: str) -> str | None:
        """Return configured Razorpay plan id for one internal paid plan."""
        return self.razorpay_plan_id_map.get(plan_code)

    @property
    def razorpay_enabled(self) -> bool:
        """Return True when billing secrets and both paid plan ids are configured."""
        plan_ids = self.razorpay_plan_id_map
        return bool(
            self.razorpay_key_id
            and self.razorpay_key_secret
            and plan_ids.get("vuvl_plus")
            and plan_ids.get("vuvl_pro")
        )
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra environment variables
    }


settings = Settings()
