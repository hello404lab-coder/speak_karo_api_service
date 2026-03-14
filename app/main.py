"""FastAPI application entry point."""
# Avoid "The current process just got forked, after parallelism has already been used" from tokenizers
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import HTTPException
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.database import init_db
from app.api.ai import router as ai_router
from app.api.auth import router as auth_router
from app.api.conversations import router as conversations_router
from app.api.subscription import router as subscription_router
from app.dependencies.auth import limiter

import torch
print("CUDA available inside app:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())


# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Return 402 body as top-level JSON (error + message) for subscription_required."""
    if exc.status_code == 402 and isinstance(exc.detail, dict):
        return JSONResponse(status_code=402, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for audio serving (local storage fallback)
os.makedirs(settings.audio_storage_path, exist_ok=True)
app.mount("/audio", StaticFiles(directory=settings.audio_storage_path), name="audio")

# Serve test page
@app.get("/test")
async def test_page():
    """Serve the voice chat test page."""
    backend_dir = Path(__file__).resolve().parents[1]  # .../backend
    file_path = backend_dir / "test_voice_chat.html"
    return FileResponse(str(file_path))


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting up AI English Practice Backend...")
    logger.info(f"APP_ENV={settings.app_env} (is_prod={settings.is_prod})")
    
    # Prod: require Gemini API key (fail fast)
    if settings.is_prod and not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required when APP_ENV=prod. Set it in .env or environment.")
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY not set. LLM features will not work. Set it in .env file.")
    else:
        logger.info("Gemini API key configured")
    
    # Log inference device (lazy; may import torch)
    try:
        from app.utils.device import get_infer_device
        logger.info(f"Infer device: {get_infer_device()}")
    except Exception as e:
        logger.warning(f"Could not resolve infer device: {e}")
    
    # Log STT configuration
    if not getattr(settings, "stt_whisper_local_enabled", True):
        model = getattr(settings, "stt_groq_model", "whisper-large-v3-turbo") or "whisper-large-v3-turbo"
        if getattr(settings, "groq_api_key", None):
            logger.info("STT: Groq Whisper API (model: %s)", model)
        else:
            logger.warning("STT: Groq Whisper API selected (model: %s) but GROQ_API_KEY not set; voice chat will fail until key is set", model)
    elif settings.stt_mode == "faster_whisper_large":
        logger.info("STT: faster_whisper_large (Systran/faster-whisper-large-v3, local)")
    elif settings.stt_mode == "openai_whisper_large_v3":
        logger.info("STT: openai_whisper_large_v3 (Hugging Face Transformers, openai/whisper-large-v3)")
    elif settings.stt_mode == "faster_whisper_medium":
        logger.info(f"STT: faster_whisper_medium (model: {settings.stt_faster_whisper_model_size}, CPU, int8)")
    else:
        logger.info(f"STT: {settings.stt_mode}")
    
    # Log LLM configuration
    if settings.gemini_api_key:
        logger.info(f"LLM: Gemini (model: {settings.llm_model})")
    else:
        logger.warning("Gemini API key not set. LLM will fail.")
    
    # Log TTS configuration (dual: English -> Chatterbox-Turbo or Gemini TTS, Indic -> IndicF5 or fallback)
    if getattr(settings, "tts_chatterbox_enabled", True):
        audio_prompt_info = f"voice cloning: {settings.tts_audio_prompt_path}" if settings.tts_audio_prompt_path else "not set (required for Turbo)"
        logger.info(f"TTS: English -> Chatterbox-Turbo ({audio_prompt_info})")
        indic_fallback = "Chatterbox-Turbo"
    else:
        logger.info("TTS: English -> Gemini TTS (model: %s, voice: %s)", getattr(settings, "tts_gemini_model", "gemini-2.5-flash-lite-preview-tts"), getattr(settings, "tts_gemini_voice", "Puck"))
        indic_fallback = "Gemini TTS"
    if getattr(settings, "tts_indicf5_enabled", False):
        indicf5_dir = getattr(settings, "tts_indicf5_ref_audio_dir", None)
        if indicf5_dir:
            logger.info(f"TTS: Indic (hi/ml/ta/...) -> IndicF5 (ref_audio_dir: {indicf5_dir}, speed: {getattr(settings, 'tts_indicf5_speed', 0.9)})")
        else:
            logger.info("TTS: Indic (hi/ml/ta/...) -> %s fallback (IndicF5 ref dir not set)", indic_fallback)
    else:
        logger.info("TTS: Indic (hi/ml/ta/...) -> %s only (IndicF5 disabled)", indic_fallback)
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    # Cache and Redis
    cache_enabled = settings.cache_enabled
    if cache_enabled:
        try:
            from app.services.cache import redis_available
            if redis_available:
                logger.info("Cache enabled (Redis available)")
            else:
                logger.warning("Cache enabled but Redis not available, continuing without cache")
        except Exception as e:
            logger.warning(f"Redis check failed: {e}")
    else:
        logger.info("Cache disabled (CACHE_ENABLED=false)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down...")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "service": settings.app_name
    }


# Include API routers
app.include_router(ai_router, prefix="/api/v1/ai", tags=["AI"])
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(conversations_router, prefix="/api/v1/conversations", tags=["Conversations"])
app.include_router(subscription_router, prefix="/api/v1/subscription", tags=["Subscription"])


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler - never expose stack traces."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."}
    )
