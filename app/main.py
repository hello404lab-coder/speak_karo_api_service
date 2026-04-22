"""FastAPI application entry point."""
# Avoid "The current process just got forked, after parallelism has already been used" from tokenizers
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio
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
from app.api.live import router as live_router
from app.api.auth import router as auth_router
from app.api.conversations import router as conversations_router
from app.api.subscription import router as subscription_router
from app.api.social import router as social_router
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
    """Return structured JSON for subscription (402), rate limits (429), and Live session errors (410)."""
    if isinstance(exc.detail, dict) and exc.status_code in (402, 410, 429):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
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
    
    cloud_tts_provider = getattr(settings, "tts_cloud_provider", "gemini")
    chirp_voice = getattr(settings, "tts_chirp_voice", "Charon")
    chirp_status = {"available": False, "reason": None}
    if cloud_tts_provider == "chirp3_hd":
        try:
            from app.services.tts import chirp_runtime_status

            chirp_status = chirp_runtime_status()
        except Exception as e:
            chirp_status = {"available": False, "reason": str(e)}
    # Log TTS configuration (English -> Chatterbox-Turbo or configured cloud provider; Indic -> IndicF5 optional then configured cloud provider)
    if getattr(settings, "tts_chatterbox_enabled", True):
        audio_prompt_info = f"voice cloning: {settings.tts_audio_prompt_path}" if settings.tts_audio_prompt_path else "not set (required for Turbo)"
        logger.info(f"TTS: English -> Chatterbox-Turbo ({audio_prompt_info})")
    else:
        if cloud_tts_provider == "chirp3_hd":
            if chirp_status["available"]:
                logger.info(
                    "TTS: English -> Chirp 3 HD (voice: %s, region: %s, sample_rate_hz: %s)",
                    chirp_voice,
                    getattr(settings, "tts_chirp_region", "global"),
                    getattr(settings, "tts_chirp_sample_rate_hz", 24000),
                )
            else:
                logger.warning(
                    "TTS: English Chirp 3 HD configured but unavailable (%s); requests will fall back to Gemini",
                    chirp_status["reason"] or "unknown reason",
                )
        else:
            logger.info(
                "TTS: English -> Gemini TTS (model: %s, voice: %s)",
                getattr(settings, "tts_gemini_model", "gemini-2.5-flash-lite-preview-tts"),
                getattr(settings, "tts_gemini_voice", "Puck"),
            )
    indic_gemini = getattr(settings, "tts_gemini_model_indic", "gemini-2.5-flash-preview-tts")
    if getattr(settings, "tts_indicf5_enabled", False):
        indicf5_dir = getattr(settings, "tts_indicf5_ref_audio_dir", None)
        if indicf5_dir:
            from app.services.tts import _get_indicf5_torch_device

            logger.info(
                "TTS: Indic (hi/ml/ta/...) -> IndicF5 (ref_audio_dir: %s, speed: %s, torch: %s); fallback Gemini Indic (model: %s)",
                indicf5_dir,
                getattr(settings, "tts_indicf5_speed", 0.9),
                _get_indicf5_torch_device(),
                indic_gemini,
            )
        else:
            logger.info(
                "TTS: Indic (hi/ml/ta/...) -> configured cloud provider (%s); IndicF5 ref dir not set",
                cloud_tts_provider,
            )
    else:
        if cloud_tts_provider == "chirp3_hd":
            if chirp_status["available"]:
                logger.info(
                    "TTS: Indic (hi/ml/ta/...) -> Chirp 3 HD (voice: %s, region: %s)",
                    chirp_voice,
                    getattr(settings, "tts_chirp_region", "global"),
                )
            else:
                logger.warning(
                    "TTS: Indic Chirp 3 HD configured but unavailable (%s); requests will fall back to Gemini",
                    chirp_status["reason"] or "unknown reason",
                )
        else:
            logger.info(
                "TTS: Indic (hi/ml/ta/...) -> Gemini Indic (model: %s; requires GEMINI_API_KEY)",
                indic_gemini,
            )
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    try:
        from app.database import SessionLocal
        from app.services.voice_drafts import cleanup_expired_voice_drafts

        def _cleanup_voice_drafts_once() -> int:
            db = SessionLocal()
            try:
                return cleanup_expired_voice_drafts(db)
            finally:
                db.close()

        cleaned = await asyncio.to_thread(_cleanup_voice_drafts_once)
        if cleaned:
            logger.info("Voice draft startup cleanup expired %s draft(s)", cleaned)
    except Exception as e:
        logger.warning("Voice draft startup cleanup failed: %s", e)
    
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

    # Async Redis for social voice matchmaking (separate from sync cache client)
    try:
        from app.services.redis_social import create_social_redis

        app.state.social_redis = create_social_redis()
        await app.state.social_redis.ping()
        logger.info("Social matchmaking Redis connected")
    except Exception as e:
        logger.warning("Social matchmaking Redis unavailable: %s", e)
        app.state.social_redis = None

    # Async Redis for Gemini Live token rate limiting
    try:
        from app.services.redis_live import create_live_redis

        app.state.live_redis = create_live_redis()
        await app.state.live_redis.ping()
        logger.info("Live Redis connected (token rate limit)")
    except Exception as e:
        logger.warning("Live Redis unavailable: %s", e)
        app.state.live_redis = None

    app.state._live_reaper_stop = asyncio.Event()
    app.state.reaper_task = None
    if getattr(settings, "gemini_live_reaper_enabled", True):
        reaper_interval = max(15.0, float(getattr(settings, "gemini_live_reaper_interval_seconds", 60) or 60))
        reaper_stale_seconds = int(getattr(settings, "gemini_live_heartbeat_stale_seconds", 90) or 90)

        async def _live_reaper_worker() -> None:
            while True:
                try:
                    await asyncio.wait_for(app.state._live_reaper_stop.wait(), timeout=reaper_interval)
                    break
                except asyncio.TimeoutError:
                    pass

                def _reap_once() -> int:
                    from app.database import SessionLocal
                    from app.services.live_session_service import reap_stale_live_sessions

                    db = SessionLocal()
                    try:
                        return reap_stale_live_sessions(db, stale_seconds=reaper_stale_seconds)
                    finally:
                        db.close()

                try:
                    n = await asyncio.to_thread(_reap_once)
                    if n:
                        logger.info("live_reaper closed %s stale session(s)", n)
                except Exception as ex:
                    logger.warning("live_reaper error: %s", ex)

        app.state.reaper_task = asyncio.create_task(_live_reaper_worker())
        logger.info(
            "Live session reaper started (interval=%ss, stale=%ss)",
            reaper_interval,
            reaper_stale_seconds,
        )

    if settings.is_prod and (not settings.agora_app_id or not settings.agora_app_certificate):
        logger.warning(
            "AGORA_APP_ID / AGORA_APP_CERTIFICATE not set; social voice tokens will fail until configured."
        )


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down...")
    evt = getattr(app.state, "_live_reaper_stop", None)
    if evt is not None and not evt.is_set():
        evt.set()
    task = getattr(app.state, "reaper_task", None)
    if task is not None:
        try:
            await asyncio.wait_for(task, timeout=8.0)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning("Live reaper task shutdown: %s", e)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    r = getattr(app.state, "social_redis", None)
    if r is not None:
        try:
            await r.aclose()
        except Exception as e:
            logger.warning("Social Redis close failed: %s", e)

    lr = getattr(app.state, "live_redis", None)
    if lr is not None:
        try:
            await lr.aclose()
        except Exception as e:
            logger.warning("Live Redis close failed: %s", e)


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
app.include_router(live_router, prefix="/api/v1/ai")
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(conversations_router, prefix="/api/v1/conversations", tags=["Conversations"])
app.include_router(subscription_router, prefix="/api/v1/subscription", tags=["Subscription"])
app.include_router(social_router, prefix="/api/v1/social", tags=["Social"])


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler - never expose stack traces."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."}
    )
