"""FastAPI application entry point."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.database import init_db
from app.api.ai import router as ai_router

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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for audio serving (local storage fallback)
import os
os.makedirs(settings.audio_storage_path, exist_ok=True)
app.mount("/audio", StaticFiles(directory=settings.audio_storage_path), name="audio")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting up AI English Practice Backend...")
    
    # Check OpenAI API key
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not set. AI features will not work. Set it in .env file.")
    else:
        logger.info("OpenAI API key configured")
    
    # Log STT provider configuration
    logger.info(f"STT Provider configured: {settings.stt_provider}")
    if settings.stt_provider.lower() == "qubrid":
        if settings.qubrid_api_key:
            logger.info("Qubrid API key configured")
        else:
            logger.warning("Qubrid API key not set. STT will fail if Qubrid is selected.")
    
    # Log TTS provider configuration
    logger.info(f"TTS Provider configured: {settings.tts_provider}")
    if settings.tts_provider.lower() == "gemini":
        if settings.gemini_api_key:
            logger.info(f"Gemini API key configured (model: {settings.tts_gemini_model}, voice: {settings.tts_gemini_voice})")
        else:
            logger.warning("Gemini API key not set. TTS will fail if Gemini is selected.")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    # Test Redis connection
    try:
        from app.services.cache import redis_available
        if redis_available:
            logger.info("Redis cache available")
        else:
            logger.warning("Redis cache not available, continuing without cache")
    except Exception as e:
        logger.warning(f"Redis check failed: {e}")


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


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler - never expose stack traces."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."}
    )
