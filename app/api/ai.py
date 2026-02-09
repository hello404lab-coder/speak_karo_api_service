"""AI chat endpoints. Sync inference (LLM, STT, TTS) runs in thread pool with timeouts."""
import asyncio
import logging
import uuid
from datetime import date
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.core.config import settings
from app.database import get_db
from app.schemas.ai import TextChatRequest, AIChatResponse
from app.services.llm import generate_reply, init_llm_client
from app.services.stt import transcribe_audio, init_stt_models
from app.services.tts import text_to_speech, init_tts_models
from app.utils.language import get_response_language
from app.models.usage import Conversation, Message, Usage

logger = logging.getLogger(__name__)

# User-safe message for timeout (no stack traces or internal detail)
TIMEOUT_MESSAGE = "Request took too long. Please try again."

router = APIRouter()

# Max time for init-models (first-time load can be slow)
INIT_MODELS_TIMEOUT_SECONDS = 30000


def get_or_create_conversation(user_id: str, conversation_id: Optional[str], db: Session) -> Conversation:
    """Get existing conversation or create new one."""
    if conversation_id:
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conversation:
            return conversation
    
    # Create new conversation
    new_id = str(uuid.uuid4())
    conversation = Conversation(id=new_id, user_id=user_id)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def get_conversation_history(conversation_id: str, db: Session) -> list:
    """Get last 5 messages from conversation for context."""
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.desc()).limit(5).all()
    
    # Reverse to get chronological order
    messages.reverse()
    
    history = []
    for msg in messages:
        history.append({
            "role": "user",
            "content": msg.user_message
        })
        history.append({
            "role": "assistant",
            "content": msg.ai_reply
        })
    
    return history


def update_usage_stats(user_id: str, db: Session, duration_seconds: float = 0.0):
    """Update daily usage statistics."""
    today = date.today()
    
    usage = db.query(Usage).filter(
        Usage.user_id == user_id,
        Usage.date == today
    ).first()
    
    if not usage:
        usage = Usage(user_id=user_id, date=today, minutes_used=0.0, request_count=0)
        db.add(usage)
    
    usage.request_count += 1
    usage.minutes_used += duration_seconds / 60.0
    
    db.commit()


def _run_init_models_sync() -> dict:
    """Run all model initializers sequentially (called from thread)."""
    stt = init_stt_models()
    llm = init_llm_client()
    tts = init_tts_models()
    return {"stt": stt, "llm": llm, "tts": tts}


@router.post("/init-models")
async def init_models():
    """
    Initialize (warm up) all models: STT, LLM client, and TTS (Turbo + IndicF5 if configured).
    Call this after startup to avoid cold-start latency on first user request.
    Runs in a thread with a 5-minute timeout.
    """
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_run_init_models_sync),
            timeout=float(INIT_MODELS_TIMEOUT_SECONDS),
        )
        return result
    except asyncio.TimeoutError:
        logger.warning("init-models timed out")
        raise HTTPException(
            status_code=504,
            detail="Model initialization timed out. Try again or check server logs.",
        )


@router.post("/text-chat", response_model=AIChatResponse)
async def text_chat(
    request: TextChatRequest,
    db: Session = Depends(get_db)
):
    """
    Text chat endpoint - accepts text message and returns AI reply.
    """
    try:
        # Get or create conversation
        conversation = get_or_create_conversation(request.user_id, request.conversation_id, db)
        
        # Get conversation history for context
        history = get_conversation_history(conversation.id, db)

        # Response language from text (script detection; no STT)
        response_language = get_response_language(request.message, None)

        # Run sync inference in thread pool with timeouts so event loop is not blocked
        try:
            ai_response = await asyncio.wait_for(
                asyncio.to_thread(generate_reply, request.message, history, response_language),
                timeout=float(settings.llm_timeout_seconds),
            )
        except asyncio.TimeoutError:
            logger.warning("LLM request timed out")
            raise HTTPException(status_code=504, detail=TIMEOUT_MESSAGE)

        # Generate TTS audio - combine reply_text and correction if correction exists
        reply_text_for_tts = ai_response["reply_text"]
        correction_text = ai_response.get("correction", "").strip()
        hinglish_explanation_text = ai_response.get("hinglish_explanation", "").strip()

        if correction_text:
            combined_text = f"{reply_text_for_tts}. ... {hinglish_explanation_text}"
        else:
            combined_text = reply_text_for_tts

        try:
            audio_url = await asyncio.wait_for(
                asyncio.to_thread(text_to_speech, combined_text, response_language),
                timeout=float(settings.tts_timeout_seconds),
            )
        except asyncio.TimeoutError:
            logger.warning("TTS request timed out")
            raise HTTPException(status_code=504, detail=TIMEOUT_MESSAGE)
        
        # Save message to database
        message = Message(
            conversation_id=conversation.id,
            user_message=request.message,
            ai_reply=ai_response["reply_text"],
            correction=ai_response.get("correction", ""),
            hinglish_explanation=ai_response.get("hinglish_explanation", ""),
            score=ai_response.get("score", 0)
        )
        db.add(message)
        db.commit()
        
        # Update usage stats
        update_usage_stats(request.user_id, db)
        
        return AIChatResponse(
            reply_text=ai_response["reply_text"],
            correction=ai_response.get("correction", ""),
            hinglish_explanation=ai_response.get("hinglish_explanation", ""),
            score=ai_response.get("score", 75),
            audio_url=audio_url,
            conversation_id=conversation.id
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in text_chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred processing your request.")


@router.post("/voice-chat", response_model=AIChatResponse)
async def voice_chat(
    user_id: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    audio_file: UploadFile = File(...),
    stt_mode: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Voice chat endpoint - accepts audio file and returns AI reply.
    """
    try:
        # Validate file type
        if not audio_file.filename.endswith('.wav'):
            raise HTTPException(status_code=400, detail="Only WAV files are supported.")
        
        # Read audio file
        audio_bytes = await audio_file.read()

        effective_stt_mode = stt_mode if stt_mode in ("faster_whisper_medium", "faster_whisper_large") else settings.stt_mode
        try:
            transcribed_text, detected_lang = await asyncio.wait_for(
                asyncio.to_thread(
                    transcribe_audio,
                    audio_bytes,
                    audio_file.filename,
                    effective_stt_mode,
                ),
                timeout=float(settings.stt_timeout_seconds),
            )
        except asyncio.TimeoutError:
            logger.warning("STT request timed out")
            raise HTTPException(status_code=504, detail=TIMEOUT_MESSAGE)

        # Get or create conversation
        conversation = get_or_create_conversation(user_id, conversation_id, db)

        # Get conversation history
        history = get_conversation_history(conversation.id, db)

        # Response language from STT + script detection (en -> Chatterbox, hi/ml/ta -> IndicF5)
        response_language = get_response_language(transcribed_text, detected_lang)

        try:
            ai_response = await asyncio.wait_for(
                asyncio.to_thread(generate_reply, transcribed_text, history, response_language),
                timeout=float(settings.llm_timeout_seconds),
            )
        except asyncio.TimeoutError:
            logger.warning("LLM request timed out")
            raise HTTPException(status_code=504, detail=TIMEOUT_MESSAGE)

        # Generate TTS audio - combine reply_text and correction if correction exists
        reply_text_for_tts = ai_response["reply_text"]
        correction_text = ai_response.get("correction", "").strip()
        hinglish_explanation_text = ai_response.get("hinglish_explanation", "").strip()

        if correction_text:
            combined_text = f"{reply_text_for_tts}. ... {hinglish_explanation_text}"
        else:
            combined_text = reply_text_for_tts

        try:
            audio_url = await asyncio.wait_for(
                asyncio.to_thread(text_to_speech, combined_text, response_language),
                timeout=float(settings.tts_timeout_seconds),
            )
        except asyncio.TimeoutError:
            logger.warning("TTS request timed out")
            raise HTTPException(status_code=504, detail=TIMEOUT_MESSAGE)

        # Save message to database
        message = Message(
            conversation_id=conversation.id,
            user_message=transcribed_text,
            ai_reply=ai_response["reply_text"],
            correction=ai_response.get("correction", ""),
            hinglish_explanation=ai_response.get("hinglish_explanation", ""),
            score=ai_response.get("score", 0)
        )
        db.add(message)
        db.commit()
        
        # Update usage stats (estimate duration from audio)
        # Rough estimate: 1 second per 10 bytes (very rough)
        estimated_duration = len(audio_bytes) / 10000.0
        update_usage_stats(user_id, db, estimated_duration)
        
        return AIChatResponse(
            reply_text=ai_response["reply_text"],
            correction=ai_response.get("correction", ""),
            hinglish_explanation=ai_response.get("hinglish_explanation", ""),
            score=ai_response.get("score", 75),
            audio_url=audio_url,
            conversation_id=conversation.id
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in voice_chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred processing your request.")
