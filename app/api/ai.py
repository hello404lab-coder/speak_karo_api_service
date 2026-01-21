"""AI chat endpoints."""
import logging
import uuid
from datetime import date
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.ai import TextChatRequest, AIChatResponse
from app.services.llm import generate_reply
from app.services.stt import transcribe_audio
from app.services.tts import text_to_speech
from app.models.usage import Conversation, Message, Usage

logger = logging.getLogger(__name__)

router = APIRouter()


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
        
        # Generate AI reply
        ai_response = generate_reply(request.message, history)
        
        # Generate TTS audio - combine reply_text and correction if correction exists
        reply_text_for_tts = ai_response["reply_text"]
        correction_text = ai_response.get("correction", "").strip()
        
        if correction_text:
            # Combine reply and correction with a natural pause
            # Using period and ellipsis for a clear pause between reply and correction
            combined_text = f"{reply_text_for_tts}. ... {correction_text}"
        else:
            combined_text = reply_text_for_tts
        
        audio_url = text_to_speech(combined_text)
        
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
    except Exception as e:
        logger.error(f"Unexpected error in text_chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred processing your request.")


@router.post("/voice-chat", response_model=AIChatResponse)
async def voice_chat(
    user_id: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
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
        
        # Transcribe audio
        transcribed_text = transcribe_audio(audio_bytes, audio_file.filename)
        
        # Get or create conversation
        conversation = get_or_create_conversation(user_id, conversation_id, db)
        
        # Get conversation history
        history = get_conversation_history(conversation.id, db)
        
        # Generate AI reply (same as text chat)
        ai_response = generate_reply(transcribed_text, history)
        
        # Generate TTS audio - combine reply_text and correction if correction exists
        reply_text_for_tts = ai_response["reply_text"]
        correction_text = ai_response.get("correction", "").strip()
        
        if correction_text:
            # Combine reply and correction with a natural pause
            # Using period and ellipsis for a clear pause between reply and correction
            combined_text = f"{reply_text_for_tts}. ... {correction_text}"
        else:
            combined_text = reply_text_for_tts
        
        audio_url = text_to_speech(combined_text)
        
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
