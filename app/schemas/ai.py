"""Request and response schemas for AI endpoints."""
from pydantic import BaseModel, Field
from typing import Optional


class TextChatRequest(BaseModel):
    """Request schema for text chat endpoint."""
    user_id: str = Field(..., description="User identifier")
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")


class VoiceChatRequest(BaseModel):
    """Request schema for voice chat endpoint."""
    user_id: str = Field(..., description="User identifier")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")


class AIChatResponse(BaseModel):
    """Response schema for AI chat endpoints."""
    reply_text: str = Field(..., description="AI's natural reply")
    correction: str = Field(..., description="One correction (if any)")
    hinglish_explanation: str = Field(..., description="Explanation in Hinglish")
    score: int = Field(..., ge=0, le=100, description="Score out of 100")
    audio_url: Optional[str] = Field(
        None,
        description="Always null in chat response. Get the audio URL from the audio_ready SSE event when calling POST /api/ai/tts/stream.",
    )
    conversation_id: Optional[str] = Field(None, description="Conversation ID for this session")


class TTSStreamRequest(BaseModel):
    """Request schema for TTS streaming endpoint."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    response_language: str = Field(default="en", description="Language code (en, hi, ml, ta, etc.)")
