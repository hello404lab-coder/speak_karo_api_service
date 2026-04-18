"""Request and response schemas for AI endpoints."""
from pydantic import BaseModel, Field
from typing import Optional


class UserAnalysis(BaseModel):
    """Feedback attached to the learner's own message."""

    correction: str = Field(default="", description="One correction for the learner's message")
    explanation: Optional[str] = Field(None, description="Short explanation of the correction")
    example: Optional[str] = Field(None, description="Example sentence for correct usage")
    score: int = Field(default=70, ge=0, le=100, description="Score out of 100")


class TextChatRequest(BaseModel):
    """Request schema for text chat endpoint."""
    user_id: str = Field(..., description="User identifier")
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")
    learner_context: Optional[str] = Field(None, description="Optional long-term context (e.g. preparing for IELTS); stored on conversation and injected every turn")
    reply_language: Optional[str] = Field(
        None,
        description="Optional ISO 639-1 code for the assistant reply/TTS language. Defaults to detected behavior when omitted.",
    )
    translation_language: Optional[str] = Field(
        None,
        description="Optional ISO 639-1 code for translated display text. Falls back to the user's native language when omitted.",
    )
    response_language: Optional[str] = Field(
        "en",
        description="Deprecated alias for reply_language. Optional ISO 639-1 code (hi, ml, ta, ...).",
    )


class VoiceChatRequest(BaseModel):
    """Request schema for voice chat endpoint."""
    user_id: str = Field(..., description="User identifier")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")
    learner_context: Optional[str] = Field(None, description="Optional long-term context (e.g. preparing for IELTS); stored on conversation and injected every turn")
    reply_language: Optional[str] = Field(
        None,
        description="Optional ISO 639-1 code for the assistant reply/TTS language. Defaults to detected behavior when omitted.",
    )
    translation_language: Optional[str] = Field(
        None,
        description="Optional ISO 639-1 code for translated display text. Falls back to the user's native language when omitted.",
    )
    response_language: Optional[str] = Field(
        None,
        description="Deprecated alias for reply_language. Optional ISO 639-1 code (hi, ml, ta, ...).",
    )


class AIChatResponse(BaseModel):
    """Response schema for AI chat endpoints."""
    reply_text: str = Field(..., description="AI's natural reply")
    translated_reply_text: Optional[str] = Field(None, description="Saved translation of the assistant reply for display")
    reply_language: str = Field(..., description="Language code of the assistant reply")
    translation_language: Optional[str] = Field(None, description="Language code of the translated reply, if present")
    user_analysis: UserAnalysis = Field(..., description="Feedback attached to the learner's message")
    correction: str = Field(..., description="Legacy compatibility alias for user_analysis.correction")
    explanation: Optional[str] = Field(None, description="Legacy compatibility alias for user_analysis.explanation")
    example: Optional[str] = Field(None, description="Legacy compatibility alias for user_analysis.example")
    score: int = Field(..., ge=0, le=100, description="Legacy compatibility alias for user_analysis.score")
    audio_url: Optional[str] = Field(
        None,
        description="Always null in chat response. Get the audio URL from the audio_ready SSE event when calling POST /api/ai/tts/stream.",
    )
    response_language: Optional[str] = Field(
        None,
        description="Deprecated alias for reply_language. Pass to POST /api/ai/tts/stream when requesting audio.",
    )
    conversation_id: Optional[str] = Field(None, description="Conversation ID for this session")


class TTSStreamRequest(BaseModel):
    """Request schema for TTS streaming endpoint."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    response_language: str = Field(default="en", description="Language code (en, hi, ml, ta, etc.)")
