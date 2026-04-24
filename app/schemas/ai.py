"""Request and response schemas for AI endpoints."""
from pydantic import BaseModel, Field
from typing import Literal, Optional


class UserAnalysis(BaseModel):
    """Feedback attached to the learner's own message."""

    correction: Optional[str] = Field(None, description="A meaningful correction or better natural phrasing for the learner's message")
    explanation: Optional[str] = Field(None, description="Short explanation of the correction")
    example: Optional[str] = Field(None, description="Example sentence for correct usage")
    score: int = Field(default=70, ge=0, le=100, description="Score out of 100")


class TextChatRequest(BaseModel):
    """Request schema for text chat endpoint."""
    user_id: str = Field(..., description="User identifier")
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    client_turn_id: Optional[str] = Field(
        None,
        description="Optional client-generated UUID used to reconcile optimistic chat state.",
    )
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
    include_audio_stream: bool = Field(
        True,
        description="When false, stream only text/metadata and skip eager TTS generation for this turn.",
    )
    voice_draft_id: Optional[str] = Field(
        None,
        description="Optional one-time voice draft id created by POST /voice-drafts/finalize.",
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
    client_turn_id: Optional[str] = Field(None, description="Echoed client turn identifier when provided")
    translated_reply_text: Optional[str] = Field(None, description="Saved translation of the assistant reply for display")
    reply_language: str = Field(..., description="Language code of the assistant reply")
    translation_language: Optional[str] = Field(None, description="Language code of the translated reply, if present")
    user_analysis: UserAnalysis = Field(..., description="Feedback attached to the learner's message")
    correction: Optional[str] = Field(None, description="Legacy compatibility alias for user_analysis.correction")
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


class VoiceDraftFinalizeResponse(BaseModel):
    """Response schema for POST /voice-drafts/finalize."""

    voice_draft_id: str = Field(..., description="One-time reusable draft id for a subsequent chat send")
    transcript_text: str = Field(..., description="Final transcript text for review/edit before send")
    detected_lang: Optional[str] = Field(None, description="Detected language code from backend STT or browser fallback")
    transcript_source: Literal["backend_final", "browser_fallback"] = Field(
        ...,
        description="Whether transcript_text came from backend STT or browser fallback text",
    )
    warning: Optional[str] = Field(None, description="Optional warning when backend STT failed and browser fallback was used")


class TTSStreamRequest(BaseModel):
    """Request schema for TTS streaming endpoint."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    response_language: str = Field(default="en", description="Language code (en, hi, ml, ta, etc.)")


class MessageAudioRequest(BaseModel):
    """Request schema for lazily generated message audio."""

    segment: Literal["reply", "translation", "explanation", "example"] = Field(
        ...,
        description="Which message section to synthesize or reuse stored audio for.",
    )


class MessageAudioResponse(BaseModel):
    """Response schema for lazily generated message audio."""

    audio_url: str = Field(..., description="Fresh playback URL for the requested audio")
    segment: Literal["reply", "translation", "explanation", "example"] = Field(
        ...,
        description="Requested section",
    )
    generated: bool = Field(
        ...,
        description="True when audio was generated on this request, false when stored audio was reused.",
    )
