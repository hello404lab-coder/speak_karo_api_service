"""Request and response schemas for conversation and chat history endpoints."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field
from app.schemas.ai import UserAnalysis


class ConversationListItem(BaseModel):
    """Single conversation in list response."""
    id: str = Field(..., description="Conversation UUID")
    title: Optional[str] = Field(None, description="Optional title (e.g. AI-generated)")
    last_message: Optional[str] = Field(None, description="Last assistant message excerpt")
    updated_at: datetime = Field(..., description="Last update time")


class ConversationListResponse(BaseModel):
    """Response for GET /conversations."""
    conversations: list[ConversationListItem] = Field(
        default_factory=list,
        description="Conversations ordered by updated_at DESC",
    )


class ConversationDetail(BaseModel):
    """Response for GET /conversations/{id}."""
    id: str = Field(..., description="Conversation UUID")
    title: Optional[str] = Field(None, description="Optional title")
    created_at: datetime = Field(..., description="Creation time")
    updated_at: datetime = Field(..., description="Last update time")


class ChatMessage(BaseModel):
    """Single message in chat history (user or assistant)."""
    index: int = Field(..., ge=0, description="Zero-based display order. Always sort by this field; do not use id for ordering.")
    id: str = Field(..., description="Stable UUID for this message (e.g. React keys). Use index for ordering.")
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: Optional[str] = Field(None, description="User message text (role=user)")
    user_audio_url: Optional[str] = Field(None, description="URL of user voice recording (role=user, voice-chat only)")
    user_analysis: Optional[UserAnalysis] = Field(None, description="Feedback attached to the learner's message (role=user)")
    reply_text: Optional[str] = Field(None, description="Assistant reply (role=assistant)")
    translated_reply_text: Optional[str] = Field(None, description="Translated assistant reply (role=assistant)")
    reply_language: Optional[str] = Field(None, description="Language code for assistant reply text (role=assistant)")
    translation_language: Optional[str] = Field(None, description="Language code for translated assistant reply (role=assistant)")
    created_at: datetime = Field(..., description="Message timestamp")


class MessagesResponse(BaseModel):
    """Response for GET /conversations/{id}/messages."""
    messages: list[ChatMessage] = Field(
        default_factory=list,
        description="Messages in display order (chronological, user then assistant per exchange). Use each message's index for ordering; do not rely on id.",
    )
    next_cursor: Optional[str] = Field(
        None,
        description="Opaque cursor for next page (older messages). Omit when no more pages.",
    )


class TitleResponse(BaseModel):
    """Response for POST /conversations/{id}/title."""
    title: str = Field(..., description="Generated or updated conversation title")
