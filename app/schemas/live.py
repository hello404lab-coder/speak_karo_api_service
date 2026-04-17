"""Pydantic schemas for Gemini Live control-plane API."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class LiveConfigResponse(BaseModel):
    """Safe client configuration for Gemini Live (no API keys)."""

    model: str = Field(..., description="Gemini Live model id for WebSocket setup")
    system_instruction: str = Field(..., description="System instruction to send in Live setup")
    prompt_version: str = Field(..., description="Server prompt template version for logging/cache")
    voice: Optional[str] = Field(None, description="Prebuilt voice name when using native audio TTS")
    language_code: Optional[str] = Field(
        None,
        description="BCP-47 locale hint for the client only (not embedded in server Live speech_config)",
    )
    temperature: float = Field(..., description="Generation temperature for Live connect config")
    response_modalities: List[str] = Field(
        default_factory=lambda: ["AUDIO"],
        description="Modalities for model output (e.g. AUDIO, TEXT)",
    )


class LiveSessionStartRequest(BaseModel):
    """Start a server-tracked Live session."""

    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation id to attach learner context from long_term_context",
    )
    client_platform: Optional[str] = Field(
        None,
        max_length=64,
        description="Optional client hint (e.g. ios, web)",
    )


class LiveSessionStartResponse(BaseModel):
    session_id: str
    started_at: str = Field(..., description="ISO 8601 UTC")
    reused_existing: bool = Field(
        default=False,
        description="True when an open session already existed (single active session per user)",
    )


class LiveSessionHeartbeatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class LiveSessionHeartbeatResponse(BaseModel):
    session_id: str
    last_seen_at: str = Field(..., description="ISO 8601 UTC")


class LiveSessionActiveResponse(BaseModel):
    active: bool
    session_id: Optional[str] = None
    started_at: Optional[str] = Field(None, description="ISO 8601 UTC when active")
    last_seen_at: Optional[str] = Field(None, description="ISO 8601 UTC when active")
    conversation_id: Optional[str] = None


class LiveSessionEndRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Id returned from session/start")


class LiveSessionEndResponse(BaseModel):
    session_id: str
    duration_seconds: float = Field(..., description="Server-computed session length used for usage")
    ended_at: str = Field(..., description="ISO 8601 UTC")


class LiveTokenRequest(BaseModel):
    """Optional session correlation when minting an ephemeral Live auth token."""

    session_id: Optional[str] = Field(
        None,
        description="If set, must be an open Live session owned by the caller",
    )


class LiveTokenResponse(BaseModel):
    auth_token: str = Field(..., description="Ephemeral token resource name for Live WebSocket auth")
    new_session_expire_seconds: int = Field(
        default=60,
        description="Hint: new sessions with this token are typically rejected after this window",
    )
