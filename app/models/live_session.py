"""Server-tracked Gemini Live sessions (metadata only; audio is client-to-Google)."""
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text, text

from app.models.usage import Base


class LiveSession(Base):
    __tablename__ = "live_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=True, index=True)

    started_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    # Liveness for reaper; updated by POST /live/session/heartbeat
    last_seen_at = Column(DateTime, nullable=True, index=True)
    # active | ended | auto_ended
    status = Column(String(16), nullable=False, default="active", server_default=text("'active'"))

    prompt_version = Column(String(32), nullable=True)
    client_platform = Column(String(64), nullable=True)

    __table_args__ = ({"schema": None},)
