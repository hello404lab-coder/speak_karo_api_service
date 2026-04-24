"""Social voice matchmaking session (Agora channel pairing)."""
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String

from app.models.usage import Base


class SocialSession(Base):
    __tablename__ = "social_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user1_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    user2_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)

    status = Column(String(32), nullable=False, default="waiting")  # waiting, active, ended

    agora_channel = Column(String(128), unique=True, nullable=False, index=True)

    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = ({"schema": None},)
