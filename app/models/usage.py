"""Database models for usage tracking and conversations."""
import uuid
from datetime import datetime, date

from sqlalchemy import Column, String, Integer, Float, DateTime, Date, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Usage(Base):
    """Daily usage statistics per user."""
    __tablename__ = "usage"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    date = Column(Date, default=date.today, nullable=False, index=True)
    minutes_used = Column(Float, default=0.0)
    request_count = Column(Integer, default=0)
    chat_count = Column(Integer, default=0)
    voice_count = Column(Integer, default=0)

    __table_args__ = (
        {"schema": None},
    )


class Conversation(Base):
    """Conversation sessions."""
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    # Long-term learner context (e.g. "preparing for IELTS"); injected into system instruction every turn
    long_term_context = Column(Text, nullable=True)
    title = Column(String(255), nullable=True)
    
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    __table_args__ = (
        {"schema": None},
    )


class Message(Base):
    """Individual messages in conversations."""
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False, index=True)
    user_message = Column(Text, nullable=False)
    ai_reply = Column(Text, nullable=False)
    correction = Column(Text, nullable=True)
    hinglish_explanation = Column(Text, nullable=True)
    example = Column(Text, nullable=True)
    score = Column(Integer, nullable=True)  # 0-100
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    user_audio_url = Column(String(512), nullable=True)  # URL for user voice recording (voice-chat only)
    
    conversation = relationship("Conversation", back_populates="messages")
    
    __table_args__ = (
        {"schema": None},
    )
