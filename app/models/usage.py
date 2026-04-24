"""Database models for usage tracking and conversations."""
import uuid
from datetime import datetime, date

import sqlalchemy as sa
from sqlalchemy import Column, String, Integer, Float, DateTime, Date, ForeignKey, Index, Text
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
    llm_output_tokens = Column(Integer, nullable=False, default=0, server_default=sa.text("0"))
    stt_seconds = Column(Float, nullable=False, default=0.0, server_default=sa.text("0"))
    tts_seconds = Column(Float, nullable=False, default=0.0, server_default=sa.text("0"))
    request_count = Column(Integer, default=0)
    chat_count = Column(Integer, nullable=False, default=0, server_default=sa.text("0"))
    voice_count = Column(Integer, nullable=False, default=0, server_default=sa.text("0"))


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
        Index("ix_conversations_user_updated", "user_id", "updated_at"),
    )


class Message(Base):
    """Individual messages in conversations."""
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False, index=True)
    client_turn_id = Column(String(36), nullable=True, index=True)
    user_message = Column(Text, nullable=False)
    ai_reply = Column(Text, nullable=False)
    reply_language = Column(String(16), nullable=True)
    translated_ai_reply = Column(Text, nullable=True)
    translation_language_code = Column(String(16), nullable=True)
    correction = Column(Text, nullable=True)
    hinglish_explanation = Column(Text, nullable=True)
    example = Column(Text, nullable=True)
    score = Column(Integer, nullable=True)  # 0-100
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    user_audio_url = Column(String(512), nullable=True)  # URL for user voice recording (voice-chat only)
    ai_reply_audio_storage_ref = Column(String(1024), nullable=True)
    translated_ai_reply_audio_storage_ref = Column(String(1024), nullable=True)
    explanation_audio_storage_ref = Column(String(1024), nullable=True)
    example_audio_storage_ref = Column(String(1024), nullable=True)
    
    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
        Index("ix_messages_conversation_id_id", "conversation_id", "id"),
    )


class VoiceInputDraft(Base):
    """Pending voice transcript draft created before a message is sent to the AI."""
    __tablename__ = "voice_input_drafts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=True, index=True)
    user_audio_url = Column(String(512), nullable=False)
    user_audio_storage_key = Column(String(1024), nullable=True)
    transcript_text = Column(Text, nullable=False)
    detected_lang = Column(String(16), nullable=True)
    transcript_source = Column(String(32), nullable=False, default="backend_final", server_default=sa.text("'backend_final'"))
    warning = Column(Text, nullable=True)
    status = Column(String(16), nullable=False, default="pending", server_default=sa.text("'pending'"), index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    consumed_at = Column(DateTime, nullable=True)

    conversation = relationship("Conversation")
