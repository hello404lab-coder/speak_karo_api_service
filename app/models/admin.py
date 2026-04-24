"""Admin account model for backend administration."""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, String, text

from app.models.usage import Base


class Admin(Base):
    """Dedicated admin account authenticated with email/password."""

    __tablename__ = "admins"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, server_default=text("1"), default=True)
    last_login_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = ({"schema": None},)
