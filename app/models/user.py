"""User model for OAuth authentication."""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String

from app.models.usage import Base


class User(Base):
    """User account from OAuth provider (Google or Apple)."""

    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    provider = Column(String(32), nullable=False)  # "google" | "apple"
    provider_id = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Onboarding (step 0 = not started, 1-5 = step completed; 5 => onboarding_completed = True)
    nickname = Column(String(100), nullable=True)
    native_language = Column(String(100), nullable=True)
    student_type = Column(String(32), nullable=True)  # adult | kid
    occupation = Column(String(32), nullable=True)  # college | work | home_maker | teacher | other
    goal = Column(String(64), nullable=True)
    english_level = Column(String(32), nullable=True)
    onboarding_step = Column(Integer, default=0, nullable=False)
    onboarding_completed = Column(Boolean, default=False, nullable=False)

    # Subscription: free | trial | premium
    plan = Column(String(32), default="free", nullable=False)
    trial_expires_at = Column(DateTime, nullable=True)
    subscription_expires_at = Column(DateTime, nullable=True)
    is_trial_used = Column(Boolean, default=False, nullable=False)

    __table_args__ = ({"schema": None},)
