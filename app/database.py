"""Database connection and session management (PostgreSQL only)."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings
from app.models.usage import Base  # noqa: F401 - exported for tests / scripts
from app.models import user  # noqa: F401 - register User model with Base.metadata
from app.models import admin  # noqa: F401 - register Admin model with Base.metadata
from app.models import social_session  # noqa: F401 - register SocialSession
from app.models import live_session  # noqa: F401 - register LiveSession
from app.models import billing  # noqa: F401 - register Billing models


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """FastAPI dependency yielding a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
