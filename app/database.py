"""Database connection and session management."""
import os
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from app.core.config import settings
from app.models.usage import Base


def _create_engine():
    """Create engine: SQLite uses NullPool and check_same_thread=False; PostgreSQL uses pooling."""
    url = make_url(settings.database_url)
    if url.get_backend_name() == "sqlite":
        # File-based SQLite: ensure parent directory exists (skip for :memory:)
        database_path = url.database
        if database_path and database_path != ":memory:":
            parent = os.path.dirname(database_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        return create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
        )
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


engine = _create_engine()

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
