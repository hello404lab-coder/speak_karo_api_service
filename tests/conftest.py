"""Shared pytest fixtures for the PostgreSQL-only test suite.

Design:

* A single session-scoped engine points at ``TEST_DATABASE_URL`` (default:
  ``postgresql+psycopg2://postgres:postgres@localhost:5432/english_practice_test``).
* On first use we run ``alembic downgrade base`` then ``alembic upgrade head``
  against that engine so the schema exactly matches production migrations.
* Each test gets its own ``db_session`` that runs inside an outer transaction;
  the transaction is rolled back at teardown so tests stay isolated without
  truncating tables.
* ``override_get_db`` returns a dependency-override callable that binds the
  FastAPI ``get_db`` dependency to the same transactional session, so tests
  that hit HTTP endpoints see uncommitted writes from the surrounding test.

Create the database once per machine::

    createdb english_practice_test
"""

from __future__ import annotations

import os
from typing import Callable, Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/english_practice_test",
)


def _alembic_config(url: str) -> Config:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """Expose the resolved test database URL for tests that need to configure
    app settings (for example overriding ``settings.database_url`` before the
    FastAPI app boots)."""
    return TEST_DATABASE_URL


@pytest.fixture(scope="session")
def test_engine(test_database_url: str) -> Iterator[Engine]:
    """Session-scoped engine; migrations are applied once per test session."""
    # Point the app's engine at the test DB so any code path that reuses
    # ``app.database.engine`` observes the same schema. This must happen before
    # importing ``app.database``.
    os.environ["DATABASE_URL"] = test_database_url

    engine = create_engine(test_database_url, pool_pre_ping=True)

    cfg = _alembic_config(test_database_url)
    # Reset to a known state. ``downgrade base`` is a no-op on an empty DB.
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")

    yield engine

    engine.dispose()


@pytest.fixture
def db_session(test_engine: Engine) -> Iterator[Session]:
    """Per-test transactional session; rolled back at teardown.

    Uses ``join_transaction_mode="create_savepoint"`` so that tests which call
    ``session.commit()`` only release a SAVEPOINT; the outer connection-level
    transaction is then rolled back at teardown, keeping tests isolated.
    """
    connection = test_engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(
        bind=connection,
        autocommit=False,
        autoflush=False,
        join_transaction_mode="create_savepoint",
    )
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()


@pytest.fixture
def override_get_db(db_session: Session) -> Callable[[], Iterator[Session]]:
    """Return a callable suitable for ``app.dependency_overrides[get_db]``."""

    def _get_db_override() -> Iterator[Session]:
        try:
            yield db_session
        finally:
            # Do not close here; the outer fixture rolls back and closes.
            pass

    return _get_db_override
