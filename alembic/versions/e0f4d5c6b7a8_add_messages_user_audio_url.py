"""add user_audio_url to messages

Revision ID: e0f4d5c6b7a8
Revises: d9e0f3c4b5a6
Create Date: 2026-03-11 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


revision: str = "e0f4d5c6b7a8"
down_revision: Union[str, None] = "d9e0f3c4b5a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists_sqlite(connection, table: str, column: str) -> bool:
    result = connection.execute(text(f"PRAGMA table_info({table})"))
    return any(row[1] == column for row in result)


def _column_exists_pg(connection, table: str, column: str) -> bool:
    result = connection.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :t AND column_name = :c"
        ),
        {"t": table, "c": column},
    )
    return result.scalar() is not None


def _column_exists(connection, table: str, column: str) -> bool:
    dialect = connection.dialect.name
    if dialect == "sqlite":
        return _column_exists_sqlite(connection, table, column)
    return _column_exists_pg(connection, table, column)


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, "messages", "user_audio_url"):
        op.add_column("messages", sa.Column("user_audio_url", sa.String(512), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "messages", "user_audio_url"):
        op.drop_column("messages", "user_audio_url")
