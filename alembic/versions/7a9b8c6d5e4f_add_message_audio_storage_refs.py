"""add message audio storage refs

Revision ID: 7a9b8c6d5e4f
Revises: e0e1e1945835
Create Date: 2026-04-24 18:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


revision: str = "7a9b8c6d5e4f"
down_revision: Union[str, None] = "e0e1e1945835"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists_sqlite(connection, table: str, column: str) -> bool:
    result = connection.execute(text(f"PRAGMA table_info({table})"))
    return any(row[1] == column for row in result)


def _column_exists_other(connection, table: str, column: str) -> bool:
    result = connection.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :table AND column_name = :column"
        ),
        {"table": table, "column": column},
    )
    return result.scalar() is not None


def _column_exists(connection, table: str, column: str) -> bool:
    if connection.dialect.name == "sqlite":
        return _column_exists_sqlite(connection, table, column)
    return _column_exists_other(connection, table, column)


def upgrade() -> None:
    conn = op.get_bind()
    columns = [
        "ai_reply_audio_storage_ref",
        "translated_ai_reply_audio_storage_ref",
        "explanation_audio_storage_ref",
        "example_audio_storage_ref",
    ]
    for column in columns:
        if not _column_exists(conn, "messages", column):
            op.add_column("messages", sa.Column(column, sa.String(length=1024), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    columns = [
        "example_audio_storage_ref",
        "explanation_audio_storage_ref",
        "translated_ai_reply_audio_storage_ref",
        "ai_reply_audio_storage_ref",
    ]
    for column in columns:
        if _column_exists(conn, "messages", column):
            op.drop_column("messages", column)
