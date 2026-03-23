"""add conversation title and pagination indexes

Revision ID: d9e0f3c4b5a6
Revises: c8d9e0f2b3a4
Create Date: 2026-03-11 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


revision: str = "d9e0f3c4b5a6"
down_revision: Union[str, None] = "c8d9e0f2b3a4"
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


def _index_exists(connection, table: str, index_name: str) -> bool:
    dialect = connection.dialect.name
    if dialect == "sqlite":
        result = connection.execute(text(f"PRAGMA index_list({table})"))
        return any(row[1] == index_name for row in result)
    result = connection.execute(
        text(
            "SELECT 1 FROM pg_indexes WHERE tablename = :t AND indexname = :idx"
        ),
        {"t": table, "idx": index_name},
    )
    return result.scalar() is not None


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, "conversations", "title"):
        op.add_column("conversations", sa.Column("title", sa.String(255), nullable=True))

    if not _index_exists(conn, "messages", "ix_messages_conversation_created"):
        op.create_index(
            "ix_messages_conversation_created",
            "messages",
            ["conversation_id", "created_at"],
            unique=False,
        )
    if not _index_exists(conn, "conversations", "ix_conversations_user_updated"):
        op.create_index(
            "ix_conversations_user_updated",
            "conversations",
            ["user_id", "updated_at"],
            unique=False,
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _index_exists(conn, "conversations", "ix_conversations_user_updated"):
        op.drop_index("ix_conversations_user_updated", table_name="conversations")
    if _index_exists(conn, "messages", "ix_messages_conversation_created"):
        op.drop_index("ix_messages_conversation_created", table_name="messages")
    if _column_exists(conn, "conversations", "title"):
        op.drop_column("conversations", "title")
