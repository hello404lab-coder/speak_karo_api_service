"""add messages conversation_id id index for pagination

Revision ID: f1a5e6d7c8b9
Revises: e0f4d5c6b7a8
Create Date: 2026-03-11 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy import text


revision: str = "f1a5e6d7c8b9"
down_revision: Union[str, None] = "e0f4d5c6b7a8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


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
    if not _index_exists(conn, "messages", "ix_messages_conversation_id_id"):
        op.create_index(
            "ix_messages_conversation_id_id",
            "messages",
            ["conversation_id", "id"],
            unique=False,
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _index_exists(conn, "messages", "ix_messages_conversation_id_id"):
        op.drop_index("ix_messages_conversation_id_id", table_name="messages")
