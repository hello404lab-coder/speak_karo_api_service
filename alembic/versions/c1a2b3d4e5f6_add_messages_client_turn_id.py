"""add client_turn_id to messages

Revision ID: c1a2b3d4e5f6
Revises: 6f1a2b3c4d5e
Create Date: 2026-04-19 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


revision: str = "c1a2b3d4e5f6"
down_revision: Union[str, None] = "6f1a2b3c4d5e"
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


def _index_exists_sqlite(connection, index_name: str) -> bool:
    result = connection.execute(
        text("SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = :name"),
        {"name": index_name},
    )
    return result.scalar() is not None


def _index_exists_other(connection, index_name: str) -> bool:
    result = connection.execute(
        text(
            "SELECT 1 FROM information_schema.statistics "
            "WHERE index_name = :name"
        ),
        {"name": index_name},
    )
    return result.scalar() is not None


def _index_exists(connection, index_name: str) -> bool:
    if connection.dialect.name == "sqlite":
        return _index_exists_sqlite(connection, index_name)
    return _index_exists_other(connection, index_name)


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, "messages", "client_turn_id"):
        op.add_column("messages", sa.Column("client_turn_id", sa.String(length=36), nullable=True))

    index_name = op.f("ix_messages_client_turn_id")
    if not _index_exists(conn, index_name):
        op.create_index(index_name, "messages", ["client_turn_id"], unique=False)


def downgrade() -> None:
    conn = op.get_bind()
    index_name = op.f("ix_messages_client_turn_id")
    if _index_exists(conn, index_name):
        op.drop_index(index_name, table_name="messages")
    if _column_exists(conn, "messages", "client_turn_id"):
        op.drop_column("messages", "client_turn_id")
