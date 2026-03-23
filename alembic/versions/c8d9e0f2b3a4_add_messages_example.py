"""add example column to messages

Revision ID: c8d9e0f2b3a4
Revises: b7c8d9e0f1a2
Create Date: 2026-03-11 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


revision: str = "c8d9e0f2b3a4"
down_revision: Union[str, None] = "b7c8d9e0f1a2"
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
    if not _column_exists(conn, "messages", "example"):
        op.add_column("messages", sa.Column("example", sa.Text(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "messages", "example"):
        op.drop_column("messages", "example")
