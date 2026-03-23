"""add subscription fields (users + usage)

Revision ID: b7c8d9e0f1a2
Revises: 5b5afc83126f
Create Date: 2026-03-10 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


revision: str = "b7c8d9e0f1a2"
down_revision: Union[str, None] = "5b5afc83126f"
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

    # Users: subscription columns
    if not _column_exists(conn, "users", "plan"):
        op.add_column(
            "users",
            sa.Column("plan", sa.String(length=32), nullable=False, server_default="free"),
        )
    if not _column_exists(conn, "users", "trial_expires_at"):
        op.add_column("users", sa.Column("trial_expires_at", sa.DateTime(), nullable=True))
    if not _column_exists(conn, "users", "subscription_expires_at"):
        op.add_column(
            "users",
            sa.Column("subscription_expires_at", sa.DateTime(), nullable=True),
        )
    if not _column_exists(conn, "users", "is_trial_used"):
        op.add_column(
            "users",
            sa.Column(
                "is_trial_used",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )

    # Usage: chat_count, voice_count
    if not _column_exists(conn, "usage", "chat_count"):
        op.add_column(
            "usage",
            sa.Column("chat_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        )
    if not _column_exists(conn, "usage", "voice_count"):
        op.add_column(
            "usage",
            sa.Column("voice_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        )


def downgrade() -> None:
    conn = op.get_bind()
    for col in ("is_trial_used", "subscription_expires_at", "trial_expires_at", "plan"):
        if _column_exists(conn, "users", col):
            op.drop_column("users", col)
    for col in ("voice_count", "chat_count"):
        if _column_exists(conn, "usage", col):
            op.drop_column("usage", col)
