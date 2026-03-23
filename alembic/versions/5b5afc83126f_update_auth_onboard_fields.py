"""update: auth onboard fields

Revision ID: 5b5afc83126f
Revises: aa0cb32bb206
Create Date: 2026-03-10 01:25:14.983126

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = '5b5afc83126f'
down_revision: Union[str, None] = 'aa0cb32bb206'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(connection, table: str, column: str) -> bool:
    """Return True if column exists on table (SQLite)."""
    result = connection.execute(text(f"PRAGMA table_info({table})"))
    return any(row[1] == column for row in result)


def upgrade() -> None:
    conn = op.get_bind()
    # Only add columns that don't exist (idempotent for existing DBs).
    if not _column_exists(conn, "users", "nickname"):
        op.add_column("users", sa.Column("nickname", sa.String(length=100), nullable=True))
    if not _column_exists(conn, "users", "native_language"):
        op.add_column("users", sa.Column("native_language", sa.String(length=100), nullable=True))
    if not _column_exists(conn, "users", "student_type"):
        op.add_column("users", sa.Column("student_type", sa.String(length=32), nullable=True))
    if not _column_exists(conn, "users", "occupation"):
        op.add_column("users", sa.Column("occupation", sa.String(length=32), nullable=True))
    if not _column_exists(conn, "users", "goal"):
        op.add_column("users", sa.Column("goal", sa.String(length=64), nullable=True))
    if not _column_exists(conn, "users", "english_level"):
        op.add_column("users", sa.Column("english_level", sa.String(length=32), nullable=True))
    if not _column_exists(conn, "users", "onboarding_step"):
        op.add_column("users", sa.Column("onboarding_step", sa.Integer(), nullable=False, server_default="0"))
    if not _column_exists(conn, "users", "onboarding_completed"):
        op.add_column("users", sa.Column("onboarding_completed", sa.Boolean(), nullable=False, server_default="0"))


def downgrade() -> None:
    conn = op.get_bind()
    for col in (
        "onboarding_completed",
        "onboarding_step",
        "english_level",
        "goal",
        "occupation",
        "student_type",
        "native_language",
        "nickname",
    ):
        if _column_exists(conn, "users", col):
            op.drop_column("users", col)
