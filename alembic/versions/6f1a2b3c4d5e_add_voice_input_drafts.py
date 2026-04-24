"""add voice input drafts

Revision ID: 6f1a2b3c4d5e
Revises: 4c5d6e7f8a90
Create Date: 2026-04-18 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = "6f1a2b3c4d5e"
down_revision: Union[str, None] = "4c5d6e7f8a90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists_sqlite(connection, table: str) -> bool:
    result = connection.execute(
        text("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = :table"),
        {"table": table},
    )
    return result.scalar() is not None


def _table_exists_other(connection, table: str) -> bool:
    result = connection.execute(
        text(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = :table"
        ),
        {"table": table},
    )
    return result.scalar() is not None


def _table_exists(connection, table: str) -> bool:
    if connection.dialect.name == "sqlite":
        return _table_exists_sqlite(connection, table)
    return _table_exists_other(connection, table)


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

    if not _table_exists(conn, "voice_input_drafts"):
        op.create_table(
            "voice_input_drafts",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("user_id", sa.String(), nullable=False),
            sa.Column("conversation_id", sa.String(), nullable=True),
            sa.Column("user_audio_url", sa.String(length=512), nullable=False),
            sa.Column("user_audio_storage_key", sa.String(length=1024), nullable=True),
            sa.Column("transcript_text", sa.Text(), nullable=False),
            sa.Column("detected_lang", sa.String(length=16), nullable=True),
            sa.Column(
                "transcript_source",
                sa.String(length=32),
                nullable=False,
                server_default=sa.text("'backend_final'"),
            ),
            sa.Column(
                "warning",
                sa.Text(),
                nullable=True,
            ),
            sa.Column(
                "status",
                sa.String(length=16),
                nullable=False,
                server_default=sa.text("'pending'"),
            ),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("expires_at", sa.DateTime(), nullable=False),
            sa.Column("consumed_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
            sa.PrimaryKeyConstraint("id"),
        )

    indexes = [
        (op.f("ix_voice_input_drafts_user_id"), ["user_id"]),
        (op.f("ix_voice_input_drafts_conversation_id"), ["conversation_id"]),
        (op.f("ix_voice_input_drafts_status"), ["status"]),
        (op.f("ix_voice_input_drafts_created_at"), ["created_at"]),
        (op.f("ix_voice_input_drafts_expires_at"), ["expires_at"]),
    ]
    for index_name, columns in indexes:
        if not _index_exists(conn, index_name):
            op.create_index(index_name, "voice_input_drafts", columns, unique=False)


def downgrade() -> None:
    conn = op.get_bind()

    for index_name in (
        op.f("ix_voice_input_drafts_expires_at"),
        op.f("ix_voice_input_drafts_created_at"),
        op.f("ix_voice_input_drafts_status"),
        op.f("ix_voice_input_drafts_conversation_id"),
        op.f("ix_voice_input_drafts_user_id"),
    ):
        if _index_exists(conn, index_name):
            op.drop_index(index_name, table_name="voice_input_drafts")
    if _table_exists(conn, "voice_input_drafts"):
        op.drop_table("voice_input_drafts")
