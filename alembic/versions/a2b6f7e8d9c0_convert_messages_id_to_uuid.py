"""convert messages.id from integer to UUID (string)

Revision ID: a2b6f7e8d9c0
Revises: f1a5e6d7c8b9
Create Date: 2026-03-11 20:00:00.000000

Assumes messages table is empty (cleared). Use keyset pagination (created_at, id) after this.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


revision: str = "a2b6f7e8d9c0"
down_revision: Union[str, None] = "f1a5e6d7c8b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _is_sqlite(connection) -> bool:
    return connection.dialect.name == "sqlite"


def _index_exists(connection, table: str, index_name: str) -> bool:
    dialect = connection.dialect.name
    if dialect == "sqlite":
        result = connection.execute(text(f"PRAGMA index_list({table})"))
        return any(row[1] == index_name for row in result)
    result = connection.execute(
        text("SELECT 1 FROM pg_indexes WHERE tablename = :t AND indexname = :idx"),
        {"t": table, "idx": index_name},
    )
    return result.scalar() is not None


def upgrade() -> None:
    conn = op.get_bind()
    if _is_sqlite(conn):
        # SQLite: recreate table (no ALTER COLUMN type). Table is assumed empty.
        op.create_table(
            "messages_new",
            sa.Column("id", sa.String(36), nullable=False),
            sa.Column("conversation_id", sa.String(), nullable=False),
            sa.Column("user_message", sa.Text(), nullable=False),
            sa.Column("ai_reply", sa.Text(), nullable=False),
            sa.Column("correction", sa.Text(), nullable=True),
            sa.Column("hinglish_explanation", sa.Text(), nullable=True),
            sa.Column("example", sa.Text(), nullable=True),
            sa.Column("score", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("user_audio_url", sa.String(512), nullable=True),
            sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.drop_table("messages")
        op.rename_table("messages_new", "messages")
        op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"], unique=False)
        op.create_index("ix_messages_created_at", "messages", ["created_at"], unique=False)
        op.create_index("ix_messages_conversation_id_id", "messages", ["conversation_id", "id"], unique=False)
    else:
        # PostgreSQL: alter column type via new column + swap (or recreate if empty)
        result = conn.execute(text("SELECT COUNT(*) FROM messages"))
        if result.scalar() == 0:
            op.drop_table("messages")
            op.create_table(
                "messages",
                sa.Column("id", sa.String(36), nullable=False),
                sa.Column("conversation_id", sa.String(), nullable=False),
                sa.Column("user_message", sa.Text(), nullable=False),
                sa.Column("ai_reply", sa.Text(), nullable=False),
                sa.Column("correction", sa.Text(), nullable=True),
                sa.Column("hinglish_explanation", sa.Text(), nullable=True),
                sa.Column("example", sa.Text(), nullable=True),
                sa.Column("score", sa.Integer(), nullable=True),
                sa.Column("created_at", sa.DateTime(), nullable=False),
                sa.Column("user_audio_url", sa.String(512), nullable=True),
                sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
                sa.PrimaryKeyConstraint("id"),
            )
            op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"], unique=False)
            op.create_index("ix_messages_created_at", "messages", ["created_at"], unique=False)
            op.create_index("ix_messages_conversation_id_id", "messages", ["conversation_id", "id"], unique=False)
        else:
            op.add_column("messages", sa.Column("id_new", sa.String(36), nullable=True))
            op.execute(text("UPDATE messages SET id_new = gen_random_uuid()::text WHERE id_new IS NULL"))
            op.drop_constraint("messages_pkey", "messages", type_="primary")
            op.drop_column("messages", "id")
            op.alter_column("messages", "id_new", new_column_name="id", nullable=False)
            op.create_primary_key("messages_pkey", "messages", ["id"])


def downgrade() -> None:
    conn = op.get_bind()
    if _is_sqlite(conn):
        op.create_table(
            "messages_old",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("conversation_id", sa.String(), nullable=False),
            sa.Column("user_message", sa.Text(), nullable=False),
            sa.Column("ai_reply", sa.Text(), nullable=False),
            sa.Column("correction", sa.Text(), nullable=True),
            sa.Column("hinglish_explanation", sa.Text(), nullable=True),
            sa.Column("example", sa.Text(), nullable=True),
            sa.Column("score", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("user_audio_url", sa.String(512), nullable=True),
            sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.execute(
            text(
                "INSERT INTO messages_old (conversation_id, user_message, ai_reply, correction, "
                "hinglish_explanation, example, score, created_at, user_audio_url) "
                "SELECT conversation_id, user_message, ai_reply, correction, "
                "hinglish_explanation, example, score, created_at, user_audio_url FROM messages"
            )
        )
        op.drop_table("messages")
        op.rename_table("messages_old", "messages")
        op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"], unique=False)
        op.create_index("ix_messages_created_at", "messages", ["created_at"], unique=False)
    else:
        op.add_column("messages", sa.Column("id_int", sa.Integer(), nullable=True))
        op.execute(
            text(
                "WITH ordered AS (SELECT id, row_number() OVER (ORDER BY created_at, id) AS rn FROM messages) "
                "UPDATE messages SET id_int = ordered.rn FROM ordered WHERE messages.id = ordered.id"
            )
        )
        op.drop_constraint("messages_pkey", "messages", type_="primary")
        op.drop_column("messages", "id")
        op.alter_column("messages", "id_int", new_column_name="id", nullable=False)
        op.execute(text("CREATE SEQUENCE IF NOT EXISTS messages_id_seq"))
        op.execute(text("SELECT setval('messages_id_seq', COALESCE((SELECT MAX(id) FROM messages), 1))"))
        op.create_primary_key("messages_pkey", "messages", ["id"])
