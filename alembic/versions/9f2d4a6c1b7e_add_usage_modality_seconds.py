"""add_usage_modality_seconds

Revision ID: 9f2d4a6c1b7e
Revises: c89cb5b7d265
Create Date: 2026-04-23 13:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9f2d4a6c1b7e"
down_revision: Union[str, None] = "c89cb5b7d265"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "usage",
        sa.Column("llm_seconds", sa.Float(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "usage",
        sa.Column("stt_seconds", sa.Float(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "usage",
        sa.Column("tts_seconds", sa.Float(), nullable=False, server_default=sa.text("0")),
    )
    op.execute("UPDATE usage SET llm_seconds = 0 WHERE llm_seconds IS NULL")
    op.execute("UPDATE usage SET stt_seconds = 0 WHERE stt_seconds IS NULL")
    op.execute("UPDATE usage SET tts_seconds = 0 WHERE tts_seconds IS NULL")


def downgrade() -> None:
    op.drop_column("usage", "tts_seconds")
    op.drop_column("usage", "stt_seconds")
    op.drop_column("usage", "llm_seconds")
