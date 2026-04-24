"""replace_llm_seconds_with_output_tokens

Revision ID: f4a1c2d3e4b5
Revises: 9f2d4a6c1b7e
Create Date: 2026-04-23 14:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f4a1c2d3e4b5"
down_revision: Union[str, None] = "9f2d4a6c1b7e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "usage",
        sa.Column("llm_output_tokens", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.drop_column("usage", "llm_seconds")


def downgrade() -> None:
    op.add_column(
        "usage",
        sa.Column("llm_seconds", sa.Float(), nullable=False, server_default=sa.text("0")),
    )
    op.drop_column("usage", "llm_output_tokens")
