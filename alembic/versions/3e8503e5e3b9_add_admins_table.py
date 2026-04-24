"""add_admins_table

Revision ID: 3e8503e5e3b9
Revises: c1a2b3d4e5f6
Create Date: 2026-04-23 00:17:06.334664

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3e8503e5e3b9'
down_revision: Union[str, None] = 'c1a2b3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "admins" in inspector.get_table_names():
        return

    op.create_table(
        "admins",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_admins_email"), "admins", ["email"], unique=True)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "admins" not in inspector.get_table_names():
        return
    op.drop_index(op.f("ix_admins_email"), table_name="admins")
    op.drop_table("admins")
