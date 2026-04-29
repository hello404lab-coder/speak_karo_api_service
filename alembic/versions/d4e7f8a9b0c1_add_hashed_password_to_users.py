"""add hashed_password to users

Revision ID: d4e7f8a9b0c1
Revises: 52f68d1edebc
Create Date: 2026-04-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e7f8a9b0c1'
down_revision: Union[str, None] = '52f68d1edebc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('hashed_password', sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column('users', 'hashed_password')
