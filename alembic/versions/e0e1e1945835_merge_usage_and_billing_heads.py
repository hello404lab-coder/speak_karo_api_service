"""merge usage and billing heads

Revision ID: e0e1e1945835
Revises: 1dc53012ca93, f4a1c2d3e4b5
Create Date: 2026-04-23 20:28:29.344517

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e0e1e1945835'
down_revision: Union[str, None] = ('1dc53012ca93', 'f4a1c2d3e4b5')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
