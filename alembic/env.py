from logging.config import fileConfig
import sys
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine

# Add the parent directory to the path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings

# Import Base and all models - this must happen before target_metadata is set.
# Importing the models registers them with ``Base.metadata``.
from app.models.usage import Base

from app.models.usage import Usage, Conversation, Message  # noqa: F401
from app.models.admin import Admin  # noqa: F401
from app.models.user import User  # noqa: F401
from app.models.social_session import SocialSession  # noqa: F401
from app.models.live_session import LiveSession  # noqa: F401
from app.models.billing import (  # noqa: F401
    BillingCoupon,
    BillingCouponRedemption,
    BillingSubscription,
    BillingWebhookEvent,
)

# this is the Alembic Config object, which provides access to the values
# within the .ini file in use.
config = context.config

# Resolve the target DB URL. If a caller (e.g. tests/conftest.py or
# ``alembic -x ...``) already set ``sqlalchemy.url`` on this config, keep it so
# tests can point at a dedicated database. Otherwise fall back to the app
# settings so interactive ``alembic`` invocations target the dev/prod DB.
_existing_url = config.get_main_option("sqlalchemy.url")
if not _existing_url or _existing_url == "driver://user:pass@localhost/dbname":
    config.set_main_option("sqlalchemy.url", settings.database_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# All models must be imported above before this line
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Always creates a dedicated engine bound to whatever URL the config resolves
    to. This keeps Alembic independent of ``app.database.engine`` so tests can
    point Alembic at a test database without mutating the app-wide engine.
    """
    url = config.get_main_option("sqlalchemy.url")
    connectable = create_engine(url, pool_pre_ping=True)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
