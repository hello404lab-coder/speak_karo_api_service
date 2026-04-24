"""add translation fields and native language code

Revision ID: 4c5d6e7f8a90
Revises: af9750fb2202
Create Date: 2026-04-17 11:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = "4c5d6e7f8a90"
down_revision: Union[str, None] = "af9750fb2202"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


LANGUAGE_ALIASES = {
    "en": "en",
    "english": "en",
    "hi": "hi",
    "hindi": "hi",
    "hindustani": "hi",
    "ml": "ml",
    "malayalam": "ml",
    "ta": "ta",
    "tamil": "ta",
    "te": "te",
    "telugu": "te",
    "kn": "kn",
    "kannada": "kn",
    "bn": "bn",
    "bengali": "bn",
    "bangla": "bn",
    "mr": "mr",
    "marathi": "mr",
    "gu": "gu",
    "gujarati": "gu",
    "pa": "pa",
    "punjabi": "pa",
    "ur": "ur",
    "urdu": "ur",
    "es": "es",
    "spanish": "es",
    "fr": "fr",
    "french": "fr",
    "de": "de",
    "german": "de",
    "zh": "zh",
    "chinese": "zh",
    "mandarin": "zh",
    "mandarin chinese": "zh",
    "ja": "ja",
    "japanese": "ja",
    "ko": "ko",
    "korean": "ko",
    "ar": "ar",
    "arabic": "ar",
    "pt": "pt",
    "portuguese": "pt",
}


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
    if connection.dialect.name == "sqlite":
        return _column_exists_sqlite(connection, table, column)
    return _column_exists_pg(connection, table, column)


def _normalize_language_code(value: str | None) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None

    normalized = raw.lower().replace("_", "-")
    primary = normalized.split("-", 1)[0]
    if primary in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[primary]
    return LANGUAGE_ALIASES.get(normalized)


def upgrade() -> None:
    conn = op.get_bind()

    if not _column_exists(conn, "users", "native_language_code"):
        op.add_column("users", sa.Column("native_language_code", sa.String(length=16), nullable=True))

    if not _column_exists(conn, "messages", "reply_language"):
        op.add_column("messages", sa.Column("reply_language", sa.String(length=16), nullable=True))
    if not _column_exists(conn, "messages", "translated_ai_reply"):
        op.add_column("messages", sa.Column("translated_ai_reply", sa.Text(), nullable=True))
    if not _column_exists(conn, "messages", "translation_language_code"):
        op.add_column("messages", sa.Column("translation_language_code", sa.String(length=16), nullable=True))

    rows = conn.execute(text("SELECT id, native_language FROM users WHERE native_language IS NOT NULL"))
    for user_id, native_language in rows:
        code = _normalize_language_code(native_language)
        if code:
            conn.execute(
                text("UPDATE users SET native_language_code = :code WHERE id = :user_id"),
                {"code": code, "user_id": user_id},
            )


def downgrade() -> None:
    conn = op.get_bind()
    for column in ("translation_language_code", "translated_ai_reply", "reply_language"):
        if _column_exists(conn, "messages", column):
            op.drop_column("messages", column)
    if _column_exists(conn, "users", "native_language_code"):
        op.drop_column("users", "native_language_code")
