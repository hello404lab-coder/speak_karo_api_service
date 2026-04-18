"""Persistence helpers for transcript-first voice input drafts."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.usage import VoiceInputDraft
from app.services.tts import delete_stored_audio

logger = logging.getLogger(__name__)

VOICE_DRAFT_STATUS_PENDING = "pending"
VOICE_DRAFT_STATUS_CONSUMED = "consumed"
VOICE_DRAFT_STATUS_DISCARDED = "discarded"
VOICE_DRAFT_STATUS_EXPIRED = "expired"

VOICE_DRAFT_SOURCE_BACKEND_FINAL = "backend_final"
VOICE_DRAFT_SOURCE_BROWSER_FALLBACK = "browser_fallback"


def build_voice_draft_expiry(now: Optional[datetime] = None) -> datetime:
    """Return the expiry timestamp for a newly created draft."""
    created_at = now or datetime.utcnow()
    return created_at + timedelta(hours=max(1, int(settings.voice_draft_ttl_hours or 24)))


def create_voice_input_draft(
    db: Session,
    *,
    user_id: str,
    conversation_id: Optional[str],
    user_audio_url: str,
    user_audio_storage_key: Optional[str],
    transcript_text: str,
    detected_lang: Optional[str],
    transcript_source: str,
    warning: Optional[str],
) -> VoiceInputDraft:
    """Persist a pending voice input draft and return the refreshed row."""
    now = datetime.utcnow()
    draft = VoiceInputDraft(
        user_id=user_id,
        conversation_id=conversation_id,
        user_audio_url=user_audio_url,
        user_audio_storage_key=user_audio_storage_key,
        transcript_text=transcript_text,
        detected_lang=detected_lang,
        transcript_source=transcript_source,
        warning=warning,
        status=VOICE_DRAFT_STATUS_PENDING,
        created_at=now,
        expires_at=build_voice_draft_expiry(now),
    )
    db.add(draft)
    db.commit()
    db.refresh(draft)
    return draft


def cleanup_expired_voice_drafts(db: Session) -> int:
    """Expire pending drafts that have passed their TTL and delete stored audio when possible."""
    now = datetime.utcnow()
    drafts = (
        db.query(VoiceInputDraft)
        .filter(
            VoiceInputDraft.status == VOICE_DRAFT_STATUS_PENDING,
            VoiceInputDraft.expires_at <= now,
        )
        .all()
    )
    if not drafts:
        return 0

    for draft in drafts:
        delete_stored_audio(draft.user_audio_storage_key)
        draft.status = VOICE_DRAFT_STATUS_EXPIRED

    db.commit()
    logger.info("voice_draft_cleanup_expired", extra={"count": len(drafts)})
    return len(drafts)


def get_pending_voice_input_draft(db: Session, user_id: str, draft_id: str) -> Optional[VoiceInputDraft]:
    """Return a still-pending draft for the user, expiring it first if needed."""
    draft = (
        db.query(VoiceInputDraft)
        .filter(VoiceInputDraft.id == draft_id, VoiceInputDraft.user_id == user_id)
        .first()
    )
    if not draft:
        return None

    if (
        draft.status == VOICE_DRAFT_STATUS_PENDING
        and draft.expires_at is not None
        and draft.expires_at <= datetime.utcnow()
    ):
        delete_stored_audio(draft.user_audio_storage_key)
        draft.status = VOICE_DRAFT_STATUS_EXPIRED
        db.commit()
        return None

    if draft.status != VOICE_DRAFT_STATUS_PENDING:
        return None
    return draft


def discard_voice_input_draft(db: Session, user_id: str, draft_id: str) -> bool:
    """Discard a pending draft and delete its stored audio."""
    draft = (
        db.query(VoiceInputDraft)
        .filter(VoiceInputDraft.id == draft_id, VoiceInputDraft.user_id == user_id)
        .first()
    )
    if not draft:
        return False

    if draft.status == VOICE_DRAFT_STATUS_PENDING:
        delete_stored_audio(draft.user_audio_storage_key)
        draft.status = VOICE_DRAFT_STATUS_DISCARDED
        db.commit()
        return True

    return False
