"""Unit tests for transcript-first voice draft persistence helpers."""
from datetime import datetime, timedelta
from unittest.mock import patch

from app.models.usage import VoiceInputDraft
from app.services.voice_drafts import (
    VOICE_DRAFT_SOURCE_BACKEND_FINAL,
    VOICE_DRAFT_STATUS_DISCARDED,
    VOICE_DRAFT_STATUS_EXPIRED,
    VOICE_DRAFT_STATUS_PENDING,
    cleanup_expired_voice_drafts,
    create_voice_input_draft,
    discard_voice_input_draft,
    get_pending_voice_input_draft,
)


def test_create_voice_input_draft_defaults_to_pending(db_session):
    db = db_session
    try:
        draft = create_voice_input_draft(
            db,
            user_id="user-1",
            conversation_id=None,
            user_audio_url="https://example.com/audio.wav",
            user_audio_storage_key="local:audio.wav",
            transcript_text="Hello there",
            detected_lang="en",
            transcript_source=VOICE_DRAFT_SOURCE_BACKEND_FINAL,
            warning=None,
        )
        assert draft.status == VOICE_DRAFT_STATUS_PENDING
        assert draft.transcript_text == "Hello there"
        assert draft.expires_at > draft.created_at
    finally:
        db.close()


def test_cleanup_expired_voice_drafts_marks_and_deletes_audio(db_session):
    db = db_session
    try:
        draft = VoiceInputDraft(
            user_id="user-1",
            user_audio_url="https://example.com/audio.wav",
            user_audio_storage_key="local:audio.wav",
            transcript_text="Hello there",
            detected_lang="en",
            transcript_source=VOICE_DRAFT_SOURCE_BACKEND_FINAL,
            status=VOICE_DRAFT_STATUS_PENDING,
            created_at=datetime.utcnow() - timedelta(days=2),
            expires_at=datetime.utcnow() - timedelta(minutes=1),
        )
        db.add(draft)
        db.commit()

        with patch("app.services.voice_drafts.delete_stored_audio") as mock_delete:
            cleaned = cleanup_expired_voice_drafts(db)

        db.refresh(draft)
        assert cleaned == 1
        assert draft.status == VOICE_DRAFT_STATUS_EXPIRED
        mock_delete.assert_called_once_with("local:audio.wav")
    finally:
        db.close()


def test_discard_voice_input_draft_marks_row_discarded(db_session):
    db = db_session
    try:
        draft = create_voice_input_draft(
            db,
            user_id="user-1",
            conversation_id=None,
            user_audio_url="https://example.com/audio.wav",
            user_audio_storage_key="local:audio.wav",
            transcript_text="Hello there",
            detected_lang="en",
            transcript_source=VOICE_DRAFT_SOURCE_BACKEND_FINAL,
            warning=None,
        )

        with patch("app.services.voice_drafts.delete_stored_audio") as mock_delete:
            discarded = discard_voice_input_draft(db, "user-1", draft.id)

        db.refresh(draft)
        assert discarded is True
        assert draft.status == VOICE_DRAFT_STATUS_DISCARDED
        mock_delete.assert_called_once_with("local:audio.wav")
    finally:
        db.close()


def test_get_pending_voice_input_draft_expires_stale_row(db_session):
    db = db_session
    try:
        draft = create_voice_input_draft(
            db,
            user_id="user-1",
            conversation_id=None,
            user_audio_url="https://example.com/audio.wav",
            user_audio_storage_key="local:audio.wav",
            transcript_text="Hello there",
            detected_lang="en",
            transcript_source=VOICE_DRAFT_SOURCE_BACKEND_FINAL,
            warning=None,
        )
        draft.expires_at = datetime.utcnow() - timedelta(minutes=1)
        db.commit()

        with patch("app.services.voice_drafts.delete_stored_audio") as mock_delete:
            resolved = get_pending_voice_input_draft(db, "user-1", draft.id)

        db.refresh(draft)
        assert resolved is None
        assert draft.status == VOICE_DRAFT_STATUS_EXPIRED
        mock_delete.assert_called_once_with("local:audio.wav")
    finally:
        db.close()
