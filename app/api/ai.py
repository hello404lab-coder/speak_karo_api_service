"""AI chat endpoints. Sync inference (LLM, STT, TTS) runs in thread pool with timeouts."""
import asyncio
import base64
import hashlib
import json
import logging
import queue as queue_lib
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional
from io import BytesIO
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Response, status
from fastapi.responses import StreamingResponse
from pydub import AudioSegment
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.prompts import parse_gemini_response
from app.database import get_db
from app.dependencies.subscription import require_active_plan
from app.schemas.ai import TextChatRequest, AIChatResponse, TTSStreamRequest, VoiceDraftFinalizeResponse
from app.services.llm import (
    estimate_output_tokens_for_text,
    finalize_llm_reply,
    generate_reply_with_usage,
    init_llm_client,
    stream_gemini_tokens,
)
from app.services.stt import transcribe_audio, init_stt_models
from app.services.translation import attach_translated_reply_text
from app.services.tts import (
    CHIRP_STREAM_EVENT_RAW_PCM,
    chirp_streaming_enabled_for_text,
    chirp_pcm_to_wav,
    delete_stored_audio,
    feed_tts_stream_to_queue,
    feed_chirp_stream_to_queue,
    feed_smallest_stream_to_queue,
    _convert_wav_to_mp3,
    generate_tts_bytes,
    init_tts_models,
    resolve_stored_audio_playback_url,
    smallest_streaming_enabled_for_text,
    split_text_for_chirp_stream,
    split_text_for_smallest_stream,
    store_audio_mp3_record,
    store_user_voice_wav,
    store_user_voice_wav_record,
    CHIRP_DEFAULT_SAMPLE_RATE_HZ,
)
from app.services.voice_drafts import (
    VOICE_DRAFT_SOURCE_BACKEND_FINAL,
    VOICE_DRAFT_SOURCE_BROWSER_FALLBACK,
    VOICE_DRAFT_STATUS_CONSUMED,
    cleanup_expired_voice_drafts,
    create_voice_input_draft,
    discard_voice_input_draft,
    get_pending_voice_input_draft,
)
from app.services.subscription_service import apply_usage_delta, update_usage_stats
from app.utils.audio import wav_bytes_duration_seconds
from app.utils.language import normalize_language_code, resolve_reply_language, resolve_translation_language
from app.models.usage import Conversation, Message, VoiceInputDraft
from app.models.user import User

logger = logging.getLogger(__name__)

# User-safe message for timeout (no stack traces or internal detail)
TIMEOUT_MESSAGE = "Request took too long. Please try again."
ONBOARDING_REQUIRED_MESSAGE = "User onboarding not completed"

router = APIRouter()

# Max time for init-models (first-time load can be slow)
INIT_MODELS_TIMEOUT_SECONDS = 30000


# Max length for long_term_context (append when client sends learner_context on existing conversation)
LONG_TERM_CONTEXT_MAX_CHARS = 500


def get_or_create_conversation(
    user_id: str,
    conversation_id: Optional[str],
    db: Session,
    learner_context: Optional[str] = None,
) -> Conversation:
    """Get existing conversation or create new one. Optionally set or append learner_context."""
    if conversation_id:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
            .first()
        )
        if conversation:
            if learner_context and learner_context.strip():
                existing = (conversation.long_term_context or "").strip()
                if existing:
                    new_context = (existing + "\n" + learner_context.strip()).strip()[:LONG_TERM_CONTEXT_MAX_CHARS]
                else:
                    new_context = learner_context.strip()[:LONG_TERM_CONTEXT_MAX_CHARS]
                conversation.long_term_context = new_context or None
                db.commit()
                db.refresh(conversation)
            return conversation

    # Create new conversation
    new_id = str(uuid.uuid4())
    conversation = Conversation(id=new_id, user_id=user_id)
    if learner_context and learner_context.strip():
        conversation.long_term_context = learner_context.strip()[:LONG_TERM_CONTEXT_MAX_CHARS]
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    logger.info("conversation_created", extra={"conversation_id": conversation.id, "user_id": user_id})
    return conversation


def _require_owned_conversation(
    db: Session,
    user_id: str,
    conversation_id: str,
) -> Conversation:
    """Return an owned conversation or raise 404."""
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def get_conversation_history(conversation_id: str, db: Session) -> list:
    """Load last N exchanges from conversation (cap by count; LLM layer trims by token budget)."""
    limit = settings.llm_history_max_exchanges
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.desc()).limit(limit).all()

    # Reverse to get chronological order
    messages.reverse()

    history = []
    for msg in messages:
        history.append({
            "role": "user",
            "content": msg.user_message
        })
        history.append({
            "role": "assistant",
            "content": msg.ai_reply
        })

    return history


def _build_user_analysis_payload(ai_response: dict) -> dict:
    """Normalize learner feedback into the nested API shape."""
    score = ai_response.get("score", 75)
    if not isinstance(score, int):
        try:
            score = int(score) if score is not None else 75
        except (TypeError, ValueError):
            score = 75
    score = max(0, min(100, score))
    return {
        "correction": ai_response.get("correction") or None,
        "explanation": ai_response.get("explanation") or None,
        "example": ai_response.get("example") or None,
        "score": score,
    }


def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_turn_languages(
    text: str,
    detected_lang: Optional[str],
    reply_language_override: Optional[str],
    response_language_override: Optional[str],
    translation_language_override: Optional[str],
    current_user: User,
) -> tuple[str, Optional[str]]:
    """Resolve reply/TTS language and optional assistant translation language for the turn."""
    reply_language = resolve_reply_language(
        text,
        detected_lang,
        reply_language_override or response_language_override,
    )
    translation_language = resolve_translation_language(
        translation_language_override,
        current_user.native_language_code or current_user.native_language,
        reply_language,
    )
    return reply_language, translation_language


def _save_exchange_message(
    db: Session,
    conversation: Conversation,
    user_message: str,
    ai_response: dict,
    reply_language: str,
    translation_language: Optional[str],
    client_turn_id: Optional[str] = None,
    created_at: Optional[datetime] = None,
    user_audio_url: Optional[str] = None,
    voice_draft: Optional[VoiceInputDraft] = None,
) -> Message:
    """Persist one user/assistant exchange."""
    resolved_user_audio_url = user_audio_url
    if voice_draft is not None:
        if voice_draft.conversation_id is None:
            voice_draft.conversation_id = conversation.id
        voice_draft.status = VOICE_DRAFT_STATUS_CONSUMED
        voice_draft.consumed_at = datetime.utcnow()
        resolved_user_audio_url = resolve_stored_audio_playback_url(
            voice_draft.user_audio_storage_key,
            voice_draft.user_audio_url,
        )

    message = Message(
        conversation_id=conversation.id,
        client_turn_id=client_turn_id,
        user_message=user_message,
        ai_reply=ai_response["reply_text"],
        reply_language=reply_language,
        translated_ai_reply=ai_response.get("translated_reply_text"),
        translation_language_code=translation_language,
        correction=ai_response.get("correction"),
        hinglish_explanation=ai_response.get("explanation"),
        example=ai_response.get("example"),
        score=ai_response.get("score", 0),
        created_at=created_at or datetime.utcnow(),
        user_audio_url=resolved_user_audio_url,
    )
    db.add(message)
    conversation.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(message)
    logger.info(
        "Message saved for conversation %s",
        conversation.id,
        extra={"conversation_id": conversation.id, "message_id": message.id},
    )
    return message


def _persist_message_audio_storage_ref(
    db: Session,
    message: Optional[Message],
    storage_ref: Optional[str],
) -> None:
    """Persist a generated assistant reply audio storage ref for later replay reuse."""
    if message is None or not storage_ref:
        return
    message.ai_reply_audio_storage_ref = storage_ref
    db.commit()
    db.refresh(message)


def _resolve_voice_draft_for_send(
    db: Session,
    current_user: User,
    draft_id: Optional[str],
    requested_conversation_id: Optional[str],
) -> tuple[Optional[VoiceInputDraft], Optional[str]]:
    """Resolve a pending voice draft for a text send, validating ownership and conversation consistency."""
    cleanup_expired_voice_drafts(db)
    if not draft_id:
        return None, requested_conversation_id

    draft = get_pending_voice_input_draft(db, current_user.id, draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Voice draft not found or no longer available")

    if requested_conversation_id and draft.conversation_id and requested_conversation_id != draft.conversation_id:
        raise HTTPException(status_code=400, detail="Voice draft belongs to a different conversation")

    effective_conversation_id = requested_conversation_id or draft.conversation_id
    return draft, effective_conversation_id


def _build_ai_chat_response(
    ai_response: dict,
    conversation_id: str,
    reply_language: str,
    translation_language: Optional[str],
    client_turn_id: Optional[str] = None,
) -> AIChatResponse:
    """Build the sync response payload while preserving flat compatibility fields."""
    user_analysis = _build_user_analysis_payload(ai_response)
    return AIChatResponse(
        reply_text=ai_response["reply_text"],
        client_turn_id=client_turn_id,
        translated_reply_text=ai_response.get("translated_reply_text"),
        reply_language=reply_language,
        translation_language=translation_language,
        user_analysis=user_analysis,
        correction=user_analysis["correction"],
        explanation=user_analysis["explanation"],
        example=user_analysis["example"],
        score=user_analysis["score"],
        audio_url=None,
        response_language=reply_language,
        conversation_id=conversation_id,
    )


def _build_stream_metadata_payload(
    ai_response: dict,
    conversation_id: str,
    reply_language: str,
    translation_language: Optional[str],
    client_turn_id: Optional[str] = None,
) -> dict:
    """Build final SSE metadata payload for the completed turn."""
    user_analysis = _build_user_analysis_payload(ai_response)
    return {
        "translated_reply_text": ai_response.get("translated_reply_text"),
        "reply_language": reply_language,
        "translation_language": translation_language,
        "user_analysis": user_analysis,
        "correction": user_analysis["correction"],
        "explanation": user_analysis["explanation"],
        "example": user_analysis["example"],
        "score": user_analysis["score"],
        "conversation_id": conversation_id,
        "client_turn_id": client_turn_id,
        "response_language": reply_language,
    }


def _run_init_models_sync() -> dict:
    """Run all model initializers sequentially (called from thread)."""
    stt = init_stt_models()
    llm = init_llm_client()
    tts = init_tts_models()
    return {"stt": stt, "llm": llm, "tts": tts}


@router.post("/init-models")
async def init_models(
    # current_user: User = Depends(require_active_plan),
):
    """
    Initialize (warm up) all models: STT, LLM client, and TTS (local Turbo, Resemble API, IndicF5, or Gemini per config).
    Call this after startup to avoid cold-start latency on first user request.
    Runs in a thread with a 5-minute timeout.
    """
    # if not current_user.onboarding_completed:
        # raise HTTPException(status_code=403, detail=ONBOARDING_REQUIRED_MESSAGE)
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_run_init_models_sync),
            timeout=float(INIT_MODELS_TIMEOUT_SECONDS),
        )
        return result
    except asyncio.TimeoutError:
        logger.warning("init-models timed out")
        raise HTTPException(
            status_code=504,
            detail="Model initialization timed out. Try again or check server logs.",
        )


@router.post("/text-chat", response_model=AIChatResponse)
async def text_chat(
    request: TextChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_active_plan),
):
    """
    Text chat endpoint - accepts text message and returns AI reply.
    """
    if not current_user.onboarding_completed:
        raise HTTPException(status_code=403, detail=ONBOARDING_REQUIRED_MESSAGE)
    user_id = current_user.id
    try:
        voice_draft, effective_conversation_id = _resolve_voice_draft_for_send(
            db,
            current_user,
            request.voice_draft_id,
            request.conversation_id,
        )
        # Get or create conversation
        conversation = get_or_create_conversation(
            user_id, effective_conversation_id, db, learner_context=request.learner_context
        )

        # Get conversation history for context
        history = get_conversation_history(conversation.id, db)

        reply_language, translation_language = _resolve_turn_languages(
            request.message,
            voice_draft.detected_lang if voice_draft else None,
            request.reply_language,
            request.response_language,
            request.translation_language,
            current_user,
        )

        # Run sync inference in thread pool with timeouts so event loop is not blocked
        try:
            ai_response, llm_output_tokens = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_reply_with_usage,
                    request.message,
                    history,
                    reply_language,
                    translation_language,
                    long_term_context=conversation.long_term_context,
                ),
                timeout=float(settings.llm_timeout_seconds),
            )
        except asyncio.TimeoutError:
            logger.warning("LLM request timed out")
            raise HTTPException(status_code=504, detail=TIMEOUT_MESSAGE)
        _save_exchange_message(
            db,
            conversation,
            request.message,
            ai_response,
            reply_language,
            translation_language,
            client_turn_id=request.client_turn_id,
            voice_draft=voice_draft,
        )
        
        # Update usage stats
        update_usage_stats(
            user_id,
            db,
            0.0,
            "voice" if voice_draft else "chat",
            llm_output_tokens=llm_output_tokens,
        )
        
        return _build_ai_chat_response(
            ai_response,
            conversation.id,
            reply_language,
            translation_language,
            client_turn_id=request.client_turn_id,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in text_chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred processing your request.")


@router.post("/voice-chat", response_model=AIChatResponse)
async def voice_chat(
    user_id: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    learner_context: Optional[str] = Form(None),
    reply_language: Optional[str] = Form(None),
    translation_language: Optional[str] = Form(None),
    response_language: Optional[str] = Form(None),
    audio_file: UploadFile = File(...),
    stt_mode: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_active_plan),
):
    """
    Voice chat endpoint - accepts audio file and returns AI reply.
    """
    if not current_user.onboarding_completed:
        raise HTTPException(status_code=403, detail=ONBOARDING_REQUIRED_MESSAGE)
    user_id = current_user.id
    try:
        # Validate file type
        if not audio_file.filename.endswith('.wav'):
            raise HTTPException(status_code=400, detail="Only WAV files are supported.")
        
        # Read audio file
        audio_bytes = await audio_file.read()

        # Use server STT_MODE (env) as single source of truth so voice-chat respects STT_MODE=openai_whisper_large_v3
        effective_stt_mode = settings.stt_mode
        stt_input_audio_seconds = wav_bytes_duration_seconds(audio_bytes)
        try:
            transcribed_text, detected_lang = await asyncio.wait_for(
                asyncio.to_thread(
                    transcribe_audio,
                    audio_bytes,
                    audio_file.filename,
                    effective_stt_mode,
                ),
                timeout=float(settings.stt_timeout_seconds),
            )
        except asyncio.TimeoutError:
            logger.warning("STT request timed out")
            raise HTTPException(status_code=504, detail=TIMEOUT_MESSAGE)

        # Get or create conversation
        conversation = get_or_create_conversation(
            user_id, conversation_id, db, learner_context=learner_context
        )

        # Get conversation history
        history = get_conversation_history(conversation.id, db)

        reply_language_resolved, translation_language_resolved = _resolve_turn_languages(
            transcribed_text,
            detected_lang,
            reply_language,
            response_language,
            translation_language,
            current_user,
        )

        user_audio_url_sync: Optional[str] = None
        try:
            user_audio_url_sync = store_user_voice_wav(
                audio_bytes,
                f"user_voice_{conversation.id}_{uuid.uuid4().hex[:12]}.wav",
            )
        except Exception as e:
            logger.warning("User voice storage failed: %s", e)

        try:
            ai_response, llm_output_tokens = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_reply_with_usage,
                    transcribed_text,
                    history,
                    reply_language_resolved,
                    translation_language_resolved,
                    long_term_context=conversation.long_term_context,
                ),
                timeout=float(settings.llm_timeout_seconds),
            )
        except asyncio.TimeoutError:
            logger.warning("LLM request timed out")
            raise HTTPException(status_code=504, detail=TIMEOUT_MESSAGE)
        _save_exchange_message(
            db,
            conversation,
            transcribed_text,
            ai_response,
            reply_language_resolved,
            translation_language_resolved,
            user_audio_url=user_audio_url_sync,
        )
        
        update_usage_stats(
            user_id,
            db,
            0.0,
            "voice",
            llm_output_tokens=llm_output_tokens,
            stt_seconds=stt_input_audio_seconds,
        )
        
        return _build_ai_chat_response(
            ai_response,
            conversation.id,
            reply_language_resolved,
            translation_language_resolved,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in voice_chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred processing your request.")


@router.post("/voice-drafts/finalize", response_model=VoiceDraftFinalizeResponse)
async def finalize_voice_draft(
    user_id: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    browser_draft_text: Optional[str] = Form(None),
    browser_language: Optional[str] = Form(None),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_active_plan),
):
    """Store a recorded clip, run backend STT, and return a reusable voice draft for review-before-send."""
    del user_id  # auth user is authoritative

    if not current_user.onboarding_completed:
        raise HTTPException(status_code=403, detail=ONBOARDING_REQUIRED_MESSAGE)
    if not audio_file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")

    cleanup_expired_voice_drafts(db)

    if conversation_id:
        _require_owned_conversation(db, current_user.id, conversation_id)

    audio_bytes = await audio_file.read()
    stored_audio = await asyncio.to_thread(
        store_user_voice_wav_record,
        audio_bytes,
        f"user_voice_{current_user.id}_{uuid.uuid4().hex[:12]}.wav",
    )

    transcript_text: Optional[str] = None
    detected_lang: Optional[str] = None
    transcript_source = VOICE_DRAFT_SOURCE_BACKEND_FINAL
    warning: Optional[str] = None

    try:
        transcript_text, detected_lang = await asyncio.wait_for(
            asyncio.to_thread(
                transcribe_audio,
                audio_bytes,
                audio_file.filename,
                settings.stt_mode,
            ),
            timeout=float(settings.stt_timeout_seconds),
        )
        transcript_text = transcript_text.strip()
        detected_lang = normalize_language_code(detected_lang) or detected_lang
    except asyncio.TimeoutError:
        transcript_text = None
        warning = "Final transcription timed out. Please review the draft carefully."
    except ValueError as e:
        logger.warning("voice_draft_stt_failed: %s", e)
        transcript_text = None
        warning = "Final transcription unavailable. Please review the draft carefully."
    except Exception as e:
        logger.exception("voice_draft_stt_unexpected_error: %s", e)
        transcript_text = None
        warning = "Final transcription unavailable. Please review the draft carefully."

    browser_fallback_text = (browser_draft_text or "").strip()
    if not transcript_text:
        if not browser_fallback_text:
            delete_stored_audio(stored_audio.storage_ref)
            raise HTTPException(status_code=400, detail="Could not transcribe audio. Please try again.")
        transcript_text = browser_fallback_text
        detected_lang = normalize_language_code(browser_language)
        transcript_source = VOICE_DRAFT_SOURCE_BROWSER_FALLBACK
        warning = warning or "Final transcription unavailable. Please review the draft carefully."

    draft = create_voice_input_draft(
        db,
        user_id=current_user.id,
        conversation_id=conversation_id,
        user_audio_url=stored_audio.playback_url,
        user_audio_storage_key=stored_audio.storage_ref,
        transcript_text=transcript_text,
        detected_lang=detected_lang,
        transcript_source=transcript_source,
        warning=warning,
    )
    return VoiceDraftFinalizeResponse(
        voice_draft_id=draft.id,
        transcript_text=draft.transcript_text,
        detected_lang=draft.detected_lang,
        transcript_source=draft.transcript_source,
        warning=draft.warning,
    )


@router.delete("/voice-drafts/{draft_id}", status_code=status.HTTP_204_NO_CONTENT)
async def discard_voice_draft(
    draft_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_active_plan),
):
    """Discard a pending voice input draft without sending it to the AI."""
    cleanup_expired_voice_drafts(db)
    discard_voice_input_draft(db, current_user.id, draft_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Heartbeat interval for chat/stream SSE (keep connection alive during long LLM pauses)
SSE_HEARTBEAT_SECONDS = 0.5

# SSE headers for streaming TTS
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _concat_wav_chunks_and_store(chunks: list[bytes], text: str) -> tuple[Optional[str], Optional[str]]:
    """Sync helper: concatenate WAV chunks, export to MP3, store. Returns (audio_url, error_message)."""
    record, error = _concat_wav_chunks_and_store_record(chunks, text)
    if error:
        return (None, error)
    return (record.playback_url if record else None, None)


def _concat_wav_chunks_and_store_record(chunks: list[bytes], text: str):
    """Sync helper: concatenate WAV chunks, export to MP3, and return a durable stored-audio record."""
    try:
        full = AudioSegment.empty()
        for b in chunks:
            full += AudioSegment.from_wav(BytesIO(b))
        out = BytesIO()
        full.export(out, format="mp3", bitrate="128k")
        full_bytes = out.getvalue()
        filename = f"{hashlib.md5(text.encode()).hexdigest()}.mp3"
        record = store_audio_mp3_record(full_bytes, filename)
        return (record, None)
    except Exception as e:
        logger.exception("TTS stream concatenation error")
        return (None, str(e))


def _store_pcm_stream_and_store(audio_bytes: bytes, text: str) -> tuple[Optional[str], Optional[str]]:
    """Sync helper: convert streamed PCM to MP3 once and store it for replay."""
    record, error = _store_pcm_stream_and_store_record(audio_bytes, text)
    if error:
        return (None, error)
    return (record.playback_url if record else None, None)


def _store_pcm_stream_and_store_record(audio_bytes: bytes, text: str):
    """Sync helper: convert streamed PCM to MP3 once and return a durable stored-audio record."""
    try:
        filename = f"{hashlib.md5(text.encode()).hexdigest()}.mp3"
        wav_bytes = chirp_pcm_to_wav(audio_bytes)
        record = store_audio_mp3_record(_convert_wav_to_mp3(wav_bytes), filename)
        return (record, None)
    except Exception as e:
        logger.exception("TTS PCM stream storage error")
        return (None, str(e))


def _wav_chunk_duration_seconds(wav_bytes: bytes) -> float:
    """Best-effort duration for WAV chunk bytes."""
    return wav_bytes_duration_seconds(wav_bytes or b"")


def _pcm_duration_seconds(
    pcm_bytes: bytes,
    *,
    sample_rate_hz: int = CHIRP_DEFAULT_SAMPLE_RATE_HZ,
    channels: int = 1,
    sample_width_bytes: int = 2,
) -> float:
    """Duration for raw PCM payload."""
    if not pcm_bytes:
        return 0.0
    frame_width = max(1, int(channels) * int(sample_width_bytes))
    total_frames = len(pcm_bytes) / frame_width
    if sample_rate_hz <= 0:
        return 0.0
    return max(0.0, float(total_frames) / float(sample_rate_hz))


def _is_section_header(line: str) -> bool:
    """True if line starts with correction/hinglish/explanation/example (reply-only TTS boundary)."""
    lower = line.strip().lower()
    if not lower:
        return False
    return (
        lower.startswith("correction") or
        lower.startswith("hinglish") or
        lower.startswith("explanation") or
        lower.startswith("example") or
        lower.startswith("**correction") or
        lower.startswith("**hinglish") or
        lower.startswith("**explanation") or
        lower.startswith("**example")
    )


# Minimum chars before sending a sentence to TTS (avoids single-word fragments, improves prosody; ~18 prevents "Ok." from flushing alone)
_MIN_SENTENCE_CHARS = 18
_SENTENCE_BOUNDARIES = (".", "?", "!", "\u0964", "\n")  # Purna Viram (।) = \u0964


def _clean_sentence_for_tts(s: str) -> str:
    """Strip whitespace, quotes, and JSON cruft so TTS never sees characters that cause alignment pauses."""
    if not s:
        return ""
    s = s.strip().strip('"')
    # Remove internal quotes and trailing JSON/control chars
    s = s.replace('"', "").replace("}", "").replace("\n", " ").strip()
    return s


_CHIRP_MIN_FRAGMENT_CHARS = 36
_CHIRP_FRAGMENT_STRONG_BOUNDARIES = (".", "?", "!", ";", ":", "\n", "\u0964")
_CHIRP_FRAGMENT_WEAK_BOUNDARIES = (",", " ")


class _ReplyTextStreamExtractor:
    """Extract reply_text characters from the streamed JSON response without waiting for full completion."""

    marker = '"reply_text": "'

    def __init__(self) -> None:
        self._scan_buffer = ""
        self._json_started = False
        self._in_reply = False
        self._plain_text = False
        self._escaped = False
        self._done = False

    def feed(self, token: str) -> list[str]:
        if not token or self._done:
            return []
        if self._plain_text:
            return [token]

        if not self._json_started:
            self._scan_buffer += token
            marker_idx = self._scan_buffer.find(self.marker)
            if marker_idx >= 0:
                self._json_started = True
                self._in_reply = True
                tail = self._scan_buffer[marker_idx + len(self.marker):]
                self._scan_buffer = ""
                return self._consume_reply_chars(tail)
            stripped = self._scan_buffer.lstrip()
            if stripped and not stripped.startswith("{"):
                out = self._scan_buffer
                self._scan_buffer = ""
                self._plain_text = True
                return [out]
            if len(self._scan_buffer) > len(self.marker) * 2:
                self._scan_buffer = self._scan_buffer[-len(self.marker) :]
            return []

        if self._in_reply:
            return self._consume_reply_chars(token)
        return []

    def flush(self) -> list[str]:
        if self._plain_text and self._scan_buffer:
            out = self._scan_buffer
            self._scan_buffer = ""
            return [out]
        return []

    def _consume_reply_chars(self, text: str) -> list[str]:
        out: list[str] = []
        for ch in text:
            if self._escaped:
                mapping = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
                out.append(mapping.get(ch, ch))
                self._escaped = False
                continue
            if ch == "\\":
                self._escaped = True
                continue
            if ch == '"':
                self._in_reply = False
                self._done = True
                break
            out.append(ch)
        return ["".join(out)] if out else []


def _pop_chirp_ready_fragments(buffer: str, force: bool = False) -> tuple[list[str], str]:
    """Flush small phrase chunks for Chirp while keeping display text sentence-based."""
    ready: list[str] = []
    working = buffer
    while working:
        if force:
            flushed = working.strip()
            if flushed:
                ready.append(flushed)
            return ready, ""

        condensed = working.strip()
        if len(condensed) < _CHIRP_MIN_FRAGMENT_CHARS:
            return ready, working

        strong_split_at = max(working.rfind(sep) for sep in _CHIRP_FRAGMENT_STRONG_BOUNDARIES)
        weak_split_at = max(working.rfind(sep) for sep in _CHIRP_FRAGMENT_WEAK_BOUNDARIES)
        split_at = strong_split_at
        if split_at <= 0 and len(condensed) >= (_CHIRP_MIN_FRAGMENT_CHARS * 2):
            split_at = weak_split_at
        if split_at <= 0:
            return ready, working

        fragment = working[: split_at + 1].strip()
        if len(fragment) < _CHIRP_MIN_FRAGMENT_CHARS:
            return ready, working
        ready.append(fragment)
        working = working[split_at + 1 :].lstrip()
    return ready, working


async def _llm_tts_streaming_pipeline(
    user_message: str,
    history: list,
    reply_language: str,
    translation_language: Optional[str],
    conversation: Conversation,
    db: Session,
    user_id: str,
    long_term_context: Optional[str] = None,
    usage_type: Literal["chat", "voice"] = "chat",
    client_turn_id: Optional[str] = None,
    created_at: Optional[datetime] = None,
    user_audio_url: Optional[str] = None,
    voice_draft: Optional[VoiceInputDraft] = None,
    stt_seconds: float = 0.0,
    include_audio_stream: bool = True,
):
    """
    Reusable async generator: LLM stream -> sentence buffer -> per-sentence TTS -> SSE events.
    Only reply_text is streamed to TTS; correction/score emitted via a metadata event.

    Yields SSE-formatted strings:
      text_chunk, audio_chunk, metadata, done, audio_ready, error, : keep-alive
    """
    token_queue: asyncio.Queue = asyncio.Queue()
    sentence_queue: asyncio.Queue = asyncio.Queue()
    main_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    use_smallest_stream = include_audio_stream and smallest_streaming_enabled_for_text("", reply_language)
    use_chirp_stream = include_audio_stream and chirp_streaming_enabled_for_text("", reply_language)
    use_cloud_tts_streaming = include_audio_stream and (use_smallest_stream or use_chirp_stream)
    chirp_fragment_queue: Optional[queue_lib.Queue] = None
    chirp_stop_event: Optional[threading.Event] = None
    chirp_started = False
    llm_output_tokens_accumulated = 0
    tts_seconds_accumulated = 0.0
    llm_stream_usage: dict[str, int] = {"output_tokens": 0}

    def _start_cloud_tts_stream_worker_if_needed() -> None:
        nonlocal chirp_started, chirp_fragment_queue, chirp_stop_event
        if not use_cloud_tts_streaming or chirp_started or chirp_fragment_queue is None or chirp_stop_event is None:
            return
        target = (
            feed_smallest_stream_to_queue
            if use_smallest_stream
            else feed_chirp_stream_to_queue
        )
        threading.Thread(
            target=target,
            args=(chirp_fragment_queue, reply_language, main_queue, loop, chirp_stop_event),
            daemon=True,
        ).start()
        chirp_started = True

    def gemini_producer() -> None:
        nonlocal llm_output_tokens_accumulated
        try:
            for token in stream_gemini_tokens(
                user_message,
                history,
                reply_language,
                translation_language,
                long_term_context=long_term_context,
                usage_sink=llm_stream_usage,
            ):
                loop.call_soon_threadsafe(token_queue.put_nowait, token)
            loop.call_soon_threadsafe(token_queue.put_nowait, None)
        except Exception as e:
            logger.exception("Gemini stream error")
            loop.call_soon_threadsafe(main_queue.put_nowait, ("error", str(e)))
        llm_output_tokens_accumulated = max(0, int(llm_stream_usage.get("output_tokens", 0) or 0))

    async def buffer_consumer() -> None:
        buffer = ""
        in_reply = False
        json_reply_started = False
        full_reply_text_parts: list[str] = []
        chirp_extractor = _ReplyTextStreamExtractor() if use_cloud_tts_streaming else None
        chirp_pending = ""
        try:
            while True:
                try:
                    token = await asyncio.wait_for(token_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if token is None:
                    break
                full_reply_text_parts.append(token)
                buffer += token

                if use_cloud_tts_streaming and chirp_extractor is not None and chirp_fragment_queue is not None:
                    for extracted in chirp_extractor.feed(token):
                        chirp_pending += extracted
                        ready_fragments, chirp_pending = _pop_chirp_ready_fragments(chirp_pending)
                        for fragment in ready_fragments:
                            _start_cloud_tts_stream_worker_if_needed()
                            chirp_fragment_queue.put_nowait(fragment)

                if not json_reply_started and '"reply_text": "' in buffer:
                    json_reply_started = True
                    in_reply = True
                    buffer = buffer.split('"reply_text": "')[-1]
                elif not json_reply_started and buffer.strip() and not buffer.strip().startswith("{"):
                    in_reply = True

                if not in_reply:
                    continue

                # Intra-quote flushing: when inside JSON reply_text, flush on sentence boundaries
                if json_reply_started:
                    while True:
                        idx = -1
                        for sep in _SENTENCE_BOUNDARIES:
                            i = buffer.find(sep)
                            if i >= 0 and (idx < 0 or i < idx):
                                idx = i
                        if idx < 0:
                            break
                        segment = buffer[: idx + 1].strip()
                        if len(segment) < _MIN_SENTENCE_CHARS:
                            break
                        cleaned = _clean_sentence_for_tts(segment)
                        if cleaned:
                            main_queue.put_nowait(("text", cleaned))
                            if include_audio_stream:
                                sentence_queue.put_nowait(cleaned)
                        buffer = buffer[idx + 1 :].lstrip()
                    # Closing quote: flush any remaining content before the quote, then exit reply
                    if buffer.startswith('"'):
                        in_reply = False
                        buffer = buffer[1:].lstrip()
                        continue
                    end_quote_idx = buffer.find('"')
                    if end_quote_idx >= 0:
                        segment = buffer[:end_quote_idx].strip()
                        cleaned = _clean_sentence_for_tts(segment)
                        if cleaned:
                            main_queue.put_nowait(("text", cleaned))
                            if include_audio_stream:
                                sentence_queue.put_nowait(cleaned)
                        in_reply = False
                        buffer = buffer[end_quote_idx + 1 :].lstrip()
                    continue

                # Plain-text reply path: sentence-boundary flush
                idx = -1
                for sep in _SENTENCE_BOUNDARIES:
                    i = buffer.find(sep)
                    if i >= 0 and (idx < 0 or i < idx):
                        idx = i
                if idx >= 0:
                    sentence = buffer[: idx + 1].strip()
                    if len(sentence) >= _MIN_SENTENCE_CHARS:
                        if not json_reply_started:
                            for line in sentence.split("\n"):
                                if _is_section_header(line.strip()):
                                    in_reply = False
                                    break
                            if not in_reply:
                                continue
                        cleaned = _clean_sentence_for_tts(sentence)
                        if cleaned:
                            main_queue.put_nowait(("text", cleaned))
                            if include_audio_stream:
                                sentence_queue.put_nowait(cleaned)
                        buffer = buffer[idx + 1 :].lstrip()

            if buffer.strip() and in_reply:
                sent = _clean_sentence_for_tts(buffer.strip())
                if sent:
                    main_queue.put_nowait(("text", sent))
                    if not use_cloud_tts_streaming:
                        sentence_queue.put_nowait(sent)
            if use_cloud_tts_streaming and chirp_extractor is not None and chirp_fragment_queue is not None:
                for extracted in chirp_extractor.flush():
                    chirp_pending += extracted
                ready_fragments, chirp_pending = _pop_chirp_ready_fragments(chirp_pending, force=True)
                for fragment in ready_fragments:
                    _start_cloud_tts_stream_worker_if_needed()
                    chirp_fragment_queue.put_nowait(fragment)
            full_reply_text = "".join(full_reply_text_parts)
            main_queue.put_nowait(("full_text", full_reply_text))
            if use_cloud_tts_streaming and chirp_fragment_queue is not None:
                if chirp_started:
                    chirp_fragment_queue.put_nowait(None)
                else:
                    main_queue.put_nowait((None, None))
            else:
                if include_audio_stream:
                    sentence_queue.put_nowait(None)
                else:
                    main_queue.put_nowait((None, None))
        except Exception as e:
            logger.exception("Buffer consumer error")
            main_queue.put_nowait(("error", str(e)))
            if use_cloud_tts_streaming and chirp_fragment_queue is not None:
                try:
                    if chirp_started:
                        chirp_fragment_queue.put_nowait(None)
                except Exception:
                    pass
            else:
                if include_audio_stream:
                    sentence_queue.put_nowait(None)
                else:
                    main_queue.put_nowait((None, None))

    async def tts_worker() -> None:
        nonlocal tts_seconds_accumulated
        try:
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    main_queue.put_nowait((None, None))
                    return
                try:
                    wav_bytes = await asyncio.to_thread(generate_tts_bytes, sentence, reply_language)
                    main_queue.put_nowait(("audio", wav_bytes))
                except Exception as e:
                    logger.exception("TTS worker error")
                    main_queue.put_nowait(("error", str(e)))
                    return
        except Exception as e:
            logger.exception("TTS worker error")
            main_queue.put_nowait(("error", str(e)))

    if use_cloud_tts_streaming:
        chirp_fragment_queue = queue_lib.Queue()
        chirp_stop_event = threading.Event()

    threading.Thread(target=gemini_producer, daemon=True).start()
    buffer_task = asyncio.create_task(buffer_consumer())
    tts_task = asyncio.create_task(tts_worker()) if include_audio_stream and not use_cloud_tts_streaming else None

    audio_chunks_collected: list[bytes] = []
    raw_pcm_audio: Optional[bytes] = None
    full_reply_text = ""
    saved_message: Optional[Message] = None

    try:
        while True:
            try:
                item = await asyncio.wait_for(main_queue.get(), timeout=SSE_HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if item[0] == "error":
                yield f"event: error\ndata: {json.dumps({'error': item[1]})}\n\n"
                return
            if item[0] is None:
                break
            if item[0] == "text":
                yield f"event: text_chunk\ndata: {json.dumps({'text': item[1]})}\n\n"
            elif item[0] == "audio":
                audio_chunks_collected.append(item[1])
                tts_seconds_accumulated += _wav_chunk_duration_seconds(item[1])
                b64 = base64.b64encode(item[1]).decode("ascii")
                yield f"event: audio_chunk\ndata: {b64}\n\n"
            elif item[0] == CHIRP_STREAM_EVENT_RAW_PCM:
                raw_pcm_audio = item[1] or b""
            elif item[0] == "full_text":
                full_reply_text = item[1] or ""

        if tts_task is not None:
            await asyncio.gather(buffer_task, tts_task)
        else:
            await buffer_task
    except Exception as e:
        logger.exception("streaming pipeline error")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        return
    finally:
        if chirp_stop_event is not None:
            chirp_stop_event.set()
        if chirp_fragment_queue is not None:
            try:
                chirp_fragment_queue.put_nowait(None)
            except Exception:
                pass

    # Parse full LLM response -> metadata event -> DB save
    if full_reply_text:
        try:
            parsed_base = await asyncio.to_thread(
                finalize_llm_reply,
                parse_gemini_response(full_reply_text),
                user_message,
                reply_language,
                translation_language,
            )
            parsed = await asyncio.to_thread(
                attach_translated_reply_text,
                parsed_base,
                reply_language,
                translation_language,
            )
            if llm_output_tokens_accumulated <= 0:
                llm_output_tokens_accumulated = await asyncio.to_thread(
                    estimate_output_tokens_for_text,
                    full_reply_text,
                )
            yield (
                f"event: metadata\ndata: "
                f"{json.dumps(_build_stream_metadata_payload(parsed, conversation.id, reply_language, translation_language, client_turn_id=client_turn_id))}\n\n"
            )
            saved_message = _save_exchange_message(
                db,
                conversation,
                user_message,
                {
                    "reply_text": parsed.get("reply_text", ""),
                    "translated_reply_text": parsed.get("translated_reply_text"),
                    "correction": parsed.get("correction"),
                    "explanation": parsed.get("explanation"),
                    "example": parsed.get("example"),
                    "score": parsed.get("score", 75),
                },
                reply_language,
                translation_language,
                client_turn_id=client_turn_id,
                created_at=created_at,
                user_audio_url=user_audio_url,
                voice_draft=voice_draft,
            )
            update_usage_stats(
                user_id,
                db,
                0.0,
                usage_type,
                llm_output_tokens=llm_output_tokens_accumulated,
                stt_seconds=stt_seconds,
                tts_seconds=tts_seconds_accumulated,
            )
        except Exception as e:
            logger.exception("streaming pipeline save error: %s", e)

    if include_audio_stream and audio_chunks_collected:
        yield f"event: done\ndata: {json.dumps({'audio_url': None, 'saving_in_background': True})}\n\n"
        if raw_pcm_audio:
            audio_record, err = await asyncio.to_thread(
                _store_pcm_stream_and_store_record, raw_pcm_audio, full_reply_text
            )
        else:
            audio_record, err = await asyncio.to_thread(
                _concat_wav_chunks_and_store_record, audio_chunks_collected, full_reply_text
            )
        if err:
            yield f"event: error\ndata: {json.dumps({'error': err})}\n\n"
        else:
            _persist_message_audio_storage_ref(
                db,
                saved_message,
                audio_record.storage_ref if audio_record else None,
            )
            yield f"event: audio_ready\ndata: {json.dumps({'audio_url': audio_record.playback_url if audio_record else None})}\n\n"
    else:
        yield f"event: done\ndata: {json.dumps({'audio_url': None})}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    request: TextChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_active_plan),
):
    """
    Sentence-level pipelined chat (text input): LLM stream -> sentence buffer -> TTS per sentence.
    SSE events: turn_ack, text_chunk, audio_chunk, metadata, done, audio_ready.
    """
    if not current_user.onboarding_completed:
        raise HTTPException(status_code=403, detail=ONBOARDING_REQUIRED_MESSAGE)
    user_id = current_user.id
    message = request.message.strip()
    voice_draft, effective_conversation_id = _resolve_voice_draft_for_send(
        db,
        current_user,
        request.voice_draft_id,
        request.conversation_id,
    )
    conversation = get_or_create_conversation(
        user_id, effective_conversation_id, db, learner_context=request.learner_context
    )
    history = get_conversation_history(conversation.id, db)
    reply_language, translation_language = _resolve_turn_languages(
        message,
        voice_draft.detected_lang if voice_draft else None,
        request.reply_language,
        request.response_language,
        request.translation_language,
        current_user,
    )
    turn_created_at = datetime.now(timezone.utc)
    turn_ack_payload = {
        "conversation_id": conversation.id,
        "client_turn_id": request.client_turn_id,
        "created_at": _iso_z(turn_created_at),
        "reply_language": reply_language,
        "translation_language": translation_language,
    }

    async def event_gen():
        yield f"event: turn_ack\ndata: {json.dumps(turn_ack_payload)}\n\n"
        async for chunk in _llm_tts_streaming_pipeline(
            message,
            history,
            reply_language,
            translation_language,
            conversation,
            db,
            user_id,
            long_term_context=conversation.long_term_context,
            usage_type="voice" if voice_draft else "chat",
            client_turn_id=request.client_turn_id,
            created_at=turn_created_at.replace(tzinfo=None),
            voice_draft=voice_draft,
            include_audio_stream=request.include_audio_stream,
        ):
            yield chunk

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/tts/stream")
async def tts_stream(
    request: TTSStreamRequest,
    current_user: User = Depends(require_active_plan),
    db: Session = Depends(get_db),
):
    """
    Stream TTS audio over Server-Sent Events (SSE).
    Producer (thread): feed_tts_stream_to_queue puts ("audio", chunk) per sentence, then (None, None) sentinel.
    Consumer (async gen): yields event: audio_chunk as soon as ("audio", chunk) is received; done and audio_ready
    only after queue is exhausted. media_type is text/event-stream.
    """
    if not current_user.onboarding_completed:
        raise HTTPException(status_code=403, detail=ONBOARDING_REQUIRED_MESSAGE)
    text = request.text.strip()
    response_language = request.response_language or "en"
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    use_smallest_stream = smallest_streaming_enabled_for_text(text, response_language)
    use_chirp_stream = chirp_streaming_enabled_for_text(text, response_language)
    use_cloud_tts_streaming = use_smallest_stream or use_chirp_stream
    chirp_fragment_queue: Optional[queue_lib.Queue] = None
    chirp_stop_event: Optional[threading.Event] = None

    if use_cloud_tts_streaming:
        chirp_fragment_queue = queue_lib.Queue()
        chirp_stop_event = threading.Event()
        if use_smallest_stream:
            for fragment in split_text_for_smallest_stream(text):
                chirp_fragment_queue.put_nowait(fragment)
            chirp_fragment_queue.put_nowait(None)
            threading.Thread(
                target=feed_smallest_stream_to_queue,
                args=(chirp_fragment_queue, response_language, queue, loop, chirp_stop_event),
                daemon=True,
            ).start()
        else:
            for fragment in split_text_for_chirp_stream(text):
                chirp_fragment_queue.put_nowait(fragment)
            chirp_fragment_queue.put_nowait(None)
            threading.Thread(
                target=feed_chirp_stream_to_queue,
                args=(chirp_fragment_queue, response_language, queue, loop, chirp_stop_event),
                daemon=True,
            ).start()
    else:
        threading.Thread(
            target=feed_tts_stream_to_queue,
            args=(text, response_language, queue, loop),
            daemon=True,
        ).start()

    async def event_gen():
        chunks = []
        raw_pcm_audio: Optional[bytes] = None
        had_error = False
        tts_generated_seconds = 0.0
        try:
            while True:
                item = await queue.get()
                if item == (None, None) or (isinstance(item, tuple) and item[0] is None):
                    break
                if isinstance(item, tuple) and item[0] == "error":
                    had_error = True
                    yield f"event: error\ndata: {json.dumps({'error': item[1]})}\n\n"
                    return
                if isinstance(item, tuple) and item[0] == "audio":
                    chunks.append(item[1])
                    tts_generated_seconds += _wav_chunk_duration_seconds(item[1])
                    b64 = base64.b64encode(item[1]).decode("ascii")
                    yield f"event: audio_chunk\ndata: {b64}\n\n"
                if isinstance(item, tuple) and item[0] == CHIRP_STREAM_EVENT_RAW_PCM:
                    raw_pcm_audio = item[1] or b""

            if chunks:
                yield f"event: done\ndata: {json.dumps({'audio_url': None, 'saving_in_background': True})}\n\n"
                if raw_pcm_audio:
                    audio_url, err = await asyncio.to_thread(_store_pcm_stream_and_store, raw_pcm_audio, text)
                else:
                    audio_url, err = await asyncio.to_thread(_concat_wav_chunks_and_store, chunks, text)
                if err:
                    yield f"event: error\ndata: {json.dumps({'error': err})}\n\n"
                else:
                    yield f"event: audio_ready\ndata: {json.dumps({'audio_url': audio_url})}\n\n"
            else:
                yield f"event: done\ndata: {json.dumps({'audio_url': None})}\n\n"
        finally:
            if chirp_stop_event is not None:
                chirp_stop_event.set()
            if not had_error:
                apply_usage_delta(
                    current_user.id,
                    db,
                    request_delta=1,
                    tts_seconds_delta=max(0.0, float(tts_generated_seconds)),
                    chat_delta=0,
                    voice_delta=0,
                    commit=True,
                )

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/voice-chat/stream")
async def voice_chat_stream(
    user_id: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    learner_context: Optional[str] = Form(None),
    reply_language: Optional[str] = Form(None),
    translation_language: Optional[str] = Form(None),
    response_language: Optional[str] = Form(None),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_active_plan),
):
    """
    Sentence-level pipelined voice chat: STT (Groq) -> LLM stream -> sentence buffer -> TTS per sentence.
    SSE events: stt_result, text_chunk, audio_chunk, metadata, done, audio_ready.
    """
    if not current_user.onboarding_completed:
        raise HTTPException(status_code=403, detail=ONBOARDING_REQUIRED_MESSAGE)
    user_id = current_user.id
    if not audio_file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")

    audio_bytes = await audio_file.read()
    effective_stt_mode = settings.stt_mode
    stt_input_audio_seconds = wav_bytes_duration_seconds(audio_bytes)

    try:
        transcribed_text, detected_lang = await asyncio.wait_for(
            asyncio.to_thread(transcribe_audio, audio_bytes, audio_file.filename, effective_stt_mode),
            timeout=float(settings.stt_timeout_seconds),
        )
    except asyncio.TimeoutError:
        logger.warning("STT request timed out")
        raise HTTPException(status_code=504, detail=TIMEOUT_MESSAGE)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    reply_language_resolved, translation_language_resolved = _resolve_turn_languages(
        transcribed_text,
        detected_lang,
        reply_language,
        response_language,
        translation_language,
        current_user,
    )
    conversation = get_or_create_conversation(user_id, conversation_id, db, learner_context=learner_context)
    history = get_conversation_history(conversation.id, db)

    user_audio_url: Optional[str] = None
    try:
        user_audio_url = await asyncio.to_thread(
            store_user_voice_wav,
            audio_bytes,
            f"user_voice_{conversation.id}_{uuid.uuid4().hex[:12]}.wav",
        )
    except Exception as e:
        logger.warning("User voice storage failed: %s", e)

    async def event_gen():
        yield (
            f"event: stt_result\ndata: "
            f"{json.dumps({'text': transcribed_text, 'detected_lang': detected_lang, 'response_language': reply_language_resolved, 'reply_language': reply_language_resolved, 'translation_language': translation_language_resolved})}\n\n"
        )
        async for chunk in _llm_tts_streaming_pipeline(
            transcribed_text,
            history,
            reply_language_resolved,
            translation_language_resolved,
            conversation,
            db,
            user_id,
            long_term_context=conversation.long_term_context,
            usage_type="voice",
            user_audio_url=user_audio_url,
            stt_seconds=stt_input_audio_seconds,
            include_audio_stream=True,
        ):
            yield chunk

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
