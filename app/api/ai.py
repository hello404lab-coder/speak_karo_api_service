"""AI chat endpoints. Sync inference (LLM, STT, TTS) runs in thread pool with timeouts."""
import asyncio
import base64
import hashlib
import json
import logging
import re
import threading
import uuid
from datetime import datetime
from typing import Literal, Optional
from io import BytesIO
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydub import AudioSegment
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.prompts import parse_gemini_response
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.dependencies.subscription import require_active_plan
from app.schemas.ai import TextChatRequest, AIChatResponse, TTSStreamRequest
from app.services.llm import finalize_llm_reply, generate_reply, init_llm_client, stream_gemini_tokens
from app.services.stt import transcribe_audio, init_stt_models
from app.services.translation import attach_translated_reply_text
from app.services.tts import text_to_speech_stream, store_audio_mp3, store_user_voice_wav, generate_tts_bytes, feed_tts_stream_to_queue, init_tts_models
from app.services.subscription_service import update_usage_stats
from app.utils.language import resolve_reply_language, resolve_translation_language
from app.models.usage import Conversation, Message
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
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
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
        "correction": ai_response.get("correction", "") or "",
        "explanation": ai_response.get("explanation") or None,
        "example": ai_response.get("example") or None,
        "score": score,
    }


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
    user_audio_url: Optional[str] = None,
) -> Message:
    """Persist one user/assistant exchange."""
    message = Message(
        conversation_id=conversation.id,
        user_message=user_message,
        ai_reply=ai_response["reply_text"],
        reply_language=reply_language,
        translated_ai_reply=ai_response.get("translated_reply_text"),
        translation_language_code=translation_language,
        correction=ai_response.get("correction", ""),
        hinglish_explanation=ai_response.get("explanation", ""),
        example=ai_response.get("example", ""),
        score=ai_response.get("score", 0),
        user_audio_url=user_audio_url,
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


def _build_ai_chat_response(
    ai_response: dict,
    conversation_id: str,
    reply_language: str,
    translation_language: Optional[str],
) -> AIChatResponse:
    """Build the sync response payload while preserving flat compatibility fields."""
    user_analysis = _build_user_analysis_payload(ai_response)
    return AIChatResponse(
        reply_text=ai_response["reply_text"],
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
        # Get or create conversation
        conversation = get_or_create_conversation(
            user_id, request.conversation_id, db, learner_context=request.learner_context
        )

        # Get conversation history for context
        history = get_conversation_history(conversation.id, db)

        reply_language, translation_language = _resolve_turn_languages(
            request.message,
            None,
            request.reply_language,
            request.response_language,
            request.translation_language,
            current_user,
        )

        # Run sync inference in thread pool with timeouts so event loop is not blocked
        try:
            ai_response = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_reply,
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
        )
        
        # Update usage stats
        update_usage_stats(user_id, db, 0.0, "chat")
        
        return _build_ai_chat_response(
            ai_response,
            conversation.id,
            reply_language,
            translation_language,
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
            ai_response = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_reply,
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
        
        update_usage_stats(user_id, db, 0.0, "voice")
        
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
    try:
        full = AudioSegment.empty()
        for b in chunks:
            full += AudioSegment.from_wav(BytesIO(b))
        out = BytesIO()
        full.export(out, format="mp3", bitrate="128k")
        full_bytes = out.getvalue()
        filename = f"{hashlib.md5(text.encode()).hexdigest()}.mp3"
        audio_url = store_audio_mp3(full_bytes, filename)
        return (audio_url, None)
    except Exception as e:
        logger.exception("TTS stream concatenation error")
        return (None, str(e))


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
    user_audio_url: Optional[str] = None,
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

    def gemini_producer() -> None:
        try:
            for token in stream_gemini_tokens(
                user_message,
                history,
                reply_language,
                translation_language,
                long_term_context=long_term_context,
            ):
                loop.call_soon_threadsafe(token_queue.put_nowait, token)
            loop.call_soon_threadsafe(token_queue.put_nowait, None)
        except Exception as e:
            logger.exception("Gemini stream error")
            loop.call_soon_threadsafe(main_queue.put_nowait, ("error", str(e)))

    async def buffer_consumer() -> None:
        buffer = ""
        in_reply = False
        json_reply_started = False
        full_reply_text_parts: list[str] = []
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
                            sentence_queue.put_nowait(cleaned)
                        buffer = buffer[idx + 1 :].lstrip()

            if buffer.strip() and in_reply:
                sent = _clean_sentence_for_tts(buffer.strip())
                if sent:
                    main_queue.put_nowait(("text", sent))
                    sentence_queue.put_nowait(sent)
            full_reply_text = "".join(full_reply_text_parts)
            main_queue.put_nowait(("full_text", full_reply_text))
            sentence_queue.put_nowait(None)
        except Exception as e:
            logger.exception("Buffer consumer error")
            main_queue.put_nowait(("error", str(e)))
            sentence_queue.put_nowait(None)

    async def tts_worker() -> None:
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

    threading.Thread(target=gemini_producer, daemon=True).start()
    buffer_task = asyncio.create_task(buffer_consumer())
    tts_task = asyncio.create_task(tts_worker())

    audio_chunks_collected: list[bytes] = []
    full_reply_text = ""

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
                b64 = base64.b64encode(item[1]).decode("ascii")
                yield f"event: audio_chunk\ndata: {b64}\n\n"
            elif item[0] == "full_text":
                full_reply_text = item[1] or ""

        await asyncio.gather(buffer_task, tts_task)
    except Exception as e:
        logger.exception("streaming pipeline error")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        return

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
            yield (
                f"event: metadata\ndata: "
                f"{json.dumps(_build_stream_metadata_payload(parsed, conversation.id, reply_language, translation_language))}\n\n"
            )
            _save_exchange_message(
                db,
                conversation,
                user_message,
                {
                    "reply_text": parsed.get("reply_text", ""),
                    "translated_reply_text": parsed.get("translated_reply_text"),
                    "correction": parsed.get("correction", ""),
                    "explanation": parsed.get("explanation", ""),
                    "example": parsed.get("example", ""),
                    "score": parsed.get("score", 75),
                },
                reply_language,
                translation_language,
                user_audio_url=user_audio_url,
            )
            update_usage_stats(user_id, db, 0.0, usage_type)
        except Exception as e:
            logger.exception("streaming pipeline save error: %s", e)

    if audio_chunks_collected:
        yield f"event: done\ndata: {json.dumps({'audio_url': None, 'saving_in_background': True})}\n\n"
        audio_url, err = await asyncio.to_thread(
            _concat_wav_chunks_and_store, audio_chunks_collected, full_reply_text
        )
        if err:
            yield f"event: error\ndata: {json.dumps({'error': err})}\n\n"
        else:
            yield f"event: audio_ready\ndata: {json.dumps({'audio_url': audio_url})}\n\n"
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
    SSE events: text_chunk, audio_chunk, metadata, done, audio_ready.
    """
    if not current_user.onboarding_completed:
        raise HTTPException(status_code=403, detail=ONBOARDING_REQUIRED_MESSAGE)
    user_id = current_user.id
    message = request.message.strip()
    conversation = get_or_create_conversation(
        user_id, request.conversation_id, db, learner_context=request.learner_context
    )
    history = get_conversation_history(conversation.id, db)
    reply_language, translation_language = _resolve_turn_languages(
        message,
        None,
        request.reply_language,
        request.response_language,
        request.translation_language,
        current_user,
    )

    return StreamingResponse(
        _llm_tts_streaming_pipeline(
            message,
            history,
            reply_language,
            translation_language,
            conversation,
            db,
            user_id,
            long_term_context=conversation.long_term_context,
            usage_type="chat",
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/tts/stream")
async def tts_stream(
    request: TTSStreamRequest,
    current_user: User = Depends(require_active_plan),
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

    threading.Thread(
        target=feed_tts_stream_to_queue,
        args=(text, response_language, queue, loop),
        daemon=True,
    ).start()

    async def event_gen():
        chunks = []
        while True:
            item = await queue.get()
            if item == (None, None) or (isinstance(item, tuple) and item[0] is None):
                break
            if isinstance(item, tuple) and item[0] == "error":
                yield f"event: error\ndata: {json.dumps({'error': item[1]})}\n\n"
                return
            if isinstance(item, tuple) and item[0] == "audio":
                chunks.append(item[1])
                b64 = base64.b64encode(item[1]).decode("ascii")
                yield f"event: audio_chunk\ndata: {b64}\n\n"

        if chunks:
            yield f"event: done\ndata: {json.dumps({'audio_url': None, 'saving_in_background': True})}\n\n"
            audio_url, err = await asyncio.to_thread(_concat_wav_chunks_and_store, chunks, text)
            if err:
                yield f"event: error\ndata: {json.dumps({'error': err})}\n\n"
            else:
                yield f"event: audio_ready\ndata: {json.dumps({'audio_url': audio_url})}\n\n"
        else:
            yield f"event: done\ndata: {json.dumps({'audio_url': None})}\n\n"

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
        ):
            yield chunk

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
