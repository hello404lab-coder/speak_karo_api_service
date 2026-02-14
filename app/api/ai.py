"""AI chat endpoints. Sync inference (LLM, STT, TTS) runs in thread pool with timeouts."""
import asyncio
import base64
import hashlib
import json
import logging
import re
import threading
import uuid
from datetime import date
from io import BytesIO
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydub import AudioSegment
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.prompts import parse_gemini_response
from app.database import get_db
from app.schemas.ai import TextChatRequest, AIChatResponse, TTSStreamRequest
from app.services.llm import generate_reply, stream_gemini_tokens, init_llm_client
from app.services.stt import transcribe_audio, init_stt_models
from app.services.tts import text_to_speech_stream, store_audio_mp3, generate_tts_bytes, feed_tts_stream_to_queue, init_tts_models
from app.utils.language import get_response_language
from app.models.usage import Conversation, Message, Usage

logger = logging.getLogger(__name__)

# User-safe message for timeout (no stack traces or internal detail)
TIMEOUT_MESSAGE = "Request took too long. Please try again."

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


def update_usage_stats(user_id: str, db: Session, duration_seconds: float = 0.0):
    """Update daily usage statistics."""
    today = date.today()
    
    usage = db.query(Usage).filter(
        Usage.user_id == user_id,
        Usage.date == today
    ).first()
    
    if not usage:
        usage = Usage(user_id=user_id, date=today, minutes_used=0.0, request_count=0)
        db.add(usage)
    
    usage.request_count += 1
    usage.minutes_used += duration_seconds / 60.0
    
    db.commit()


def _run_init_models_sync() -> dict:
    """Run all model initializers sequentially (called from thread)."""
    stt = init_stt_models()
    llm = init_llm_client()
    tts = init_tts_models()
    return {"stt": stt, "llm": llm, "tts": tts}


@router.post("/init-models")
async def init_models():
    """
    Initialize (warm up) all models: STT, LLM client, and TTS (Turbo + IndicF5 if configured).
    Call this after startup to avoid cold-start latency on first user request.
    Runs in a thread with a 5-minute timeout.
    """
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
    db: Session = Depends(get_db)
):
    """
    Text chat endpoint - accepts text message and returns AI reply.
    """
    try:
        # Get or create conversation
        conversation = get_or_create_conversation(
            request.user_id, request.conversation_id, db, learner_context=request.learner_context
        )

        # Get conversation history for context
        history = get_conversation_history(conversation.id, db)

        # Response language from text (script detection; no STT)
        response_language = get_response_language(request.message, None)

        # Run sync inference in thread pool with timeouts so event loop is not blocked
        try:
            ai_response = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_reply,
                    request.message,
                    history,
                    response_language,
                    long_term_context=conversation.long_term_context,
                ),
                timeout=float(settings.llm_timeout_seconds),
            )
        except asyncio.TimeoutError:
            logger.warning("LLM request timed out")
            raise HTTPException(status_code=504, detail=TIMEOUT_MESSAGE)

        # Save message to database
        message = Message(
            conversation_id=conversation.id,
            user_message=request.message,
            ai_reply=ai_response["reply_text"],
            correction=ai_response.get("correction", ""),
            hinglish_explanation="",
            score=ai_response.get("score", 0)
        )
        db.add(message)
        db.commit()
        
        # Update usage stats
        update_usage_stats(request.user_id, db)
        
        return AIChatResponse(
            reply_text=ai_response["reply_text"],
            correction=ai_response.get("correction", ""),
            score=ai_response.get("score", 75),
            audio_url=None,
            response_language=response_language,
            conversation_id=conversation.id
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
    audio_file: UploadFile = File(...),
    stt_mode: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Voice chat endpoint - accepts audio file and returns AI reply.
    """
    try:
        # Validate file type
        if not audio_file.filename.endswith('.wav'):
            raise HTTPException(status_code=400, detail="Only WAV files are supported.")
        
        # Read audio file
        audio_bytes = await audio_file.read()

        effective_stt_mode = stt_mode if stt_mode in ("faster_whisper_medium", "faster_whisper_large") else settings.stt_mode
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

        # Response language from STT + script detection (en -> Chatterbox, hi/ml/ta -> IndicF5)
        response_language = get_response_language(transcribed_text, detected_lang)

        try:
            ai_response = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_reply,
                    transcribed_text,
                    history,
                    response_language,
                    long_term_context=conversation.long_term_context,
                ),
                timeout=float(settings.llm_timeout_seconds),
            )
        except asyncio.TimeoutError:
            logger.warning("LLM request timed out")
            raise HTTPException(status_code=504, detail=TIMEOUT_MESSAGE)

        # Save message to database
        message = Message(
            conversation_id=conversation.id,
            user_message=transcribed_text,
            ai_reply=ai_response["reply_text"],
            correction=ai_response.get("correction", ""),
            hinglish_explanation="",
            score=ai_response.get("score", 0)
        )
        db.add(message)
        db.commit()
        
        update_usage_stats(user_id, db, 0.0)
        
        return AIChatResponse(
            reply_text=ai_response["reply_text"],
            correction=ai_response.get("correction", ""),
            score=ai_response.get("score", 75),
            audio_url=None,
            response_language=response_language,
            conversation_id=conversation.id
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
    """True if line starts with correction/hinglish/explanation (reply-only TTS boundary)."""
    lower = line.strip().lower()
    if not lower:
        return False
    return (
        lower.startswith("correction") or
        lower.startswith("hinglish") or
        lower.startswith("explanation") or
        lower.startswith("**correction") or
        lower.startswith("**hinglish") or
        lower.startswith("**explanation")
    )


@router.post("/chat/stream")
async def chat_stream(
    request: TextChatRequest,
    db: Session = Depends(get_db),
):
    """
    Sentence-level pipelined chat: stream Gemini tokens, buffer into sentences, TTS per sentence.
    Only reply_text is streamed and sent to TTS; correction is never spoken.
    SSE: text_chunk (sentence for display), audio_chunk (WAV), then done, then audio_ready.
    Heartbeat every 500ms when waiting for next event.
    """
    user_id = request.user_id
    message = request.message.strip()
    conversation_id = request.conversation_id
    conversation = get_or_create_conversation(
        user_id, conversation_id, db, learner_context=request.learner_context
    )
    history = get_conversation_history(conversation.id, db)
    response_language = get_response_language(message, None)

    token_queue: asyncio.Queue = asyncio.Queue()
    sentence_queue: asyncio.Queue = asyncio.Queue()
    main_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def gemini_producer() -> None:
        try:
            for token in stream_gemini_tokens(
                message, history, response_language, long_term_context=conversation.long_term_context
            ):
                loop.call_soon_threadsafe(token_queue.put_nowait, token)
            loop.call_soon_threadsafe(token_queue.put_nowait, None)
        except Exception as e:
            logger.exception("Gemini stream error")
            loop.call_soon_threadsafe(main_queue.put_nowait, ("error", str(e)))

    # Minimum chars before sending a sentence to TTS (avoids single-word fragments, improves prosody)
    MIN_SENTENCE_CHARS = 25
    SENTENCE_BOUNDARIES = (".", "?", "\u0964", "\n")  # Purna Viram (।) = \u0964

    async def buffer_consumer() -> None:
        # Only reply_text is streamed and sent to TTS; correction is never pushed.
        buffer = ""
        in_reply = False  # Only True after we see reply_text value (JSON) or detect free-form (no leading {)
        json_reply_started = False  # True after we strip "reply_text": " once
        full_reply_text_parts = []
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

                # JSON: strip "reply_text": " prefix once so TTS/stream never see other keys (e.g. correction)
                if not json_reply_started and '"reply_text": "' in buffer:
                    json_reply_started = True
                    in_reply = True
                    buffer = buffer.split('"reply_text": "')[-1]
                # Free-form: no JSON prefix; treat as reply once we have content that doesn't look like JSON start
                elif not json_reply_started and buffer.strip() and not buffer.strip().startswith("{"):
                    in_reply = True

                if not in_reply:
                    continue

                # JSON: end of reply_text value (closing quote)
                end_quote_idx = buffer.find('"')
                if json_reply_started and end_quote_idx >= 0:
                    sentence = buffer[:end_quote_idx].strip()
                    if sentence:
                        main_queue.put_nowait(("text", sentence))
                        sentence_queue.put_nowait(sentence)
                    in_reply = False
                    buffer = buffer[end_quote_idx + 1 :].lstrip()
                    continue

                # Sentence boundaries: first of . ! ? । \n
                idx = -1
                for sep in SENTENCE_BOUNDARIES:
                    i = buffer.find(sep)
                    if i >= 0 and (idx < 0 or i < idx):
                        idx = i
                if idx >= 0:
                    sentence = buffer[: idx + 1].strip()
                    # Smoothness: don't send very short fragments unless we'll never get more
                    if len(sentence) < MIN_SENTENCE_CHARS:
                        continue
                    # Free-form only: stop at Correction/Hinglish section headers
                    if not json_reply_started:
                        for line in sentence.split("\n"):
                            if _is_section_header(line.strip()):
                                in_reply = False
                                break
                        if not in_reply:
                            continue
                    buffer = buffer[idx + 1 :].lstrip()
                    if sentence:
                        main_queue.put_nowait(("text", sentence))
                        sentence_queue.put_nowait(sentence)

            # End-of-stream flush
            if buffer.strip() and in_reply:
                sent = buffer.strip()
                if json_reply_started:
                    sent = sent.replace('"', '').replace('}', '').strip()
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
                    wav_bytes = await asyncio.to_thread(generate_tts_bytes, sentence, response_language)
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

    async def event_gen():
        nonlocal full_reply_text
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
            logger.exception("chat/stream event_gen error")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            return

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

        if full_reply_text:
            try:
                parsed = parse_gemini_response(full_reply_text)
                msg = Message(
                    conversation_id=conversation.id,
                    user_message=message,
                    ai_reply=parsed.get("reply_text", ""),
                    correction=parsed.get("correction", ""),
                    hinglish_explanation="",
                    score=parsed.get("score", 75),
                )
                db.add(msg)
                db.commit()
                update_usage_stats(user_id, db, 0.0)
            except Exception as e:
                logger.exception("chat/stream save message error: %s", e)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/tts/stream")
async def tts_stream(request: TTSStreamRequest):
    """
    Stream TTS audio over Server-Sent Events (SSE).
    Producer (thread): feed_tts_stream_to_queue puts ("audio", chunk) per sentence, then (None, None) sentinel.
    Consumer (async gen): yields event: audio_chunk as soon as ("audio", chunk) is received; done and audio_ready
    only after queue is exhausted. media_type is text/event-stream.
    """
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
