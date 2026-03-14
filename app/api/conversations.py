"""Conversation list, detail, messages history, and title generation endpoints."""
import base64
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.dependencies.subscription import require_active_plan
from app.models.usage import Conversation, Message
from app.models.user import User
from app.schemas.chat import (
    ChatMessage,
    ConversationDetail,
    ConversationListItem,
    ConversationListResponse,
    MessagesResponse,
    TitleResponse,
)
from app.services.llm import generate_conversation_title

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_CONVERSATIONS_LIMIT = 20
MAX_MESSAGES_LIMIT = 50

# Namespace for deterministic message UUIDs (user/assistant per row)
MESSAGE_IDS_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "ai-english-practice.messages")


def _encode_cursor(created_at: datetime, message_id: str) -> str:
    """Encode keyset cursor as base64 JSON."""
    payload = {"created_at": created_at.isoformat(), "id": message_id}
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    """Decode keyset cursor; raises ValueError if invalid."""
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        dt = datetime.fromisoformat(payload["created_at"].replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return (dt, payload["id"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid cursor: {e}") from e


def get_conversation_for_user(
    conversation_id: str,
    user_id: str,
    db: Session,
) -> Conversation:
    """Return conversation if it exists and belongs to user; else raise 404."""
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.get("", response_model=ConversationListResponse)
def list_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_active_plan),
    limit: int = Query(default=DEFAULT_CONVERSATIONS_LIMIT, ge=1, le=100),
) -> ConversationListResponse:
    """
    List conversations for the authenticated user, ordered by updated_at DESC.
    """
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .all()
    )
    if not conversations:
        return ConversationListResponse(conversations=[])

    conv_ids = [c.id for c in conversations]
    # Latest message per conversation: all messages for these convs ordered by created_at desc,
    # then take first per conversation_id in Python
    latest_messages = (
        db.query(Message)
        .filter(Message.conversation_id.in_(conv_ids))
        .order_by(Message.created_at.desc())
        .all()
    )
    last_message_by_conv: dict[str, str] = {}
    for msg in latest_messages:
        if msg.conversation_id not in last_message_by_conv:
            last_message_by_conv[msg.conversation_id] = msg.ai_reply or ""

    items = [
        ConversationListItem(
            id=c.id,
            title=c.title,
            last_message=last_message_by_conv.get(c.id),
            updated_at=c.updated_at,
        )
        for c in conversations
    ]
    return ConversationListResponse(conversations=items)


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_active_plan),
) -> ConversationDetail:
    """Get a single conversation by id. Returns 404 if not found or not owned by user."""
    conversation = get_conversation_for_user(conversation_id, current_user.id, db)
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.get("/{conversation_id}/messages", response_model=MessagesResponse)
def list_messages(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_active_plan),
    limit: int = Query(default=20, ge=1, le=MAX_MESSAGES_LIMIT),
    cursor: Optional[str] = Query(None, description="Opaque cursor for pagination (older messages)"),
) -> MessagesResponse:
    """
    Load chat history for the conversation. Each DB row is one exchange (user + assistant);
    responses are flattened to alternating user/assistant messages in chronological order
    (oldest first). Use next_cursor to load older messages.
    """
    get_conversation_for_user(conversation_id, current_user.id, db)

    row_limit = (limit + 1) // 2  # number of Message rows to fetch (each yields 2 chat messages)
    query = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
    )
    if cursor is not None:
        try:
            cursor_dt, cursor_id = _decode_cursor(cursor)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        query = query.filter(
            or_(
                Message.created_at < cursor_dt,
                and_(Message.created_at == cursor_dt, Message.id < cursor_id),
            )
        )
    rows = query.limit(row_limit).all()

    # Chronological order (oldest first): reverse so pair order is always user then assistant
    rows = list(reversed(rows))

    messages: list[ChatMessage] = []
    for idx, row in enumerate(rows):
        row_id_str = str(row.id)
        user_msg_id = str(uuid.uuid5(MESSAGE_IDS_NAMESPACE, row_id_str + ":user"))
        assistant_msg_id = str(uuid.uuid5(MESSAGE_IDS_NAMESPACE, row_id_str + ":assistant"))
        messages.append(
            ChatMessage(
                index=idx * 2,
                id=user_msg_id,
                role="user",
                content=row.user_message,
                user_audio_url=row.user_audio_url,
                reply_text=None,
                correction=None,
                explanation=None,
                example=None,
                score=None,
                created_at=row.created_at,
            )
        )
        messages.append(
            ChatMessage(
                index=idx * 2 + 1,
                id=assistant_msg_id,
                role="assistant",
                content=None,
                user_audio_url=None,
                reply_text=row.ai_reply,
                correction=row.correction,
                explanation=row.hinglish_explanation,
                example=row.example,
                score=row.score,
                created_at=row.created_at,
            )
        )

    next_cursor: Optional[str] = None
    if len(rows) == row_limit and rows:
        next_cursor = _encode_cursor(rows[0].created_at, rows[0].id)

    logger.debug(
        "Loaded %d message rows for conversation %s",
        len(rows),
        conversation_id,
    )
    return MessagesResponse(messages=messages, next_cursor=next_cursor)


@router.post("/{conversation_id}/title", response_model=TitleResponse)
def generate_title(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_active_plan),
) -> TitleResponse:
    """
    Generate and save an AI conversation title from the first few messages.
    """
    conversation = get_conversation_for_user(conversation_id, current_user.id, db)

    first_messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .limit(3)
        .all()
    )
    excerpt_parts: list[str] = []
    for msg in first_messages:
        excerpt_parts.append(f"User: {msg.user_message}")
        excerpt_parts.append(f"Assistant: {msg.ai_reply}")
    excerpt = "\n".join(excerpt_parts).strip() if excerpt_parts else ""

    title = generate_conversation_title(excerpt)
    conversation.title = title
    db.commit()
    db.refresh(conversation)

    logger.info(
        "conversation_title_generated",
        extra={"conversation_id": conversation_id, "title": title},
    )
    return TitleResponse(title=title)
