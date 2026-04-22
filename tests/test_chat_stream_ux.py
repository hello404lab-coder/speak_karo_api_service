from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.ai import _build_stream_metadata_payload
from app.database import get_db
from app.dependencies.subscription import require_active_plan
from app.main import app
from app.models.live_session import LiveSession  # noqa: F401
from app.models.social_session import SocialSession  # noqa: F401
from app.models.usage import Base, Conversation, Message
from app.models.user import User

client = TestClient(app)


def _make_memory_db() -> Session:
  engine = create_engine(
      "sqlite:///:memory:",
      connect_args={"check_same_thread": False},
      poolclass=StaticPool,
  )
  Base.metadata.create_all(bind=engine)
  SessionLocal = sessionmaker(
      autocommit=False,
      autoflush=False,
      bind=engine,
      expire_on_commit=False,
  )
  return SessionLocal()


def _install_overrides(memory_db: Session, user: User) -> None:
  def _db():
      yield memory_db

  def _active_plan():
      return user

  app.dependency_overrides[get_db] = _db
  app.dependency_overrides[require_active_plan] = _active_plan


def test_build_stream_metadata_payload_echoes_client_turn_id():
  payload = _build_stream_metadata_payload(
      {
          "translated_reply_text": "Hola",
          "correction": "I am going home.",
          "explanation": "Use present continuous for current plans.",
          "example": "I am meeting her tonight.",
          "score": 91,
      },
      "conv-123",
      "en",
      "es",
      client_turn_id="turn-123",
  )

  assert payload["conversation_id"] == "conv-123"
  assert payload["client_turn_id"] == "turn-123"
  assert payload["translation_language"] == "es"


def test_chat_stream_emits_turn_ack_before_text_chunk(monkeypatch):
  memory_db = _make_memory_db()
  user = User(
      id="ux-user-1",
      email="ux@example.com",
      name="UX User",
      provider="google",
      provider_id="ux-google",
      onboarding_completed=True,
  )
  memory_db.add(user)
  memory_db.commit()
  _install_overrides(memory_db, user)
  try:
      conversation = Conversation(id="conv-turn-ack", user_id=user.id)
      memory_db.add(conversation)
      memory_db.commit()

      def fake_get_or_create_conversation(*args, **kwargs):
          return conversation

      async def fake_pipeline(*args, **kwargs):
          yield 'event: text_chunk\ndata: {"text":"Hello there."}\n\n'
          yield 'event: done\ndata: {"audio_url":null}\n\n'

      monkeypatch.setattr("app.api.ai.get_or_create_conversation", fake_get_or_create_conversation)
      monkeypatch.setattr("app.api.ai.get_conversation_history", lambda *args, **kwargs: [])
      monkeypatch.setattr("app.api.ai._llm_tts_streaming_pipeline", fake_pipeline)

      response = client.post(
          "/api/v1/ai/chat/stream",
          json={
              "user_id": user.id,
              "message": "Hi",
              "client_turn_id": "turn-ack-1",
          },
      )

      body = response.text
      assert response.status_code == 200
      assert "event: turn_ack" in body
      assert '"client_turn_id": "turn-ack-1"' in body
      assert body.index("event: turn_ack") < body.index("event: text_chunk")
  finally:
      app.dependency_overrides.clear()
      memory_db.close()


def test_list_messages_returns_client_turn_id_for_both_rows():
  memory_db = _make_memory_db()
  user = User(
      id="ux-user-2",
      email="history@example.com",
      name="History User",
      provider="google",
      provider_id="history-google",
      onboarding_completed=True,
  )
  conversation = Conversation(id="conv-history-1", user_id=user.id)
  message = Message(
      conversation_id=conversation.id,
      client_turn_id="turn-history-1",
      user_message="How are you?",
      ai_reply="I am doing well.",
      reply_language="en",
      translated_ai_reply=None,
      translation_language_code=None,
      correction="",
      hinglish_explanation="",
      example="",
      score=82,
      created_at=datetime.utcnow(),
  )
  memory_db.add(user)
  memory_db.add(conversation)
  memory_db.add(message)
  memory_db.commit()
  _install_overrides(memory_db, user)
  try:
      response = client.get(f"/api/v1/conversations/{conversation.id}/messages")

      assert response.status_code == 200
      payload = response.json()
      assert len(payload["messages"]) == 2
      assert payload["messages"][0]["client_turn_id"] == "turn-history-1"
      assert payload["messages"][1]["client_turn_id"] == "turn-history-1"
  finally:
      app.dependency_overrides.clear()
      memory_db.close()
