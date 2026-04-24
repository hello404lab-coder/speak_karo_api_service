from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

pytest.importorskip("slowapi")

from app.api.ai import _build_stream_metadata_payload, _persist_message_audio_storage_ref
from app.database import get_db
from app.dependencies.subscription import require_active_plan
from app.main import app
from app.models.live_session import LiveSession  # noqa: F401
from app.models.social_session import SocialSession  # noqa: F401
from app.models.usage import Base, Conversation, Message
from app.models.user import User
from app.services.llm import stream_gemini_tokens
from app.services.tts import StoredAudioRecord

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


def test_build_stream_metadata_payload_preserves_null_feedback():
  payload = _build_stream_metadata_payload(
      {
          "translated_reply_text": None,
          "correction": None,
          "explanation": None,
          "example": None,
          "score": 96,
      },
      "conv-456",
      "en",
      None,
      client_turn_id="turn-456",
  )

  assert payload["user_analysis"]["correction"] is None
  assert payload["correction"] is None
  assert payload["explanation"] is None
  assert payload["example"] is None


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
      assert payload["messages"][0]["exchange_id"] == message.id
      assert payload["messages"][1]["exchange_id"] == message.id
      assert payload["messages"][0]["client_turn_id"] == "turn-history-1"
      assert payload["messages"][1]["client_turn_id"] == "turn-history-1"
  finally:
      app.dependency_overrides.clear()
      memory_db.close()


def test_chat_stream_skips_audio_events_when_include_audio_stream_is_false(monkeypatch):
  memory_db = _make_memory_db()
  user = User(
      id="ux-user-3",
      email="noaudio@example.com",
      name="No Audio User",
      provider="google",
      provider_id="noaudio-google",
      onboarding_completed=True,
  )
  memory_db.add(user)
  memory_db.commit()
  _install_overrides(memory_db, user)
  try:
      conversation = Conversation(id="conv-no-audio", user_id=user.id)
      memory_db.add(conversation)
      memory_db.commit()

      def fake_get_or_create_conversation(*args, **kwargs):
          return conversation

      async def fake_pipeline(*args, **kwargs):
          assert kwargs["include_audio_stream"] is False
          yield 'event: text_chunk\ndata: {"text":"Hello there."}\n\n'
          yield 'event: metadata\ndata: {"conversation_id":"conv-no-audio","client_turn_id":"turn-no-audio","correction":null,"explanation":null,"example":null,"score":92}\n\n'
          yield 'event: done\ndata: {"audio_url":null}\n\n'

      monkeypatch.setattr("app.api.ai.get_or_create_conversation", fake_get_or_create_conversation)
      monkeypatch.setattr("app.api.ai.get_conversation_history", lambda *args, **kwargs: [])
      monkeypatch.setattr("app.api.ai._llm_tts_streaming_pipeline", fake_pipeline)

      response = client.post(
          "/api/v1/ai/chat/stream",
          json={
              "user_id": user.id,
              "message": "Hi",
              "client_turn_id": "turn-no-audio",
              "include_audio_stream": False,
          },
      )

      body = response.text
      assert response.status_code == 200
      assert "event: text_chunk" in body
      assert "event: metadata" in body
      assert "event: done" in body
      assert "event: audio_chunk" not in body
      assert "event: audio_ready" not in body
  finally:
      app.dependency_overrides.clear()
      memory_db.close()


def test_message_audio_endpoint_reuses_stored_audio(monkeypatch):
  memory_db = _make_memory_db()
  user = User(
      id="ux-user-4",
      email="reuse@example.com",
      name="Reuse User",
      provider="google",
      provider_id="reuse-google",
      onboarding_completed=True,
  )
  conversation = Conversation(id="conv-audio-reuse", user_id=user.id)
  message = Message(
      id="exchange-reuse-1",
      conversation_id=conversation.id,
      user_message="Hello",
      ai_reply="Hello there.",
      reply_language="en",
      translated_ai_reply="Hola",
      translation_language_code="es",
      correction=None,
      hinglish_explanation="Use a natural greeting.",
      example="Hello, how are you?",
      score=88,
      ai_reply_audio_storage_ref="s3:ai/audio/reply.mp3",
  )
  memory_db.add_all([user, conversation, message])
  memory_db.commit()
  _install_overrides(memory_db, user)
  try:
      monkeypatch.setattr(
          "app.api.conversations.resolve_stored_audio_playback_url",
          lambda storage_ref, fallback_url=None: "https://audio.example/reply.mp3",
      )

      response = client.post(
          f"/api/v1/conversations/{conversation.id}/messages/{message.id}/audio",
          json={"segment": "reply"},
      )

      assert response.status_code == 200
      payload = response.json()
      assert payload == {
          "audio_url": "https://audio.example/reply.mp3",
          "segment": "reply",
          "generated": False,
      }
  finally:
      app.dependency_overrides.clear()
      memory_db.close()


def test_message_audio_endpoint_generates_and_persists_audio(monkeypatch):
  memory_db = _make_memory_db()
  user = User(
      id="ux-user-5",
      email="generate@example.com",
      name="Generate User",
      provider="google",
      provider_id="generate-google",
      onboarding_completed=True,
  )
  conversation = Conversation(id="conv-audio-generate", user_id=user.id)
  message = Message(
      id="exchange-generate-1",
      conversation_id=conversation.id,
      user_message="Hello",
      ai_reply="Hello there.",
      reply_language="en",
      translated_ai_reply="Hola",
      translation_language_code="es",
      correction=None,
      hinglish_explanation="Use a natural greeting.",
      example="Hello, how are you?",
      score=88,
  )
  memory_db.add_all([user, conversation, message])
  memory_db.commit()
  _install_overrides(memory_db, user)
  try:
      monkeypatch.setattr(
          "app.api.conversations.text_to_speech_record",
          lambda text, response_language: StoredAudioRecord(
              playback_url="https://audio.example/generated.mp3",
              storage_ref="s3:ai/audio/generated.mp3",
          ),
      )

      response = client.post(
          f"/api/v1/conversations/{conversation.id}/messages/{message.id}/audio",
          json={"segment": "example"},
      )

      assert response.status_code == 200
      payload = response.json()
      assert payload == {
          "audio_url": "https://audio.example/generated.mp3",
          "segment": "example",
          "generated": True,
      }
      refreshed = memory_db.get(Message, message.id)
      assert refreshed is not None
      assert refreshed.example_audio_storage_ref == "s3:ai/audio/generated.mp3"
  finally:
      app.dependency_overrides.clear()
      memory_db.close()


def test_persist_message_audio_storage_ref_updates_exchange_row():
  memory_db = _make_memory_db()
  user = User(
      id="ux-user-6",
      email="persist@example.com",
      name="Persist User",
      provider="google",
      provider_id="persist-google",
      onboarding_completed=True,
  )
  conversation = Conversation(id="conv-persist-audio", user_id=user.id)
  message = Message(
      id="exchange-persist-1",
      conversation_id=conversation.id,
      user_message="Hello",
      ai_reply="Hello there.",
      reply_language="en",
      score=90,
  )
  memory_db.add_all([user, conversation, message])
  memory_db.commit()
  try:
      _persist_message_audio_storage_ref(memory_db, message, "s3:ai/audio/persisted.mp3")
      refreshed = memory_db.get(Message, message.id)
      assert refreshed is not None
      assert refreshed.ai_reply_audio_storage_ref == "s3:ai/audio/persisted.mp3"
  finally:
      memory_db.close()


def test_stream_gemini_tokens_normalizes_retryable_provider_errors(monkeypatch):
  class FakeGeminiBusyError(Exception):
      status_code = 503

  class FakeModels:
      def generate_content_stream(self, **kwargs):
          raise FakeGeminiBusyError("503 UNAVAILABLE. high demand")

  class FakeClient:
      models = FakeModels()

  monkeypatch.setattr("app.services.llm._get_gemini_client", lambda: FakeClient())
  monkeypatch.setattr("app.services.llm._build_trimmed_contents", lambda *args, **kwargs: [])
  monkeypatch.setattr("app.services.llm._build_safety_settings", lambda: None)
  monkeypatch.setattr("app.services.llm.get_system_instruction", lambda *args, **kwargs: "system")

  with pytest.raises(RuntimeError) as exc_info:
      list(stream_gemini_tokens("Hi", []))

  assert str(exc_info.value) == "The AI is busy right now. Please try again in a few seconds."
