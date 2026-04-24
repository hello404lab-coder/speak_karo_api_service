import asyncio
import queue as queue_lib
from types import SimpleNamespace

import pytest
from app.services import tts


def test_resolve_tts_backend_uses_chirp_when_configured(monkeypatch):
    monkeypatch.setattr(tts.settings, "tts_chatterbox_enabled", False)
    monkeypatch.setattr(tts.settings, "tts_indicf5_enabled", False)
    monkeypatch.setattr(tts.settings, "tts_cloud_provider", "chirp3_hd")

    assert tts.resolve_tts_backend("Hello there", "en") == "chirp3_hd"


def test_resolve_tts_backend_falls_back_to_gemini_for_unsupported_chirp_language(monkeypatch):
    monkeypatch.setattr(tts.settings, "tts_chatterbox_enabled", False)
    monkeypatch.setattr(tts.settings, "tts_indicf5_enabled", False)
    monkeypatch.setattr(tts.settings, "tts_cloud_provider", "chirp3_hd")

    assert tts.resolve_tts_backend("Bonjour", "fr") == "gemini"


@pytest.mark.asyncio
async def test_feed_chirp_stream_to_queue_wraps_pcm_and_emits_raw_pcm(monkeypatch):
    class _StreamingAudioConfig:
        class AudioEncoding:
            PCM = "pcm"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _VoiceSelectionParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _StreamingSynthesizeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _StreamingSynthesisInput:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _StreamingSynthesizeRequest:
        def __init__(self, streaming_config=None, input=None):
            self.streaming_config = streaming_config
            self.input = input

    fake_module = SimpleNamespace(
        StreamingAudioConfig=_StreamingAudioConfig,
        AudioEncoding=SimpleNamespace(PCM="pcm"),
        VoiceSelectionParams=_VoiceSelectionParams,
        StreamingSynthesizeConfig=_StreamingSynthesizeConfig,
        StreamingSynthesisInput=_StreamingSynthesisInput,
        StreamingSynthesizeRequest=_StreamingSynthesizeRequest,
    )

    class _FakeClient:
        def streaming_synthesize(self, requests, timeout=None):
            captured = list(requests)
            assert captured[0].streaming_config is not None
            assert captured[1].input.kwargs["text"] == "Hello world"
            yield SimpleNamespace(audio_content=b"\x01\x02")
            yield SimpleNamespace(audio_content=b"\x03\x04")

    monkeypatch.setattr(tts, "_import_chirp_texttospeech", lambda: fake_module)
    monkeypatch.setattr(tts, "_get_chirp_client", lambda: _FakeClient())
    monkeypatch.setattr(tts, "_postprocess_chirp_pcm_chunk", lambda audio_data, sample_rate_hz=None, state=None: audio_data)

    fragments = queue_lib.Queue()
    fragments.put_nowait("Hello world")
    fragments.put_nowait(None)
    out_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    await asyncio.to_thread(
        tts.feed_chirp_stream_to_queue,
        fragments,
        "en",
        out_queue,
        loop,
        asyncio.Event(),
    )

    items = []
    while not out_queue.empty():
        items.append(await out_queue.get())

    assert items[0][0] == "audio"
    assert items[0][1].startswith(b"RIFF")
    assert items[1][0] == tts.CHIRP_STREAM_EVENT_RAW_PCM
    assert items[1][1] == b"\x01\x02\x03\x04"
    assert items[2] == (None, None)


def test_tts_stream_endpoint_uses_chirp_streaming_path(monkeypatch):
    pytest.importorskip("slowapi")
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.database import get_db
    from app.dependencies.subscription import require_active_plan
    from app.main import app
    from app.models.live_session import LiveSession  # noqa: F401
    from app.models.social_session import SocialSession  # noqa: F401
    from app.models.usage import Base
    from app.models.user import User

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

    client = TestClient(app)
    memory_db = _make_memory_db()
    user = User(
        id="chirp-user-1",
        email="chirp@example.com",
        name="Chirp User",
        provider="google",
        provider_id="chirp-google",
        onboarding_completed=True,
    )
    memory_db.add(user)
    memory_db.commit()
    _install_overrides(memory_db, user)
    try:
        monkeypatch.setattr("app.api.ai.chirp_streaming_enabled_for_text", lambda *args, **kwargs: True)
        monkeypatch.setattr("app.api.ai.smallest_streaming_enabled_for_text", lambda *args, **kwargs: False)
        monkeypatch.setattr("app.api.ai.split_text_for_chirp_stream", lambda text: ["Hello", "world"])
        monkeypatch.setattr("app.api.ai._store_pcm_stream_and_store", lambda audio, text: ("https://audio.test/final.mp3", None))

        def _fake_feed(fragments, response_language, queue, loop, stop_event):
            seen = []
            while True:
                item = fragments.get()
                if item is None:
                    break
                seen.append(item)
            assert seen == ["Hello", "world"]
            loop.call_soon_threadsafe(queue.put_nowait, ("audio", b"RIFFfake"))
            loop.call_soon_threadsafe(queue.put_nowait, ("raw_pcm", b"\x00\x01"))
            loop.call_soon_threadsafe(queue.put_nowait, (None, None))

        monkeypatch.setattr("app.api.ai.feed_chirp_stream_to_queue", _fake_feed)

        response = client.post(
            "/api/v1/ai/tts/stream",
            json={"text": "Hello world", "response_language": "en"},
        )

        assert response.status_code == 200
        body = response.text
        assert "event: audio_chunk" in body
        assert "event: done" in body
        assert "event: audio_ready" in body
        assert "https://audio.test/final.mp3" in body
    finally:
        app.dependency_overrides.clear()
        memory_db.close()


def test_chat_stream_endpoint_keeps_existing_sse_contract_with_chirp(monkeypatch):
    pytest.importorskip("slowapi")
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.database import get_db
    from app.dependencies.subscription import require_active_plan
    from app.main import app
    from app.models.live_session import LiveSession  # noqa: F401
    from app.models.social_session import SocialSession  # noqa: F401
    from app.models.usage import Base, Conversation
    from app.models.user import User

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

    client = TestClient(app)
    memory_db = _make_memory_db()
    user = User(
        id="chirp-user-2",
        email="chirp2@example.com",
        name="Chirp User 2",
        provider="google",
        provider_id="chirp-google-2",
        onboarding_completed=True,
    )
    conversation = Conversation(id="conv-chirp-1", user_id=user.id)
    memory_db.add(user)
    memory_db.add(conversation)
    memory_db.commit()
    _install_overrides(memory_db, user)
    try:
        monkeypatch.setattr("app.api.ai.chirp_streaming_enabled_for_text", lambda *args, **kwargs: True)
        monkeypatch.setattr("app.api.ai.smallest_streaming_enabled_for_text", lambda *args, **kwargs: False)
        monkeypatch.setattr(
            "app.api.ai.stream_gemini_tokens",
            lambda *args, **kwargs: iter([
                '{"reply_text": "Hello there. How are you?", "correction": "", "explanation": "", "example": "", "score": 88}'
            ]),
        )
        monkeypatch.setattr("app.api.ai.attach_translated_reply_text", lambda payload, *args, **kwargs: payload)
        monkeypatch.setattr("app.api.ai.finalize_llm_reply", lambda payload, *args, **kwargs: payload)
        monkeypatch.setattr("app.api.ai.update_usage_stats", lambda *args, **kwargs: None)
        monkeypatch.setattr("app.api.ai._store_pcm_stream_and_store", lambda audio, text: ("https://audio.test/chat.mp3", None))
        monkeypatch.setattr("app.api.ai.get_conversation_history", lambda *args, **kwargs: [])
        monkeypatch.setattr("app.api.ai.get_or_create_conversation", lambda *args, **kwargs: conversation)

        def _fake_feed(fragments, response_language, queue, loop, stop_event):
            consumed = []
            while True:
                item = fragments.get()
                if item is None:
                    break
                consumed.append(item)
            assert consumed
            loop.call_soon_threadsafe(queue.put_nowait, ("audio", b"RIFFchat"))
            loop.call_soon_threadsafe(queue.put_nowait, ("raw_pcm", b"\x01\x02"))
            loop.call_soon_threadsafe(queue.put_nowait, (None, None))

        monkeypatch.setattr("app.api.ai.feed_chirp_stream_to_queue", _fake_feed)

        response = client.post(
            "/api/v1/ai/chat/stream",
            json={"user_id": user.id, "message": "Hi there"},
        )

        assert response.status_code == 200
        body = response.text
        assert "event: turn_ack" in body
        assert "event: text_chunk" in body
        assert "event: audio_chunk" in body
        assert "event: metadata" in body
        assert "event: done" in body
        assert "event: audio_ready" in body
        assert body.index("event: audio_chunk") < body.index("event: done")
    finally:
        app.dependency_overrides.clear()
        memory_db.close()
