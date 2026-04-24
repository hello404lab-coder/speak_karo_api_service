"""Tests for Smallest.ai Lightning TTS integration."""
import asyncio
import base64
import json
import queue as queue_lib

import pytest
from app.services import tts


def test_resolve_tts_backend_uses_smallest_when_configured(monkeypatch):
    monkeypatch.setattr(tts.settings, "tts_chatterbox_enabled", False)
    monkeypatch.setattr(tts.settings, "tts_indicf5_enabled", False)
    monkeypatch.setattr(tts.settings, "tts_cloud_provider", "smallest")
    monkeypatch.setattr(tts.settings, "smallest_api_key", "test-key")

    assert tts.resolve_tts_backend("Hello", "en") == "smallest"
    assert tts.resolve_tts_backend("नमस्ते", "hi") == "smallest"


def test_resolve_tts_backend_falls_back_to_chirp_for_unsupported_smallest_lang(monkeypatch):
    monkeypatch.setattr(tts.settings, "tts_chatterbox_enabled", False)
    monkeypatch.setattr(tts.settings, "tts_indicf5_enabled", False)
    monkeypatch.setattr(tts.settings, "tts_cloud_provider", "smallest")
    monkeypatch.setattr(tts.settings, "smallest_api_key", "test-key")

    # Telugu: not in Smallest set; Chirp has a voice mapping -> cloud fallback is Chirp 3 HD
    assert tts.resolve_tts_backend("Hello", "te") == "chirp3_hd"


def test_stream_body_parser_two_data_lines_no_blank_between():
    """Common Smallest shape: consecutive data: lines without double-newline events."""
    body = (
        'data: {"status": "chunk", "data": {"audio": "AAAA"}}\n'
        'data: {"status": "complete", "done": true}\n'
    ).encode("utf-8")
    objs = tts._smallest_stream_body_to_json_objects(body)
    assert len(objs) == 2
    assert objs[0]["status"] == "chunk"


def test_smallest_voice_per_lang_ta_override(monkeypatch):
    monkeypatch.setattr(tts.settings, "smallest_api_key", "k")
    monkeypatch.setattr(tts.settings, "tts_smallest_voice", "magnus")
    monkeypatch.setattr(tts.settings, "tts_smallest_voice_per_lang", "ta:custom")
    assert tts._smallest_voice_for_lang("ta") == "custom"
    assert tts._smallest_voice_for_lang("en") == "magnus"


def test_resolve_ml_uses_chirp_not_smallest(monkeypatch):
    monkeypatch.setattr(tts.settings, "tts_chatterbox_enabled", False)
    monkeypatch.setattr(tts.settings, "tts_indicf5_enabled", False)
    monkeypatch.setattr(tts.settings, "tts_cloud_provider", "smallest")
    monkeypatch.setattr(tts.settings, "smallest_api_key", "test-key")
    assert tts.resolve_tts_backend("നമസ്കാരം", "ml") == "chirp3_hd"


def test_tts_with_smallest_posts_json_and_returns_wav(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200
        content = b"RIFF" + b"\x00" * 20

    class _C:
        def post(self, path, **kwargs):
            captured["path"] = path
            captured["json"] = kwargs.get("json")
            return _Resp()

    monkeypatch.setattr(tts, "_get_smallest_client", lambda: _C())
    monkeypatch.setattr(tts.settings, "smallest_api_key", "k")
    monkeypatch.setattr(tts.settings, "tts_smallest_model", "lightning-v3.1")
    out = tts._tts_with_smallest("Hi there", "en")
    assert out.startswith(b"RIFF")
    assert captured["path"] == "lightning-v3.1/get_speech"
    assert captured["json"]["voice_id"] == "magnus"
    assert captured["json"]["text"] == "Hi there"
    assert captured["json"]["language"] == "en"


@pytest.mark.asyncio
async def test_feed_smallest_stream_to_queue_emits_wav_and_raw_pcm(monkeypatch):
    sample_rate = 24000
    # Two samples of 16-bit silence -> 4 bytes
    b64 = base64.b64encode(b"\x00\x00\x00\x00").decode("ascii")

    sse_body = (
        f'data: {json.dumps({"status": "chunk", "data": {"audio": b64}})}\n'
        f'data: {json.dumps({"status": "complete", "done": True})}\n'
    ).encode("utf-8")

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        status_code = 200

        def read(self):
            return b""

        def iter_bytes(self):
            # Simulate TCP chunking: split one logical SSE line across two packets
            mid = max(1, len(sse_body) // 2)
            yield sse_body[:mid]
            yield sse_body[mid:]

    class _C:
        def stream(self, method, path, **kwargs):
            assert "stream" in path
            j = kwargs.get("json", {})
            assert "text" in j and "voice_id" in j
            return _Stream()

    monkeypatch.setattr(tts, "_get_smallest_client", lambda: _C())
    monkeypatch.setattr(tts, "_postprocess_chirp_pcm_chunk", lambda *a, **k: a[0])
    monkeypatch.setattr(tts.settings, "smallest_api_key", "k")
    monkeypatch.setattr(tts.settings, "tts_smallest_sample_rate_hz", sample_rate)

    fragments = queue_lib.Queue()
    fragments.put_nowait("Hello")
    fragments.put_nowait(None)
    out_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    ev = tts.threading.Event()

    await asyncio.to_thread(
        tts.feed_smallest_stream_to_queue,
        fragments,
        "en",
        out_queue,
        loop,
        ev,
    )
    items = []
    while not out_queue.empty():
        items.append(await out_queue.get())
    assert any(x[0] == "audio" and x[1].startswith(b"RIFF") for x in items)
    assert any(x[0] == tts.CHIRP_STREAM_EVENT_RAW_PCM and x[1] for x in items)
    assert items[-1] == (None, None)


def test_tts_stream_endpoint_uses_smallest_path(monkeypatch):
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
        id="small-user-1",
        email="s@example.com",
        name="S",
        provider="google",
        provider_id="s-google",
        onboarding_completed=True,
    )
    memory_db.add(user)
    memory_db.commit()
    _install_overrides(memory_db, user)
    try:
        monkeypatch.setattr("app.api.ai.smallest_streaming_enabled_for_text", lambda *a, **k: True)
        monkeypatch.setattr("app.api.ai.chirp_streaming_enabled_for_text", lambda *a, **k: False)
        monkeypatch.setattr("app.api.ai.split_text_for_smallest_stream", lambda text: ["One"])
        monkeypatch.setattr("app.api.ai._store_pcm_stream_and_store", lambda audio, text: ("https://a.test/s.mp3", None))

        def _fake_smallest(fragments, response_language, queue, loop, stop_event):
            fragments.get()  # "One"
            assert fragments.get() is None
            loop.call_soon_threadsafe(queue.put_nowait, ("audio", b"RIFFs"))
            loop.call_soon_threadsafe(queue.put_nowait, ("raw_pcm", b"\xab"))
            loop.call_soon_threadsafe(queue.put_nowait, (None, None))

        monkeypatch.setattr("app.api.ai.feed_smallest_stream_to_queue", _fake_smallest)
        r = client.post(
            "/api/v1/ai/tts/stream",
            json={"text": "Test", "response_language": "en"},
        )
        assert r.status_code == 200
        assert "https://a.test/s.mp3" in r.text
    finally:
        app.dependency_overrides.clear()
        memory_db.close()
