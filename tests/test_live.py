"""Tests for Gemini Live control-plane endpoints."""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from fastapi import HTTPException, status

from app.core.config import settings
from app.core.security import create_access_token
from app.database import get_db
from app.dependencies.live_access import require_gemini_live_plan
from app.main import app
from app.services.subscription_service import (
    check_usage_limit,
    get_usage_today,
    plan_rank,
    resolve_user_plan,
)
from app.models.live_session import LiveSession
from app.models.usage import Conversation
from app.models.user import User

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_dependency_overrides(monkeypatch):
    monkeypatch.setattr(settings, "gemini_live_min_plan", "free")
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def memory_db(test_engine):
    """Transactional PG session seeded with a Live-eligible user.

    Named ``memory_db`` to avoid churn in existing test parameters; the
    underlying engine is the shared session-scoped PostgreSQL engine from
    ``conftest.py``.
    """
    connection = test_engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(
        bind=connection,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    db = SessionLocal()
    try:
        u = User(
            id="live-user-1",
            email="live@example.com",
            name="Live User",
            provider="google",
            provider_id="g-live",
            onboarding_completed=True,
        )
        db.add(u)
        db.commit()
        yield db
    finally:
        db.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()


def _auth_headers(user_id: str = "live-user-1"):
    token = create_access_token(subject=user_id)
    return {"Authorization": f"Bearer {token}"}


def _make_live_user_override(memory_db: Session, user_id: str):
    """Full Live gate without JWT (mirrors require_active_plan + require_gemini_live_plan)."""

    def _override():
        u = memory_db.query(User).filter(User.id == user_id).first()
        assert u is not None
        usage_today = get_usage_today(u.id, memory_db)
        check_usage_limit(u, usage_today)
        effective = resolve_user_plan(u)
        minimum = settings.gemini_live_min_plan
        if plan_rank(effective) < plan_rank(minimum):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Gemini Live is not available on your current plan.",
            )
        return u

    return _override


def _install_live_test_overrides(memory_db, user: User):
    """Inject in-memory DB and bypass JWT by replacing the Live dependency chain."""

    def _db():
        yield memory_db

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[require_gemini_live_plan] = _make_live_user_override(memory_db, user.id)


def test_live_config_401_without_auth():
    r = client.get("/api/v1/ai/live/config")
    assert r.status_code == 401


def test_live_config_403_onboarding(memory_db):
    u = memory_db.query(User).filter(User.id == "live-user-1").first()
    u.onboarding_completed = False
    memory_db.commit()

    _install_live_test_overrides(memory_db, u)
    try:
        r = client.get("/api/v1/ai/live/config", headers=_auth_headers())
        assert r.status_code == 403
        assert "onboarding" in r.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_live_config_happy(memory_db):
    u = memory_db.query(User).filter(User.id == "live-user-1").first()
    u.onboarding_completed = True
    memory_db.commit()
    _install_live_test_overrides(memory_db, u)
    try:
        r = client.get("/api/v1/ai/live/config", headers=_auth_headers())
        assert r.status_code == 200
        data = r.json()
        assert "model" in data
        assert "system_instruction" in data
        assert len(data["system_instruction"]) > 50
        assert data["prompt_version"]
        assert "AUDIO" in data["response_modalities"]
        assert "gemini_api_key" not in str(data).lower()
    finally:
        app.dependency_overrides.clear()


def test_live_session_start_end_updates_usage(memory_db):
    u = memory_db.query(User).filter(User.id == "live-user-1").first()
    u.onboarding_completed = True
    memory_db.commit()
    _install_live_test_overrides(memory_db, u)
    try:
        r = client.post(
            "/api/v1/ai/live/session/start",
            json={},
            headers=_auth_headers(),
        )
        assert r.status_code == 200
        sid = r.json()["session_id"]

        past = datetime.now(timezone.utc) - timedelta(seconds=30)
        row = memory_db.query(LiveSession).filter(LiveSession.id == sid).first()
        row.started_at = past.replace(tzinfo=None)
        row.last_seen_at = past.replace(tzinfo=None)
        memory_db.commit()

        r2 = client.post(
            "/api/v1/ai/live/session/end",
            json={"session_id": sid},
            headers=_auth_headers(),
        )
        assert r2.status_code == 200
        assert r2.json()["duration_seconds"] >= 29

        from app.models.usage import Usage
        from datetime import date

        usage = (
            memory_db.query(Usage)
            .filter(Usage.user_id == "live-user-1", Usage.date == date.today())
            .first()
        )
        assert usage is not None
        assert usage.voice_count >= 1
        assert usage.minutes_used > 0
    finally:
        app.dependency_overrides.clear()


def test_live_token_mint_mocked(memory_db):
    u = memory_db.query(User).filter(User.id == "live-user-1").first()
    u.onboarding_completed = True
    memory_db.commit()
    _install_live_test_overrides(memory_db, u)
    try:
        with (
            patch(
                "app.api.live.mint_live_ephemeral_auth_token",
                new_callable=AsyncMock,
                return_value="auth-tokens/fake-token-name",
            ),
            patch("app.api.live._enforce_token_rate_limit", new_callable=AsyncMock),
        ):
            r = client.post(
                "/api/v1/ai/live/token",
                json={},
                headers=_auth_headers(),
            )
        assert r.status_code == 200
        assert r.json()["auth_token"] == "auth-tokens/fake-token-name"
        assert r.json()["new_session_expire_seconds"] >= 10
    finally:
        app.dependency_overrides.clear()


def test_live_config_requires_vuvl_pro_when_min_plan_is_pro(memory_db, monkeypatch):
    u = memory_db.query(User).filter(User.id == "live-user-1").first()
    u.onboarding_completed = True
    u.plan = "vuvl_plus"
    u.subscription_expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    u.subscription_expires_at = u.subscription_expires_at.replace(tzinfo=None)
    memory_db.commit()
    monkeypatch.setattr(settings, "gemini_live_min_plan", "vuvl_pro")
    _install_live_test_overrides(memory_db, u)
    try:
        blocked = client.get("/api/v1/ai/live/config", headers=_auth_headers())
        assert blocked.status_code == 403

        u.plan = "vuvl_pro"
        memory_db.commit()
        allowed = client.get("/api/v1/ai/live/config", headers=_auth_headers())
        assert allowed.status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_live_config_with_conversation(memory_db):
    cid = "conv-live-1"
    memory_db.add(
        Conversation(
            id=cid,
            user_id="live-user-1",
            long_term_context="Prepare for IELTS speaking",
        )
    )
    memory_db.commit()

    u = memory_db.query(User).filter(User.id == "live-user-1").first()
    u.onboarding_completed = True
    memory_db.commit()
    _install_live_test_overrides(memory_db, u)
    try:
        r = client.get(
            f"/api/v1/ai/live/config?conversation_id={cid}",
            headers=_auth_headers(),
        )
        assert r.status_code == 200
        assert "IELTS" in r.json()["system_instruction"]
    finally:
        app.dependency_overrides.clear()


def test_live_session_start_reuses_active(memory_db):
    u = memory_db.query(User).filter(User.id == "live-user-1").first()
    u.onboarding_completed = True
    memory_db.commit()
    _install_live_test_overrides(memory_db, u)
    try:
        r1 = client.post("/api/v1/ai/live/session/start", json={}, headers=_auth_headers())
        assert r1.status_code == 200
        assert r1.json()["reused_existing"] is False
        sid = r1.json()["session_id"]

        r2 = client.post("/api/v1/ai/live/session/start", json={}, headers=_auth_headers())
        assert r2.status_code == 200
        assert r2.json()["reused_existing"] is True
        assert r2.json()["session_id"] == sid
    finally:
        app.dependency_overrides.clear()


def test_live_session_active_and_heartbeat(memory_db):
    u = memory_db.query(User).filter(User.id == "live-user-1").first()
    u.onboarding_completed = True
    memory_db.commit()
    _install_live_test_overrides(memory_db, u)
    try:
        client.post("/api/v1/ai/live/session/start", json={}, headers=_auth_headers())
        ra = client.get("/api/v1/ai/live/session/active", headers=_auth_headers())
        assert ra.status_code == 200
        assert ra.json()["active"] is True
        sid = ra.json()["session_id"]

        rh = client.post(
            "/api/v1/ai/live/session/heartbeat",
            json={"session_id": sid},
            headers=_auth_headers(),
        )
        assert rh.status_code == 200
        assert rh.json()["session_id"] == sid

        bad = client.post(
            "/api/v1/ai/live/session/heartbeat",
            json={"session_id": "00000000-0000-0000-0000-000000000000"},
            headers=_auth_headers(),
        )
        assert bad.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_live_session_end_idempotent(memory_db):
    u = memory_db.query(User).filter(User.id == "live-user-1").first()
    u.onboarding_completed = True
    # Premium so a second /session/end (idempotent) is not blocked by free daily voice cap after first end.
    u.subscription_expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    u.subscription_expires_at = u.subscription_expires_at.replace(tzinfo=None)
    memory_db.commit()
    _install_live_test_overrides(memory_db, u)
    try:
        r1 = client.post("/api/v1/ai/live/session/start", json={}, headers=_auth_headers())
        sid = r1.json()["session_id"]

        e1 = client.post(
            "/api/v1/ai/live/session/end",
            json={"session_id": sid},
            headers=_auth_headers(),
        )
        assert e1.status_code == 200
        d1 = e1.json()["duration_seconds"]

        e2 = client.post(
            "/api/v1/ai/live/session/end",
            json={"session_id": sid},
            headers=_auth_headers(),
        )
        assert e2.status_code == 200
        assert e2.json()["duration_seconds"] == d1

        from app.models.usage import Usage
        from datetime import date

        usage = (
            memory_db.query(Usage)
            .filter(Usage.user_id == "live-user-1", Usage.date == date.today())
            .first()
        )
        assert usage.voice_count == 1
    finally:
        app.dependency_overrides.clear()


def test_live_token_rejects_stale_session(memory_db):
    u = memory_db.query(User).filter(User.id == "live-user-1").first()
    u.onboarding_completed = True
    memory_db.commit()
    _install_live_test_overrides(memory_db, u)
    try:
        r1 = client.post("/api/v1/ai/live/session/start", json={}, headers=_auth_headers())
        sid = r1.json()["session_id"]
        row = memory_db.query(LiveSession).filter(LiveSession.id == sid).first()
        old = datetime.now(timezone.utc) - timedelta(seconds=500)
        row.last_seen_at = old.replace(tzinfo=None)
        memory_db.commit()

        with (
            patch("app.api.live._enforce_token_rate_limit", new_callable=AsyncMock),
            patch("app.api.live.mint_live_ephemeral_auth_token", new_callable=AsyncMock),
        ):
            r = client.post(
                "/api/v1/ai/live/token",
                json={"session_id": sid},
                headers=_auth_headers(),
            )
        assert r.status_code == 410
        assert r.json().get("type") == "SESSION_STALE"
    finally:
        app.dependency_overrides.clear()


def test_live_token_rate_limited(memory_db):
    u = memory_db.query(User).filter(User.id == "live-user-1").first()
    u.onboarding_completed = True
    memory_db.commit()
    _install_live_test_overrides(memory_db, u)
    # Drive real check_live_token_rate_limit: INCR above GEMINI_LIVE_TOKEN_REQUESTS_PER_MINUTE -> 429.
    fake_redis = AsyncMock()
    fake_redis.incr = AsyncMock(return_value=99)
    fake_redis.expire = AsyncMock(return_value=True)
    old_redis = getattr(app.state, "live_redis", None)
    app.state.live_redis = fake_redis
    try:
        with patch(
            "app.api.live.mint_live_ephemeral_auth_token",
            new_callable=AsyncMock,
            return_value="auth-tokens/fake",
        ):
            r = client.post("/api/v1/ai/live/token", json={}, headers=_auth_headers())
        assert r.status_code == 429
        body = r.json()
        assert body.get("type") == "RATE_LIMITED"
        fake_redis.incr.assert_awaited()
    finally:
        app.state.live_redis = old_redis
        app.dependency_overrides.clear()
