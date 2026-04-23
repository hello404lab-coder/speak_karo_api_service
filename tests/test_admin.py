"""Tests for admin auth and admin user management endpoints."""
from datetime import date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.security import (
    ADMIN_PRINCIPAL_TYPE,
    USER_PRINCIPAL_TYPE,
    create_access_token,
    create_refresh_token,
    hash_password,
)
from app.database import get_db
from app.main import app
from app.models.admin import Admin
from app.models.usage import Base, Conversation, Message, Usage
from app.models.user import User
from app.services.admin_auth_service import bootstrap_admin_account

client = TestClient(app)


@pytest.fixture()
def db_session():
    """Provide an isolated in-memory DB and override FastAPI's get_db dependency."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()

    def override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield session
    finally:
        app.dependency_overrides.clear()
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def _make_admin(db_session, *, email="admin@example.com", password="secret123", is_active=True) -> Admin:
    admin = Admin(
        email=email,
        password_hash=hash_password(password),
        is_active=is_active,
    )
    db_session.add(admin)
    db_session.commit()
    db_session.refresh(admin)
    return admin


def _make_user(
    db_session,
    *,
    email: str,
    provider: str = "google",
    provider_id: str | None = None,
    name: str | None = None,
    nickname: str | None = None,
    onboarding_completed: bool = False,
    onboarding_step: int = 0,
    plan: str = "free",
    trial_expires_at: datetime | None = None,
    subscription_expires_at: datetime | None = None,
) -> User:
    user = User(
        email=email,
        name=name,
        provider=provider,
        provider_id=provider_id or f"{provider}-{email}",
        nickname=nickname,
        onboarding_completed=onboarding_completed,
        onboarding_step=onboarding_step,
        plan=plan,
        trial_expires_at=trial_expires_at,
        subscription_expires_at=subscription_expires_at,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_bootstrap_admin_creates_admin_from_settings(db_session, monkeypatch):
    monkeypatch.setattr(settings, "admin_bootstrap_email", "BOOTSTRAP@Example.com")
    monkeypatch.setattr(settings, "admin_bootstrap_password", "bootstrap-pass")

    admin = bootstrap_admin_account(db_session)

    assert admin is not None
    assert admin.email == "bootstrap@example.com"
    assert admin.is_active is True


def test_bootstrap_admin_updates_existing_password_and_reactivates(db_session, monkeypatch):
    admin = _make_admin(db_session, email="admin@example.com", password="old-pass", is_active=False)
    old_hash = admin.password_hash
    monkeypatch.setattr(settings, "admin_bootstrap_email", "admin@example.com")
    monkeypatch.setattr(settings, "admin_bootstrap_password", "new-pass")

    updated_admin = bootstrap_admin_account(db_session)

    assert updated_admin is not None
    assert updated_admin.id == admin.id
    assert updated_admin.password_hash != old_hash
    assert updated_admin.is_active is True


def test_admin_login_success(db_session):
    _make_admin(db_session, email="admin@example.com", password="secret123")

    response = client.post(
        "/api/v1/admin/auth/login",
        json={"email": "Admin@example.com", "password": "secret123"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["admin"]["email"] == "admin@example.com"
    assert data["admin"]["is_active"] is True
    assert data["token_type"] == "bearer"


def test_admin_login_rejects_invalid_password_and_inactive_admin(db_session):
    _make_admin(db_session, email="admin@example.com", password="secret123", is_active=False)

    inactive_response = client.post(
        "/api/v1/admin/auth/login",
        json={"email": "admin@example.com", "password": "secret123"},
    )
    assert inactive_response.status_code == 401

    db_session.query(Admin).delete()
    db_session.commit()
    _make_admin(db_session, email="admin@example.com", password="secret123", is_active=True)
    wrong_password_response = client.post(
        "/api/v1/admin/auth/login",
        json={"email": "admin@example.com", "password": "wrong-pass"},
    )
    assert wrong_password_response.status_code == 401


def test_admin_refresh_me_logout_and_token_isolation(db_session):
    admin = _make_admin(db_session, email="admin@example.com", password="secret123")
    user = _make_user(db_session, email="user@example.com", name="User One")
    admin_refresh = create_refresh_token(admin.id, principal_type=ADMIN_PRINCIPAL_TYPE)
    admin_access = create_access_token(admin.id, principal_type=ADMIN_PRINCIPAL_TYPE)
    user_access = create_access_token(user.id, principal_type=USER_PRINCIPAL_TYPE)

    refresh_response = client.post(
        "/api/v1/admin/auth/refresh",
        json={"refresh_token": admin_refresh},
    )
    assert refresh_response.status_code == 200
    assert "access_token" in refresh_response.json()

    me_response = client.get("/api/v1/admin/auth/me", headers=_auth_header(admin_access))
    assert me_response.status_code == 200
    assert me_response.json()["email"] == "admin@example.com"

    logout_response = client.post("/api/v1/admin/auth/logout", headers=_auth_header(admin_access))
    assert logout_response.status_code == 200

    user_on_admin_response = client.get("/api/v1/admin/auth/me", headers=_auth_header(user_access))
    assert user_on_admin_response.status_code == 401

    admin_on_user_response = client.get("/api/v1/auth/me", headers=_auth_header(admin_access))
    assert admin_on_user_response.status_code == 401


def test_admin_users_list_supports_pagination_search_filters_sorting_and_aggregates(db_session):
    admin = _make_admin(db_session)
    now = datetime.utcnow()
    free_user = _make_user(
        db_session,
        email="free@example.com",
        name="Free User",
        nickname="starter",
        provider="google",
        onboarding_completed=False,
        onboarding_step=1,
    )
    premium_user = _make_user(
        db_session,
        email="premium@example.com",
        name="Premium User",
        nickname="boss",
        provider="apple",
        onboarding_completed=True,
        onboarding_step=5,
        plan="vuvl_plus",
        subscription_expires_at=now + timedelta(days=10),
    )
    trial_user = _make_user(
        db_session,
        email="trial@example.com",
        name="Trial User",
        nickname="helper",
        provider="google",
        onboarding_completed=True,
        onboarding_step=4,
        trial_expires_at=now + timedelta(days=2),
    )

    db_session.add_all(
        [
            Usage(user_id=free_user.id, date=date.today() - timedelta(days=1), request_count=1, chat_count=1, voice_count=0, minutes_used=0.5),
            Usage(user_id=free_user.id, date=date.today(), request_count=2, chat_count=1, voice_count=1, minutes_used=1.0),
            Usage(user_id=premium_user.id, date=date.today(), request_count=10, chat_count=8, voice_count=2, minutes_used=6.0),
        ]
    )
    db_session.add_all(
        [
            Conversation(id="conv-free", user_id=free_user.id, updated_at=now - timedelta(hours=2), created_at=now - timedelta(days=2)),
            Conversation(id="conv-premium", user_id=premium_user.id, updated_at=now - timedelta(hours=1), created_at=now - timedelta(days=1)),
        ]
    )
    db_session.commit()

    token = create_access_token(admin.id, principal_type=ADMIN_PRINCIPAL_TYPE)
    response = client.get(
        "/api/v1/admin/users",
        headers=_auth_header(token),
        params={"search": "boss", "provider": "apple", "plan": "vuvl_plus"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 1
    assert data["page_size"] == 20
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["email"] == "premium@example.com"
    assert data["items"][0]["total_request_count"] == 10
    assert data["items"][0]["last_activity_at"] is not None

    sorted_response = client.get(
        "/api/v1/admin/users",
        headers=_auth_header(token),
        params={"sort_by": "last_activity_at", "sort_order": "desc", "page_size": 2},
    )
    assert sorted_response.status_code == 200
    sorted_items = sorted_response.json()["items"]
    assert len(sorted_items) == 2
    assert sorted_items[0]["email"] == "premium@example.com"
    assert sorted_items[1]["email"] == "free@example.com"

    google_onboarded_response = client.get(
        "/api/v1/admin/users",
        headers=_auth_header(token),
        params={"provider": "google", "onboarding_completed": True},
    )
    assert google_onboarded_response.status_code == 200
    google_items = google_onboarded_response.json()["items"]
    assert len(google_items) == 1
    assert google_items[0]["email"] == "trial@example.com"
    assert google_items[0]["total_request_count"] == 0


def test_admin_user_detail_returns_profile_usage_history_recent_activity_and_404(db_session):
    admin = _make_admin(db_session)
    user = _make_user(
        db_session,
        email="learner@example.com",
        name="Learner",
        nickname="practice-pro",
        onboarding_completed=True,
        onboarding_step=5,
    )
    now = datetime.utcnow()
    conversation = Conversation(
        id="conv-1",
        user_id=user.id,
        title="Interview practice",
        created_at=now - timedelta(days=3),
        updated_at=now - timedelta(hours=3),
    )
    db_session.add(conversation)
    db_session.flush()
    db_session.add_all(
        [
            Message(
                conversation_id=conversation.id,
                user_message="I has interview tomorrow",
                ai_reply="You can say: I have an interview tomorrow.",
                correction="I has interview tomorrow -> I have an interview tomorrow",
                created_at=now - timedelta(hours=3),
            ),
            Usage(
                user_id=user.id,
                date=date.today() - timedelta(days=1),
                request_count=2,
                chat_count=1,
                voice_count=1,
                minutes_used=1.25,
            ),
            Usage(
                user_id=user.id,
                date=date.today(),
                request_count=3,
                chat_count=2,
                voice_count=1,
                minutes_used=2.5,
            ),
        ]
    )
    db_session.commit()

    token = create_access_token(admin.id, principal_type=ADMIN_PRINCIPAL_TYPE)
    response = client.get(f"/api/v1/admin/users/{user.id}", headers=_auth_header(token))

    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "learner@example.com"
    assert data["usage_summary"]["today"]["request_count"] == 3
    assert data["usage_summary"]["totals"]["request_count"] == 5
    assert len(data["usage_history"]) == 2
    assert data["usage_history"][0]["date"] == (date.today() - timedelta(days=1)).isoformat()
    assert len(data["recent_activity"]) == 1
    assert data["recent_activity"][0]["title"] == "Interview practice"
    assert data["recent_activity"][0]["message_count"] == 1
    assert data["recent_activity"][0]["last_user_message"] == "I has interview tomorrow"
    assert data["recent_activity"][0]["last_ai_reply"] == "You can say: I have an interview tomorrow."

    missing_response = client.get("/api/v1/admin/users/missing-user", headers=_auth_header(token))
    assert missing_response.status_code == 404
