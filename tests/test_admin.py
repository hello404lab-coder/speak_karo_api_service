"""Tests for admin auth and admin user management endpoints."""
from datetime import date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

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
from app.models.billing import BillingCoupon, BillingCouponRedemption, BillingSubscription
from app.models.usage import Conversation, Message, Usage
from app.models.user import User
from app.services.admin_auth_service import bootstrap_admin_account

client = TestClient(app)


@pytest.fixture()
def db_session(test_engine):
    """Per-test transactional PG session bound to FastAPI's ``get_db`` dependency.

    Overrides the plain ``db_session`` fixture from ``conftest.py`` so the
    TestClient shares the same connection (and sees uncommitted writes) as the
    test body.
    """
    connection = test_engine.connect()
    transaction = connection.begin()
    TestingSessionLocal = sessionmaker(
        bind=connection,
        autocommit=False,
        autoflush=False,
        join_transaction_mode="create_savepoint",
    )
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
        app.dependency_overrides.pop(get_db, None)
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()


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
            Usage(
                user_id=free_user.id,
                date=date.today(),
                request_count=2,
                chat_count=1,
                voice_count=1,
                minutes_used=1.0,
                llm_output_tokens=140,
                stt_seconds=3.0,
                tts_seconds=6.0,
            ),
            Usage(
                user_id=premium_user.id,
                date=date.today(),
                request_count=10,
                chat_count=8,
                voice_count=2,
                minutes_used=6.0,
                llm_output_tokens=900,
                stt_seconds=12.0,
                tts_seconds=40.0,
            ),
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
    assert data["items"][0]["total_llm_output_tokens"] == 900
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
                llm_output_tokens=120,
                stt_seconds=3.0,
                tts_seconds=4.0,
            ),
            Usage(
                user_id=user.id,
                date=date.today(),
                request_count=3,
                chat_count=2,
                voice_count=1,
                minutes_used=2.5,
                llm_output_tokens=150,
                stt_seconds=4.0,
                tts_seconds=7.0,
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
    assert data["usage_summary"]["today"]["llm_output_tokens"] == 150
    assert data["usage_summary"]["totals"]["request_count"] == 5
    assert data["usage_summary"]["totals"]["llm_output_tokens"] == 270
    assert data["usage_summary"]["totals"]["tts_seconds"] == 11.0
    assert len(data["usage_history"]) == 2
    assert data["usage_history"][0]["date"] == (date.today() - timedelta(days=1)).isoformat()
    assert data["usage_history"][0]["stt_seconds"] == 3.0
    assert len(data["recent_activity"]) == 1
    assert data["recent_activity"][0]["title"] == "Interview practice"
    assert data["recent_activity"][0]["message_count"] == 1
    assert data["recent_activity"][0]["last_user_message"] == "I has interview tomorrow"
    assert data["recent_activity"][0]["last_ai_reply"] == "You can say: I have an interview tomorrow."

    missing_response = client.get("/api/v1/admin/users/missing-user", headers=_auth_header(token))
    assert missing_response.status_code == 404


def test_admin_coupon_crud_and_redemptions(db_session):
    admin = _make_admin(db_session)
    token = create_access_token(admin.id, principal_type=ADMIN_PRINCIPAL_TYPE)

    create_response = client.post(
        "/api/v1/admin/coupons",
        headers=_auth_header(token),
        json={
            "code": "VUVL20",
            "name": "Two Months Free",
            "mode": "trial_extension",
            "applies_to_plan_code": "vuvl_plus",
            "status": "enabled",
            "free_cycles": 2,
            "max_redemptions_total": 20,
            "max_redemptions_per_user": 1,
            "metadata_json": {"campaign": "launch"},
        },
    )
    assert create_response.status_code == 201
    coupon = create_response.json()
    assert coupon["code"] == "VUVL20"
    assert coupon["free_cycles"] == 2
    assert coupon["total_redemptions"] == 0

    list_response = client.get("/api/v1/admin/coupons", headers=_auth_header(token))
    assert list_response.status_code == 200
    assert list_response.json()["items"][0]["code"] == "VUVL20"

    detail_response = client.get(f"/api/v1/admin/coupons/{coupon['id']}", headers=_auth_header(token))
    assert detail_response.status_code == 200
    assert detail_response.json()["name"] == "Two Months Free"

    patch_response = client.patch(
        f"/api/v1/admin/coupons/{coupon['id']}",
        headers=_auth_header(token),
        json={"name": "Two Free Months", "max_redemptions_total": 25},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["name"] == "Two Free Months"
    assert patch_response.json()["max_redemptions_total"] == 25

    user = _make_user(db_session, email="coupon-user@example.com", onboarding_completed=True)
    subscription = BillingSubscription(
        user_id=user.id,
        provider="razorpay",
        plan_code="vuvl_plus",
        provider_plan_id="plan_plus_test",
        provider_subscription_id="sub_coupon_admin_1",
        status="authenticated",
        billing_phase="trial",
        coupon_id=coupon["id"],
        coupon_code_snapshot="VUVL20",
        trial_access_until=datetime.utcnow() + timedelta(days=30),
    )
    db_session.add(subscription)
    db_session.flush()
    redemption = BillingCouponRedemption(
        coupon_id=coupon["id"],
        user_id=user.id,
        billing_subscription_id=subscription.id,
        status="consumed",
        coupon_code_snapshot="VUVL20",
        effect_snapshot={"applied_mode": "coupon_trial_extension", "free_cycles": 2},
        verified_at=datetime.utcnow(),
        consumed_at=datetime.utcnow(),
    )
    db_session.add(redemption)
    db_session.commit()

    redemptions_response = client.get(
        f"/api/v1/admin/coupons/{coupon['id']}/redemptions",
        headers=_auth_header(token),
    )
    assert redemptions_response.status_code == 200
    items = redemptions_response.json()["items"]
    assert len(items) == 1
    assert items[0]["coupon_code_snapshot"] == "VUVL20"
    assert items[0]["user_email"] == "coupon-user@example.com"

    disable_response = client.post(
        f"/api/v1/admin/coupons/{coupon['id']}/disable",
        headers=_auth_header(token),
    )
    assert disable_response.status_code == 200
    assert disable_response.json()["status"] == "disabled"
