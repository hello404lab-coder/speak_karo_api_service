"""Tests for Razorpay-backed subscription flows."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app
from app.models.billing import BillingCoupon
from app.models.usage import Base, Usage
from app.models.user import User
from app.services.billing_service import (
    verify_razorpay_checkout_signature,
    verify_razorpay_webhook_signature,
)
from app.services.subscription_service import resolve_user_plan, update_usage_stats


def _unix_ts(delta_days: int = 0, delta_minutes: int = 0) -> int:
    now = datetime.now(timezone.utc) + timedelta(days=delta_days, minutes=delta_minutes)
    return int(now.timestamp())


def _configure_razorpay(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "dev")
    monkeypatch.setattr(settings, "razorpay_key_id", "rzp_test_123")
    monkeypatch.setattr(settings, "razorpay_key_secret", "secret_123")
    monkeypatch.setattr(settings, "razorpay_webhook_secret", "whsec_current")
    monkeypatch.setattr(settings, "razorpay_webhook_secret_previous", "whsec_previous")
    monkeypatch.setattr(settings, "razorpay_plan_id_vuvl_plus_test", "plan_plus_test")
    monkeypatch.setattr(settings, "razorpay_plan_id_vuvl_pro_test", "plan_pro_test")
    monkeypatch.setattr(settings, "razorpay_plan_id_vuvl_plus_live", "plan_plus_live")
    monkeypatch.setattr(settings, "razorpay_plan_id_vuvl_pro_live", "plan_pro_live")
    monkeypatch.setattr(settings, "razorpay_checkout_reuse_minutes", 30)
    monkeypatch.setattr(settings, "razorpay_monthly_total_count", 1200)


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def memory_db():
    from app.models import billing as _billing  # noqa: F401
    from app.models import live_session as _live  # noqa: F401
    from app.models import social_session as _social  # noqa: F401
    from app.models import admin as _admin  # noqa: F401

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
    db = SessionLocal()
    try:
        user = User(
            id="sub-user-1",
            email="sub@example.com",
            name="Sub User",
            provider="google",
            provider_id="google-sub-user-1",
            onboarding_completed=True,
        )
        db.add(user)
        db.commit()
        yield db
    finally:
        db.close()
        engine.dispose()


def _install_auth_overrides(memory_db: Session, user_id: str = "sub-user-1") -> User:
    def _db():
        yield memory_db

    def _user():
        return memory_db.query(User).filter(User.id == user_id).first()

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_current_user] = _user
    return memory_db.query(User).filter(User.id == user_id).first()


def test_signature_helpers_require_exact_payload(monkeypatch):
    _configure_razorpay(monkeypatch)

    payment_id = "pay_123"
    subscription_id = "sub_123"
    signature = hmac.new(
        settings.razorpay_key_secret.encode("utf-8"),
        f"{payment_id}|{subscription_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert verify_razorpay_checkout_signature(
        payment_id=payment_id,
        subscription_id=subscription_id,
        signature=signature,
    ) is True
    assert verify_razorpay_checkout_signature(
        payment_id=payment_id,
        subscription_id=subscription_id,
        signature="bad-signature",
    ) is False

    raw_body = b'{"event":"subscription.charged"}'
    webhook_signature = hmac.new(
        settings.razorpay_webhook_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    assert verify_razorpay_webhook_signature(raw_body, webhook_signature) is True
    assert verify_razorpay_webhook_signature(raw_body + b" ", webhook_signature) is False


def test_resolve_user_plan_handles_new_paid_tiers_and_legacy_premium():
    future = datetime.utcnow() + timedelta(days=10)
    pro_user = User(
        id="u-pro",
        email="pro@example.com",
        name="Pro",
        provider="google",
        provider_id="google-pro",
        plan="vuvl_pro",
        subscription_expires_at=future,
    )
    legacy_user = User(
        id="u-legacy",
        email="legacy@example.com",
        name="Legacy",
        provider="google",
        provider_id="google-legacy",
        plan="premium",
        subscription_expires_at=future,
    )
    trial_paid_user = User(
        id="u-trial-paid",
        email="trial-paid@example.com",
        name="Trial Paid",
        provider="google",
        provider_id="google-trial-paid",
        plan="vuvl_pro",
        trial_expires_at=future,
    )

    assert resolve_user_plan(pro_user) == "vuvl_pro"
    assert resolve_user_plan(legacy_user) == "vuvl_plus"
    assert resolve_user_plan(trial_paid_user) == "vuvl_pro"


def test_update_usage_stats_records_modality_seconds(memory_db):
    update_usage_stats(
        "sub-user-1",
        memory_db,
        duration_seconds=30.0,
        usage_type="voice",
        llm_output_tokens=125,
        stt_seconds=4.0,
        tts_seconds=6.5,
    )

    usage = memory_db.query(Usage).filter(Usage.user_id == "sub-user-1").first()
    assert usage is not None
    assert usage.minutes_used == 0.5
    assert usage.voice_count == 1
    assert usage.chat_count == 0
    assert usage.request_count == 1
    assert usage.llm_output_tokens == 125
    assert usage.stt_seconds == 4.0
    assert usage.tts_seconds == 6.5


def test_subscription_checkout_verify_status_and_cancel(memory_db, monkeypatch):
    _configure_razorpay(monkeypatch)
    _install_auth_overrides(memory_db)
    client = TestClient(app)

    class _FakeRazorpayClient:
        def create_subscription(self, payload):
            assert payload["plan_id"] == "plan_plus_test"
            return {
                "id": "sub_checkout_1",
                "plan_id": payload["plan_id"],
                "status": "created",
                "short_url": "https://rzp.test/checkout-1",
                "customer_id": None,
                "current_start": None,
                "current_end": None,
                "charge_at": _unix_ts(delta_minutes=5),
                "start_at": _unix_ts(delta_minutes=5),
                "end_at": _unix_ts(delta_days=365),
                "expire_by": _unix_ts(delta_minutes=30),
                "notes": payload["notes"],
            }

        def fetch_subscription(self, provider_subscription_id):
            assert provider_subscription_id == "sub_checkout_1"
            return {
                "id": provider_subscription_id,
                "plan_id": "plan_plus_test",
                "status": "active",
                "customer_id": "cust_123",
                "short_url": "https://rzp.test/checkout-1",
                "current_start": _unix_ts(),
                "current_end": _unix_ts(delta_days=30),
                "charge_at": _unix_ts(delta_days=30),
                "start_at": _unix_ts(),
                "end_at": _unix_ts(delta_days=365),
                "expire_by": _unix_ts(delta_minutes=30),
                "notes": {"user_id": "sub-user-1", "plan_code": "vuvl_plus"},
            }

        def cancel_subscription(self, provider_subscription_id, *, cancel_at_cycle_end):
            assert provider_subscription_id == "sub_checkout_1"
            assert cancel_at_cycle_end is True
            return {
                "id": provider_subscription_id,
                "plan_id": "plan_plus_test",
                "status": "active",
                "customer_id": "cust_123",
                "short_url": "https://rzp.test/checkout-1",
                "current_start": _unix_ts(),
                "current_end": _unix_ts(delta_days=30),
                "charge_at": _unix_ts(delta_days=30),
                "start_at": _unix_ts(),
                "end_at": _unix_ts(delta_days=365),
                "expire_by": _unix_ts(delta_minutes=30),
                "cancel_at_cycle_end": True,
                "notes": {"user_id": "sub-user-1", "plan_code": "vuvl_plus"},
            }

    monkeypatch.setattr("app.api.subscription.get_razorpay_client", lambda: _FakeRazorpayClient())

    checkout = client.post("/api/v1/subscription/checkout", json={"plan_code": "vuvl_plus"})
    assert checkout.status_code == 200
    assert checkout.json()["subscription_id"] == "sub_checkout_1"
    assert checkout.json()["reuse_existing"] is False

    reused = client.post("/api/v1/subscription/checkout", json={"plan_code": "vuvl_plus"})
    assert reused.status_code == 200
    assert reused.json()["reuse_existing"] is True

    signature = hmac.new(
        settings.razorpay_key_secret.encode("utf-8"),
        b"pay_checkout_1|sub_checkout_1",
        hashlib.sha256,
    ).hexdigest()
    verify = client.post(
        "/api/v1/subscription/verify",
        json={
            "razorpay_payment_id": "pay_checkout_1",
            "razorpay_subscription_id": "sub_checkout_1",
            "razorpay_signature": signature,
        },
    )
    assert verify.status_code == 200
    assert verify.json()["plan"] == "vuvl_plus"
    assert verify.json()["billing_status"] == "active"

    status_response = client.get("/api/v1/subscription/status")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["plan"] == "vuvl_plus"
    assert status_payload["billing_status"] == "active"
    assert status_payload["active_subscription_id"] == "sub_checkout_1"
    assert status_payload["manage_actions"]["can_cancel"] is True

    cancel = client.post("/api/v1/subscription/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["cancel_at_cycle_end"] is True


def test_razorpay_webhook_is_idempotent_and_updates_entitlement(memory_db, monkeypatch):
    _configure_razorpay(monkeypatch)
    client = TestClient(app)

    def _db():
        yield memory_db

    app.dependency_overrides[get_db] = _db

    payload = {
        "event": "subscription.charged",
        "payload": {
            "subscription": {
                "entity": {
                    "id": "sub_wh_1",
                    "plan_id": "plan_pro_test",
                    "status": "active",
                    "customer_id": "cust_wh_1",
                    "current_start": _unix_ts(),
                    "current_end": _unix_ts(delta_days=30),
                    "charge_at": _unix_ts(delta_days=30),
                    "start_at": _unix_ts(),
                    "end_at": _unix_ts(delta_days=365),
                    "expire_by": _unix_ts(delta_minutes=30),
                    "notes": {"user_id": "sub-user-1", "plan_code": "vuvl_pro"},
                }
            },
            "payment": {"entity": {"id": "pay_wh_1"}},
        },
    }
    raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(
        settings.razorpay_webhook_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    first = client.post(
        "/api/v1/subscription/webhooks/razorpay",
        data=raw_body,
        headers={"X-Razorpay-Signature": signature, "Content-Type": "application/json"},
    )
    assert first.status_code == 200
    assert first.json()["duplicate"] is False

    second = client.post(
        "/api/v1/subscription/webhooks/razorpay",
        data=raw_body,
        headers={"X-Razorpay-Signature": signature, "Content-Type": "application/json"},
    )
    assert second.status_code == 200
    assert second.json()["duplicate"] is True

    user = memory_db.query(User).filter(User.id == "sub-user-1").first()
    assert user.plan == "vuvl_pro"
    assert resolve_user_plan(user) == "vuvl_pro"
    assert user.subscription_expires_at is not None


def test_intro_trial_checkout_verify_and_invoice_paid(memory_db, monkeypatch):
    _configure_razorpay(monkeypatch)
    _install_auth_overrides(memory_db)
    client = TestClient(app)

    class _FakeRazorpayClient:
        def __init__(self):
            self.fetch_count = 0

        def create_subscription(self, payload):
            assert payload["notes"]["checkout_mode"] == "intro_trial"
            assert payload["notes"]["coupon_code"] == ""
            assert "start_at" in payload
            return {
                "id": "sub_trial_1",
                "plan_id": payload["plan_id"],
                "status": "created",
                "short_url": "https://rzp.test/trial-1",
                "customer_id": None,
                "current_start": None,
                "current_end": None,
                "charge_at": payload["start_at"],
                "start_at": payload["start_at"],
                "end_at": _unix_ts(delta_days=365),
                "expire_by": _unix_ts(delta_minutes=30),
                "notes": payload["notes"],
            }

        def fetch_subscription(self, provider_subscription_id):
            assert provider_subscription_id == "sub_trial_1"
            self.fetch_count += 1
            if self.fetch_count == 1:
                return {
                    "id": provider_subscription_id,
                    "plan_id": "plan_plus_test",
                    "status": "authenticated",
                    "customer_id": "cust_trial_1",
                    "short_url": "https://rzp.test/trial-1",
                    "current_start": None,
                    "current_end": None,
                    "charge_at": _unix_ts(delta_days=3),
                    "start_at": _unix_ts(delta_days=3),
                    "end_at": _unix_ts(delta_days=365),
                    "expire_by": _unix_ts(delta_minutes=30),
                    "notes": {
                        "user_id": "sub-user-1",
                        "plan_code": "vuvl_plus",
                        "checkout_mode": "intro_trial",
                        "coupon_code": "",
                    },
                }
            return {
                "id": provider_subscription_id,
                "plan_id": "plan_plus_test",
                "status": "active",
                "customer_id": "cust_trial_1",
                "short_url": "https://rzp.test/trial-1",
                "current_start": _unix_ts(),
                "current_end": _unix_ts(delta_days=30),
                "charge_at": _unix_ts(delta_days=30),
                "start_at": _unix_ts(),
                "end_at": _unix_ts(delta_days=365),
                "expire_by": _unix_ts(delta_minutes=30),
                "notes": {
                    "user_id": "sub-user-1",
                    "plan_code": "vuvl_plus",
                    "checkout_mode": "intro_trial",
                    "coupon_code": "",
                },
            }

    fake_client = _FakeRazorpayClient()
    monkeypatch.setattr("app.api.subscription.get_razorpay_client", lambda: fake_client)

    checkout = client.post("/api/v1/subscription/checkout", json={"plan_code": "vuvl_plus"})
    assert checkout.status_code == 200
    assert checkout.json()["applied_mode"] == "intro_trial"
    assert checkout.json()["coupon_code"] is None
    assert checkout.json()["trial_ends_at"] is not None

    signature = hmac.new(
        settings.razorpay_key_secret.encode("utf-8"),
        b"pay_trial_1|sub_trial_1",
        hashlib.sha256,
    ).hexdigest()
    verify = client.post(
        "/api/v1/subscription/verify",
        json={
            "razorpay_payment_id": "pay_trial_1",
            "razorpay_subscription_id": "sub_trial_1",
            "razorpay_signature": signature,
        },
    )
    assert verify.status_code == 200
    assert verify.json()["plan"] == "vuvl_plus"
    assert verify.json()["billing_phase"] == "trial"
    assert verify.json()["trial_expires_at"] is not None

    status_before = client.get("/api/v1/subscription/status")
    assert status_before.status_code == 200
    assert status_before.json()["billing_phase"] == "trial"

    user = memory_db.query(User).filter(User.id == "sub-user-1").first()
    assert user.is_trial_used is True
    assert resolve_user_plan(user) == "vuvl_plus"
    assert user.trial_expires_at is not None

    invoice_payload = {
        "event": "invoice.paid",
        "payload": {
            "invoice": {
                "entity": {
                    "id": "inv_trial_1",
                    "subscription_id": "sub_trial_1",
                    "payment_id": "pay_invoice_1",
                    "status": "paid",
                }
            },
            "payment": {"entity": {"id": "pay_invoice_1"}},
        },
    }
    raw_body = json.dumps(invoice_payload, separators=(",", ":")).encode("utf-8")
    webhook_signature = hmac.new(
        settings.razorpay_webhook_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    webhook = client.post(
        "/api/v1/subscription/webhooks/razorpay",
        data=raw_body,
        headers={"X-Razorpay-Signature": webhook_signature, "Content-Type": "application/json"},
    )
    assert webhook.status_code == 200
    assert webhook.json()["duplicate"] is False

    user = memory_db.query(User).filter(User.id == "sub-user-1").first()
    assert user.trial_expires_at is None
    assert user.subscription_expires_at is not None
    assert resolve_user_plan(user) == "vuvl_plus"

    status_after = client.get("/api/v1/subscription/status")
    assert status_after.status_code == 200
    assert status_after.json()["billing_phase"] == "active"


def test_coupon_trial_extension_validate_checkout_and_limit(memory_db, monkeypatch):
    _configure_razorpay(monkeypatch)
    _install_auth_overrides(memory_db)
    client = TestClient(app)

    coupon = BillingCoupon(
        code="VUVL20",
        name="Two Months Free",
        status="enabled",
        mode="trial_extension",
        applies_to_plan_code="vuvl_plus",
        free_cycles=2,
        max_redemptions_total=1,
        max_redemptions_per_user=1,
    )
    memory_db.add(coupon)
    memory_db.commit()

    class _FakeRazorpayClient:
        def create_subscription(self, payload):
            assert payload["notes"]["coupon_code"] == "VUVL20"
            assert payload["notes"]["checkout_mode"] == "coupon_trial_extension"
            assert "start_at" in payload
            assert "offer_id" not in payload
            return {
                "id": "sub_coupon_trial_1",
                "plan_id": payload["plan_id"],
                "status": "created",
                "short_url": "https://rzp.test/coupon-trial-1",
                "customer_id": None,
                "current_start": None,
                "current_end": None,
                "charge_at": payload["start_at"],
                "start_at": payload["start_at"],
                "end_at": _unix_ts(delta_days=365),
                "expire_by": _unix_ts(delta_minutes=30),
                "notes": payload["notes"],
            }

        def fetch_subscription(self, provider_subscription_id):
            assert provider_subscription_id == "sub_coupon_trial_1"
            return {
                "id": provider_subscription_id,
                "plan_id": "plan_plus_test",
                "status": "authenticated",
                "customer_id": "cust_coupon_trial_1",
                "short_url": "https://rzp.test/coupon-trial-1",
                "current_start": None,
                "current_end": None,
                "charge_at": _unix_ts(delta_days=60),
                "start_at": _unix_ts(delta_days=60),
                "end_at": _unix_ts(delta_days=365),
                "expire_by": _unix_ts(delta_minutes=30),
                "notes": {
                    "user_id": "sub-user-1",
                    "plan_code": "vuvl_plus",
                    "checkout_mode": "coupon_trial_extension",
                    "coupon_code": "VUVL20",
                },
            }

    monkeypatch.setattr("app.api.subscription.get_razorpay_client", lambda: _FakeRazorpayClient())

    validation = client.post(
        "/api/v1/subscription/coupons/validate",
        json={"plan_code": "vuvl_plus", "coupon_code": "VUVL20"},
    )
    assert validation.status_code == 200
    assert validation.json()["eligible"] is True
    assert validation.json()["applied_mode"] == "coupon_trial_extension"
    assert validation.json()["free_cycles"] == 2

    checkout = client.post(
        "/api/v1/subscription/checkout",
        json={"plan_code": "vuvl_plus", "coupon_code": "VUVL20"},
    )
    assert checkout.status_code == 200
    assert checkout.json()["applied_mode"] == "coupon_trial_extension"
    assert checkout.json()["coupon_code"] == "VUVL20"

    signature = hmac.new(
        settings.razorpay_key_secret.encode("utf-8"),
        b"pay_coupon_trial_1|sub_coupon_trial_1",
        hashlib.sha256,
    ).hexdigest()
    verify = client.post(
        "/api/v1/subscription/verify",
        json={
            "razorpay_payment_id": "pay_coupon_trial_1",
            "razorpay_subscription_id": "sub_coupon_trial_1",
            "razorpay_signature": signature,
        },
    )
    assert verify.status_code == 200
    assert verify.json()["billing_phase"] == "trial"
    assert verify.json()["coupon_code"] == "VUVL20"

    status_response = client.get("/api/v1/subscription/status")
    assert status_response.status_code == 200
    assert status_response.json()["coupon_code"] == "VUVL20"

    second_user = User(
        id="sub-user-2",
        email="sub2@example.com",
        name="Sub User 2",
        provider="google",
        provider_id="google-sub-user-2",
        onboarding_completed=True,
    )
    memory_db.add(second_user)
    memory_db.commit()
    _install_auth_overrides(memory_db, user_id="sub-user-2")
    exhausted = client.post(
        "/api/v1/subscription/coupons/validate",
        json={"plan_code": "vuvl_plus", "coupon_code": "VUVL20"},
    )
    assert exhausted.status_code == 200
    assert exhausted.json()["eligible"] is False


def test_offer_coupon_checkout_applies_razorpay_offer(memory_db, monkeypatch):
    _configure_razorpay(monkeypatch)
    _install_auth_overrides(memory_db)
    client = TestClient(app)

    coupon = BillingCoupon(
        code="HALFOFF",
        name="Half Off",
        status="enabled",
        mode="razorpay_offer",
        applies_to_plan_code="vuvl_pro",
        razorpay_offer_id="offer_123",
        discount_type="percentage",
        discount_value=50,
        max_redemptions_total=10,
        max_redemptions_per_user=1,
    )
    memory_db.add(coupon)
    memory_db.commit()

    class _FakeRazorpayClient:
        def create_subscription(self, payload):
            assert payload["offer_id"] == "offer_123"
            assert "start_at" not in payload
            return {
                "id": "sub_offer_1",
                "plan_id": payload["plan_id"],
                "status": "created",
                "short_url": "https://rzp.test/offer-1",
                "customer_id": None,
                "current_start": None,
                "current_end": None,
                "charge_at": _unix_ts(delta_minutes=5),
                "start_at": _unix_ts(delta_minutes=5),
                "end_at": _unix_ts(delta_days=365),
                "expire_by": _unix_ts(delta_minutes=30),
                "offer_id": "offer_123",
                "notes": payload["notes"],
            }

        def fetch_subscription(self, provider_subscription_id):
            assert provider_subscription_id == "sub_offer_1"
            return {
                "id": provider_subscription_id,
                "plan_id": "plan_pro_test",
                "status": "active",
                "customer_id": "cust_offer_1",
                "short_url": "https://rzp.test/offer-1",
                "current_start": _unix_ts(),
                "current_end": _unix_ts(delta_days=30),
                "charge_at": _unix_ts(delta_days=30),
                "start_at": _unix_ts(),
                "end_at": _unix_ts(delta_days=365),
                "expire_by": _unix_ts(delta_minutes=30),
                "offer_id": "offer_123",
                "notes": {
                    "user_id": "sub-user-1",
                    "plan_code": "vuvl_pro",
                    "checkout_mode": "coupon_offer_discount",
                    "coupon_code": "HALFOFF",
                },
            }

    monkeypatch.setattr("app.api.subscription.get_razorpay_client", lambda: _FakeRazorpayClient())

    checkout = client.post(
        "/api/v1/subscription/checkout",
        json={"plan_code": "vuvl_pro", "coupon_code": "HALFOFF"},
    )
    assert checkout.status_code == 200
    assert checkout.json()["applied_mode"] == "coupon_offer_discount"
    assert checkout.json()["coupon_code"] == "HALFOFF"

    signature = hmac.new(
        settings.razorpay_key_secret.encode("utf-8"),
        b"pay_offer_1|sub_offer_1",
        hashlib.sha256,
    ).hexdigest()
    verify = client.post(
        "/api/v1/subscription/verify",
        json={
            "razorpay_payment_id": "pay_offer_1",
            "razorpay_subscription_id": "sub_offer_1",
            "razorpay_signature": signature,
        },
    )
    assert verify.status_code == 200
    assert verify.json()["billing_phase"] == "active"
    assert verify.json()["coupon_code"] == "HALFOFF"

    status_response = client.get("/api/v1/subscription/status")
    assert status_response.status_code == 200
    assert status_response.json()["coupon_code"] == "HALFOFF"
    assert status_response.json()["active_plan_code"] == "vuvl_pro"
