"""Billing service for Razorpay subscription lifecycle and entitlement sync."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.billing import BillingSubscription, BillingWebhookEvent
from app.models.user import User
from app.services.subscription_service import (
    FREE_PLAN,
    PAID_PLAN_CODES,
    PLAN_CATALOG,
    normalize_paid_plan_code,
    normalize_plan_code,
)

RAZORPAY_PROVIDER = "razorpay"
REUSABLE_CHECKOUT_STATUSES = {"created", "authenticated"}
ACCESS_GRANTING_STATUSES = {
    "active",
    "authenticated",
    "pending",
    "halted",
    "paused",
    "cancelled",
    "completed",
}
NON_ACCESS_STATUSES = {"created", "expired"}


def utcnow() -> datetime:
    """Return naive UTC datetime for DB compatibility."""
    return datetime.utcnow()


def unix_to_utc_naive(value: Any) -> datetime | None:
    """Convert Razorpay unix timestamp into naive UTC datetime."""
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(parsed, tz=timezone.utc).replace(tzinfo=None)


def get_supported_paid_plan_codes() -> tuple[str, ...]:
    """Return paid plans supported by backend catalog."""
    return tuple(PLAN_CATALOG.keys())


def get_configured_plan_id_map() -> dict[str, str]:
    """Return only configured Razorpay plan ids keyed by internal plan code."""
    return {
        plan_code: plan_id
        for plan_code, plan_id in settings.razorpay_plan_id_map.items()
        if plan_id
    }


def get_reverse_plan_id_map() -> dict[str, str]:
    """Return provider plan id -> internal paid plan code mapping."""
    return {plan_id: plan_code for plan_code, plan_id in get_configured_plan_id_map().items()}


def resolve_plan_code_for_provider_plan_id(provider_plan_id: str | None) -> str:
    """Map a Razorpay plan id to an internal plan code."""
    reverse_map = get_reverse_plan_id_map()
    if not provider_plan_id or provider_plan_id not in reverse_map:
        raise ValueError("Unsupported Razorpay plan id")
    return reverse_map[provider_plan_id]


def verify_razorpay_checkout_signature(
    *,
    payment_id: str,
    subscription_id: str,
    signature: str,
) -> bool:
    """Verify Standard Checkout signature for subscription authorization."""
    secret = settings.razorpay_key_secret or ""
    expected = hmac.new(
        secret.encode("utf-8"),
        f"{payment_id}|{subscription_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def verify_razorpay_webhook_signature(raw_body: bytes, signature: str | None) -> bool:
    """Verify webhook signature against current and previous secrets."""
    if not signature:
        return False
    secrets = [settings.razorpay_webhook_secret, settings.razorpay_webhook_secret_previous]
    for secret in secrets:
        if not secret:
            continue
        expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected, signature):
            return True
    return False


def build_checkout_prefill(user: User) -> dict[str, str | None]:
    """Build safe frontend prefill values for Razorpay Checkout."""
    return {
        "name": user.name,
        "email": user.email,
        "contact": None,
    }


def build_checkout_payload(user: User, plan_code: str) -> dict[str, Any]:
    """Create the Razorpay subscription payload for Standard Checkout."""
    provider_plan_id = settings.razorpay_plan_id_for(plan_code)
    if not provider_plan_id:
        raise ValueError("Razorpay plan id is not configured for this plan")
    expiry = utcnow() + timedelta(minutes=settings.razorpay_checkout_reuse_minutes)
    return {
        "plan_id": provider_plan_id,
        "total_count": settings.razorpay_monthly_total_count,
        "quantity": 1,
        "customer_notify": True,
        "expire_by": int(expiry.replace(tzinfo=timezone.utc).timestamp()),
        "notes": {
            "user_id": user.id,
            "plan_code": plan_code,
            "user_email": user.email,
        },
    }


def get_active_paid_subscription(db: Session, user_id: str) -> BillingSubscription | None:
    """Return the current paid subscription granting access, if any."""
    now = utcnow()
    rows = (
        db.query(BillingSubscription)
        .filter(BillingSubscription.user_id == user_id, BillingSubscription.provider == RAZORPAY_PROVIDER)
        .order_by(BillingSubscription.current_end_at.desc(), BillingSubscription.updated_at.desc())
        .all()
    )
    for row in rows:
        if (
            row.plan_code in PAID_PLAN_CODES
            and row.status in ACCESS_GRANTING_STATUSES
            and row.current_end_at
            and row.current_end_at > now
        ):
            return row
    return None


def get_latest_paid_subscription(db: Session, user_id: str) -> BillingSubscription | None:
    """Return the most recently updated paid subscription for a user."""
    return (
        db.query(BillingSubscription)
        .filter(BillingSubscription.user_id == user_id, BillingSubscription.provider == RAZORPAY_PROVIDER)
        .order_by(BillingSubscription.updated_at.desc(), BillingSubscription.created_at.desc())
        .first()
    )


def get_reusable_checkout_subscription(db: Session, user_id: str, plan_code: str) -> BillingSubscription | None:
    """Return a recent pending checkout subscription we can safely reuse."""
    cutoff = utcnow() - timedelta(minutes=settings.razorpay_checkout_reuse_minutes)
    return (
        db.query(BillingSubscription)
        .filter(
            BillingSubscription.user_id == user_id,
            BillingSubscription.provider == RAZORPAY_PROVIDER,
            BillingSubscription.plan_code == plan_code,
            BillingSubscription.status.in_(tuple(REUSABLE_CHECKOUT_STATUSES)),
            BillingSubscription.created_at >= cutoff,
        )
        .order_by(BillingSubscription.created_at.desc())
        .first()
    )


def get_manageable_subscription(db: Session, user_id: str) -> BillingSubscription | None:
    """Return the current paid subscription that can be cancelled."""
    active = get_active_paid_subscription(db, user_id)
    if active:
        return active
    latest = get_latest_paid_subscription(db, user_id)
    if latest and latest.status in {"active", "pending", "halted", "paused"}:
        return latest
    return None


def project_user_paid_entitlement(db: Session, user: User) -> BillingSubscription | None:
    """Recompute paid entitlement cache on the user record from billing rows."""
    active = get_active_paid_subscription(db, user.id)
    if active:
        user.plan = active.plan_code
        user.subscription_expires_at = active.current_end_at
        return active

    latest_paid = get_latest_paid_subscription(db, user.id)
    if latest_paid and latest_paid.plan_code in PAID_PLAN_CODES:
        user.plan = latest_paid.plan_code
    else:
        normalized = normalize_plan_code(user.plan)
        user.plan = normalized if normalized in {FREE_PLAN, "trial"} else FREE_PLAN
    user.subscription_expires_at = None
    return latest_paid


def sync_subscription_from_razorpay(
    db: Session,
    subscription_payload: dict[str, Any],
    *,
    expected_user_id: str | None = None,
    payment_id: str | None = None,
    raw_payload: dict[str, Any] | None = None,
) -> BillingSubscription:
    """Create or update one local billing subscription from Razorpay payload."""
    provider_subscription_id = subscription_payload.get("id")
    if not provider_subscription_id:
        raise ValueError("Razorpay subscription id missing")

    plan_code = resolve_plan_code_for_provider_plan_id(subscription_payload.get("plan_id"))
    existing = (
        db.query(BillingSubscription)
        .filter(BillingSubscription.provider_subscription_id == provider_subscription_id)
        .first()
    )

    note_user_id = ((subscription_payload.get("notes") or {}).get("user_id") or expected_user_id)
    user_id = existing.user_id if existing else note_user_id
    if expected_user_id and user_id and user_id != expected_user_id:
        raise ValueError("Razorpay subscription does not belong to the current user")
    if not user_id:
        raise ValueError("Unable to resolve user for Razorpay subscription")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise ValueError("User for Razorpay subscription was not found")

    row = existing or BillingSubscription(
        user_id=user_id,
        provider=RAZORPAY_PROVIDER,
        provider_subscription_id=provider_subscription_id,
        provider_plan_id=subscription_payload["plan_id"],
        plan_code=plan_code,
        status=str(subscription_payload.get("status") or "created"),
    )
    row.user_id = user_id
    row.provider = RAZORPAY_PROVIDER
    row.plan_code = plan_code
    row.provider_plan_id = subscription_payload["plan_id"]
    row.provider_customer_id = subscription_payload.get("customer_id")
    row.provider_payment_id = payment_id or row.provider_payment_id
    row.status = str(subscription_payload.get("status") or row.status or "created")
    row.short_url = subscription_payload.get("short_url")
    row.current_start_at = unix_to_utc_naive(subscription_payload.get("current_start"))
    row.current_end_at = unix_to_utc_naive(subscription_payload.get("current_end"))
    row.charge_at = unix_to_utc_naive(subscription_payload.get("charge_at"))
    row.start_at = unix_to_utc_naive(subscription_payload.get("start_at"))
    row.end_at = unix_to_utc_naive(subscription_payload.get("end_at"))
    row.expire_by = unix_to_utc_naive(subscription_payload.get("expire_by"))
    row.cancel_at_cycle_end = bool(subscription_payload.get("cancel_at_cycle_end") or False)
    row.cancelled_at = unix_to_utc_naive(subscription_payload.get("cancelled_at"))
    row.ended_at = unix_to_utc_naive(subscription_payload.get("ended_at"))
    row.raw_last_payload = raw_payload or subscription_payload
    db.add(row)
    db.flush()

    project_user_paid_entitlement(db, user)
    db.commit()
    db.refresh(row)
    db.refresh(user)
    return row


def process_razorpay_webhook(db: Session, *, raw_body: bytes, payload: dict[str, Any]) -> tuple[BillingWebhookEvent, bool]:
    """Process one verified Razorpay webhook idempotently."""
    delivery_hash = hashlib.sha256(raw_body).hexdigest()
    event_type = str(payload.get("event") or "unknown")
    subscription_payload = ((payload.get("payload") or {}).get("subscription") or {}).get("entity") or {}
    payment_payload = ((payload.get("payload") or {}).get("payment") or {}).get("entity") or {}
    provider_subscription_id = subscription_payload.get("id")

    row = (
        db.query(BillingWebhookEvent)
        .filter(BillingWebhookEvent.delivery_hash == delivery_hash)
        .first()
    )
    if row and row.processing_status == "processed":
        return row, True

    if not row:
        row = BillingWebhookEvent(
            provider=RAZORPAY_PROVIDER,
            event_type=event_type,
            delivery_hash=delivery_hash,
            provider_subscription_id=provider_subscription_id,
            payload=payload,
            processing_status="processing",
        )
        db.add(row)
    else:
        row.event_type = event_type
        row.provider_subscription_id = provider_subscription_id
        row.payload = payload
        row.processing_status = "processing"

    try:
        if subscription_payload:
            sync_subscription_from_razorpay(
                db,
                subscription_payload,
                payment_id=payment_payload.get("id"),
                raw_payload=payload,
            )
        row.processing_status = "processed"
        row.processed_at = utcnow()
        db.add(row)
        db.commit()
        db.refresh(row)
        return row, False
    except Exception:
        db.rollback()
        db.add(row)
        row.processing_status = "failed"
        row.processed_at = utcnow()
        db.commit()
        raise


def build_catalog_payload(*, current_plan: str, can_checkout: bool) -> list[dict[str, Any]]:
    """Build public plan catalog with dynamic checkout eligibility."""
    items: list[dict[str, Any]] = []
    for plan_code, meta in PLAN_CATALOG.items():
        items.append(
            {
                **meta,
                "eligible_for_checkout": can_checkout and current_plan != plan_code,
            }
        )
    return items


def serialize_subscription_summary(row: BillingSubscription | None) -> dict[str, Any]:
    """Serialize a billing subscription into status response fields."""
    if not row:
        return {
            "billing_status": None,
            "active_subscription_id": None,
            "active_plan_code": None,
            "current_period_end": None,
            "cancel_at_cycle_end": False,
            "manage_actions": {"can_cancel": False},
        }
    return {
        "billing_status": row.status,
        "active_subscription_id": row.provider_subscription_id,
        "active_plan_code": normalize_paid_plan_code(row.plan_code) if row.plan_code in PAID_PLAN_CODES else None,
        "current_period_end": row.current_end_at,
        "cancel_at_cycle_end": bool(row.cancel_at_cycle_end),
        "manage_actions": {
            "can_cancel": row.status in {"active", "pending", "halted", "paused"} and not row.cancel_at_cycle_end,
        },
    }
