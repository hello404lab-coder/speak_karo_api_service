"""Coupon validation, reservation, and admin management helpers."""
from __future__ import annotations

import calendar
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.admin import Admin
from app.models.billing import BillingCoupon, BillingCouponRedemption, BillingSubscription
from app.models.user import User
from app.services.subscription_service import PAID_PLAN_CODES

COUPON_MODE_TRIAL_EXTENSION = "trial_extension"
COUPON_MODE_RAZORPAY_OFFER = "razorpay_offer"
COUPON_STATUS_ENABLED = "enabled"
COUPON_STATUS_DISABLED = "disabled"

CHECKOUT_MODE_STANDARD = "standard"
CHECKOUT_MODE_INTRO_TRIAL = "intro_trial"
CHECKOUT_MODE_COUPON_TRIAL_EXTENSION = "coupon_trial_extension"
CHECKOUT_MODE_COUPON_OFFER_DISCOUNT = "coupon_offer_discount"

REDEMPTION_STATUS_RESERVED = "reserved"
REDEMPTION_STATUS_VERIFIED = "verified"
REDEMPTION_STATUS_CONSUMED = "consumed"
REDEMPTION_STATUS_EXPIRED = "expired"
REDEMPTION_STATUS_REVOKED = "revoked"

INTRO_TRIAL_DAYS = 3
_ACTIVE_REDEMPTION_STATUSES = {
    REDEMPTION_STATUS_RESERVED,
    REDEMPTION_STATUS_VERIFIED,
    REDEMPTION_STATUS_CONSUMED,
}


def utcnow() -> datetime:
    """Return naive UTC datetime for DB compatibility."""
    return datetime.utcnow()


def normalize_coupon_code(code: str | None) -> str | None:
    """Normalize a coupon code for storage and comparison."""
    if code is None:
        return None
    normalized = "".join(str(code).strip().upper().split())
    return normalized or None


def add_months(value: datetime, months: int) -> datetime:
    """Add whole calendar months while preserving the clock portion."""
    total_month = (value.month - 1) + max(0, int(months))
    year = value.year + total_month // 12
    month = (total_month % 12) + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def expire_stale_coupon_redemptions(db: Session) -> None:
    """Release coupon reservations tied to abandoned checkout attempts."""
    cutoff = utcnow() - timedelta(minutes=settings.razorpay_checkout_reuse_minutes)
    stale_rows = (
        db.query(BillingCouponRedemption)
        .filter(
            BillingCouponRedemption.status == REDEMPTION_STATUS_RESERVED,
            BillingCouponRedemption.created_at < cutoff,
        )
        .all()
    )
    if not stale_rows:
        return
    now = utcnow()
    for row in stale_rows:
        row.status = REDEMPTION_STATUS_EXPIRED
        row.updated_at = now
    db.flush()


def get_coupon_by_code(db: Session, code: str | None) -> BillingCoupon | None:
    """Return a coupon by normalized code."""
    normalized = normalize_coupon_code(code)
    if not normalized:
        return None
    return db.query(BillingCoupon).filter(BillingCoupon.code == normalized).first()


def _serialize_effect(
    *,
    coupon: BillingCoupon | None,
    applied_mode: str,
    trial_ends_at: datetime | None = None,
) -> dict[str, Any]:
    return {
        "applied_mode": applied_mode,
        "coupon_id": coupon.id if coupon else None,
        "coupon_code": coupon.code if coupon else None,
        "free_cycles": int(coupon.free_cycles or 0) if coupon else 0,
        "razorpay_offer_id": coupon.razorpay_offer_id if coupon else None,
        "trial_ends_at": trial_ends_at,
    }


def validate_coupon_for_user(
    db: Session,
    *,
    user: User,
    plan_code: str,
    coupon_code: str | None,
) -> dict[str, Any]:
    """Validate a coupon against plan, time window, and redemption caps."""
    expire_stale_coupon_redemptions(db)
    normalized = normalize_coupon_code(coupon_code)
    if not normalized:
        return {
            "eligible": False,
            "coupon": None,
            "reason": "Coupon code is required",
        }

    coupon = get_coupon_by_code(db, normalized)
    if not coupon:
        return {"eligible": False, "coupon": None, "reason": "Coupon not found"}

    now = utcnow()
    if coupon.status != COUPON_STATUS_ENABLED:
        return {"eligible": False, "coupon": coupon, "reason": "Coupon is disabled"}
    if coupon.applies_to_plan_code != plan_code:
        return {"eligible": False, "coupon": coupon, "reason": "Coupon does not apply to this plan"}
    if coupon.starts_at and coupon.starts_at > now:
        return {"eligible": False, "coupon": coupon, "reason": "Coupon is not active yet"}
    if coupon.ends_at and coupon.ends_at <= now:
        return {"eligible": False, "coupon": coupon, "reason": "Coupon has expired"}

    active_user_redemptions = (
        db.query(BillingCouponRedemption)
        .filter(
            BillingCouponRedemption.coupon_id == coupon.id,
            BillingCouponRedemption.user_id == user.id,
            BillingCouponRedemption.status.in_(tuple(_ACTIVE_REDEMPTION_STATUSES)),
        )
        .count()
    )
    if coupon.max_redemptions_per_user and active_user_redemptions >= int(coupon.max_redemptions_per_user):
        return {"eligible": False, "coupon": coupon, "reason": "Coupon usage limit reached for this user"}

    active_total_redemptions = (
        db.query(BillingCouponRedemption)
        .filter(
            BillingCouponRedemption.coupon_id == coupon.id,
            BillingCouponRedemption.status.in_(tuple(_ACTIVE_REDEMPTION_STATUSES)),
        )
        .count()
    )
    if coupon.max_redemptions_total and active_total_redemptions >= int(coupon.max_redemptions_total):
        return {"eligible": False, "coupon": coupon, "reason": "Coupon redemption limit reached"}

    if coupon.mode == COUPON_MODE_TRIAL_EXTENSION:
        free_cycles = int(coupon.free_cycles or 0)
        if free_cycles <= 0:
            return {"eligible": False, "coupon": coupon, "reason": "Coupon is misconfigured"}
        trial_ends_at = add_months(now, free_cycles)
        return {
            "eligible": True,
            "coupon": coupon,
            "reason": None,
            "effect": _serialize_effect(
                coupon=coupon,
                applied_mode=CHECKOUT_MODE_COUPON_TRIAL_EXTENSION,
                trial_ends_at=trial_ends_at,
            ),
        }

    if coupon.mode == COUPON_MODE_RAZORPAY_OFFER:
        if not coupon.razorpay_offer_id:
            return {"eligible": False, "coupon": coupon, "reason": "Coupon is missing a Razorpay offer"}
        return {
            "eligible": True,
            "coupon": coupon,
            "reason": None,
            "effect": _serialize_effect(
                coupon=coupon,
                applied_mode=CHECKOUT_MODE_COUPON_OFFER_DISCOUNT,
            ),
        }

    return {"eligible": False, "coupon": coupon, "reason": "Unsupported coupon mode"}


def resolve_checkout_strategy(
    db: Session,
    *,
    user: User,
    plan_code: str,
    coupon_code: str | None,
) -> dict[str, Any]:
    """Choose the checkout mode for the request."""
    normalized_coupon = normalize_coupon_code(coupon_code)
    if normalized_coupon:
        validation = validate_coupon_for_user(
            db,
            user=user,
            plan_code=plan_code,
            coupon_code=normalized_coupon,
        )
        if not validation["eligible"]:
            raise ValueError(validation["reason"])
        effect = dict(validation["effect"])
        effect["coupon"] = validation["coupon"]
        return effect

    now = utcnow()
    if not user.is_trial_used:
        return {
            "applied_mode": CHECKOUT_MODE_INTRO_TRIAL,
            "coupon": None,
            "coupon_code": None,
            "free_cycles": 0,
            "razorpay_offer_id": None,
            "trial_ends_at": now + timedelta(days=INTRO_TRIAL_DAYS),
        }

    return {
        "applied_mode": CHECKOUT_MODE_STANDARD,
        "coupon": None,
        "coupon_code": None,
        "free_cycles": 0,
        "razorpay_offer_id": None,
        "trial_ends_at": None,
    }


def reserve_coupon_redemption(
    db: Session,
    *,
    user: User,
    coupon: BillingCoupon | None,
    billing_subscription: BillingSubscription,
    effect: dict[str, Any] | None,
) -> BillingCouponRedemption | None:
    """Reserve a coupon spot for a just-created checkout."""
    if not coupon:
        return None

    existing = (
        db.query(BillingCouponRedemption)
        .filter(
            BillingCouponRedemption.coupon_id == coupon.id,
            BillingCouponRedemption.user_id == user.id,
            BillingCouponRedemption.billing_subscription_id == billing_subscription.id,
        )
        .first()
    )
    if existing:
        return existing

    serialized_effect: dict[str, Any] | None = None
    if effect:
        serialized_effect = {}
        for key, value in effect.items():
            if key == "coupon":
                continue
            if isinstance(value, datetime):
                serialized_effect[key] = value.isoformat()
            else:
                serialized_effect[key] = value

    row = BillingCouponRedemption(
        coupon_id=coupon.id,
        user_id=user.id,
        billing_subscription_id=billing_subscription.id,
        status=REDEMPTION_STATUS_RESERVED,
        coupon_code_snapshot=coupon.code,
        effect_snapshot=serialized_effect,
    )
    db.add(row)
    db.flush()
    return row


def sync_coupon_redemption_state(db: Session, subscription: BillingSubscription) -> None:
    """Promote linked coupon reservations as the subscription progresses."""
    if not subscription.coupon_id:
        return
    row = (
        db.query(BillingCouponRedemption)
        .filter(BillingCouponRedemption.billing_subscription_id == subscription.id)
        .order_by(BillingCouponRedemption.created_at.desc())
        .first()
    )
    if not row:
        return

    now = utcnow()
    if subscription.status == "expired" and row.status == REDEMPTION_STATUS_RESERVED:
        row.status = REDEMPTION_STATUS_EXPIRED
        row.updated_at = now
        return

    if subscription.status == "cancelled" and row.status == REDEMPTION_STATUS_RESERVED:
        row.status = REDEMPTION_STATUS_REVOKED
        row.updated_at = now
        return

    if subscription.status == "authenticated" and row.status == REDEMPTION_STATUS_RESERVED:
        row.status = REDEMPTION_STATUS_VERIFIED
        row.verified_at = row.verified_at or now

    if subscription.billing_phase == "trial" or subscription.status in {
        "active",
        "pending",
        "halted",
        "paused",
        "completed",
    }:
        row.status = REDEMPTION_STATUS_CONSUMED
        row.verified_at = row.verified_at or now
        row.consumed_at = row.consumed_at or now

    row.updated_at = now


def serialize_coupon_validation(
    *,
    plan_code: str,
    coupon_code: str | None,
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Serialize coupon validation for the public API."""
    effect = validation.get("effect") or {}
    coupon = validation.get("coupon")
    return {
        "plan_code": plan_code,
        "coupon_code": normalize_coupon_code(coupon_code),
        "eligible": bool(validation.get("eligible")),
        "reason": validation.get("reason"),
        "applied_mode": effect.get("applied_mode"),
        "free_cycles": effect.get("free_cycles"),
        "trial_ends_at": effect.get("trial_ends_at"),
        "discount_type": getattr(coupon, "discount_type", None),
        "discount_value": getattr(coupon, "discount_value", None),
        "razorpay_offer_id": effect.get("razorpay_offer_id"),
    }


def _serialize_coupon_counts(db: Session, coupon_id: str) -> tuple[int, int]:
    total = (
        db.query(func.count(BillingCouponRedemption.id))
        .filter(BillingCouponRedemption.coupon_id == coupon_id)
        .scalar()
        or 0
    )
    active = (
        db.query(func.count(BillingCouponRedemption.id))
        .filter(
            BillingCouponRedemption.coupon_id == coupon_id,
            BillingCouponRedemption.status.in_(tuple(_ACTIVE_REDEMPTION_STATUSES)),
        )
        .scalar()
        or 0
    )
    return int(total), int(active)


def serialize_coupon_for_admin(db: Session, coupon: BillingCoupon) -> dict[str, Any]:
    """Serialize coupon metadata for admin APIs."""
    total_redemptions, active_redemptions = _serialize_coupon_counts(db, coupon.id)
    return {
        "id": coupon.id,
        "code": coupon.code,
        "name": coupon.name,
        "status": coupon.status,
        "mode": coupon.mode,
        "applies_to_plan_code": coupon.applies_to_plan_code,
        "free_cycles": coupon.free_cycles,
        "discount_type": coupon.discount_type,
        "discount_value": coupon.discount_value,
        "max_redemptions_total": coupon.max_redemptions_total,
        "max_redemptions_per_user": coupon.max_redemptions_per_user,
        "starts_at": coupon.starts_at,
        "ends_at": coupon.ends_at,
        "razorpay_offer_id": coupon.razorpay_offer_id,
        "metadata_json": coupon.metadata_json,
        "raw_config": coupon.raw_config,
        "created_by_admin_id": coupon.created_by_admin_id,
        "updated_by_admin_id": coupon.updated_by_admin_id,
        "created_at": coupon.created_at,
        "updated_at": coupon.updated_at,
        "total_redemptions": total_redemptions,
        "active_redemptions": active_redemptions,
    }


def list_coupons_for_admin(db: Session) -> list[dict[str, Any]]:
    """List all coupons for the admin panel."""
    rows = db.query(BillingCoupon).order_by(BillingCoupon.created_at.desc(), BillingCoupon.id.desc()).all()
    return [serialize_coupon_for_admin(db, row) for row in rows]


def get_coupon_detail_for_admin(db: Session, coupon_id: str) -> dict[str, Any] | None:
    """Return one coupon with counters."""
    row = db.query(BillingCoupon).filter(BillingCoupon.id == coupon_id).first()
    if not row:
        return None
    return serialize_coupon_for_admin(db, row)


def list_coupon_redemptions_for_admin(db: Session, coupon_id: str) -> list[dict[str, Any]]:
    """Return coupon redemption history for the admin panel."""
    rows = (
        db.query(BillingCouponRedemption, User.email)
        .outerjoin(User, User.id == BillingCouponRedemption.user_id)
        .filter(BillingCouponRedemption.coupon_id == coupon_id)
        .order_by(BillingCouponRedemption.created_at.desc(), BillingCouponRedemption.id.desc())
        .all()
    )
    return [
        {
            "id": redemption.id,
            "coupon_id": redemption.coupon_id,
            "user_id": redemption.user_id,
            "user_email": email,
            "billing_subscription_id": redemption.billing_subscription_id,
            "status": redemption.status,
            "coupon_code_snapshot": redemption.coupon_code_snapshot,
            "effect_snapshot": redemption.effect_snapshot,
            "created_at": redemption.created_at,
            "verified_at": redemption.verified_at,
            "consumed_at": redemption.consumed_at,
            "updated_at": redemption.updated_at,
        }
        for redemption, email in rows
    ]


def create_coupon(
    db: Session,
    *,
    admin: Admin,
    payload: dict[str, Any],
) -> BillingCoupon:
    """Create a coupon after validating business constraints."""
    code = normalize_coupon_code(payload.get("code"))
    if not code:
        raise ValueError("Coupon code is required")
    if db.query(BillingCoupon).filter(BillingCoupon.code == code).first():
        raise ValueError("Coupon code already exists")

    mode = payload.get("mode")
    plan_code = payload.get("applies_to_plan_code")
    if plan_code not in PAID_PLAN_CODES:
        raise ValueError("Coupon must target a paid plan")
    if mode not in {COUPON_MODE_TRIAL_EXTENSION, COUPON_MODE_RAZORPAY_OFFER}:
        raise ValueError("Unsupported coupon mode")

    row = BillingCoupon(
        code=code,
        name=str(payload.get("name") or code),
        status=payload.get("status") or COUPON_STATUS_ENABLED,
        mode=mode,
        applies_to_plan_code=plan_code,
        free_cycles=payload.get("free_cycles"),
        discount_type=payload.get("discount_type"),
        discount_value=payload.get("discount_value"),
        max_redemptions_total=payload.get("max_redemptions_total"),
        max_redemptions_per_user=payload.get("max_redemptions_per_user") or 1,
        starts_at=payload.get("starts_at"),
        ends_at=payload.get("ends_at"),
        razorpay_offer_id=payload.get("razorpay_offer_id"),
        metadata_json=payload.get("metadata_json"),
        raw_config=payload.get("raw_config"),
        created_by_admin_id=admin.id,
        updated_by_admin_id=admin.id,
    )
    _validate_coupon_config(row)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def update_coupon(
    db: Session,
    *,
    coupon: BillingCoupon,
    admin: Admin,
    payload: dict[str, Any],
) -> BillingCoupon:
    """Patch a coupon while enforcing mode-specific configuration."""
    if "code" in payload:
        new_code = normalize_coupon_code(payload.get("code"))
        if not new_code:
            raise ValueError("Coupon code cannot be empty")
        existing = db.query(BillingCoupon).filter(BillingCoupon.code == new_code, BillingCoupon.id != coupon.id).first()
        if existing:
            raise ValueError("Coupon code already exists")
        coupon.code = new_code

    for field in (
        "name",
        "status",
        "mode",
        "applies_to_plan_code",
        "free_cycles",
        "discount_type",
        "discount_value",
        "max_redemptions_total",
        "max_redemptions_per_user",
        "starts_at",
        "ends_at",
        "razorpay_offer_id",
        "metadata_json",
        "raw_config",
    ):
        if field in payload:
            setattr(coupon, field, payload[field])

    coupon.updated_by_admin_id = admin.id
    _validate_coupon_config(coupon)
    db.add(coupon)
    db.commit()
    db.refresh(coupon)
    return coupon


def disable_coupon(db: Session, *, coupon: BillingCoupon, admin: Admin) -> BillingCoupon:
    """Disable a coupon without deleting historical redemption data."""
    coupon.status = COUPON_STATUS_DISABLED
    coupon.updated_by_admin_id = admin.id
    db.add(coupon)
    db.commit()
    db.refresh(coupon)
    return coupon


def _validate_coupon_config(coupon: BillingCoupon) -> None:
    if coupon.applies_to_plan_code not in PAID_PLAN_CODES:
        raise ValueError("Coupon must target a paid plan")
    if coupon.mode == COUPON_MODE_TRIAL_EXTENSION:
        if not coupon.free_cycles or int(coupon.free_cycles) <= 0:
            raise ValueError("Trial extension coupons require free_cycles > 0")
    elif coupon.mode == COUPON_MODE_RAZORPAY_OFFER:
        if not coupon.razorpay_offer_id:
            raise ValueError("Razorpay offer coupons require razorpay_offer_id")
    else:
        raise ValueError("Unsupported coupon mode")
