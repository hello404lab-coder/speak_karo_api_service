"""Billing models for provider-backed subscriptions, coupons, and webhook audit."""
from __future__ import annotations

from datetime import datetime
import uuid

import sqlalchemy as sa
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text

from app.models.usage import Base


class BillingSubscription(Base):
    """Provider-backed recurring billing subscription for one user."""

    __tablename__ = "billing_subscriptions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    provider = Column(String(32), nullable=False, index=True)
    plan_code = Column(String(32), nullable=False, index=True)
    provider_plan_id = Column(String(64), nullable=False, index=True)
    provider_subscription_id = Column(String(64), nullable=False, unique=True, index=True)
    provider_customer_id = Column(String(64), nullable=True, index=True)
    provider_payment_id = Column(String(64), nullable=True, index=True)
    provider_offer_id = Column(String(64), nullable=True, index=True)
    status = Column(String(32), nullable=False, index=True)
    billing_phase = Column(String(32), nullable=False, server_default=sa.text("'free'"), default="free", index=True)
    short_url = Column(Text, nullable=True)
    current_start_at = Column(DateTime, nullable=True)
    current_end_at = Column(DateTime, nullable=True)
    charge_at = Column(DateTime, nullable=True)
    start_at = Column(DateTime, nullable=True)
    end_at = Column(DateTime, nullable=True)
    expire_by = Column(DateTime, nullable=True)
    trial_access_until = Column(DateTime, nullable=True)
    authenticated_at = Column(DateTime, nullable=True)
    cancel_at_cycle_end = Column(Boolean, nullable=False, server_default=sa.false(), default=False)
    cancelled_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    coupon_id = Column(String(36), ForeignKey("billing_coupons.id"), nullable=True, index=True)
    coupon_code_snapshot = Column(String(64), nullable=True, index=True)
    last_invoice_id = Column(String(64), nullable=True, index=True)
    last_payment_id = Column(String(64), nullable=True, index=True)
    raw_last_payload = Column(sa.JSON(), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = ({"schema": None},)


class BillingCoupon(Base):
    """Admin-managed coupon configuration for subscription checkout."""

    __tablename__ = "billing_coupons"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    code = Column(String(64), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=False)
    status = Column(String(32), nullable=False, server_default=sa.text("'enabled'"), default="enabled", index=True)
    mode = Column(String(32), nullable=False, index=True)
    applies_to_plan_code = Column(String(32), nullable=False, index=True)
    free_cycles = Column(Integer, nullable=True)
    discount_type = Column(String(32), nullable=True)
    discount_value = Column(Integer, nullable=True)
    max_redemptions_total = Column(Integer, nullable=True)
    max_redemptions_per_user = Column(Integer, nullable=False, server_default=sa.text("1"), default=1)
    starts_at = Column(DateTime, nullable=True)
    ends_at = Column(DateTime, nullable=True)
    razorpay_offer_id = Column(String(64), nullable=True, index=True)
    metadata_json = Column(sa.JSON(), nullable=True)
    raw_config = Column(sa.JSON(), nullable=True)
    created_by_admin_id = Column(String(36), ForeignKey("admins.id"), nullable=True, index=True)
    updated_by_admin_id = Column(String(36), ForeignKey("admins.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = ({"schema": None},)


class BillingCouponRedemption(Base):
    """One user redemption attempt / reservation for a coupon."""

    __tablename__ = "billing_coupon_redemptions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    coupon_id = Column(String(36), ForeignKey("billing_coupons.id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    billing_subscription_id = Column(
        String(36),
        ForeignKey("billing_subscriptions.id"),
        nullable=True,
        index=True,
    )
    status = Column(String(32), nullable=False, server_default=sa.text("'reserved'"), default="reserved", index=True)
    coupon_code_snapshot = Column(String(64), nullable=False, index=True)
    effect_snapshot = Column(sa.JSON(), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    verified_at = Column(DateTime, nullable=True)
    consumed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = ({"schema": None},)


class BillingWebhookEvent(Base):
    """Idempotent audit record for processed billing webhooks."""

    __tablename__ = "billing_webhook_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    provider = Column(String(32), nullable=False, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    delivery_hash = Column(String(64), nullable=False, unique=True, index=True)
    provider_subscription_id = Column(String(64), nullable=True, index=True)
    payload = Column(sa.JSON(), nullable=False)
    processed_at = Column(DateTime, nullable=True)
    processing_status = Column(String(32), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = ({"schema": None},)
