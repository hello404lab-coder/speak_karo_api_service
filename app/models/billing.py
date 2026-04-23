"""Billing models for provider-backed subscriptions and webhook audit."""
from __future__ import annotations

from datetime import datetime
import uuid

import sqlalchemy as sa
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Text

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
    status = Column(String(32), nullable=False, index=True)
    short_url = Column(Text, nullable=True)
    current_start_at = Column(DateTime, nullable=True)
    current_end_at = Column(DateTime, nullable=True)
    charge_at = Column(DateTime, nullable=True)
    start_at = Column(DateTime, nullable=True)
    end_at = Column(DateTime, nullable=True)
    expire_by = Column(DateTime, nullable=True)
    cancel_at_cycle_end = Column(Boolean, nullable=False, server_default=sa.false(), default=False)
    cancelled_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    raw_last_payload = Column(sa.JSON(), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
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
