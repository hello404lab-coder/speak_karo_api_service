"""Subscription API: catalog, checkout, verification, status, coupons, and webhooks."""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database import get_db
from app.dependencies.auth import get_current_user, limiter
from app.models.user import User
from app.schemas.subscription import (
    CouponValidateRequest,
    CouponValidateResponse,
    RazorpayWebhookResponse,
    SubscriptionCancelResponse,
    SubscriptionCatalogItemResponse,
    SubscriptionCatalogResponse,
    SubscriptionCheckoutPrefillResponse,
    SubscriptionCheckoutRequest,
    SubscriptionCheckoutResponse,
    SubscriptionManageActions,
    SubscriptionStatusResponse,
    SubscriptionVerifyRequest,
    SubscriptionVerifyResponse,
    UsageTodayResponse,
)
from app.services.billing_service import (
    build_catalog_payload,
    build_checkout_payload,
    build_checkout_prefill,
    get_active_paid_subscription,
    get_manageable_subscription,
    get_reusable_checkout_subscription,
    get_subscription_for_status,
    process_razorpay_webhook,
    serialize_subscription_summary,
    sync_subscription_from_razorpay,
    verify_razorpay_checkout_signature,
    verify_razorpay_webhook_signature,
)
from app.services.coupon_service import (
    CHECKOUT_MODE_INTRO_TRIAL,
    reserve_coupon_redemption,
    resolve_checkout_strategy,
    serialize_coupon_validation,
    validate_coupon_for_user,
)
from app.services.razorpay_client import RazorpayAPIError, get_razorpay_client
from app.services.subscription_service import (
    FREE_MAX_CHATS_PER_DAY,
    FREE_MAX_VOICE_PER_DAY,
    get_usage_today_for_display,
    resolve_user_plan,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _ensure_razorpay_enabled() -> None:
    if not settings.razorpay_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Razorpay billing is not configured",
        )


def _build_usage_response(user: User, db: Session) -> UsageTodayResponse:
    plan = resolve_user_plan(user)
    usage_data = get_usage_today_for_display(user.id, db)
    chat_limit: int | None = FREE_MAX_CHATS_PER_DAY if plan == "free" else None
    voice_limit: int | None = FREE_MAX_VOICE_PER_DAY if plan == "free" else None
    return UsageTodayResponse(
        date=usage_data["date"],
        chat_count=usage_data["chat_count"],
        voice_count=usage_data["voice_count"],
        chat_limit=chat_limit,
        voice_limit=voice_limit,
        minutes_used=usage_data["minutes_used"],
    )


@router.get("/catalog", response_model=SubscriptionCatalogResponse)
@limiter.limit("30/minute")
async def subscription_catalog(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SubscriptionCatalogResponse:
    """Return the public paid plan catalog and checkout eligibility."""
    del request
    active = get_active_paid_subscription(db, current_user.id)
    current_plan = resolve_user_plan(current_user)
    can_checkout = settings.razorpay_enabled and active is None
    plans = [
        SubscriptionCatalogItemResponse.model_validate(item)
        for item in build_catalog_payload(current_plan=current_plan, can_checkout=can_checkout)
    ]
    return SubscriptionCatalogResponse(
        enabled=settings.razorpay_enabled,
        current_plan=current_plan,
        can_checkout=can_checkout,
        plans=plans,
    )


@router.get("/status", response_model=SubscriptionStatusResponse)
@limiter.limit("30/minute")
async def subscription_status(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SubscriptionStatusResponse:
    """Return current plan, billing summary, and today's usage."""
    del request
    billing_row = get_subscription_for_status(db, current_user.id)
    summary = serialize_subscription_summary(billing_row, user=current_user)
    return SubscriptionStatusResponse(
        plan=resolve_user_plan(current_user),
        trial_expires_at=current_user.trial_expires_at,
        subscription_expires_at=current_user.subscription_expires_at,
        billing_phase=summary["billing_phase"],
        usage=_build_usage_response(current_user, db),
        billing_status=summary["billing_status"],
        active_subscription_id=summary["active_subscription_id"],
        active_plan_code=summary["active_plan_code"],
        current_period_end=summary["current_period_end"],
        cancel_at_cycle_end=summary["cancel_at_cycle_end"],
        coupon_code=summary["coupon_code"],
        manage_actions=SubscriptionManageActions.model_validate(summary["manage_actions"]),
    )


@router.post("/coupons/validate", response_model=CouponValidateResponse)
@limiter.limit("20/minute")
async def validate_coupon(
    request: Request,
    body: CouponValidateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CouponValidateResponse:
    """Validate a coupon and preview the checkout effect."""
    del request
    validation = validate_coupon_for_user(
        db,
        user=current_user,
        plan_code=body.plan_code,
        coupon_code=body.coupon_code,
    )
    return CouponValidateResponse.model_validate(
        serialize_coupon_validation(
            plan_code=body.plan_code,
            coupon_code=body.coupon_code,
            validation=validation,
        )
    )


@router.post("/start-trial", deprecated=True)
@limiter.limit("10/minute")
async def start_trial_deprecated(request: Request) -> dict[str, str]:
    """Legacy endpoint removed in favor of checkout-backed trials."""
    del request
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="This endpoint is deprecated. Start trial access through /api/v1/subscription/checkout.",
    )


@router.post("/checkout", response_model=SubscriptionCheckoutResponse)
@limiter.limit("10/minute")
async def create_subscription_checkout(
    request: Request,
    body: SubscriptionCheckoutRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SubscriptionCheckoutResponse:
    """Create or reuse a Razorpay subscription checkout."""
    del request
    _ensure_razorpay_enabled()

    active = get_active_paid_subscription(db, current_user.id)
    if active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User already has an active paid subscription",
        )

    try:
        checkout_strategy = resolve_checkout_strategy(
            db,
            user=current_user,
            plan_code=body.plan_code,
            coupon_code=body.coupon_code,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    reusable = get_reusable_checkout_subscription(
        db,
        current_user.id,
        body.plan_code,
        coupon_code=checkout_strategy["coupon_code"],
        checkout_mode=checkout_strategy["applied_mode"],
    )
    if reusable:
        return SubscriptionCheckoutResponse(
            key_id=settings.razorpay_key_id or "",
            subscription_id=reusable.provider_subscription_id,
            plan_code=body.plan_code,
            status=reusable.status,
            applied_mode=checkout_strategy["applied_mode"],
            coupon_code=checkout_strategy["coupon_code"],
            trial_ends_at=reusable.trial_access_until,
            short_url=reusable.short_url,
            prefill=SubscriptionCheckoutPrefillResponse.model_validate(build_checkout_prefill(current_user)),
            reuse_existing=True,
        )

    payload = build_checkout_payload(
        current_user,
        body.plan_code,
        checkout_mode=checkout_strategy["applied_mode"],
        trial_ends_at=checkout_strategy["trial_ends_at"],
        razorpay_offer_id=checkout_strategy["razorpay_offer_id"],
        coupon_code=checkout_strategy["coupon_code"],
    )
    client = get_razorpay_client()
    try:
        remote_subscription = await asyncio.to_thread(client.create_subscription, payload)
    except (ValueError, RazorpayAPIError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to create Razorpay checkout subscription",
        ) from exc

    local = sync_subscription_from_razorpay(
        db,
        remote_subscription,
        expected_user_id=current_user.id,
        raw_payload=remote_subscription,
    )
    reserve_coupon_redemption(
        db,
        user=current_user,
        coupon=checkout_strategy.get("coupon"),
        billing_subscription=local,
        effect=checkout_strategy,
    )
    db.commit()
    db.refresh(local)

    return SubscriptionCheckoutResponse(
        key_id=settings.razorpay_key_id or "",
        subscription_id=local.provider_subscription_id,
        plan_code=body.plan_code,
        status=local.status,
        applied_mode=checkout_strategy["applied_mode"],
        coupon_code=checkout_strategy["coupon_code"],
        trial_ends_at=local.trial_access_until,
        short_url=local.short_url,
        prefill=SubscriptionCheckoutPrefillResponse.model_validate(build_checkout_prefill(current_user)),
        reuse_existing=False,
    )


@router.post("/verify", response_model=SubscriptionVerifyResponse)
@limiter.limit("20/minute")
async def verify_subscription_checkout(
    request: Request,
    body: SubscriptionVerifyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SubscriptionVerifyResponse:
    """Verify checkout signature and sync subscription state from Razorpay."""
    del request
    _ensure_razorpay_enabled()
    if not verify_razorpay_checkout_signature(
        payment_id=body.razorpay_payment_id,
        subscription_id=body.razorpay_subscription_id,
        signature=body.razorpay_signature,
    ):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Razorpay signature")

    client = get_razorpay_client()
    try:
        remote_subscription = await asyncio.to_thread(client.fetch_subscription, body.razorpay_subscription_id)
    except RazorpayAPIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch Razorpay subscription",
        ) from exc

    local = sync_subscription_from_razorpay(
        db,
        remote_subscription,
        expected_user_id=current_user.id,
        payment_id=body.razorpay_payment_id,
        raw_payload=remote_subscription,
    )
    notes = remote_subscription.get("notes") or {}
    if notes.get("checkout_mode") == CHECKOUT_MODE_INTRO_TRIAL and not current_user.is_trial_used:
        current_user.is_trial_used = True
        db.add(current_user)
        db.commit()
    db.refresh(current_user)
    return SubscriptionVerifyResponse(
        message="Subscription verified",
        plan=resolve_user_plan(current_user),
        billing_phase=local.billing_phase,
        billing_status=local.status,
        active_subscription_id=local.provider_subscription_id,
        coupon_code=local.coupon_code_snapshot,
        subscription_expires_at=current_user.subscription_expires_at,
        trial_expires_at=current_user.trial_expires_at,
    )


@router.post("/cancel", response_model=SubscriptionCancelResponse)
@limiter.limit("10/minute")
async def cancel_subscription(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SubscriptionCancelResponse:
    """Cancel the current paid subscription at cycle end."""
    del request
    _ensure_razorpay_enabled()
    billing_row = get_manageable_subscription(db, current_user.id)
    if not billing_row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No paid subscription found")

    if billing_row.cancel_at_cycle_end:
        return SubscriptionCancelResponse(
            message="Cancellation already scheduled",
            plan=resolve_user_plan(current_user),
            billing_phase=billing_row.billing_phase,
            billing_status=billing_row.status,
            cancel_at_cycle_end=True,
            current_period_end=billing_row.current_end_at,
        )

    client = get_razorpay_client()
    try:
        remote_subscription = await asyncio.to_thread(
            client.cancel_subscription,
            billing_row.provider_subscription_id,
            cancel_at_cycle_end=True,
        )
    except RazorpayAPIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to cancel Razorpay subscription",
        ) from exc

    updated = sync_subscription_from_razorpay(
        db,
        remote_subscription,
        expected_user_id=current_user.id,
        raw_payload=remote_subscription,
    )
    db.refresh(current_user)
    return SubscriptionCancelResponse(
        message="Cancellation scheduled",
        plan=resolve_user_plan(current_user),
        billing_phase=updated.billing_phase,
        billing_status=updated.status,
        cancel_at_cycle_end=updated.cancel_at_cycle_end,
        current_period_end=updated.current_end_at,
    )


@router.post("/webhooks/razorpay", response_model=RazorpayWebhookResponse)
async def razorpay_webhook(
    request: Request,
    db: Session = Depends(get_db),
    x_razorpay_signature: str | None = Header(default=None),
) -> RazorpayWebhookResponse:
    """Process a verified Razorpay webhook idempotently."""
    raw_body = await request.body()
    if not verify_razorpay_webhook_signature(raw_body, x_razorpay_signature):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook signature")
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid webhook JSON") from exc

    client = get_razorpay_client() if settings.razorpay_enabled else None
    fetch_subscription_fn = client.fetch_subscription if client else None
    try:
        _, duplicate = process_razorpay_webhook(
            db,
            raw_body=raw_body,
            payload=payload,
            fetch_subscription_fn=fetch_subscription_fn,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RazorpayAPIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to sync Razorpay webhook subscription state",
        ) from exc

    return RazorpayWebhookResponse(status="ok", duplicate=duplicate)


@router.post("/activate-premium", deprecated=True)
@limiter.limit("10/minute")
async def activate_premium_deprecated(request: Request) -> dict[str, str]:
    """Deprecated placeholder kept for backward compatibility."""
    del request
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="This endpoint is deprecated. Use /api/v1/subscription/checkout instead.",
    )
