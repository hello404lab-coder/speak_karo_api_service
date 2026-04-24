"""Request and response schemas for subscription endpoints."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


PlanCode = Literal["free", "trial", "vuvl_plus", "vuvl_pro"]
PaidPlanCode = Literal["vuvl_plus", "vuvl_pro"]
BillingPhase = Literal["free", "trial", "authenticated", "active", "pending", "halted", "paused", "cancelled", "completed", "expired"]


class UsageTodayResponse(BaseModel):
    """Today's usage and plan-based limits for display."""

    date: str = Field(..., description="ISO date (e.g. 2026-03-10)")
    chat_count: int = Field(..., description="AI chats used today")
    voice_count: int = Field(..., description="Voice conversations used today")
    chat_limit: int | None = Field(None, description="Max chats per day (null = unlimited)")
    voice_limit: int | None = Field(None, description="Max voice per day (null = unlimited)")
    minutes_used: float = Field(..., description="Minutes of usage today")


class SubscriptionManageActions(BaseModel):
    """Management affordances available to the user."""

    can_cancel: bool = False


class SubscriptionFeatureFlags(BaseModel):
    """Feature flags exposed for each plan in the public catalog."""

    unlimited_practice: bool
    gemini_live_access: bool


class SubscriptionCatalogItemResponse(BaseModel):
    """One plan in the public paid plan catalog."""

    code: PaidPlanCode
    name: str
    amount: int = Field(..., description="Amount in paise")
    currency: Literal["INR"]
    interval: Literal["monthly"]
    feature_flags: SubscriptionFeatureFlags
    eligible_for_checkout: bool


class SubscriptionCatalogResponse(BaseModel):
    """Public plan catalog and current checkout eligibility."""

    enabled: bool
    current_plan: PlanCode
    can_checkout: bool
    plans: list[SubscriptionCatalogItemResponse]


class SubscriptionStatusResponse(BaseModel):
    """Current subscription status and today's usage."""

    plan: PlanCode = Field(..., description="Resolved plan: free, trial, vuvl_plus, or vuvl_pro")
    trial_expires_at: datetime | None = Field(None, description="Trial expiration (null if not on trial)")
    subscription_expires_at: datetime | None = Field(
        None,
        description="Paid subscription entitlement end (null when inactive)",
    )
    billing_phase: BillingPhase = Field(default="free", description="High-level billing phase")
    billing_status: str | None = Field(None, description="Latest provider-backed billing status")
    active_subscription_id: str | None = Field(None, description="Provider subscription id")
    active_plan_code: PaidPlanCode | None = Field(None, description="Active paid plan code")
    current_period_end: datetime | None = Field(None, description="Current paid billing period end")
    cancel_at_cycle_end: bool = Field(False, description="Whether cancellation is already scheduled")
    coupon_code: str | None = Field(None, description="Applied checkout coupon code, if any")
    manage_actions: SubscriptionManageActions = Field(default_factory=SubscriptionManageActions)
    usage: UsageTodayResponse = Field(..., description="Today's usage and limits")


class StartTrialRequest(BaseModel):
    """Request to start trial after payment verification."""

    payment_verified: bool = Field(..., description="Must be true to activate trial")


class SubscriptionCheckoutRequest(BaseModel):
    """Create or reuse a Razorpay subscription checkout."""

    plan_code: PaidPlanCode
    coupon_code: str | None = None


class SubscriptionCheckoutPrefillResponse(BaseModel):
    """Safe prefill fields for Standard Checkout."""

    name: str | None = None
    email: str | None = None
    contact: str | None = None


class SubscriptionCheckoutResponse(BaseModel):
    """Payload required by frontend to open Razorpay Standard Checkout."""

    key_id: str
    subscription_id: str
    plan_code: PaidPlanCode
    status: str
    applied_mode: Literal["standard", "intro_trial", "coupon_trial_extension", "coupon_offer_discount"]
    coupon_code: str | None = None
    trial_ends_at: datetime | None = None
    short_url: str | None = None
    prefill: SubscriptionCheckoutPrefillResponse
    reuse_existing: bool = False


class SubscriptionVerifyRequest(BaseModel):
    """Verify a successful Standard Checkout authorization."""

    razorpay_payment_id: str = Field(..., min_length=1)
    razorpay_subscription_id: str = Field(..., min_length=1)
    razorpay_signature: str = Field(..., min_length=1)


class SubscriptionVerifyResponse(BaseModel):
    """Response after successful checkout verification and sync."""

    message: str
    plan: PlanCode
    billing_phase: BillingPhase
    billing_status: str
    active_subscription_id: str
    coupon_code: str | None = None
    subscription_expires_at: datetime | None = None
    trial_expires_at: datetime | None = None


class SubscriptionCancelResponse(BaseModel):
    """Response after a cancel-at-cycle-end request."""

    message: str
    plan: PlanCode
    billing_phase: BillingPhase
    billing_status: str
    cancel_at_cycle_end: bool
    current_period_end: datetime | None = None


class CouponValidateRequest(BaseModel):
    """Validate a coupon against a target plan."""

    plan_code: PaidPlanCode
    coupon_code: str = Field(..., min_length=1)


class CouponValidateResponse(BaseModel):
    """Public coupon validation preview."""

    plan_code: PaidPlanCode
    coupon_code: str | None = None
    eligible: bool
    reason: str | None = None
    applied_mode: Literal["coupon_trial_extension", "coupon_offer_discount"] | None = None
    free_cycles: int | None = None
    trial_ends_at: datetime | None = None
    discount_type: str | None = None
    discount_value: int | None = None
    razorpay_offer_id: str | None = None


class RazorpayWebhookResponse(BaseModel):
    """Webhook acknowledgement."""

    status: Literal["ok"]
    duplicate: bool = False
