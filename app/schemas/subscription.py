"""Request and response schemas for subscription endpoints."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UsageTodayResponse(BaseModel):
    """Today's usage and plan-based limits for display."""

    date: str = Field(..., description="ISO date (e.g. 2026-03-10)")
    chat_count: int = Field(..., description="AI chats used today")
    voice_count: int = Field(..., description="Voice conversations used today")
    chat_limit: int | None = Field(None, description="Max chats per day (null = unlimited)")
    voice_limit: int | None = Field(None, description="Max voice per day (null = unlimited)")
    minutes_used: float = Field(..., description="Minutes of usage today")


class SubscriptionStatusResponse(BaseModel):
    """Current subscription status and today's usage."""

    plan: str = Field(..., description="Resolved plan: free, trial, or premium")
    trial_expires_at: datetime | None = Field(None, description="Trial expiration (null if not on trial)")
    subscription_expires_at: datetime | None = Field(
        None, description="Premium expiration (null if not premium)"
    )
    usage: UsageTodayResponse = Field(..., description="Today's usage and limits")


class StartTrialRequest(BaseModel):
    """Request to start trial after payment verification."""

    payment_verified: bool = Field(..., description="Must be true to activate trial")


class ActivatePremiumRequest(BaseModel):
    """Request to activate premium after store purchase verification."""

    purchase_token: str = Field(..., min_length=1, description="Token from Google Play / App Store")
    provider: Literal["google", "apple"] = Field(..., description="Store provider")
