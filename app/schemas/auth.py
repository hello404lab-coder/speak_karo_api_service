"""Request and response schemas for auth endpoints."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class OAuthLoginRequest(BaseModel):
    """Request body for OAuth login (client sends ID token from Google/Apple)."""

    provider: Literal["google", "apple"] = Field(..., description="OAuth provider")
    id_token: str = Field(..., min_length=1, description="ID token from the provider")


class UserResponse(BaseModel):
    """User profile returned in auth responses."""

    id: str = Field(..., description="User UUID")
    email: str = Field(..., description="User email")
    name: str | None = Field(None, description="Display name")
    native_language: str | None = Field(None, description="Learner's native language as entered during onboarding")
    native_language_code: str | None = Field(None, description="Normalized native language code used for translation defaults")
    onboarding_completed: bool = Field(default=False, description="Whether user finished onboarding")
    onboarding_step: int = Field(default=0, description="Current onboarding step (0-5)")
    plan: Literal["free", "trial", "vuvl_plus", "vuvl_pro"] = Field(
        default="free",
        description="Resolved plan: free, trial, vuvl_plus, or vuvl_pro",
    )
    billing_phase: Literal["free", "trial", "authenticated", "active", "pending", "halted", "paused", "cancelled", "completed", "expired"] = Field(
        default="free",
        description="High-level billing phase",
    )
    trial_expires_at: datetime | None = Field(None, description="Trial expiration")
    subscription_expires_at: datetime | None = Field(None, description="Paid subscription expiration")
    billing_status: str | None = Field(None, description="Latest provider-backed billing status")
    active_plan_code: Literal["vuvl_plus", "vuvl_pro"] | None = Field(None, description="Active paid plan code")
    current_period_end: datetime | None = Field(None, description="Current paid billing period end")
    coupon_code: str | None = Field(None, description="Applied checkout coupon code")

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    """Response after successful OAuth login or token refresh."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    user: UserResponse = Field(..., description="User profile")


class RefreshTokenRequest(BaseModel):
    """Request body for refreshing the access token."""

    refresh_token: str = Field(..., min_length=1, description="JWT refresh token")


class AccessTokenResponse(BaseModel):
    """Response when refreshing only the access token (POST /refresh)."""

    access_token: str = Field(..., description="New JWT access token")
    token_type: str = Field(default="bearer", description="Token type")


class EmailRegisterRequest(BaseModel):
    """Request body for email/password registration."""

    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=6, max_length=128, description="Password (min 6 chars)")
    name: str | None = Field(None, max_length=255, description="Display name (optional)")


class EmailLoginRequest(BaseModel):
    """Request body for email/password login."""

    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=1, description="Password")
