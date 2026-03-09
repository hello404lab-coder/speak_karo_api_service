"""Request and response schemas for auth endpoints."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class OAuthLoginRequest(BaseModel):
    """Request body for OAuth login (client sends ID token from Google/Apple)."""

    provider: Literal["google", "apple"] = Field(..., description="OAuth provider")
    id_token: str = Field(..., min_length=1, description="ID token from the provider")


class UserResponse(BaseModel):
    """User profile returned in auth responses."""

    id: str = Field(..., description="User UUID")
    email: str = Field(..., description="User email")
    name: str | None = Field(None, description="Display name")
    onboarding_completed: bool = Field(default=False, description="Whether user finished onboarding")
    onboarding_step: int = Field(default=0, description="Current onboarding step (0-5)")
    plan: str = Field(default="free", description="Resolved plan: free, trial, or premium")
    trial_expires_at: datetime | None = Field(None, description="Trial expiration")
    subscription_expires_at: datetime | None = Field(None, description="Premium expiration")

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
