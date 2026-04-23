"""Schemas for admin authentication and user management endpoints."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AdminLoginRequest(BaseModel):
    """Request body for admin email/password login."""

    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=1, max_length=255)


class AdminRefreshTokenRequest(BaseModel):
    """Request body for refreshing an admin access token."""

    refresh_token: str = Field(..., min_length=1)


class AdminResponse(BaseModel):
    """Admin profile returned from admin auth endpoints."""

    id: str
    email: str
    is_active: bool
    last_login_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AdminTokenResponse(BaseModel):
    """Token pair and admin profile returned after login."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    admin: AdminResponse


class AdminAccessTokenResponse(BaseModel):
    """Response returned from admin refresh."""

    access_token: str
    token_type: str = "bearer"


class AdminUserListItem(BaseModel):
    """One user row in the admin users list."""

    id: str
    email: str
    name: str | None = None
    nickname: str | None = None
    provider: Literal["google", "apple"]
    onboarding_completed: bool
    onboarding_step: int
    plan: Literal["free", "trial", "vuvl_plus", "vuvl_pro"]
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime | None = None
    total_request_count: int
    total_chat_count: int
    total_voice_count: int
    total_minutes_used: float


class AdminUsersListResponse(BaseModel):
    """Paginated response for admin users list."""

    page: int
    page_size: int
    total: int
    items: list[AdminUserListItem]


class AdminUsagePoint(BaseModel):
    """One daily usage row for admin detail/history views."""

    date: str
    request_count: int
    chat_count: int
    voice_count: int
    minutes_used: float


class AdminUsageTotals(BaseModel):
    """Aggregated usage totals."""

    request_count: int
    chat_count: int
    voice_count: int
    minutes_used: float


class AdminUserUsageSummary(BaseModel):
    """Top-level usage summary attached to user detail."""

    today: AdminUsagePoint
    totals: AdminUsageTotals
    last_activity_at: datetime | None = None


class AdminRecentConversation(BaseModel):
    """Conversation summary used in the admin user detail response."""

    id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_user_message: str | None = None
    last_ai_reply: str | None = None


class AdminUserDetailResponse(BaseModel):
    """Complete read-only user detail payload for the admin panel."""

    id: str
    email: str
    name: str | None = None
    provider: Literal["google", "apple"]
    nickname: str | None = None
    native_language: str | None = None
    native_language_code: str | None = None
    student_type: str | None = None
    occupation: str | None = None
    goal: str | None = None
    english_level: str | None = None
    onboarding_completed: bool
    onboarding_step: int
    plan: Literal["free", "trial", "vuvl_plus", "vuvl_pro"]
    trial_expires_at: datetime | None = None
    subscription_expires_at: datetime | None = None
    is_trial_used: bool
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime | None = None
    usage_summary: AdminUserUsageSummary
    usage_history: list[AdminUsagePoint]
    recent_activity: list[AdminRecentConversation]
