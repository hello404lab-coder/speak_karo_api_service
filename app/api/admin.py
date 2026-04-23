"""Admin authentication and read-only user management endpoints."""
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.core.security import (
    ADMIN_PRINCIPAL_TYPE,
    REFRESH_TOKEN_TYPE,
    create_access_token,
    create_refresh_token,
    verify_token,
)
from app.database import get_db
from app.dependencies.admin_auth import get_current_admin
from app.dependencies.auth import limiter
from app.models.admin import Admin
from app.schemas.admin import (
    AdminAccessTokenResponse,
    AdminLoginRequest,
    AdminRefreshTokenRequest,
    AdminResponse,
    AdminTokenResponse,
    AdminUserDetailResponse,
    AdminUsersListResponse,
)
from app.services.admin_auth_service import authenticate_admin, record_admin_login
from app.services.admin_query_service import get_user_detail_for_admin, list_users_for_admin

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/auth/login", response_model=AdminTokenResponse)
@limiter.limit("10/minute")
async def admin_login(
    request: Request,
    body: AdminLoginRequest,
    db: Session = Depends(get_db),
) -> AdminTokenResponse:
    """Authenticate an admin and return admin-scoped tokens."""
    admin = authenticate_admin(db, body.email, body.password)
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    admin = record_admin_login(db, admin)
    access_token = create_access_token(admin.id, principal_type=ADMIN_PRINCIPAL_TYPE)
    refresh_token = create_refresh_token(admin.id, principal_type=ADMIN_PRINCIPAL_TYPE)
    logger.info("Admin login success: admin_id=%s", admin.id)
    return AdminTokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        admin=AdminResponse.model_validate(admin),
    )


@router.post("/auth/refresh", response_model=AdminAccessTokenResponse)
@limiter.limit("10/minute")
async def admin_refresh(
    request: Request,
    body: AdminRefreshTokenRequest,
) -> AdminAccessTokenResponse:
    """Exchange a valid admin refresh token for a new admin access token."""
    admin_id = verify_token(body.refresh_token, REFRESH_TOKEN_TYPE, ADMIN_PRINCIPAL_TYPE)
    if not admin_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    access_token = create_access_token(admin_id, principal_type=ADMIN_PRINCIPAL_TYPE)
    return AdminAccessTokenResponse(access_token=access_token, token_type="bearer")


@router.get("/auth/me", response_model=AdminResponse)
@limiter.limit("10/minute")
async def admin_me(
    request: Request,
    current_admin: Admin = Depends(get_current_admin),
) -> AdminResponse:
    """Return the currently authenticated admin."""
    return AdminResponse.model_validate(current_admin)


@router.post("/auth/logout")
@limiter.limit("10/minute")
async def admin_logout(
    request: Request,
    current_admin: Admin = Depends(get_current_admin),
) -> dict[str, str]:
    """Stateless admin logout. The client must clear stored tokens."""
    logger.info("Admin logout: admin_id=%s", current_admin.id)
    return {"message": "Successfully logged out"}


@router.get("/users", response_model=AdminUsersListResponse)
@limiter.limit("30/minute")
async def admin_list_users(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    search: str | None = Query(default=None),
    provider: Literal["google", "apple"] | None = Query(default=None),
    plan: Literal["free", "trial", "vuvl_plus", "vuvl_pro"] | None = Query(default=None),
    onboarding_completed: bool | None = Query(default=None),
    sort_by: Literal["created_at", "last_activity_at", "email"] = Query(default="created_at"),
    sort_order: Literal["asc", "desc"] = Query(default="desc"),
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_admin),
) -> AdminUsersListResponse:
    """Return a paginated, searchable, filterable user list for admins."""
    del current_admin
    payload = list_users_for_admin(
        db,
        page=page,
        page_size=page_size,
        search=search,
        provider=provider,
        plan=plan,
        onboarding_completed=onboarding_completed,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return AdminUsersListResponse.model_validate(payload)


@router.get("/users/{user_id}", response_model=AdminUserDetailResponse)
@limiter.limit("30/minute")
async def admin_get_user_detail(
    user_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_admin),
) -> AdminUserDetailResponse:
    """Return read-only profile, usage, and recent activity for one user."""
    del current_admin
    payload = get_user_detail_for_admin(db, user_id)
    if not payload:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return AdminUserDetailResponse.model_validate(payload)
