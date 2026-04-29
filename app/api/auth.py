"""Authentication API: OAuth login, refresh, and me."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies.auth import get_current_user, limiter
from app.models.billing import BillingSubscription
from app.models.user import User
from app.schemas.auth import (
    AccessTokenResponse,
    EmailLoginRequest,
    EmailRegisterRequest,
    OAuthLoginRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserResponse,
)
from app.schemas.onboarding import (
    EnglishLevelStep,
    GoalStep,
    NativeLanguageStep,
    NicknameStep,
    OnboardingStatusResponse,
    StudentDetailsStep,
)
from app.core.security import (
    REFRESH_TOKEN_TYPE,
    USER_PRINCIPAL_TYPE,
    create_access_token,
    create_refresh_token,
    verify_token,
)
from app.services.auth_service import (
    authenticate_with_email,
    get_or_create_user,
    register_with_email,
    verify_apple_token,
    verify_google_token,
)
from app.services.billing_service import get_subscription_for_status, serialize_subscription_summary
from app.services.subscription_service import resolve_user_plan
from app.utils.language import normalize_language_code

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_user_response(user: User, db: Session) -> UserResponse:
    """Serialize a user with safe defaults for optional/mock attributes."""
    plan = resolve_user_plan(user)
    summary = {
        "billing_phase": "free",
        "billing_status": None,
        "active_plan_code": None,
        "current_period_end": None,
        "coupon_code": None,
    }
    try:
        billing_row = get_subscription_for_status(db, user.id)
        if isinstance(billing_row, BillingSubscription):
            summary = serialize_subscription_summary(billing_row, user=user)
    except Exception:
        logger.warning("Billing summary unavailable for auth response: user_id=%s", user.id, exc_info=True)
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        native_language=getattr(user, "native_language", None),
        native_language_code=getattr(user, "native_language_code", None),
        onboarding_completed=bool(getattr(user, "onboarding_completed", False) or False),
        onboarding_step=int(getattr(user, "onboarding_step", 0) or 0),
        plan=plan,
        billing_phase=summary["billing_phase"],
        trial_expires_at=getattr(user, "trial_expires_at", None),
        subscription_expires_at=getattr(user, "subscription_expires_at", None),
        billing_status=summary["billing_status"],
        active_plan_code=summary["active_plan_code"],
        current_period_end=summary["current_period_end"],
        coupon_code=summary["coupon_code"],
    )


@router.post("/oauth", response_model=TokenResponse)
@limiter.limit("10/minute")
async def oauth_login(
    request: Request,
    body: OAuthLoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """
    Exchange Google or Apple ID token for our JWTs and user.
    Client performs OAuth on device and sends the ID token here.
    """
    try:
        if body.provider == "google":
            provider_info = verify_google_token(body.id_token)
        else:
            provider_info = verify_apple_token(body.id_token)
    except ValueError as e:
        logger.warning("OAuth token verification failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        ) from e

    user = get_or_create_user(db, body.provider, provider_info)
    access_token = create_access_token(subject=user.id, principal_type=USER_PRINCIPAL_TYPE)
    refresh_token = create_refresh_token(subject=user.id, principal_type=USER_PRINCIPAL_TYPE)

    logger.info("Login success: user_id=%s provider=%s", user.id, body.provider)
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=_build_user_response(user, db),
    )


@router.post("/register", response_model=TokenResponse)
@limiter.limit("10/minute")
async def email_register(
    request: Request,
    body: EmailRegisterRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Register a new account with email and password."""
    try:
        user = register_with_email(db, body.email, body.password, body.name)
    except ValueError as e:
        status_code = status.HTTP_409_CONFLICT if "already exists" in str(e) else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail=str(e)) from e

    access_token = create_access_token(subject=user.id, principal_type=USER_PRINCIPAL_TYPE)
    refresh_token = create_refresh_token(subject=user.id, principal_type=USER_PRINCIPAL_TYPE)

    logger.info("Email register success: user_id=%s email=%s", user.id, user.email)
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=_build_user_response(user, db),
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
async def email_login(
    request: Request,
    body: EmailLoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Sign in with email and password."""
    user = authenticate_with_email(db, body.email, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    access_token = create_access_token(subject=user.id, principal_type=USER_PRINCIPAL_TYPE)
    refresh_token = create_refresh_token(subject=user.id, principal_type=USER_PRINCIPAL_TYPE)

    logger.info("Email login success: user_id=%s", user.id)
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=_build_user_response(user, db),
    )


@router.post("/refresh", response_model=AccessTokenResponse)
@limiter.limit("10/minute")
async def refresh(
    request: Request,
    body: RefreshTokenRequest,
) -> AccessTokenResponse:
    """Exchange a valid refresh token for a new access token."""
    user_id = verify_token(body.refresh_token, REFRESH_TOKEN_TYPE, USER_PRINCIPAL_TYPE)
    if not user_id:
        logger.warning("Refresh token verification failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    access_token = create_access_token(subject=user_id, principal_type=USER_PRINCIPAL_TYPE)
    return AccessTokenResponse(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
@limiter.limit("10/minute")
async def me(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserResponse:
    """Return the current authenticated user (requires Bearer token)."""
    return _build_user_response(current_user, db)


@router.post("/logout")
@limiter.limit("10/minute")
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    """
    Log out the current user. Requires a valid access token.
    The client must clear the stored access and refresh tokens after calling this.
    JWTs are stateless; the server does not invalidate tokens.
    """
    logger.info("Logout: user_id=%s", current_user.id)
    return {"message": "Successfully logged out"}


# --- Onboarding (all require authentication) ---


@router.get("/onboarding/status", response_model=OnboardingStatusResponse)
@limiter.limit("10/minute")
async def onboarding_status(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> OnboardingStatusResponse:
    """Return current onboarding progress."""
    return OnboardingStatusResponse(
        onboarding_completed=current_user.onboarding_completed,
        current_step=current_user.onboarding_step,
    )


@router.post("/onboarding/nickname")
@limiter.limit("10/minute")
async def onboarding_nickname(
    request: Request,
    body: NicknameStep,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, int]:
    """Step 1: set nickname."""
    current_user.nickname = body.nickname
    current_user.onboarding_step = 1
    db.commit()
    db.refresh(current_user)
    logger.info("User %s completed onboarding step 1", current_user.id)
    return {"onboarding_step": 1}


@router.post("/onboarding/language")
@limiter.limit("10/minute")
async def onboarding_language(
    request: Request,
    body: NativeLanguageStep,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, int]:
    """Step 2: set native language."""
    current_user.native_language = body.native_language
    current_user.native_language_code = normalize_language_code(body.native_language)
    current_user.onboarding_step = 2
    db.commit()
    db.refresh(current_user)
    logger.info("User %s completed onboarding step 2", current_user.id)
    return {"onboarding_step": 2}


@router.post("/onboarding/student")
@limiter.limit("10/minute")
async def onboarding_student(
    request: Request,
    body: StudentDetailsStep,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, int]:
    """Step 3: set student type and occupation."""
    current_user.student_type = body.student_type
    current_user.occupation = body.occupation
    current_user.onboarding_step = 3
    db.commit()
    db.refresh(current_user)
    logger.info("User %s completed onboarding step 3", current_user.id)
    return {"onboarding_step": 3}


@router.post("/onboarding/goal")
@limiter.limit("10/minute")
async def onboarding_goal(
    request: Request,
    body: GoalStep,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, int]:
    """Step 4: set goal."""
    current_user.goal = body.goal
    current_user.onboarding_step = 4
    db.commit()
    db.refresh(current_user)
    logger.info("User %s completed onboarding step 4", current_user.id)
    return {"onboarding_step": 4}


@router.post("/onboarding/english-level")
@limiter.limit("10/minute")
async def onboarding_english_level(
    request: Request,
    body: EnglishLevelStep,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, int | bool]:
    """Step 5: set English level and mark onboarding completed."""
    current_user.english_level = body.english_level
    current_user.onboarding_step = 5
    current_user.onboarding_completed = True
    db.commit()
    db.refresh(current_user)
    logger.info("User %s completed onboarding step 5", current_user.id)
    logger.info("User %s completed onboarding", current_user.id)
    return {"onboarding_step": 5, "onboarding_completed": True}
