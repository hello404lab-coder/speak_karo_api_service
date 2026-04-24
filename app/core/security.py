"""JWT and password utilities for authentication."""
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

# JWT
ALGORITHM = "HS256"
ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"
USER_PRINCIPAL_TYPE = "user"
ADMIN_PRINCIPAL_TYPE = "admin"


def create_access_token(subject: str, principal_type: str = USER_PRINCIPAL_TYPE) -> str:
    """Create a JWT access token. subject is the user id (str)."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "exp": expire,
        "type": ACCESS_TOKEN_TYPE,
        "principal_type": principal_type,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def create_refresh_token(subject: str, principal_type: str = USER_PRINCIPAL_TYPE) -> str:
    """Create a JWT refresh token. subject is the user id (str)."""
    expire = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_token_expire_days)
    payload: dict[str, Any] = {
        "sub": subject,
        "exp": expire,
        "type": REFRESH_TOKEN_TYPE,
        "principal_type": principal_type,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def _principal_matches(token_principal_type: str | None, expected_principal_type: str) -> bool:
    """Allow legacy tokens without principal_type for user auth only."""
    if expected_principal_type == USER_PRINCIPAL_TYPE:
        return token_principal_type in (None, "", USER_PRINCIPAL_TYPE)
    return token_principal_type == expected_principal_type


def verify_token(
    token: str,
    token_type: str,
    principal_type: str = USER_PRINCIPAL_TYPE,
) -> str | None:
    """
    Decode and verify a JWT. Returns the subject (user id) if valid, else None.
    token_type must be 'access' or 'refresh'. principal_type scopes the token
    to user or admin routes.
    """
    if token_type not in (ACCESS_TOKEN_TYPE, REFRESH_TOKEN_TYPE):
        return None
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
        if payload.get("type") != token_type:
            return None
        if not _principal_matches(payload.get("principal_type"), principal_type):
            return None
        sub = payload.get("sub")
        if not sub or not isinstance(sub, str):
            return None
        return sub
    except JWTError:
        return None


# Passwords
# Prefer pbkdf2_sha256 for compatibility in our runtime, while still accepting
# legacy bcrypt hashes if any were created previously.
pwd_context = CryptContext(schemes=["pbkdf2_sha256", "bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a plain password for storage."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a plain password against a stored hash."""
    return pwd_context.verify(plain_password, hashed_password)
