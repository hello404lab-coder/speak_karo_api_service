"""OAuth ID token verification and user resolution."""
import logging
from typing import Any

import requests
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
from jose import jwt as jose_jwt
from jose import jwk
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.user import User

logger = logging.getLogger(__name__)

APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"
APPLE_ISSUER = "https://appleid.apple.com"


def verify_google_token(id_token: str) -> dict[str, Any]:
    """
    Verify Google ID token and return payload with email, name, sub.
    Raises ValueError if token is invalid or audience/issuer check fails.
    """
    if not settings.google_client_id:
        logger.error("Google OAuth: GOOGLE_CLIENT_ID not configured")
        raise ValueError("Google sign-in is not configured")
    try:
        request = google_requests.Request()
        id_info = google_id_token.verify_oauth2_token(
            id_token,
            request,
            settings.google_client_id,
        )
        return {
            "sub": id_info["sub"],
            "email": id_info.get("email") or "",
            "name": id_info.get("name"),
        }
    except ValueError as e:
        logger.warning("Google token verification failed: %s", e)
        raise
    except Exception as e:
        logger.exception("Google OAuth provider error: %s", e)
        raise ValueError("Invalid Google token") from e


def _get_apple_public_key(token: str) -> Any:
    """Fetch Apple JWKS and return key for the token's kid."""
    try:
        unverified = jose_jwt.get_unverified_header(token)
        kid = unverified.get("kid")
        if not kid:
            return None
    except Exception:
        return None

    resp = requests.get(APPLE_KEYS_URL, timeout=10)
    resp.raise_for_status()
    keys_payload = resp.json()
    keys_list = keys_payload.get("keys") or []

    for key_data in keys_list:
        if key_data.get("kid") == kid:
            try:
                return jwk.construct(key_data)
            except Exception as e:
                logger.warning("Apple JWK construct failed for kid=%s: %s", kid, e)
                return None
    return None


def verify_apple_token(id_token: str) -> dict[str, Any]:
    """
    Verify Apple ID token using Apple's public keys. Validates aud, iss, exp.
    Returns payload with sub, email, name (Apple may omit email/name).
    Raises ValueError if token is invalid.
    """
    if not settings.apple_client_id:
        logger.error("Apple OAuth: APPLE_CLIENT_ID not configured")
        raise ValueError("Apple sign-in is not configured")

    key = _get_apple_public_key(id_token)
    if not key:
        logger.warning("Apple token verification failed: could not get public key")
        raise ValueError("Invalid Apple token")

    try:
        payload = jose_jwt.decode(
            id_token,
            key,
            algorithms=["RS256"],
            audience=settings.apple_client_id,
            issuer=APPLE_ISSUER,
        )
        return {
            "sub": payload.get("sub") or "",
            "email": payload.get("email") or "",
            "name": payload.get("name"),
        }
    except jose_jwt.JWTError as e:
        logger.warning("Apple token verification failed: %s", e)
        raise ValueError("Invalid or expired Apple token") from e
    except Exception as e:
        logger.exception("Apple OAuth provider error: %s", e)
        raise ValueError("Invalid Apple token") from e


def get_or_create_user(
    db: Session,
    provider: str,
    provider_info: dict[str, Any],
) -> User:
    """
    Find user by provider_id; if not found, create a new user.
    provider_info must have: sub (provider_id), email, name (optional).
    """
    provider_id = provider_info.get("sub") or ""
    email = (provider_info.get("email") or "").strip()
    name = provider_info.get("name")
    if not provider_id:
        raise ValueError("Provider info missing 'sub'")

    user = db.query(User).filter(
        User.provider == provider,
        User.provider_id == provider_id,
    ).first()

    if user:
        if name and user.name != name:
            user.name = name
        if email and user.email != email:
            user.email = email
        db.commit()
        db.refresh(user)
        return user

    if not email:
        email = f"{provider}_{provider_id}@placeholder.local"
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        email = f"{provider}_{provider_id}@placeholder.local"

    user = User(
        email=email,
        name=name,
        provider=provider,
        provider_id=provider_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("Created new user: id=%s provider=%s email=%s", user.id, provider, user.email)
    return user
