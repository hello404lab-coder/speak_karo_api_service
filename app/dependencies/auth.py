"""Authentication dependencies and rate limiting."""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

import app.database as database_module
from app.core.security import ACCESS_TOKEN_TYPE, USER_PRINCIPAL_TYPE, verify_token
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/oauth", auto_error=True)

limiter = Limiter(key_func=get_remote_address)


def get_auth_db() -> Session:
    """Resolve the DB dependency at runtime so tests can patch app.database.get_db."""
    db_provider = database_module.get_db()
    if callable(db_provider):
        db_provider = db_provider()
    yield from db_provider


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_auth_db),
) -> User:
    """Decode JWT from Authorization header and return the User. Raises 401 if invalid."""
    user_id = verify_token(token, ACCESS_TOKEN_TYPE, USER_PRINCIPAL_TYPE)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
