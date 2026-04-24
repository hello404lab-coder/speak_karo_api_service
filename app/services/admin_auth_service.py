"""Admin authentication and bootstrap helpers."""
import logging
from datetime import datetime

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import hash_password, verify_password
from app.models.admin import Admin

logger = logging.getLogger(__name__)


def normalize_admin_email(email: str) -> str:
    """Normalize admin email for lookup and storage."""
    return email.strip().lower()


def get_admin_by_email(db: Session, email: str) -> Admin | None:
    """Return an admin by normalized email."""
    return db.query(Admin).filter(Admin.email == normalize_admin_email(email)).first()


def authenticate_admin(db: Session, email: str, password: str) -> Admin | None:
    """Authenticate an active admin with email/password."""
    admin = get_admin_by_email(db, email)
    if not admin or not admin.is_active:
        return None
    if not verify_password(password, admin.password_hash):
        return None
    return admin


def record_admin_login(db: Session, admin: Admin) -> Admin:
    """Persist the admin last-login timestamp."""
    admin.last_login_at = datetime.utcnow()
    db.commit()
    db.refresh(admin)
    return admin


def bootstrap_admin_account(db: Session) -> Admin | None:
    """
    Create or update the bootstrap admin from settings.
    If either bootstrap setting is missing, nothing is created.
    """
    email = (settings.admin_bootstrap_email or "").strip()
    password = settings.admin_bootstrap_password or ""

    if not email and not password:
        return None
    if not email or not password:
        logger.warning(
            "Admin bootstrap skipped because ADMIN_BOOTSTRAP_EMAIL and ADMIN_BOOTSTRAP_PASSWORD must both be set"
        )
        return None

    normalized_email = normalize_admin_email(email)
    password_hash = hash_password(password)
    admin = db.query(Admin).filter(Admin.email == normalized_email).first()

    if admin:
        admin.password_hash = password_hash
        admin.is_active = True
        db.commit()
        db.refresh(admin)
        logger.info("Admin bootstrap updated existing admin: email=%s", normalized_email)
        return admin

    admin = Admin(email=normalized_email, password_hash=password_hash, is_active=True)
    db.add(admin)
    db.commit()
    db.refresh(admin)
    logger.info("Admin bootstrap created admin: email=%s", normalized_email)
    return admin
