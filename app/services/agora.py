"""Agora RTC token generation for social voice channels."""
import logging
import time

from agora_token_builder.RtcTokenBuilder import RtcTokenBuilder, Role_Attendee

from app.core.config import settings

logger = logging.getLogger(__name__)


class AgoraConfigError(RuntimeError):
    """Raised when Agora credentials are missing or invalid for token minting."""


def generate_agora_token(channel_name: str, uid: int) -> str:
    """
    Build an RTC token for joining `channel_name` as `uid` (integer, non-zero for typical SDKs).
    """
    app_id = settings.agora_app_id
    cert = settings.agora_app_certificate
    if not app_id or not cert:
        raise AgoraConfigError("AGORA_APP_ID and AGORA_APP_CERTIFICATE must be set to generate tokens")

    ttl = max(60, int(settings.agora_token_ttl_seconds))
    current_ts = int(time.time())
    privilege_expired_ts = current_ts + ttl

    try:
        token = RtcTokenBuilder.buildTokenWithUid(
            app_id,
            cert,
            channel_name,
            uid,
            Role_Attendee,
            privilege_expired_ts,
        )
    except Exception as e:
        logger.exception("Agora token build failed for channel=%s uid=%s", channel_name, uid)
        raise AgoraConfigError("Failed to build Agora token") from e

    return token
