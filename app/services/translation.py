"""Assistant reply translation using Google Cloud Translation APIs."""
import hashlib
import html
import json
import logging
import threading
from typing import Any, Dict, Optional, Tuple

import requests

try:
    from google.auth import default as google_auth_default
    from google.auth.transport.requests import AuthorizedSession
except ImportError:  # pragma: no cover - exercised via graceful fallback in runtime/test envs
    google_auth_default = None
    AuthorizedSession = None

from app.core.config import settings
from app.services.cache import get_json, set_json
from app.utils.language import normalize_language_code, sanitize_translated_reply_text

logger = logging.getLogger(__name__)

_TRANSLATION_SCOPE = ("https://www.googleapis.com/auth/cloud-platform",)
_TRANSLATION_API_BASE = "https://translation.googleapis.com/v3"
_TRANSLATION_API_BASIC_URL = "https://translation.googleapis.com/language/translate/v2"

_translation_credentials = None
_translation_project_id: Optional[str] = None
_translation_init_lock = threading.Lock()
_translation_init_error_logged = False


def _generate_translation_cache_key(
    reply_text: str,
    source_language: Optional[str],
    target_language: str,
) -> str:
    """Cache translations independently from LLM responses."""
    payload = {
        "reply_text": reply_text,
        "source_language": source_language or "",
        "target_language": target_language,
        "location": settings.translation_google_location,
        "provider": settings.translation_provider,
        "auth_mode": "api_key" if settings.translation_api_key else "adc",
        "basic_model": "nmt" if settings.translation_api_key else "",
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return f"translate:gcloud:v1:{hashlib.md5(encoded.encode()).hexdigest()}"


def _log_init_error_once(message: str, *args: Any) -> None:
    """Avoid noisy logs when credentials/project configuration is missing."""
    global _translation_init_error_logged
    if _translation_init_error_logged:
        return
    logger.warning(message, *args)
    _translation_init_error_logged = True


def _get_translation_credentials_and_project() -> Tuple[Optional[Any], Optional[str]]:
    """Load ADC credentials and resolve the project id for Cloud Translation."""
    global _translation_credentials, _translation_project_id

    if settings.translation_provider != "google_cloud":
        return None, None

    if _translation_credentials is not None and _translation_project_id:
        return _translation_credentials, _translation_project_id

    with _translation_init_lock:
        if _translation_credentials is not None and _translation_project_id:
            return _translation_credentials, _translation_project_id

        try:
            if google_auth_default is None:
                raise ImportError("google-auth is not installed")
            credentials, detected_project_id = google_auth_default(scopes=_TRANSLATION_SCOPE)
            project_id = settings.translation_google_project_id or detected_project_id
            if not project_id:
                raise ValueError(
                    "Google Cloud project id is required for Translation. "
                    "Set TRANSLATION_GOOGLE_PROJECT_ID or provide ADC with a project."
                )
            _translation_credentials = credentials
            _translation_project_id = project_id
            logger.info(
                "Google Cloud Translation initialized",
                extra={
                    "project_id": project_id,
                    "location": settings.translation_google_location,
                },
            )
        except Exception as exc:
            _log_init_error_once("Google Cloud Translation unavailable: %s", exc)
            return None, None

    return _translation_credentials, _translation_project_id


def translate_reply_text(
    reply_text: str,
    source_language: Optional[str],
    target_language: Optional[str],
) -> Optional[str]:
    """Translate finalized assistant reply text for display only."""
    clean_reply = (reply_text or "").strip()
    source = normalize_language_code(source_language)
    target = normalize_language_code(target_language)

    if not clean_reply or not target:
        logger.info(
            "translate_reply_text skipped: %s",
            "empty_reply" if not clean_reply else "no_target_language",
        )
        return None
    if source and source == target:
        logger.info(
            "translate_reply_text skipped: same_source_and_target lang=%s",
            source,
        )
        return None

    cache_key = _generate_translation_cache_key(clean_reply, source, target)
    cached = get_json(cache_key)
    if cached is not None and "translated_reply_text" in cached:
        logger.info(
            "translate_reply_text: Google translation from cache (source=%s target=%s)",
            source,
            target,
        )
        return cached.get("translated_reply_text")

    if settings.translation_api_key:
        logger.info(
            "translate_reply_text: Google Cloud Translation Basic API v2 (API key) source=%s target=%s",
            source,
            target,
        )
        translated = _translate_with_api_key(clean_reply, source, target)
        if translated is not None:
            set_json(
                cache_key,
                {"translated_reply_text": translated},
                settings.translation_cache_ttl,
            )
        return translated

    credentials, project_id = _get_translation_credentials_and_project()
    if not credentials or not project_id:
        logger.info(
            "translate_reply_text: no Google route (no API key and ADC unavailable) source=%s target=%s",
            source,
            target,
        )
        return None
    if AuthorizedSession is None:
        _log_init_error_once("Google Cloud Translation unavailable: google-auth transport is not installed")
        return None
    logger.info(
        "translate_reply_text: Google Cloud Translation Advanced v3 (ADC) project=%s source=%s target=%s",
        project_id,
        source,
        target,
    )
    translated = _translate_with_adc(clean_reply, source, target, credentials, project_id)
    if translated is not None:
        set_json(
            cache_key,
            {"translated_reply_text": translated},
            settings.translation_cache_ttl,
        )
    return translated


def _translate_with_api_key(
    reply_text: str,
    source_language: Optional[str],
    target_language: str,
) -> Optional[str]:
    """Translate using Cloud Translation Basic v2 with API-key authentication."""
    request_body: Dict[str, Any] = {
        "q": [reply_text],
        "target": target_language,
        "format": "text",
        "model": "nmt",
    }
    if source_language:
        request_body["source"] = source_language

    try:
        response = requests.post(
            _TRANSLATION_API_BASIC_URL,
            params={"key": settings.translation_api_key},
            json=request_body,
            timeout=settings.translation_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        translations = ((payload.get("data") or {}).get("translations") or [])
        translated = None
        if translations:
            translated = html.unescape((translations[0].get("translatedText") or "").strip()) or None
        return sanitize_translated_reply_text(reply_text, translated, target_language)
    except Exception as exc:
        logger.warning(
            "Google Cloud Translation API-key request failed",
            extra={
                "source_language": source_language,
                "target_language": target_language,
                "error": str(exc),
            },
        )
        return None


def _translate_with_adc(
    reply_text: str,
    source_language: Optional[str],
    target_language: str,
    credentials: Any,
    project_id: str,
) -> Optional[str]:
    """Translate using Cloud Translation Advanced v3 with ADC/service-account auth."""
    request_body: Dict[str, Any] = {
        "contents": [reply_text],
        "mimeType": "text/plain",
        "targetLanguageCode": target_language,
    }
    if source_language:
        request_body["sourceLanguageCode"] = source_language

    url = (
        f"{_TRANSLATION_API_BASE}/projects/{project_id}/locations/"
        f"{settings.translation_google_location}:translateText"
    )

    try:
        session = AuthorizedSession(credentials)
        response = session.post(
            url,
            json=request_body,
            headers={"x-goog-user-project": project_id},
            timeout=settings.translation_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        translations = payload.get("translations") or []
        translated = None
        if translations:
            translated = html.unescape((translations[0].get("translatedText") or "").strip()) or None
        return sanitize_translated_reply_text(reply_text, translated, target_language)
    except Exception as exc:
        logger.warning(
            "Google Cloud Translation request failed",
            extra={
                "source_language": source_language,
                "target_language": target_language,
                "error": str(exc),
            },
        )
        return None


def attach_translated_reply_text(
    ai_response: Dict[str, Any],
    source_language: Optional[str],
    target_language: Optional[str],
) -> Dict[str, Any]:
    """Return a response dict with translated_reply_text resolved by the translation backend."""
    return {
        **ai_response,
        "translated_reply_text": translate_reply_text(
            ai_response.get("reply_text", ""),
            source_language,
            target_language,
        ),
    }
