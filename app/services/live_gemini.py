"""Gemini Live ephemeral auth tokens (google-genai async client)."""
import logging
from datetime import datetime, timedelta, timezone

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)


def _live_connect_config(system_instruction: str) -> types.LiveConnectConfig:
    modalities = [types.Modality.AUDIO]
    voice_name = (getattr(settings, "gemini_live_voice", None) or "").strip() or "Puck"

    # Gemini Live rejects language_code / languageCodes on speech_config for token
    # constraints; voice only. Use GEMINI_LIVE_LANGUAGE_CODE in /config for clients.
    speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name),
        ),
    )

    return types.LiveConnectConfig(
        system_instruction=system_instruction,
        response_modalities=modalities,
        temperature=float(getattr(settings, "gemini_live_temperature", 0.4) or 0.4),
        speech_config=speech_config,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )


async def mint_live_ephemeral_auth_token(system_instruction: str) -> str:
    """
    Create a short-lived auth token for browser/mobile Live WebSocket connections.
    Uses GEMINI_API_KEY server-side only.
    """
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=settings.gemini_api_key)
    constraints = types.LiveConnectConstraints(
        model=settings.gemini_live_model,
        config=_live_connect_config(system_instruction),
    )

    uses = int(getattr(settings, "gemini_live_token_uses", 1) or 1)
    if uses < 0:
        uses = 1
    new_sess_sec = int(getattr(settings, "gemini_live_token_new_session_seconds", 120) or 120)
    new_sess_sec = max(10, min(new_sess_sec, 3600))

    expire_time = datetime.now(timezone.utc) + timedelta(minutes=30)
    new_session_expire_time = datetime.now(timezone.utc) + timedelta(seconds=new_sess_sec)

    # Ephemeral auth tokens are only exposed on v1alpha; default client version 404s.
    create_kwargs: dict = {
        "http_options": types.HttpOptions(api_version="v1alpha"),
        "expire_time": expire_time,
        "new_session_expire_time": new_session_expire_time,
        "live_connect_constraints": constraints,
    }
    if uses > 0:
        create_kwargs["uses"] = uses

    token = await client.aio.auth_tokens.create(
        config=types.CreateAuthTokenConfig(**create_kwargs),
    )
    if not token.name:
        raise ValueError("Auth token response missing name")
    logger.info("Minted Gemini Live ephemeral auth token")
    return token.name
