"""
Tabbly streaming TTS (non-English path).
https://docs.tabbly.io/tts-api/tts-streaming
"""
from __future__ import annotations

import json
import logging
import threading
import time
from io import BytesIO
from typing import List, Optional

from pydub import AudioSegment

from app.core.config import settings

logger = logging.getLogger(__name__)

TABBLY_BASE_URL = "https://api.tabbly.io"
TABBLY_STREAM_PATH = "/tts/stream"

_lock = threading.Lock()
_client = None


def _get_client():
    global _client
    with _lock:
        if _client is None:
            import httpx

            api_key = getattr(settings, "tabbly_api_key", None)
            if not api_key:
                raise ValueError(
                    "TABBLY_API_KEY is not set. Required when TTS_TABBLY_FOR_NON_ENGLISH=true."
                )
            timeout_s = float(getattr(settings, "tts_timeout_seconds", 45) or 45)
            _client = httpx.Client(
                base_url=TABBLY_BASE_URL,
                headers={
                    "X-API-Key": api_key,
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=timeout_s,
                    write=10.0,
                    pool=5.0,
                ),
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                ),
            )
    return _client


def _split_consecutive_wav_files(buf: bytes) -> List[bytes]:
    """Split a buffer that may contain multiple RIFF/WAVE files back-to-back."""
    out: List[bytes] = []
    i = 0
    n = len(buf)
    while i + 12 <= n:
        if buf[i : i + 4] != b"RIFF" or buf[i + 8 : i + 12] != b"WAVE":
            i += 1
            continue
        chunk_sz = int.from_bytes(buf[i + 4 : i + 8], "little")
        wav_total = 8 + chunk_sz
        if i + wav_total > n:
            break
        out.append(bytes(buf[i : i + wav_total]))
        i += wav_total
    return out


def _merge_wav_parts(parts: List[bytes]) -> bytes:
    """Merge one or more WAV byte blobs into a single WAV (pydub re-encode)."""
    if not parts:
        raise ValueError("No WAV segments to merge")
    if len(parts) == 1:
        return parts[0]
    combined = AudioSegment.empty()
    for p in parts:
        combined += AudioSegment.from_wav(BytesIO(p))
    out = BytesIO()
    combined.export(out, format="wav")
    return out.getvalue()


def _normalize_stream_to_wav(raw: bytes) -> bytes:
    """
    Tabbly may return one WAV or several concatenated. Produce one valid WAV bytes.
    """
    if not raw:
        raise ValueError("Tabbly TTS returned empty body")
    parts = _split_consecutive_wav_files(raw)
    if parts:
        try:
            return _merge_wav_parts(parts)
        except Exception as e:
            logger.warning("Tabbly WAV split/merge failed (%s), trying single parse", e)
    try:
        seg = AudioSegment.from_wav(BytesIO(raw))
        if len(seg) == 0:
            raise ValueError("Tabbly audio is empty")
        out = BytesIO()
        seg.export(out, format="wav")
        return out.getvalue()
    except Exception as e:
        logger.error("Tabbly WAV parse failed: %s", e, exc_info=True)
        raise ValueError("Tabbly TTS returned unreadable audio") from e


def _parse_error_body(body: bytes, status_code: int) -> str:
    if not body:
        return f"Tabbly TTS request failed (HTTP {status_code})"
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
        if isinstance(data, dict):
            msg = data.get("message") or data.get("error") or data.get("detail")
            if msg:
                return str(msg)
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    preview = body[:500].decode("utf-8", errors="replace")
    return f"Tabbly TTS request failed (HTTP {status_code}): {preview}"


def synthesize_wav(text: str) -> bytes:
    """
    Call Tabbly streaming endpoint; return a single WAV file as bytes.

    Raises:
        ValueError: on empty text, client/config errors, or API errors (4xx user-facing).
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text for Tabbly TTS")
    max_chars = 2000
    if len(text) > max_chars:
        logger.warning("Tabbly TTS text length %s, truncating to %s", len(text), max_chars)
        text = text[:max_chars]

    voice_id = getattr(settings, "tts_tabbly_voice_id", "Mosina") or "Mosina"
    model_id = getattr(settings, "tts_tabbly_model_id", "tabbly-tts") or "tabbly-tts"
    payload = {
        "text": text,
        "voice_id": voice_id,
        "model_id": model_id,
    }
    max_retries = max(0, int(getattr(settings, "tts_tabbly_max_retries", 2)))

    import httpx

    client = _get_client()
    last_error_msg: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            with client.stream("POST", TABBLY_STREAM_PATH, json=payload) as response:
                if response.status_code != 200:
                    err_body = response.read()
                    msg = _parse_error_body(err_body, response.status_code)
                    if 400 <= response.status_code < 500:
                        raise ValueError(msg)
                    last_error_msg = msg
                    if attempt < max_retries:
                        delay = 0.5 * (2**attempt)
                        logger.warning(
                            "Tabbly TTS HTTP %s, retry %s/%s after %.1fs",
                            response.status_code,
                            attempt + 1,
                            max_retries,
                            delay,
                        )
                        time.sleep(delay)
                        continue
                    raise ValueError(msg)

                buf = bytearray()
                for chunk in response.iter_bytes():
                    if chunk:
                        buf.extend(chunk)
                raw = bytes(buf)
                wav = _normalize_stream_to_wav(raw)
                logger.debug("Tabbly TTS generated %s bytes WAV (raw stream %s)", len(wav), len(raw))
                return wav

        except ValueError:
            raise
        except (httpx.TimeoutException, httpx.ConnectError, httpx.TransportError) as e:
            last_error_msg = str(e)
            if attempt < max_retries:
                delay = 0.5 * (2**attempt)
                logger.warning(
                    "Tabbly TTS transport error, retry %s/%s after %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    e,
                )
                time.sleep(delay)
                continue
            logger.error("Tabbly TTS failed after retries: %s", e, exc_info=True)
            raise ValueError("Tabbly TTS temporarily unavailable. Please try again.") from e

    raise ValueError(last_error_msg or "Tabbly TTS failed.")


def warmup() -> None:
    """One minimal synthesis to validate API key and connectivity."""
    synthesize_wav("नमस्ते.")
