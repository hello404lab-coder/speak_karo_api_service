"""LLM service for generating AI replies using Gemini."""
import hashlib
import json
import logging
from typing import Iterator, List, Dict, Optional, Any
from google import genai
from google.genai import types
from app.core.config import settings
from app.core.prompts import (
    LLMReplySchema,
    LearnerAnalysisSchema,
    LEARNER_ANALYSIS_REPAIR_INSTRUCTION,
    extract_correction_candidate_from_reply,
    get_system_instruction,
    parse_gemini_response,
    postprocess_llm_reply,
    prepare_history,
    user_analysis_needs_repair,
)
from app.services.cache import get_json, set_json
from app.services.translation import attach_translated_reply_text

logger = logging.getLogger(__name__)

# Initialize Gemini client (lazy loaded)
_gemini_client = None

_GEMINI_BUSY_MESSAGE = "The AI is busy right now. Please try again in a few seconds."
_GEMINI_TIMEOUT_MESSAGE = "The AI took too long to respond. Please try again."
_GEMINI_GENERIC_ERROR_MESSAGE = "Something went wrong. Please try again."


def _get_gemini_client():
    """Lazy load Gemini client with request timeout. Client is stateless (HTTP), no lock needed."""
    global _gemini_client
    if _gemini_client is None:
        if not settings.gemini_api_key:
            raise ValueError("Gemini API key not configured. Please set GEMINI_API_KEY in environment.")
        timeout_ms = settings.llm_timeout_seconds * 1000
        try:
            from google.genai.types import HttpOptions
            _gemini_client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=HttpOptions(timeout=timeout_ms),
            )
        except (ImportError, AttributeError):
            _gemini_client = genai.Client(api_key=settings.gemini_api_key)
        logger.info("Gemini client initialized for LLM (timeout=%ss)", settings.llm_timeout_seconds)
    return _gemini_client


def _extract_exception_status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def _user_facing_gemini_error_message(exc: Exception) -> str:
    status_code = _extract_exception_status_code(exc)
    message = str(exc or "").lower()

    if status_code in {429, 500, 502, 503, 504}:
        return _GEMINI_BUSY_MESSAGE
    if any(
        marker in message
        for marker in (
            "503",
            "429",
            "unavailable",
            "high demand",
            "resource exhausted",
            "rate limit",
            "overloaded",
            "try again later",
        )
    ):
        return _GEMINI_BUSY_MESSAGE
    if "timeout" in message or "timed out" in message:
        return _GEMINI_TIMEOUT_MESSAGE
    return _GEMINI_GENERIC_ERROR_MESSAGE


def init_llm_client() -> dict:
    """
    Initialize the Gemini client (used by /init-models warmup).
    Returns {"status": "loaded"} or {"status": "failed", "error": str}.
    """
    try:
        _get_gemini_client()
        return {"status": "loaded"}
    except Exception as e:
        logger.exception("LLM client init failed")
        return {"status": "failed", "error": str(e)}


TITLE_SYSTEM_INSTRUCTION = (
    "Generate a very short conversation title (max 6 words, no quotes) "
    "based on the following excerpt. Reply with only the title, nothing else."
)
TITLE_MAX_WORDS = 6


def generate_conversation_title(excerpt: str) -> str:
    """
    Generate a short conversation title from the first messages excerpt using Gemini.
    Returns a string of at most TITLE_MAX_WORDS words, or "Conversation" if empty/failed.
    """
    if not (excerpt or "").strip():
        return "Conversation"
    try:
        client = _get_gemini_client()
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=excerpt.strip())])]
        config = types.GenerateContentConfig(
            system_instruction=TITLE_SYSTEM_INSTRUCTION,
            max_output_tokens=50,
            temperature=0.3,
        )
        response = client.models.generate_content(
            model=settings.llm_model,
            contents=contents,
            config=config,
        )
        if (
            not response.candidates
            or not response.candidates[0].content
            or not response.candidates[0].content.parts
        ):
            return "Conversation"
        text = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text += part.text
        title = (text or "").strip().strip('"\'')
        if not title:
            return "Conversation"
        words = title.split()[:TITLE_MAX_WORDS]
        return " ".join(words) if words else "Conversation"
    except Exception as e:
        logger.warning("Conversation title generation failed: %s", e)
        return "Conversation"


def _history_to_contents(history_formatted: List[Dict]) -> List[types.Content]:
    """
    Convert prepare_history() output to Gemini Content list.
    history_formatted: list of {"role": "user"|"model", "parts": [content]}
    """
    contents = []
    for msg in history_formatted:
        role = msg.get("role", "user")
        parts = msg.get("parts", [])
        text = parts[0] if parts else ""
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=text)],
            )
        )
    return contents


# Reserve tokens for system instruction when trimming history (so total input stays under budget)
SYSTEM_INSTRUCTION_RESERVE_TOKENS = 2048


def _estimate_tokens_text(text: str) -> int:
    """Conservative estimate: ~4 chars per token for English."""
    return max(1, (len(text or "") + 3) // 4)


def _count_contents_tokens(client, model: str, contents: List[types.Content]) -> int:
    """Return token count for contents. Uses Gemini count_tokens when available, else estimate."""
    try:
        resp = client.models.count_tokens(model=model, contents=contents)
        total = getattr(resp, "total_tokens", None)
        if total is not None and isinstance(total, int):
            return total
    except Exception:
        pass
    total = 0
    for c in contents:
        if c.parts:
            for p in c.parts:
                total += _estimate_tokens_text(getattr(p, "text", None) or "")
    return total


def _extract_output_tokens_from_usage_metadata(usage_metadata: Any) -> int:
    """Return candidate/output token count from Gemini usage metadata."""
    if usage_metadata is None:
        return 0
    candidates = getattr(usage_metadata, "candidates_token_count", None)
    if isinstance(candidates, int):
        return max(0, candidates)
    return 0


def estimate_output_tokens_for_text(text: str, model: Optional[str] = None, client=None) -> int:
    """Estimate output token count for generated text, preferring Gemini count_tokens."""
    normalized = (text or "").strip()
    if not normalized:
        return 0
    model_name = model or settings.llm_model
    try:
        active_client = client or _get_gemini_client()
        resp = active_client.models.count_tokens(model=model_name, contents=normalized)
        total = getattr(resp, "total_tokens", None)
        if isinstance(total, int):
            return max(0, total)
    except Exception:
        pass
    return _estimate_tokens_text(normalized)


def _build_trimmed_contents(
    conversation_history: List[Dict[str, str]],
    current_user_message: str,
    token_budget: int,
    client,
    model: str,
) -> List[types.Content]:
    """
    Build contents for Gemini: history + current user message, trimmed so total tokens <= token_budget.
    Drops oldest messages first. Keeps at least the current user message.
    """
    history_formatted = prepare_history(conversation_history or [])
    contents = _history_to_contents(history_formatted)
    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=current_user_message)])
    )
    max_contents_tokens = max(0, token_budget - SYSTEM_INSTRUCTION_RESERVE_TOKENS)
    while _count_contents_tokens(client, model, contents) > max_contents_tokens and len(contents) > 1:
        contents.pop(0)
    return contents


def _generate_cache_key(
    user_message: str,
    conversation_history: List[Dict[str, str]],
    reply_language: str = "en",
) -> str:
    """Deterministic cache key: json.dumps with sort_keys so key is stable across runs."""
    context = (
        user_message
        + json.dumps(conversation_history, sort_keys=True)
        + (reply_language or "en")
    )
    return f"llm:v5:{hashlib.md5(context.encode()).hexdigest()}"


def _generate_analysis_repair_cache_key(user_message: str, ai_response: Dict[str, any]) -> str:
    """Cache key for learner-analysis repairs."""
    context = json.dumps(
        {
            "user_message": user_message,
            "reply_text": ai_response.get("reply_text", ""),
            "correction": ai_response.get("correction", ""),
            "explanation": ai_response.get("explanation", ""),
            "example": ai_response.get("example", ""),
            "score": ai_response.get("score", 70),
        },
        sort_keys=True,
    )
    return f"llm:analysis-repair:v2:{hashlib.md5(context.encode()).hexdigest()}"


def _contents_to_history_for_cache(contents: List[types.Content]) -> List[Dict[str, str]]:
    """Convert contents (trimmed list actually sent) to history shape for cache key."""
    out = []
    for c in contents:
        role = "user" if getattr(c, "role", None) == "user" else "assistant"
        text = ""
        if c.parts:
            for p in c.parts:
                if hasattr(p, "text") and p.text:
                    text = p.text
                    break
        out.append({"role": role, "content": text})
    return out


def _build_safety_settings():
    """Build safety settings for Gemini (block only HIGH). Returns None if types not found."""
    try:
        SafetySetting = getattr(types, "SafetySetting", None)
        HarmCategory = getattr(types, "HarmCategory", None)
        HarmBlockThreshold = getattr(types, "HarmBlockThreshold", None)
        if SafetySetting is None:
            SafetySetting = getattr(genai, "SafetySetting", None)
            HarmCategory = getattr(genai, "HarmCategory", None)
            HarmBlockThreshold = getattr(genai, "HarmBlockThreshold", None)
        if SafetySetting and HarmCategory and HarmBlockThreshold:
            return [
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
            ]
    except Exception:
        pass
    return None


def _repair_user_analysis(
    user_message: str,
    ai_response: Dict[str, any],
) -> Optional[Dict[str, any]]:
    """Repair clearly broken learner feedback with a targeted second pass."""
    client = _get_gemini_client()
    cache_key = _generate_analysis_repair_cache_key(user_message, ai_response)
    cached_response = get_json(cache_key)
    if cached_response:
        logger.info("Cache hit for learner analysis repair")
        return cached_response

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=json.dumps(
                        {
                            "learner_message": user_message,
                            "current_analysis": {
                                "correction": ai_response.get("correction"),
                                "explanation": ai_response.get("explanation"),
                                "example": ai_response.get("example"),
                                "score": ai_response.get("score", 70),
                            },
                        },
                        ensure_ascii=False,
                    )
                )
            ],
        )
    ]
    config = types.GenerateContentConfig(
        system_instruction=LEARNER_ANALYSIS_REPAIR_INSTRUCTION,
        response_mime_type="application/json",
        response_json_schema=LearnerAnalysisSchema.model_json_schema(),
        max_output_tokens=min(settings.llm_max_tokens, 160),
        temperature=0.1,
    )
    response = client.models.generate_content(
        model=settings.llm_model,
        contents=contents,
        config=config,
    )
    if (
        not response.candidates
        or not response.candidates[0].content
        or not response.candidates[0].content.parts
    ):
        return None

    response_text = ""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "text") and part.text:
            response_text += part.text
    if not response_text.strip():
        return None

    repaired = parse_gemini_response(response_text)
    normalized = {
        "reply_text": ai_response.get("reply_text", ""),
        "translated_reply_text": ai_response.get("translated_reply_text"),
        "correction": repaired.get("correction"),
        "explanation": repaired.get("explanation"),
        "example": repaired.get("example"),
        "score": repaired.get("score", ai_response.get("score", 70)),
    }
    set_json(cache_key, normalized, settings.llm_cache_ttl)
    return normalized


def finalize_llm_reply(
    ai_response: Dict[str, any],
    user_message: str,
    reply_language: str = "en",
    translation_language: Optional[str] = None,
) -> Dict[str, any]:
    """Normalize AI response and selectively repair bad learner feedback."""
    normalized = postprocess_llm_reply(
        ai_response,
        user_message,
        reply_language,
        translation_language,
    )
    if not user_analysis_needs_repair(normalized, user_message):
        return normalized

    try:
        repaired = _repair_user_analysis(user_message, normalized)
        if repaired:
            merged = postprocess_llm_reply(
                {
                    "reply_text": normalized.get("reply_text", ""),
                    "translated_reply_text": normalized.get("translated_reply_text"),
                    "correction": repaired.get("correction"),
                    "explanation": repaired.get("explanation"),
                    "example": repaired.get("example"),
                    "score": repaired.get("score", normalized.get("score", 70)),
                },
                user_message,
                reply_language,
                translation_language,
            )
            if not user_analysis_needs_repair(merged, user_message):
                logger.info("Repaired learner analysis successfully")
                return merged
    except Exception as e:
        logger.warning("Learner analysis repair failed: %s", e)

    candidate = extract_correction_candidate_from_reply(normalized.get("reply_text", ""), user_message)
    if candidate:
        fallback = {
            **normalized,
            "correction": candidate,
            "explanation": normalized.get("explanation"),
            "example": normalized.get("example"),
        }
        if not user_analysis_needs_repair(fallback, user_message):
            logger.info("Recovered learner correction from assistant reply fallback")
            return fallback
    return normalized


def _build_response_with_translation(
    ai_response: Dict[str, any],
    reply_language: str,
    translation_language: Optional[str],
) -> Dict[str, any]:
    """Attach translated display text after LLM finalization using the translation backend."""
    return attach_translated_reply_text(ai_response, reply_language, translation_language)


def stream_gemini_tokens(
    user_message: str,
    conversation_history: List[Dict[str, str]],
    reply_language: str = "en",
    translation_language: Optional[str] = None,
    long_term_context: Optional[str] = None,
    usage_sink: Optional[dict[str, int]] = None,
) -> Iterator[str]:
    """
    Stream Gemini response as text deltas (tokens). No caching.
    Uses the same JSON contract as sync generation so reply_text and learner analysis stay aligned.
    """
    if conversation_history is None:
        conversation_history = []
    try:
        client = _get_gemini_client()
        system_instruction = get_system_instruction(reply_language, translation_language, long_term_context)
        contents = _build_trimmed_contents(
            conversation_history,
            user_message,
            settings.llm_context_token_budget,
            client,
            settings.llm_model,
        )
        safety_settings = _build_safety_settings()
        config_dict = {
            "thinking_config": genai.types.ThinkingConfig(thinking_budget=0),
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_json_schema": LLMReplySchema.model_json_schema(),
            "max_output_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
        }
        if safety_settings is not None:
            config_dict["safety_settings"] = safety_settings
        config = types.GenerateContentConfig(**config_dict)
        logger.info("Starting Gemini stream (model=%s)", settings.llm_model)
        output_tokens_from_usage = 0
        for chunk in client.models.generate_content_stream(
            model=settings.llm_model,
            contents=contents,
            config=config,
        ):
            output_tokens_from_usage = max(
                output_tokens_from_usage,
                _extract_output_tokens_from_usage_metadata(getattr(chunk, "usage_metadata", None)),
            )
            # Avoid chunk.text when AFC may emit function_call parts (raises ValueError)
            text = None
            try:
                text = getattr(chunk, "text", None)
            except ValueError:
                pass
            if text and isinstance(text, str) and text.strip():
                yield text
            elif chunk.candidates and len(chunk.candidates) > 0:
                c = chunk.candidates[0]
                if c.content and c.content.parts:
                    for part in c.content.parts:
                        if getattr(part, "function_call", None) is not None:
                            continue
                        t = getattr(part, "text", None)
                        if t and isinstance(t, str) and t.strip():
                            yield t
        if usage_sink is not None:
            usage_sink["output_tokens"] = max(0, int(output_tokens_from_usage or 0))
    except Exception as e:
        logger.exception("Gemini stream error: %s", e)
        raise RuntimeError(_user_facing_gemini_error_message(e)) from e


def generate_reply_with_usage(
    user_message: str,
    conversation_history: List[Dict[str, str]] = None,
    reply_language: str = "en",
    translation_language: Optional[str] = None,
    long_term_context: Optional[str] = None,
) -> tuple[Dict[str, any], int]:
    """
    Generate AI reply with correction and explanation using Gemini.

    Args:
        user_message: User's message
        conversation_history: Previous messages for context (LLM layer trims by token budget)
        reply_language: "en" for English-only, or "hi"/"ml"/"ta"/etc. for full Indic response
        translation_language: Optional display translation language for translated_reply_text
        long_term_context: Optional learner context injected into system instruction every turn

    Returns:
        Dict with reply_text, correction, score
    """
    if conversation_history is None:
        conversation_history = []

    try:
        client = _get_gemini_client()
        contents = _build_trimmed_contents(
            conversation_history,
            user_message,
            settings.llm_context_token_budget,
            client,
            settings.llm_model,
        )
        # Cache key from trimmed history actually sent so cache matches what model saw
        trimmed_history = _contents_to_history_for_cache(contents)
        cache_key = _generate_cache_key(user_message, trimmed_history, reply_language)
        if long_term_context:
            cache_key = cache_key + ":" + hashlib.md5(long_term_context.encode()).hexdigest()
        cached_response = get_json(cache_key)
        if cached_response:
            logger.info("Cache hit for LLM response")
            return _build_response_with_translation(
                {
                    **cached_response,
                    "translated_reply_text": None,
                },
                reply_language,
                translation_language,
            ), 0

        system_instruction = get_system_instruction(reply_language, translation_language, long_term_context)

        logger.info(f"Calling Gemini LLM API with model: {settings.llm_model}")

        safety_settings = _build_safety_settings()
        if safety_settings is not None:
            logger.debug("Safety settings configured to block only HIGH probability content")

        # No tools passed; if AFC is enabled by default and causes non-JSON responses, consider disabling AFC here when supported.
        config_dict = {
            "thinking_config": genai.types.ThinkingConfig(thinking_budget=0),
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_json_schema": LLMReplySchema.model_json_schema(),
            "max_output_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
        }
        if safety_settings is not None:
            config_dict["safety_settings"] = safety_settings

        logger.debug(f"Calling Gemini with {len(contents)} content items, system_instruction length: {len(system_instruction)}")
        response = client.models.generate_content(
            model=settings.llm_model,
            contents=contents,
            config=types.GenerateContentConfig(**config_dict)
        )
        output_tokens = _extract_output_tokens_from_usage_metadata(getattr(response, "usage_metadata", None))
        
        # Log full response metadata for debugging
        logger.debug(f"Response object: candidates={len(response.candidates) if response.candidates else 0}")
        if response.candidates:
            logger.debug(f"First candidate: finish_reason={getattr(response.candidates[0], 'finish_reason', 'N/A')}, "
                        f"content.parts={len(response.candidates[0].content.parts) if response.candidates[0].content and response.candidates[0].content.parts else 0}")
        
        # Extract response text
        if (
            response.candidates is None
            or len(response.candidates) == 0
            or response.candidates[0].content is None
            or response.candidates[0].content.parts is None
            or len(response.candidates[0].content.parts) == 0
        ):
            raise ValueError("No response content from Gemini LLM")
        
        candidate = response.candidates[0]
        
        # Check finish reason first
        finish_reason = getattr(candidate, 'finish_reason', None)
        finish_reason_str = str(finish_reason) if finish_reason else "UNKNOWN"
        logger.info(f"Gemini finish_reason: {finish_reason_str}")
        
        # Check safety ratings
        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
            blocked_categories = []
            for rating in candidate.safety_ratings:
                if hasattr(rating, 'category') and hasattr(rating, 'probability'):
                    prob = str(rating.probability).upper()
                    if prob in ['HIGH', 'MEDIUM']:
                        blocked_categories.append(f"{rating.category}: {prob}")
            
            if blocked_categories:
                logger.warning(f"Response may be affected by safety filters: {', '.join(blocked_categories)}")
        
        # Check for truncation issues
        if "MAX_TOKENS" in finish_reason_str:
            logger.warning(f"Response truncated due to MAX_TOKENS. Consider increasing max_output_tokens (current: {settings.llm_max_tokens})")
        elif "SAFETY" in finish_reason_str or "RECITATION" in finish_reason_str:
            logger.warning(f"Response blocked by filters: {finish_reason_str}")
            # Don't raise error - try to use partial response if available
        
        # Collect all text parts (in case there are multiple)
        response_text_parts = []
        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                response_text_parts.append(part.text)
        
        if not response_text_parts:
            raise ValueError("No text content in Gemini response parts")
        
        response_text = " ".join(response_text_parts)
        
        # Log response length and preview for debugging
        logger.info(f"Gemini response length: {len(response_text)} characters, {len(response_text.split())} words")
        logger.info(f"Gemini full response text (first 1000 chars): {response_text[:1000]}..." if len(response_text) > 1000 else f"Gemini full response text: {response_text}")
        
        # Check if response seems truncated
        if len(response_text) < 100:
            logger.warning(f"Response seems unusually short ({len(response_text)} chars). Finish reason: {finish_reason_str}")
            logger.warning(f"Full response text: {response_text}")
        
        # Parse response (JSON from Gemini)
        parsed = finalize_llm_reply(
            parse_gemini_response(response_text),
            user_message,
            reply_language,
            translation_language,
        )
        
        # Cache LLM output only; translated display text is cached separately by translation backend.
        set_json(
            cache_key,
            {
                **parsed,
                "translated_reply_text": None,
            },
            settings.llm_cache_ttl,
        )
        return _build_response_with_translation(parsed, reply_language, translation_language), int(output_tokens or 0)
        
    except ValueError as e:
        logger.error(f"Gemini LLM validation error: {e}")
        return _build_response_with_translation(
            finalize_llm_reply(
                {
                    "reply_text": "I'm having trouble responding right now. Please try again in a moment.",
                    "translated_reply_text": None,
                    "correction": "",
                    "explanation": "",
                    "example": "",
                    "score": 0,
                },
                user_message,
                reply_language,
                translation_language,
            ),
            reply_language,
            translation_language,
        ), 0
    except Exception as e:
        logger.error(f"Unexpected Gemini LLM error: {e}", exc_info=True)
        return _build_response_with_translation(
            finalize_llm_reply(
                {
                    "reply_text": _user_facing_gemini_error_message(e),
                    "translated_reply_text": None,
                    "correction": "",
                    "explanation": "",
                    "example": "",
                    "score": 0,
                },
                user_message,
                reply_language,
                translation_language,
            ),
            reply_language,
            translation_language,
        ), 0


def generate_reply(
    user_message: str,
    conversation_history: List[Dict[str, str]] = None,
    reply_language: str = "en",
    translation_language: Optional[str] = None,
    long_term_context: Optional[str] = None,
) -> Dict[str, any]:
    """Backward-compatible wrapper that returns only the response payload."""
    result, _ = generate_reply_with_usage(
        user_message,
        conversation_history,
        reply_language,
        translation_language,
        long_term_context,
    )
    return result
