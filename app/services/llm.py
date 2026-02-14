"""LLM service for generating AI replies using Gemini."""
import hashlib
import json
import logging
from typing import Iterator, List, Dict, Optional
from google import genai
from google.genai import types
from app.core.config import settings
from app.core.prompts import get_system_instruction, prepare_history, parse_gemini_response
from app.services.cache import get_json, set_json

logger = logging.getLogger(__name__)

# Initialize Gemini client (lazy loaded)
_gemini_client = None


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
    response_language: str = "en",
) -> str:
    """Deterministic cache key: json.dumps with sort_keys so key is stable across runs."""
    context = user_message + json.dumps(conversation_history, sort_keys=True) + response_language
    return f"llm:{hashlib.md5(context.encode()).hexdigest()}"


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


def stream_gemini_tokens(
    user_message: str,
    conversation_history: List[Dict[str, str]],
    response_language: str = "en",
    long_term_context: Optional[str] = None,
) -> Iterator[str]:
    """
    Stream Gemini response as text deltas (tokens). No caching.
    No JSON mode for stream; caller parses full reply with parse_gemini_response (tries JSON then fallback).
    """
    if conversation_history is None:
        conversation_history = []
    try:
        client = _get_gemini_client()
        system_instruction = get_system_instruction(response_language, long_term_context)
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
            "max_output_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
        }
        if safety_settings is not None:
            config_dict["safety_settings"] = safety_settings
        config = types.GenerateContentConfig(**config_dict)
        logger.info("Starting Gemini stream (model=%s)", settings.llm_model)
        for chunk in client.models.generate_content_stream(
            model=settings.llm_model,
            contents=contents,
            config=config,
        ):
            text = getattr(chunk, "text", None)
            if text and isinstance(text, str) and text.strip():
                yield text
            elif chunk.candidates and len(chunk.candidates) > 0:
                c = chunk.candidates[0]
                if c.content and c.content.parts:
                    for part in c.content.parts:
                        if hasattr(part, "text") and part.text:
                            yield part.text
    except Exception as e:
        logger.exception("Gemini stream error: %s", e)
        raise


def generate_reply(
    user_message: str,
    conversation_history: List[Dict[str, str]] = None,
    response_language: str = "en",
    long_term_context: Optional[str] = None,
) -> Dict[str, any]:
    """
    Generate AI reply with correction and explanation using Gemini.

    Args:
        user_message: User's message
        conversation_history: Previous messages for context (LLM layer trims by token budget)
        response_language: "en" for English-only, or "hi"/"ml"/"ta"/etc. for full Indic response
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
        cache_key = _generate_cache_key(user_message, trimmed_history, response_language)
        if long_term_context:
            cache_key = cache_key + ":" + hashlib.md5(long_term_context.encode()).hexdigest()
        cached_response = get_json(cache_key)
        if cached_response:
            logger.info("Cache hit for LLM response")
            return cached_response

        system_instruction = get_system_instruction(response_language, long_term_context)

        logger.info(f"Calling Gemini LLM API with model: {settings.llm_model}")

        safety_settings = _build_safety_settings()
        if safety_settings is not None:
            logger.debug("Safety settings configured to block only HIGH probability content")

        config_dict = {
            "thinking_config": genai.types.ThinkingConfig(thinking_budget=0),
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
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
        parsed = parse_gemini_response(response_text)
        
        # Cache result
        set_json(cache_key, parsed, settings.llm_cache_ttl)
        return parsed
        
    except ValueError as e:
        logger.error(f"Gemini LLM validation error: {e}")
        return {
            "reply_text": "I'm having trouble responding right now. Please try again in a moment.",
            "correction": "",
            "score": 0
        }
    except Exception as e:
        logger.error(f"Unexpected Gemini LLM error: {e}", exc_info=True)
        return {
            "reply_text": "Something went wrong. Please try again.",
            "correction": "",
            "score": 0
        }
