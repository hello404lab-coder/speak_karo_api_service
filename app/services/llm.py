"""LLM service for generating AI replies using Gemini."""
import hashlib
import json
import logging
from typing import Iterator, List, Dict, Optional
from google import genai
from google.genai import types
from app.core.config import settings
from app.core.prompts import build_conversation_prompt, parse_llm_response
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


def _convert_messages_to_gemini_format(messages: List[Dict[str, str]]) -> List[types.Content]:
    """
    Convert chat messages to Gemini format.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
    
    Returns:
        List of Gemini Content objects
    """
    gemini_contents = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Gemini uses "user" and "model" roles (not "system" or "assistant")
        if role == "system":
            # System messages are handled via system_instruction in Gemini
            # For now, we'll prepend it to the first user message
            continue
        elif role == "assistant":
            gemini_role = "model"
        else:
            gemini_role = "user"
        
        gemini_contents.append(
            types.Content(
                role=gemini_role,
                parts=[types.Part.from_text(text=content)]
            )
        )
    
    return gemini_contents


def _generate_cache_key(
    user_message: str,
    conversation_history: List[Dict[str, str]],
    response_language: str = "en",
) -> str:
    """Deterministic cache key: json.dumps with sort_keys so key is stable across runs."""
    context = user_message + json.dumps(conversation_history, sort_keys=True) + response_language
    return f"llm:{hashlib.md5(context.encode()).hexdigest()}"


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
) -> Iterator[str]:
    """
    Stream Gemini response as text deltas (tokens). No caching.
    Caller is responsible for sentence buffering and TTS coordination.
    """
    if conversation_history is None:
        conversation_history = []
    try:
        client = _get_gemini_client()
        messages = build_conversation_prompt(user_message, conversation_history, response_language)
        system_instruction = None
        if messages and messages[0].get("role") == "system":
            system_instruction = messages[0].get("content", "")
            messages = messages[1:]
        contents = _convert_messages_to_gemini_format(messages)
        safety_settings = _build_safety_settings()
        config_dict = {
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
) -> Dict[str, any]:
    """
    Generate AI reply with correction and explanation using Gemini.

    Args:
        user_message: User's message
        conversation_history: Previous messages for context
        response_language: "en" for English-only, or "hi"/"ml"/"ta"/etc. for full Indic response

    Returns:
        Dict with reply_text, correction, hinglish_explanation, score
    """
    if conversation_history is None:
        conversation_history = []

    # Check cache (include response_language so en vs Indic responses don't collide)
    cache_key = _generate_cache_key(user_message, conversation_history, response_language)
    cached_response = get_json(cache_key)
    if cached_response:
        logger.info("Cache hit for LLM response")
        return cached_response
    
    try:
        # Get Gemini client
        client = _get_gemini_client()
        
        # Build messages (prompt depends on response_language: English vs Indic)
        messages = build_conversation_prompt(user_message, conversation_history, response_language)
        
        # Extract system prompt (first message if role is "system")
        system_instruction = None
        if messages and messages[0].get("role") == "system":
            system_instruction = messages[0].get("content", "")
            messages = messages[1:]  # Remove system message from conversation
        
        # Convert to Gemini format
        contents = _convert_messages_to_gemini_format(messages)
        
        logger.info(f"Calling Gemini LLM API with model: {settings.llm_model}")
        
        safety_settings = _build_safety_settings()
        if safety_settings is not None:
            logger.debug("Safety settings configured to block only HIGH probability content")
        
        # Build config with optional safety settings
        config_dict = {
            "system_instruction": system_instruction,
            "max_output_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
        }
        if safety_settings is not None:
            config_dict["safety_settings"] = safety_settings
        
        # Call Gemini API
        logger.debug(f"Calling Gemini with {len(contents)} content items, system_instruction length: {len(system_instruction) if system_instruction else 0}")
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
        
        # Parse response
        parsed = parse_llm_response(response_text)
        
        # Cache result
        set_json(cache_key, parsed, settings.llm_cache_ttl)
        return parsed
        
    except ValueError as e:
        logger.error(f"Gemini LLM validation error: {e}")
        return {
            "reply_text": "I'm having trouble responding right now. Please try again in a moment.",
            "correction": "",
            "hinglish_explanation": "",
            "score": 0
        }
    except Exception as e:
        logger.error(f"Unexpected Gemini LLM error: {e}", exc_info=True)
        return {
            "reply_text": "Something went wrong. Please try again.",
            "correction": "",
            "hinglish_explanation": "",
            "score": 0
        }
