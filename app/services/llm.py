"""LLM service for generating AI replies."""
import hashlib
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from openai import APITimeoutError, APIError
from app.core.config import settings
from app.core.prompts import build_conversation_prompt, parse_llm_response
from app.services.cache import get_json, set_json

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=settings.openai_api_key)


def _generate_cache_key(user_message: str, conversation_history: List[Dict[str, str]]) -> str:
    """Generate cache key from message and history."""
    context = user_message + str(conversation_history)
    return f"llm:{hashlib.md5(context.encode()).hexdigest()}"


def generate_reply(
    user_message: str,
    conversation_history: List[Dict[str, str]] = None
) -> Dict[str, any]:
    """
    Generate AI reply with correction and explanation.
    
    Args:
        user_message: User's message
        conversation_history: Previous messages for context
    
    Returns:
        Dict with reply_text, correction, hinglish_explanation, score
    """
    if conversation_history is None:
        conversation_history = []
    
    # Check cache
    cache_key = _generate_cache_key(user_message, conversation_history)
    cached_response = get_json(cache_key)
    if cached_response:
        logger.info("Cache hit for LLM response")
        return cached_response
    
    try:
        # Build messages
        messages = build_conversation_prompt(user_message, conversation_history)
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout
        )
        
        # Extract response text
        response_text = response.choices[0].message.content
        
        # Parse response
        parsed = parse_llm_response(response_text)
        
        # Cache result
        set_json(cache_key, parsed, settings.llm_cache_ttl)
        
        return parsed
        
    except APITimeoutError:
        logger.error("LLM API timeout")
        return {
            "reply_text": "I'm taking a bit longer to respond. Please try again!",
            "correction": "",
            "hinglish_explanation": "",
            "score": 0
        }
    except APIError as e:
        logger.error(f"LLM API error: {e}")
        return {
            "reply_text": "I'm having trouble responding right now. Please try again in a moment.",
            "correction": "",
            "hinglish_explanation": "",
            "score": 0
        }
    except Exception as e:
        logger.error(f"Unexpected LLM error: {e}")
        return {
            "reply_text": "Something went wrong. Please try again.",
            "correction": "",
            "hinglish_explanation": "",
            "score": 0
        }
