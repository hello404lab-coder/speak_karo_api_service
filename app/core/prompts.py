"""Prompt templates for LLM interactions."""
import re
from typing import List, Dict

# English-only: user spoke English only -> respond only in English, Chatterbox TTS
SYSTEM_PROMPT_ENGLISH = """You are a warm English speaking buddy on a voice call with an Indian learner.

RULES:
1. Respond ONLY in English. No Hindi or other Indian languages.
2. Be brief: 2-4 short sentences. Under 50 words.
3. If they made a mistake, give one quick correction and move on. No labels like "Correction:".
4. Speak naturally. End with a short question to keep them talking.
5. Avoid filler like "Dont worry" or "Keep practicing" unless they seem discouraged."""

# Indic-only: user spoke Hindi/Malayalam/etc. -> respond entirely in that language, IndicF5 TTS
INDIC_PROMPT_TEMPLATE = """You are a warm speaking buddy on a voice call with an Indian learner.

RULES:
1. Respond ENTIRELY in {language_name}. No English mixing. The learner will hear this in their language via TTS.
2. Be brief: 2-4 short sentences. Under 50 words.
3. If they made a mistake, give one quick correction in {language_name} and move on. No labels.
4. Speak naturally. End with a short question in {language_name} to keep them talking.
5. Avoid filler unless they seem discouraged."""

# Language code to display name for prompt
INDIC_LANG_NAMES = {
    "hi": "Hindi",
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "bn": "Bengali",
}

# Legacy single prompt (used if response_language not passed)
SYSTEM_PROMPT = SYSTEM_PROMPT_ENGLISH


def build_conversation_prompt(
    user_message: str,
    conversation_history: List[Dict[str, str]],
    response_language: str = "en",
) -> List[Dict[str, str]]:
    """
    Build conversation messages for LLM API.

    Args:
        user_message: Current user message
        conversation_history: List of previous messages with 'role' and 'content'
        response_language: "en" for English-only response, or "hi"/"ml"/"ta"/etc. for full Indic

    Returns:
        List of message dicts for LLM API
    """
    if response_language == "en":
        system_content = SYSTEM_PROMPT_ENGLISH
    else:
        language_name = INDIC_LANG_NAMES.get(response_language, "Hindi")
        system_content = INDIC_PROMPT_TEMPLATE.format(language_name=language_name)

    messages = [
        {"role": "system", "content": system_content}
    ]
    
    # Add conversation history (last 5 messages for context)
    for msg in conversation_history[-5:]:
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })
    
    # Add current message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    return messages


def parse_llm_response(response_text: str) -> Dict[str, str]:
    """
    Parse LLM response into structured format.
    
    Expected format (flexible parsing):
    - Reply text
    - Correction (if any)
    - Hinglish explanation
    - Score (0-100)
    
    Args:
        response_text: Raw LLM response
    
    Returns:
        Dict with reply_text, correction, hinglish_explanation, score
    """
    # Simple parsing - LLM should follow format, but we'll be flexible
    lines = response_text.strip().split('\n')
    
    reply_text = ""
    correction = ""
    hinglish_explanation = ""
    score = 75  # Default score
    
    current_section = "reply"
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Clean markdown formatting
        line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)  # Remove **bold**
        line = re.sub(r'\*([^*]+)\*', r'\1', line)  # Remove *italic*
        line = line.strip()
        
        # Try to detect sections - look for explicit headers (but remove them)
        line_lower = line.lower()
        if line_lower.startswith("**correction:**") or line_lower.startswith("correction:") or line_lower.startswith("correction"):
            current_section = "correction"
            # Remove the header from the line
            line = re.sub(r'^(\*\*)?correction(\*\*)?:?\s*', '', line, flags=re.IGNORECASE).strip()
        elif line_lower.startswith("**hinglish explanation:**") or line_lower.startswith("hinglish explanation:") or line_lower.startswith("**hinglish:**") or line_lower.startswith("hinglish:") or line_lower.startswith("hinglish explanation"):
            current_section = "explanation"
            # Remove the header from the line
            line = re.sub(r'^(\*\*)?(hinglish\s+)?explanation(\*\*)?:?\s*', '', line, flags=re.IGNORECASE).strip()
        elif line_lower.startswith("**improved sentence:**") or line_lower.startswith("improved sentence:") or line_lower.startswith("improved sentence"):
            # Skip improved sentence section for now
            continue
        elif line_lower.startswith("**reply:**") or line_lower.startswith("reply:") or line_lower.startswith("reply"):
            current_section = "reply"
            # Remove the header from the line
            line = re.sub(r'^(\*\*)?reply(\*\*)?:?\s*', '', line, flags=re.IGNORECASE).strip()
        elif any(keyword in line_lower for keyword in ["correction", "mistake", "wrong"]) and current_section == "reply" and len(line) > 20:
            current_section = "correction"
        elif any(keyword in line_lower for keyword in ["hinglish", "explanation", "matlab", "ka matlab"]) and current_section != "explanation" and len(line) > 20:
            current_section = "explanation"
        elif any(keyword in line_lower for keyword in ["score", "rating"]):
            # Try to extract score
            try:
                score_str = ''.join(filter(str.isdigit, line))
                if score_str:
                    score = min(100, max(0, int(score_str)))
            except:
                pass
        
        # Accumulate content
        if current_section == "reply":
            reply_text += line + " "
        # elif current_section == "correction":
        #     correction += line + " "
        # elif current_section == "explanation":
        #     hinglish_explanation += line + " "
    
    # If no structured parsing worked, use entire response as reply
    if not reply_text:
        reply_text = response_text
    
    # If we got reply but no explanation, try to extract it from the response
    if reply_text and not hinglish_explanation:
        # Look for Hindi text (Devanagari script) in the response
        # Check if there's Hindi text anywhere in the response
        hindi_pattern = r'[\u0900-\u097F]+'
        if re.search(hindi_pattern, response_text):
            # If we found Hindi but no explanation was parsed, use the part with Hindi
            # Try to find the explanation part
            parts = re.split(r'(?i)(hinglish|explanation|matlab)', response_text)
            if len(parts) > 1:
                # Take everything after the keyword
                hinglish_explanation = " ".join(parts[1:]).strip()
    
    # Log parsing results for debugging
    if not hinglish_explanation and reply_text:
        # If still no explanation, check if the reply itself contains Hindi
        hindi_pattern = r'[\u0900-\u097F]+'
        if re.search(hindi_pattern, reply_text):
            # The reply might contain the explanation mixed in
            pass  # Keep it in reply_text for now
    
    # Clean up all text - remove markdown, extra quotes, and formatting
    def clean_text(text: str) -> str:
        if not text:
            return ""
        # Remove markdown bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        # Remove extra quotes around English phrases (but keep quotes that are part of the text)
        text = re.sub(r'"([^"]+)"', r'\1', text)  # Remove quotes
        text = re.sub(r"'([^']+)'", r'\1', text)  # Remove single quotes
        # Remove section headers if they somehow got included
        text = re.sub(r'^(\*\*)?(reply|correction|hinglish|explanation)(\*\*)?:?\s*', '', text, flags=re.IGNORECASE)
        # Clean up extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    return {
        "reply_text": clean_text(reply_text),
        "correction": clean_text(correction),
        "hinglish_explanation": clean_text(hinglish_explanation),
        "score": score
    }
