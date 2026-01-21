"""Prompt templates for LLM interactions."""
from typing import List, Dict


SYSTEM_PROMPT = """You are a friendly English speaking partner helping an Indian learner practice English.

IMPORTANT RULES:
1. Be warm, encouraging, and never shame the learner
2. Use simple vocabulary and speak slowly
3. Correct ONLY ONE mistake per response (the most important one)
4. Provide explanations in Hinglish (mix of Hindi and English)
5. Keep responses natural and conversational
6. Be patient and supportive

Your response format must be:
1. A natural, friendly reply to what they said
2. A gentle correction of ONE mistake (if any)
3. A Hinglish explanation of why the correction matters
4. One improved sentence showing the correct way

Keep your total response under 150 tokens. Be concise but helpful."""


def build_conversation_prompt(user_message: str, conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Build conversation messages for LLM API.
    
    Args:
        user_message: Current user message
        conversation_history: List of previous messages with 'role' and 'content'
    
    Returns:
        List of message dicts for OpenAI API
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
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
            
        # Try to detect sections
        if any(keyword in line.lower() for keyword in ["correction", "mistake", "wrong"]):
            current_section = "correction"
        elif any(keyword in line.lower() for keyword in ["hinglish", "explanation", "matlab", "ka matlab"]):
            current_section = "explanation"
        elif any(keyword in line.lower() for keyword in ["score", "rating"]):
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
        elif current_section == "correction":
            correction += line + " "
        elif current_section == "explanation":
            hinglish_explanation += line + " "
    
    # If no structured parsing worked, use entire response as reply
    if not reply_text:
        reply_text = response_text
    
    return {
        "reply_text": reply_text.strip(),
        "correction": correction.strip(),
        "hinglish_explanation": hinglish_explanation.strip(),
        "score": score
    }
