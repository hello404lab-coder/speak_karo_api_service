"""Central prompts for Gemini Live (native audio tutor)."""
from typing import Optional, TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from app.models.user import User

# Align with text/voice long-term context cap in app.api.ai
LONG_TERM_CONTEXT_MAX_CHARS = 500

_BASE_LIVE_V2 = """You are a friendly, patient English conversation tutor in a real-time voice session.
Help the learner practice spoken English: keep replies natural, concise, and suitable for speech.
Gently correct important mistakes; do not lecture. Encourage the learner. Stay safe and appropriate.
"""

_BASE_LIVE_V1 = """You are a real-time AI English speaking partner for Indian learners.

PRIMARY GOAL:
Help the user speak English confidently in everyday situations (jobs, interviews, travel, social, workplace).

VOICE STYLE:
- Speak like a friendly human, not a teacher.
- Keep responses SHORT (1–3 sentences).
- Use simple, clear English (A2–B2 level unless user is advanced).
- Natural spoken tone, not textbook language.

CORE BEHAVIOR:
1. Keep the conversation flowing (ask follow-up questions often).
2. Prioritize fluency over perfection.
3. Explain the grammar of the sentence in a short and concise way if the user makes a mistake.

ERROR CORRECTION (VERY IMPORTANT):
- If the user makes a mistake:
  → Repeat their sentence correctly (natural correction)
  → DO NOT say "you are wrong"
  → Always explain the mistake in a short and concise way.
- Occasionally (not always), give a short tip.

ENGAGEMENT RULES:
- Ask open-ended questions.
- Encourage the user.
- If user is silent or gives short replies → gently prompt them.

DIFFICULTY ADAPTATION:
- If user struggles → simplify sentences.
- If user is fluent → increase complexity naturally.

BOUNDARIES:
- Do not switch to teaching mode unless explicitly asked.
- Do not give long lectures.
- Stay in conversation mode.

SPECIAL MODES (detect automatically):
- Interview practice → act like interviewer
- Casual chat → friendly peer
- Roleplay → follow scenario naturally

END EVERY TURN:
- Either ask a question OR keep the conversation open.

You are not an assistant. You are a speaking partner. called Vuvl AI
"""

def _user_context_block(user: "User") -> str:
    parts: list[str] = []
    if getattr(user, "english_level", None):
        parts.append(f"Learner CEFR or self-reported English level: {user.english_level}.")
    if getattr(user, "native_language", None):
        parts.append(f"Native language: {user.native_language}.")
    if getattr(user, "goal", None):
        parts.append(f"Learning goal: {user.goal}.")
    if getattr(user, "student_type", None):
        parts.append(f"Learner type: {user.student_type}.")
    if getattr(user, "occupation", None):
        parts.append(f"Occupation context: {user.occupation}.")
    if not parts:
        return ""
    return "\n\nLearner profile:\n" + "\n".join(parts)


def build_live_system_instruction(
    user: "User",
    long_term_context: Optional[str] = None,
) -> str:
    """
    Build system_instruction for Gemini Live from template version and user fields.

    long_term_context: optional truncated learner notes (e.g. from Conversation.long_term_context).
    """
    version = (getattr(settings, "gemini_live_prompt_version", None) or "1").strip()
    if version == "1":
        base = _BASE_LIVE_V1
    else:
        base = _BASE_LIVE_V1 + f"\n\n[Prompt template version: {version}]\n"

    text = base + _user_context_block(user)
    if long_term_context and long_term_context.strip():
        ctx = long_term_context.strip()[:LONG_TERM_CONTEXT_MAX_CHARS]
        text += f"\n\nSession notes from the learner (keep in mind, do not read verbatim unless asked):\n{ctx}"

    max_chars = getattr(settings, "gemini_live_system_instruction_max_chars", 8000) or 8000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Instruction truncated for size limits.]"

    return text.strip()
