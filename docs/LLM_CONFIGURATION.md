# LLM Configuration & Prompts

This document describes how the Large Language Model (LLM) is configured in the AI English Practice backend: provider, prompts, system instruction, history handling, response format, caching, and API usage.

---

## Overview

- **Provider:** Google Gemini (via `google-genai` SDK).
- **Primary model:** `gemini-2.5-flash` (configurable via `LLM_MODEL`).
- **Role:** English tutor that corrects grammar/vocabulary and keeps the conversation flowing; responses are structured as JSON for TTS and on-screen correction/score.

---

## 1. Configuration (Environment & Settings)

All LLM-related settings live in `app/core/config.py` and are driven by environment variables (or `.env`).

| Setting | Env / default | Description |
|--------|----------------|-------------|
| **Model** | `LLM_MODEL` / `gemini-2.5-flash` | Gemini model name used for both sync and streaming. |
| **Max output tokens** | `LLM_MAX_TOKENS` / `200` | Maximum tokens per response (keeps replies concise for voice). |
| **Temperature** | `LLM_TEMPERATURE` / `0.2` | Sampling temperature (lower = more deterministic). |
| **Context token budget** | `LLM_CONTEXT_TOKEN_BUDGET` / `16384` | Max input tokens for system + history + current message; oldest messages are dropped to stay under this. |
| **History max exchanges** | `LLM_HISTORY_MAX_EXCHANGES` / `10` | DB layer cap: how many past user/assistant exchanges to load; actual context length is then trimmed by the token budget. |
| **Timeout** | `LLM_TIMEOUT_SECONDS` / `60` | Timeout for sync LLM calls (used with `asyncio.wait_for` in API). |
| **Cache TTL** | `LLM_CACHE_TTL` / `86400` | TTL in seconds (24h) for cached LLM responses in Redis. |
| **API key** | `GEMINI_API_KEY` | Required for Gemini; client init fails if missing. |

**Client initialization** (`app/services/llm.py`):

- Gemini client is **lazy-loaded** on first use.
- HTTP timeout is set from `llm_timeout_seconds` (in milliseconds).
- Used by `/init-models` for warmup and by all chat endpoints.

---

## 2. System Instruction (Prompts)

System instruction is built in `app/core/prompts.py` and passed to Gemini as `system_instruction` on every request.

### 2.1 Main template: `SYSTEM_PROMPT_TEMPLATE`

The model is instructed to act as an **experienced English speaking tutor on a voice call** with these goals:

1. Help the learner speak more naturally.
2. Correct mistakes clearly.
3. Teach one useful improvement at a time.
4. Keep the conversation natural and engaging.

**Language:** Respond **only in English** in `reply_text`. The placeholder `{json_format}` is replaced by the JSON format instruction (see below).

**When the learner makes a mistake:**

- Identify the most important mistake.
- Provide the corrected sentence (in `correction`).
- Give a short explanation (max 20 words) in `explanation`.
- Provide one example sentence in `example`.
- Never correct more than one mistake in a single response.

**When the sentence is correct:** Leave `correction`, `explanation`, and `example` empty and give a higher score.

**Spoken reply (`reply_text`):** Must sound natural, be conversational, be under 50 words, and end with a question. Do not include explanations inside `reply_text`. Be encouraging but avoid excessive praise. Adjust vocabulary difficulty based on learner level.

**Scoring (deductive):** Start from 100. Subtract: -10 for grammar mistake, -10 for vocabulary mistake, -5 for article or preposition mistake, -5 for word order mistake. Minimum score = 0.

### 2.2 JSON format instruction: `JSON_FORMAT_INSTRUCTION`

Injected into the system prompt so the model responds in a fixed JSON shape:

```text
Respond ONLY in valid JSON format with these keys:
{
  "reply_text": "spoken response that continues conversation (this is the ONLY part spoken by TTS)",
  "correction": "correct sentence if mistake exists, otherwise empty string (display only, not spoken)",
  "explanation": "short explanation of the mistake (max 20 words); empty if no mistake (display only)",
  "example": "one example sentence showing correct usage; empty if no mistake (display only)",
  "score": integer from 0 to 100
}
```

### 2.3 Building the full system instruction: `get_system_instruction()`

- **Signature:** `get_system_instruction(response_language: str = "en", long_term_context: Optional[str] = None) -> str`
- **Behavior:**
  - Base = `SYSTEM_PROMPT_TEMPLATE` with `json_format=JSON_FORMAT_INSTRUCTION`.
  - If `long_term_context` is non-empty, it is prepended as:  
    `"Known about this learner: {long_term_context}\n\n{base}"`.
- **Note:** `response_language` is accepted for future use (e.g. Indic-language replies); the template instructs “Respond ONLY in English in 'reply_text'”.

### 2.4 Language names

`LANGUAGE_NAMES` in `prompts.py` maps language codes (e.g. `en`, `hi`, `ml`, `ta`) to display names for use in prompts; unknown codes fall back to “the same language they are speaking”.

---

## 3. Response schema (Pydantic & Gemini)

- **Pydantic model:** `LLMReplySchema` in `app/core/prompts.py`:
  - `reply_text: str` — response to the user (spoken by TTS).
  - `correction: str` — correct English phrase (display only); default `""`.
  - `explanation: str` — short explanation of the mistake (max 20 words); default `""`.
  - `example: str` — one example sentence showing correct usage; default `""`.
  - `score: int` — 0–100; default 70.

- **Gemini structured output:** For **non-streaming** calls, the client is configured with:
  - `response_mime_type="application/json"`
  - `response_json_schema=LLMReplySchema.model_json_schema()`

So the model is asked to return JSON that conforms to this schema. **Streaming** calls do not use JSON mode; the streamed text is later parsed with `parse_gemini_response()`.

- **Persistence:** The API stores the LLM’s `explanation` in the `Message.hinglish_explanation` column and the LLM’s `example` in the new `Message.example` column.

---

## 4. Conversation history

### 4.1 Format: `prepare_history()`

- **Input:** List of `{"role": "user"|"assistant", "content": "..."}`.
- **Output:** List of `{"role": "user"|"model", "parts": [content]}` for Gemini.
- **Mapping:** `assistant` → `model`. No trimming is done here; trimming is done in the LLM layer by token budget.

### 4.2 Trimming: token budget and reserve

- **Token budget:** `llm_context_token_budget` (default 16,384) is the max total input size (system + history + current user message).
- **Reserve:** `SYSTEM_INSTRUCTION_RESERVE_TOKENS = 2048` is reserved for the system instruction when trimming. So the **contents** (history + current message) are trimmed to at most `token_budget - 2048`.
- **Trimming logic** (`_build_trimmed_contents()` in `llm.py`):
  1. Format history with `prepare_history()` and convert to Gemini `Content` list.
  2. Append the current user message.
  3. Estimate or count tokens for contents (Gemini `count_tokens` when available, else ~4 chars per token).
  4. While total contents tokens exceed the budget and there is more than one message, drop the **oldest** message (index 0).
- So the **current user message is always kept**; only older turns are dropped.

### 4.3 DB layer: how much history is loaded

- **Cap:** `llm_history_max_exchanges` (default 10) is the maximum number of **exchanges** (user + assistant pairs) to load from the conversation.
- **Usage:** In `app/api/ai.py`, `get_conversation_history()` loads the last N messages in chronological order; the LLM layer then trims by token budget as above.

---

## 5. Request configuration (Gemini API)

For both streaming and non-streaming generation, the following are set on `GenerateContentConfig`:

| Parameter | Value | Notes |
|-----------|--------|--------|
| `thinking_config` | `ThinkingConfig(thinking_budget=0)` | No “thinking” tokens; keeps latency and cost down. |
| `system_instruction` | From `get_system_instruction(response_language, long_term_context)` | See §2. |
| `max_output_tokens` | `settings.llm_max_tokens` | e.g. 200. |
| `temperature` | `settings.llm_temperature` | e.g. 0.2. |
| `safety_settings` | See below | Optional. |

**Non-streaming only:**

- `response_mime_type="application/json"`
- `response_json_schema=LLMReplySchema.model_json_schema()`

**Safety settings** (`_build_safety_settings()`):

- Built only if the SDK exposes `SafetySetting`, `HarmCategory`, `HarmBlockThreshold`.
- All of the following are set to **block only HIGH**:
  - Harassment  
  - Hate speech  
  - Sexually explicit  
  - Dangerous content  

If the types aren’t available, `safety_settings` is not set.

---

## 6. Response parsing: `parse_gemini_response()`

- **Input:** Raw response text from Gemini (full string for sync, concatenated stream for streaming).
- **Steps:**
  1. Strip optional markdown code fences (e.g. ` ```json ` … ` ``` `).
  2. Try `json.loads()`; normalize keys (`reply_text` or `reply`), read `explanation` and `example` (default `""`), clamp `score` to 0–100, default 70.
  3. If JSON parse fails (e.g. truncated or malformed), fall back to regex-based extraction of `reply_text`, `correction`, `explanation`, `example`, and `score` from the raw string; if the result doesn’t look like JSON, the whole cleaned string can be used as `reply_text`.
- **Output:** Dict with `reply_text`, `correction`, `explanation`, `example`, `score`.

---

## 7. Caching

- **Where:** Only the **non-streaming** path (`generate_reply()`) uses cache; streaming does not.
- **Key:** `_generate_cache_key(user_message, trimmed_history, response_language)` — deterministic (e.g. MD5 of message + sorted JSON of history + language). If `long_term_context` is present, its hash is appended to the key.
- **Storage:** Redis via `app/services/cache.py` (`get_json` / `set_json`), only when `settings.cache_enabled` is True and Redis is available.
- **TTL:** `settings.llm_cache_ttl` (default 86400 seconds).
- **Cache key content:** Uses the **trimmed** history actually sent to the model (so cache matches what the model saw).

---

## 8. API usage

- **Endpoints that call the LLM:**
  - **POST /api/ai/text-chat** — text message → `generate_reply()` (sync, cached).
  - **POST /api/ai/voice-chat** — STT → same `generate_reply()` (sync, cached).
  - **POST /api/ai/chat/stream** — text → `stream_gemini_tokens()` → sentence buffer → TTS → SSE (no LLM cache).
  - **POST /api/ai/voice-chat/stream** — STT → same streaming pipeline (no LLM cache).

- **Common arguments passed to LLM:**
  - `user_message` (or transcribed text).
  - `conversation_history` from `get_conversation_history(conversation.id, db)`.
  - `response_language` from `get_response_language(message_or_text, detected_lang)` (see §9).
  - `long_term_context` = `conversation.long_term_context` (optional; e.g. “preparing for IELTS”), capped at 500 chars when appending from client (`LONG_TERM_CONTEXT_MAX_CHARS`).

- **Timeouts:** Sync LLM calls are run in a thread pool and wrapped with `asyncio.wait_for(..., timeout=settings.llm_timeout_seconds)`.

---

## 9. Response language routing

- **Purpose:** Decide whether the reply (and TTS) is in English or an Indic language (hi, ml, ta, etc.).
- **Function:** `get_response_language(text, detected_lang)` in `app/utils/language.py`.
- **Logic:**
  - If the text contains Indic script (Devanagari, Malayalam, Tamil, Telugu, Kannada, Bengali), use the corresponding language code.
  - Else if STT `detected_lang` is in the Indic map, use the mapped code (e.g. Marathi → hi).
  - Else if `detected_lang == "en"` or no Indic script and no detected Indic language → `"en"`.
  - Otherwise default to `"en"`.
- **Passed into:** `get_system_instruction(response_language, long_term_context)` and used for TTS routing; the current system prompt still instructs “Respond ONLY in English in 'reply_text'”.

---

## 10. Error handling and fallbacks

- **Missing API key:** `_get_gemini_client()` raises `ValueError` if `GEMINI_API_KEY` is not set.
- **No/invalid response content:** If the candidate has no content or no parts, `generate_reply()` raises `ValueError`; caught and returned as a user-safe message (e.g. “I'm having trouble responding right now…”).
- **Finish reason / safety:** Logged (e.g. `MAX_TOKENS`, `SAFETY`, `RECITATION`); on safety/recitation the code still tries to use partial response if available.
- **Parse failures:** `parse_gemini_response()` falls back to regex extraction and, if needed, uses the full cleaned text as `reply_text`.
- **Timeouts:** API returns 504 with a generic “Request took too long” message.

---

## 11. File reference

| Concern | File(s) |
|--------|---------|
| Config (model, tokens, temperature, budget, timeout, cache TTL) | `app/core/config.py` |
| System prompt, JSON instruction, schema, history format, parsing | `app/core/prompts.py` |
| Gemini client, trimming, caching, streaming, `generate_reply()` | `app/services/llm.py` |
| Chat endpoints, history loading, streaming pipeline | `app/api/ai.py` |
| Request/response schemas | `app/schemas/ai.py` |
| Response language for LLM/TTS | `app/utils/language.py` |
| LLM response cache (Redis) | `app/services/cache.py` |

---

## 12. Summary

- **Model:** Gemini (`gemini-2.5-flash` by default), lazy-initialized, with configurable timeout and token limits.
- **Prompt:** Single system instruction: English tutor with goals (natural speech, clear correction, one improvement at a time, engaging conversation); one-mistake-per-response rule; brief spoken reply + on-screen correction, explanation (max 20 words), and example; deductive scoring (start 100, subtract for grammar/vocab/article-preposition/word-order); JSON format and optional long-term learner context.
- **Context:** History is loaded up to `llm_history_max_exchanges`, then trimmed by token budget (with a fixed reserve for system instruction) so the oldest messages are dropped first.
- **Output:** Structured JSON (`reply_text`, `correction`, `explanation`, `example`, `score`) via schema in sync mode; streaming mode streams raw text and parses at the end with the same parser and fallbacks. The API stores `explanation` in `Message.hinglish_explanation` and `example` in `Message.example`.
- **Caching:** Sync path only; key from message + trimmed history + language (+ long-term context hash); Redis TTL 24h by default.
- **Safety:** Block-only-high for standard harm categories when the SDK supports it.
- **API:** Text and voice chat (sync and streaming) pass conversation history, response language, and optional long-term context into the same LLM layer; responses and streaming `metadata` include `explanation` and `example` for the frontend to display.
