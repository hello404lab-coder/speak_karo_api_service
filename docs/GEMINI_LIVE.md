# Gemini Live (control plane)

The backend does **not** proxy audio. The client opens a **WebSocket directly to Google** for Gemini Live. This API provides authentication gates, **safe configuration** (no API keys in responses), **ephemeral auth tokens**, and **session + usage** bookkeeping.

For client apps (sequence, errors, multi-tab): **[FRONTEND_GEMINI_LIVE_INTEGRATION.md](./FRONTEND_GEMINI_LIVE_INTEGRATION.md)**.

**Ephemeral token constraints:** `LiveConnectConfig` minted with `/live/token` enables **input** and **output** audio transcription (empty `AudioTranscriptionConfig` objects). Clients should handle the corresponding Live stream messages per [Google’s Live API docs](https://ai.google.dev/gemini-api/docs/live).

## Endpoints (prefix `/api/v1/ai/live`)

All routes require `Authorization: Bearer <access_token>` unless noted.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/config` | Returns `model`, `system_instruction`, `prompt_version`, voice/language hints, `temperature`, `response_modalities`. Optional query: `conversation_id` (must belong to the user) to inject `long_term_context` into the system instruction. |
| GET | `/session/active` | Returns whether the user has an **open** Live session (`active`, `session_id`, `started_at`, `last_seen_at`, `conversation_id` when true). |
| POST | `/session/start` | Body: optional `conversation_id`, `client_platform`. **One active session per user:** if an open session exists, returns the same `session_id` with `reused_existing: true`; otherwise creates a row. Sets `last_seen_at` on create. |
| POST | `/session/heartbeat` | Body: `session_id`. Updates `last_seen_at` for an open session. Call every **20–30 seconds** while connected so the reaper does not auto-end the session. |
| POST | `/session/end` | Body: `session_id`. Closes the session and updates **Usage** once. **Idempotent:** if the session is already ended, returns **200** with the stored `duration_seconds` and `ended_at` (no second usage charge). |
| POST | `/token` | Body: optional `session_id`. **Redis rate limit** per user (see env). If `session_id` is set, rejects **410** when the session is ended or heartbeat-**stale**. Mints a short-lived Live **auth token** via `google-genai`. Requires `GEMINI_API_KEY` on the server. |

## Environment variables

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Required for `/live/token` (server-side only). |
| `GEMINI_LIVE_MODEL` | Live-capable model id (default in `config.py`). |
| `GEMINI_LIVE_PROMPT_VERSION` | Prompt template version string (echoed in `/config`). |
| `GEMINI_LIVE_VOICE` | Prebuilt voice name for `speech_config`. |
| `GEMINI_LIVE_LANGUAGE_CODE` | BCP-47 hint returned in `/config` for clients; omitted from Live `speech_config` (unsupported on token constraints). |
| `GEMINI_LIVE_TEMPERATURE` | Live generation temperature. |
| `GEMINI_LIVE_MIN_PLAN` | `free`, `trial`, or `premium` minimum tier for Live. |
| `GEMINI_LIVE_MAX_SESSION_DURATION_SECONDS` | Cap on billed session length when calling `/session/end`. |
| `GEMINI_LIVE_SYSTEM_INSTRUCTION_MAX_CHARS` | Max length of built system instruction. |
| `GEMINI_LIVE_TOKEN_USES` | Ephemeral token `uses` (see Google SDK; `0` may mean unlimited). |
| `GEMINI_LIVE_TOKEN_NEW_SESSION_SECONDS` | Window for starting new Live sessions with that token; also returned in `/token` response. |
| `GEMINI_LIVE_HEARTBEAT_STALE_SECONDS` | No heartbeat for this long ⇒ session treated as stale for **token** mint and for the **reaper** auto-close. |
| `GEMINI_LIVE_REAPER_ENABLED` | When true, the app process runs a periodic task to auto-end stale open sessions and finalize usage. |
| `GEMINI_LIVE_REAPER_INTERVAL_SECONDS` | Sleep between reaper runs (minimum 15s enforced in code). |
| `GEMINI_LIVE_TOKEN_REQUESTS_PER_MINUTE` | Max `/live/token` calls per user per rolling Redis window. |
| `GEMINI_LIVE_TOKEN_RATE_WINDOW_SECONDS` | Redis TTL for the token counter (default 60). |
| `GEMINI_LIVE_REDIS_REQUIRED_FOR_TOKEN` | In **prod**, if true and Redis is down, `/live/token` returns **503** (fail closed). When Redis is absent in dev, rate limiting is skipped. |

## Client flow (recommended)

1. **GET** `/api/v1/ai/live/config` — UI / logging; same instruction the server will lock into the token when using `/token` without `session_id`, or align with conversation context.
2. **POST** `/api/v1/ai/live/session/start` — receive `session_id` (or reuse an existing open session via `reused_existing`).
3. While connected, **POST** `/api/v1/ai/live/session/heartbeat` every **20–30s** with the same `session_id`.
4. **POST** `/api/v1/ai/live/token` with `{ "session_id": "<id>" }` — receive `auth_token`. On **429**, body includes `type: RATE_LIMITED`.
5. Open the **Gemini Live WebSocket** from the client using Google’s documented URL and the returned `auth_token` (not the raw API key).
6. On disconnect, **POST** `/api/v1/ai/live/session/end` with `session_id` (safe to retry: idempotent).

If the client omits `session_id` on `/token`, the token still uses the same system instruction rules as `/config` (no conversation-bound context unless you pass `conversation_id` on `/config` only — for token without session, use `/config` query or extend the API later).

## Database

Table `live_sessions`: base migration `g1h2i3j4k5l6`; follow-up `h2i3j4k5l6m7` adds **`last_seen_at`** and **`status`** (`active` \| `ended` \| `auto_ended`).

## Security

- Never send `GEMINI_API_KEY` to the client.
- `/token` returns only the ephemeral **auth token name** from Google’s API.
- All routes use the same onboarding and subscription checks as the rest of `/api/v1/ai`.
- Token minting is **rate-limited per user** in Redis (`live:token_rate:{user_id}`).

## Production notes

- **Multiple app processes:** Each uvicorn/gunicorn worker runs its own in-process reaper loop. That is usually fine (same DB, idempotent closes). If you prefer a single scheduler, set **`GEMINI_LIVE_REAPER_ENABLED=false`** on API workers and run reaping from one job (cron) calling the same `reap_stale_live_sessions` logic via a small script or internal route.
- **Horizontal scale:** Token rate limiting uses Redis, so it stays **per user** across workers.

## Future work

Post-call analytics, moderation hooks, transcript storage, and moving the reaper to an external worker if you scale beyond one app process.
