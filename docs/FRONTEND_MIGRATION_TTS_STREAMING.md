# Frontend migration: streaming chat and TTS (SSE)

This guide describes how to use the streaming endpoints for lower time-to-first-audio (TTFA) and better perceived performance. Three streaming endpoints are available, from most-integrated to standalone.

---

## Endpoint overview

| Method | Path | Input | SSE events | Use case |
|--------|------|-------|------------|----------|
| POST | `/api/v1/ai/voice-chat/stream` | `multipart/form-data` (audio WAV + form fields) | `stt_result`, `text_chunk`, `audio_chunk`, `metadata`, `done`, `audio_ready` | **Full pipeline:** audio in, streamed audio+text out |
| POST | `/api/v1/ai/chat/stream` | JSON `{ user_id, message, conversation_id?, learner_context? }` | `text_chunk`, `audio_chunk`, `metadata`, `done`, `audio_ready` | **Text pipeline:** text in, streamed audio+text out |
| POST | `/api/v1/ai/tts/stream` | JSON `{ text, response_language? }` | `audio_chunk`, `done`, `audio_ready` | **TTS only:** text in, streamed audio out |

---

## 1. Voice chat streaming (recommended for voice apps)

**`POST /api/v1/ai/voice-chat/stream`** — single SSE connection does everything: STT (Groq Whisper) → LLM streaming (Gemini) → sentence-level TTS (Gemini TTS).

### Request (multipart/form-data)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio_file` | File (WAV) | Yes | Audio recording |
| `user_id` | string | Yes | User identifier |
| `conversation_id` | string | No | Existing conversation ID |
| `learner_context` | string | No | Long-term learner context |

### SSE event flow

```
event: stt_result
data: {"text": "I want to practice ordering food", "detected_lang": "en", "response_language": "en"}

event: text_chunk
data: {"text": "That's a great topic!"}

event: audio_chunk
data: <base64 WAV for "That's a great topic!">

event: text_chunk
data: {"text": "Let's pretend you're at a restaurant."}

event: audio_chunk
data: <base64 WAV for "Let's pretend you're at a restaurant.">

event: text_chunk
data: {"text": "What would you like to order?"}

event: audio_chunk
data: <base64 WAV for "What would you like to order?">

event: metadata
data: {"correction": "", "explanation": "", "example": "", "score": 85, "conversation_id": "uuid-..."}

event: done
data: {"audio_url": null, "saving_in_background": true}

event: audio_ready
data: {"audio_url": "https://...full-mp3-url..."}
```

### JavaScript example

```javascript
const formData = new FormData();
formData.append('audio_file', audioBlob, 'recording.wav');
formData.append('user_id', userId);
formData.append('conversation_id', conversationId); // optional

const response = await fetch('/api/v1/ai/voice-chat/stream', {
  method: 'POST',
  body: formData,
});

await consumeStreamingChat(response, {
  onSttResult: (data) => {
    // Show transcribed text: data.text, data.detected_lang, data.response_language
    showUserMessage(data.text);
  },
  onTextChunk: (data) => {
    // Append sentence to reply bubble: data.text
    appendToReply(data.text);
  },
  onAudioChunk: (base64Wav) => {
    // Decode and play immediately
    queueChunk(base64Wav);
  },
  onMetadata: (data) => {
    // Show correction, explanation, example, score; save conversation_id
    showCorrection(data.correction);
    if (data.explanation) showExplanation(data.explanation);
    if (data.example) showExample(data.example);
    showScore(data.score);
    conversationId = data.conversation_id;
  },
  onDone: (data) => {
    // Stream finished
  },
  onAudioReady: (data) => {
    // Full MP3 URL for replay: data.audio_url
    setReplayUrl(data.audio_url);
  },
  onError: (msg) => {
    showError(msg);
  },
});
```

---

## 2. Text chat streaming

**`POST /api/v1/ai/chat/stream`** — same pipeline as voice-chat/stream but takes text input (no STT step, no `stt_result` event).

### Request (JSON)

```json
{
  "user_id": "user-123",
  "message": "I want to practice ordering food",
  "conversation_id": "uuid-...",
  "learner_context": "preparing for IELTS"
}
```

### SSE event flow

Same as voice-chat/stream minus the `stt_result` event:

```
event: text_chunk → event: audio_chunk → ... → event: metadata → event: done → event: audio_ready
```

---

## 3. TTS-only streaming (standalone)

**`POST /api/v1/ai/tts/stream`** — takes text, returns streamed audio chunks. No LLM, no DB save, no metadata.

### Request (JSON)

```json
{
  "text": "Sentence one. Sentence two.",
  "response_language": "en"
}
```

### SSE events

| Event | Data | Meaning |
|-------|------|---------|
| `audio_chunk` | Base64 WAV string | One sentence of audio |
| `done` | `{"audio_url": null, "saving_in_background": true}` | Stream finished |
| `audio_ready` | `{"audio_url": "..."}` | Full MP3 stitched and saved |
| `error` | `{"error": "message"}` | Failure |

---

## SSE event reference (all endpoints)

| Event | Data format | Present in |
|-------|-------------|------------|
| `stt_result` | `{"text": "...", "detected_lang": "en", "response_language": "en"}` | voice-chat/stream only |
| `text_chunk` | `{"text": "One sentence of reply."}` | chat/stream, voice-chat/stream |
| `audio_chunk` | Base64-encoded WAV (raw string, no JSON) | All streaming endpoints |
| `metadata` | `{"correction": "...", "explanation": "...", "example": "...", "score": 85, "conversation_id": "uuid"}` | chat/stream, voice-chat/stream |
| `done` | `{"audio_url": null, "saving_in_background": true}` or `{"audio_url": null}` | All |
| `audio_ready` | `{"audio_url": "https://..."}` | All |
| `error` | `{"error": "message"}` | All |
| `: keep-alive` | (comment, no data) | chat/stream, voice-chat/stream |

**Event order:** `stt_result` (voice only) → interleaved `text_chunk` / `audio_chunk` → `metadata` → `done` → `audio_ready`

---

## Non-streaming endpoints (still available)

**`POST /api/v1/ai/text-chat`** and **`POST /api/v1/ai/voice-chat`** return JSON with `reply_text`, `correction`, `explanation`, `example`, `score`, `conversation_id`. **`audio_url` is always `null`** — use a streaming endpoint for audio. The `metadata` SSE event in streaming endpoints includes the same fields (`correction`, `explanation`, `example`, `score`, `conversation_id`) for the frontend to display.

---

## Conversation and chat history APIs

Use these endpoints to build a **chat window with sidebar** (conversation list, history, pagination). All require authentication (Bearer token) and an active subscription.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/conversations` | List conversations for the user (ordered by `updated_at` DESC) |
| GET | `/api/v1/conversations/{conversation_id}` | Get a single conversation (title, created_at, updated_at) |
| GET | `/api/v1/conversations/{conversation_id}/messages` | Paginated chat history (cursor-based) |
| POST | `/api/v1/conversations/{conversation_id}/title` | Generate and save an AI title from first messages |

### GET /api/v1/conversations

**Query:** `limit` (optional, default 20, max 100).

**Response:**

```json
{
  "conversations": [
    {
      "id": "uuid",
      "title": "Talking about hobbies",
      "last_message": "What kind of typing do you do?",
      "updated_at": "2026-03-10T12:00:00"
    }
  ]
}
```

- `title` and `last_message` may be `null` (e.g. new conversation or no title generated yet).

### GET /api/v1/conversations/{conversation_id}

**Response:** `{ "id", "title", "created_at", "updated_at" }`. Returns 404 if the conversation does not exist or does not belong to the authenticated user.

### GET /api/v1/conversations/{conversation_id}/messages

**Query:**

- `limit` (optional, default 20, max 50) — number of **chat messages** to return (each DB exchange yields 2 messages: user + assistant).
- `cursor` (optional) — opaque string from the previous response’s `next_cursor` to load **older** messages (infinite scroll).

**Response:**

```json
{
  "messages": [
    {
      "index": 0,
      "id": "a1b2c3d4-e5f6-5789-a0b1-c2d3e4f56789",
      "role": "user",
      "content": "Mostly I do homework and type voting.",
      "created_at": "..."
    },
    {
      "index": 1,
      "id": "b2c3d4e5-f6a7-6890-b1c2-d3e4f5678901",
      "role": "assistant",
      "reply_text": "So you mostly do homework and enjoy typing. What do you like to type?",
      "correction": "Mostly I do homework and enjoy typing.",
      "explanation": "The phrase 'type voting' is incorrect.",
      "example": "I enjoy typing short stories.",
      "score": 72,
      "created_at": "..."
    }
  ],
  "next_cursor": "eyJjcmVhdGVkX2F0IjogIjIwMjYtMDMtMTFUMTI6MDA6MDAiLCAiaWQiOiAiLi4uIn0="
}
```

- Messages are returned in **display order** (chronological, user then assistant per exchange). Each message has a stable **UUID `id`** and an **`index`** for ordering; use `index` when merging pages. **`next_cursor`** is an opaque string for the next page (older messages); omit when there are no more pages.
- For `role: "user"`, use `content`. For `role: "assistant"`, use `reply_text`, `correction`, `explanation`, `example`, `score`.

### POST /api/v1/conversations/{conversation_id}/title

Generates a short title (max 6 words) from the first few messages and saves it on the conversation.

**Response:** `{ "title": "Talking about hobbies" }`.

---

## Building a chat window (recommended flow)

1. **Sidebar:** On load, call `GET /api/v1/conversations` and render the list (use `title` or fallback e.g. “New chat”, and `last_message` / `updated_at` for preview).
2. **New chat:** Start a stream with `POST /api/v1/ai/chat/stream` or `voice-chat/stream` **without** `conversation_id`; the backend creates a new conversation and returns its `conversation_id` in the `metadata` event. Store it for the current session.
3. **Existing chat:** Pass that `conversation_id` in subsequent `chat/stream` or `voice-chat/stream` requests so new messages stay in the same thread.
4. **Load history:** When opening a conversation, call `GET /api/v1/conversations/{id}/messages` (optionally with `cursor` for older messages). Render messages in **array order** or sort by the `index` field; do not sort by `id`.
5. **Title:** After the first exchange (or when the user opens the conversation), call `POST /api/v1/conversations/{id}/title` to set a title; then refresh the list or update the sidebar entry.

Streaming endpoints already save each message and update the conversation’s `updated_at`, so the conversation list order stays correct after each turn.

---

## Parsing SSE from POST requests

`EventSource` only works with GET. For POST + SSE, parse the response body stream manually:

```javascript
async function consumeStreamingChat(response, handlers) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    let eventType = null;
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith('data: ') && eventType) {
        const data = line.slice(6);
        switch (eventType) {
          case 'stt_result':
            handlers.onSttResult?.(JSON.parse(data));
            break;
          case 'text_chunk':
            handlers.onTextChunk?.(JSON.parse(data));
            break;
          case 'audio_chunk':
            handlers.onAudioChunk?.(data); // base64 string
            break;
          case 'metadata':
            handlers.onMetadata?.(JSON.parse(data));
            break;
          case 'done':
            handlers.onDone?.(JSON.parse(data));
            break;
          case 'audio_ready':
            handlers.onAudioReady?.(JSON.parse(data));
            break;
          case 'error':
            try { handlers.onError?.(JSON.parse(data).error); }
            catch (_) { handlers.onError?.(data); }
            break;
        }
        eventType = null;
      }
    }
  }
}
```

---

## Playing audio chunks

Chunks are base64-encoded WAV (typically 24 kHz, 16-bit PCM). Decode and queue playback:

```javascript
const audioContext = new (window.AudioContext || window.webkitAudioContext)();

function base64ToArrayBuffer(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

async function playWavChunk(base64Wav) {
  const arrayBuffer = base64ToArrayBuffer(base64Wav);
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);
  source.start();
  return new Promise((resolve) => source.onended = resolve);
}

let playQueue = Promise.resolve();
function queueChunk(base64Wav) {
  playQueue = playQueue.then(() => playWavChunk(base64Wav));
}
```

---

## Migration summary

| Before | After |
|--------|-------|
| `voice-chat` → JSON → `tts/stream` (two round trips) | `voice-chat/stream` (single SSE connection, lowest TTFA) |
| `text-chat` → JSON → `tts/stream` (two round trips) | `chat/stream` (single SSE connection) |
| Audio URL from chat response | Audio URL from `audio_ready` SSE event only |
| No live text streaming | `text_chunk` events for progressive display |
| Correction/score in JSON response | `metadata` SSE event with correction, explanation, example, score, conversation_id |
| No conversation list or history API | `GET /api/v1/conversations` (list), `GET .../messages` (paginated history), `POST .../title` (AI title) for full chat window support |
