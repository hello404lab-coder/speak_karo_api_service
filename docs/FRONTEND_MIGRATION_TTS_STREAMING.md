# Frontend migration: TTS from single URL to streaming (SSE)

This guide describes how to move your frontend from the **old** TTS flow (single `audio_url` in the chat response) to the **new** streaming TTS flow over Server-Sent Events (SSE) for lower time-to-first-audio and better perceived performance.

---

## Old way (still supported)

### How it works

- You call **`POST /api/ai/text-chat`** or **`POST /api/ai/voice-chat`**.
- The response is JSON with a single **`audio_url`** pointing to the full TTS file (MP3):

```json
{
  "reply_text": "That's great! You could say...",
  "correction": "...",
  "hinglish_explanation": "...",
  "score": 75,
  "audio_url": "http://localhost:8000/audio/abc123.mp3",
  "conversation_id": "uuid-..."
}
```

### Frontend usage (old)

```javascript
// 1. Send message
const res = await fetch('/api/ai/text-chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_id: 'user-1',
    message: 'Hello',
    conversation_id: null,
  }),
});
const data = await res.json();

// 2. Show text
document.getElementById('reply').textContent = data.reply_text;

// 3. Play audio from single URL (full file must be ready)
const audio = new Audio(data.audio_url);
audio.play();
```

**Limitation:** The user waits for the entire TTS file to be generated and stored before playback can start.

---

## New way: streaming TTS over SSE

### How it works

- You still use **text-chat** or **voice-chat** to get the AI reply and metadata.
- For TTS, you **optionally** call **`POST /api/ai/tts/stream`** with the text (and language) you want to speak.
- The response is a **Server-Sent Events** stream:
  - **`audio_chunk`** events: base64-encoded **WAV** audio (one chunk per sentence). You can decode and play each chunk as it arrives.
  - **`done`** event: JSON with **`audio_url`** for the full file (MP3), if you want to store or replay the whole thing.
  - **`error`** event: JSON with **`error`** message on failure.

Streaming uses **WAV** chunks for performance; only the final stored file is MP3.

### Endpoint

| Method | Path              | Body (JSON) |
|--------|-------------------|-------------|
| POST   | `/api/ai/tts/stream` | `{ "text": "Sentence one. Sentence two.", "response_language": "en" }` |

- **`text`** (required): 1–5000 characters to synthesize.
- **`response_language`** (optional): `"en"` (default), `"hi"`, `"ml"`, `"ta"`, etc. Same as chat (English → Turbo, Indic → IndicF5).

### SSE event types

| Event        | Data (after `data: `) | Meaning |
|-------------|------------------------|--------|
| `audio_chunk` | Base64 string          | One WAV audio chunk (one sentence). Decode and append/play. |
| `done`        | JSON `{ "audio_url": "..." \| null }` | Stream finished. Optional full MP3 URL. |
| `error`       | JSON `{ "error": "message" }` | Something went wrong. |

---

## Frontend migration steps

### 1. Keep existing chat flow

Continue calling **text-chat** or **voice-chat** as today. You still get **`reply_text`**, **`correction`**, **`hinglish_explanation`**, **`score`**, **`conversation_id`**, and **`audio_url`**. You can keep using **`audio_url`** as fallback or for “replay full answer.”

### 2. Add streaming TTS when you want faster playback

After you have the reply text (and optionally correction/explanation), call the streaming endpoint with the same text you would have sent to TTS:

```javascript
// Build the same string the backend would use for TTS (reply + optional explanation)
const textForTts = data.hinglish_explanation?.trim()
  ? `${data.reply_text}. ... ${data.hinglish_explanation}`
  : data.reply_text;

const response = await fetch('/api/ai/tts/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: textForTts,
    response_language: 'en', // or from your app/lang detection
  }),
});

if (!response.ok) {
  throw new Error(`TTS stream failed: ${response.status}`);
}
```

### 3. Consume the SSE stream

You can use **`EventSource`** only for GET requests. For POST + SSE you must read the body as a stream and parse SSE manually (or use a small helper).

**Example: parse SSE from a fetch stream**

```javascript
async function consumeTtsStream(response, onChunk, onDone, onError) {
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
      if (line.startsWith('event: ')) eventType = line.slice(7).trim();
      else if (line.startsWith('data: ') && eventType) {
        const data = line.slice(6);
        if (eventType === 'audio_chunk') {
          onChunk(data); // base64 WAV
        } else if (eventType === 'done') {
          try {
            onDone(JSON.parse(data));
          } catch (_) {
            onDone({ audio_url: null });
          }
        } else if (eventType === 'error') {
          try {
            onError(JSON.parse(data).error);
          } catch (_) {
            onError(data);
          }
        }
        eventType = null;
      }
    }
  }
}
```

### 4. Decode base64 WAV and play

Chunks are **base64-encoded WAV**. Decode and play in order (e.g. queue and play one after another, or append to a single buffer and play when enough data is ready).

**Option A: Play each chunk as it arrives (simple queue)**

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

// Usage: queue chunks and play in order
let playQueue = Promise.resolve();
function queueChunk(base64Wav) {
  playQueue = playQueue.then(() => playWavChunk(base64Wav));
}
```

**Option B: Use an `<audio>` element with MediaSource / Blob URLs**

You can also append each decoded WAV to a buffer, build a single WAV blob when the stream ends, and set `audio.src = URL.createObjectURL(wavBlob)` for playback or replay.

### 5. Wire it together (streaming path)

```javascript
// After you have data from text-chat or voice-chat
const textForTts = data.hinglish_explanation?.trim()
  ? `${data.reply_text}. ... ${data.hinglish_explanation}`
  : data.reply_text;

const response = await fetch('/api/ai/tts/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: textForTts, response_language: 'en' }),
});

await consumeTtsStream(
  response,
  (base64Wav) => queueChunk(base64Wav),
  (payload) => {
    console.log('Full audio URL (for replay):', payload.audio_url);
  },
  (err) => console.error('TTS stream error:', err)
);
```

### 6. Fallback to old `audio_url`

If you still get **`audio_url`** from text-chat/voice-chat, you can:

- Prefer streaming for “play as soon as possible” and use **`audio_url`** only for “replay” or if streaming is disabled.
- Or ignore **`audio_url`** when using streaming and use the **`done`** event’s **`audio_url`** for replay (same value in practice).

---

## Summary

| Aspect | Old way | New way (streaming) |
|--------|---------|----------------------|
| **Endpoint** | Same chat endpoints | `POST /api/ai/tts/stream` with `{ text, response_language }` |
| **Response** | JSON with `audio_url` | SSE stream |
| **Audio format** | MP3 at `audio_url` | Chunks: **WAV** (base64). Final file: MP3 at `done.audio_url` |
| **Playback** | `new Audio(audio_url)` after full file | Decode base64 WAV and play chunks as they arrive (e.g. Web Audio API) |
| **When to use** | Simple, no stream handling | Lower latency, start playback after first sentence |

You can migrate gradually: keep the old flow and add the new streaming path only where you want faster time-to-first-audio.
