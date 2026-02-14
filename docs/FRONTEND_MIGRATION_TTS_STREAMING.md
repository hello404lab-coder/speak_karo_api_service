# Frontend migration: TTS from single URL to streaming (SSE)

This guide describes how to use the streaming TTS flow over Server-Sent Events (SSE) for lower time-to-first-audio and better perceived performance.

---

## Chat response: no audio URL

**`POST /api/ai/text-chat`** and **`POST /api/ai/voice-chat`** return only text and metadata. **`audio_url` is always `null`** in that response. The backend does not generate audio in the chat endpoints.

You get:

```json
{
  "reply_text": "That's great! You could say...",
  "correction": "...",
  "score": 75,
  "audio_url": null,
  "conversation_id": "uuid-..."
}
```

**To get audio:** Call **`POST /api/ai/tts/stream`** with the reply text (and optional `response_language`). **`audio_url` will always be `null` in the initial chat response**; it is **only** provided via the **`audio_ready`** SSE event from the streaming endpoint, after the full MP3 has been stitched and saved.

---

## New way: streaming TTS over SSE

### How it works

- You still use **text-chat** or **voice-chat** to get the AI reply and metadata.
- For TTS, you **optionally** call **`POST /api/ai/tts/stream`** with the text (and language) you want to speak.
- The response is a **Server-Sent Events** stream:
  - **`audio_chunk`** events: base64-encoded **WAV** audio (one chunk per sentence). Decode and play each chunk as it arrives.
  - **`done`** event: Sent **immediately** after the last chunk. Data is `{ "audio_url": null, "saving_in_background": true }` when the server is stitching and saving the full MP3 in the background. Use this to know streaming is complete; do not expect a URL here.
  - **`audio_ready`** event: Sent when the full MP3 has been stitched and saved. Data is `{ "audio_url": "..." }`. Use this URL for "replay" or saving the full file.
  - **`error`** event: JSON with **`error`** message on failure.

Streaming uses **WAV** chunks for performance; the full file is stitched and saved in the background, then delivered via **`audio_ready`**.

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
| `done`        | JSON `{ "audio_url": null, "saving_in_background": true }` or `{ "audio_url": null }` | Stream finished; sent immediately after last chunk. No URL here when saving in background. |
| `audio_ready`  | JSON `{ "audio_url": "..." }` | Full MP3 stitched and saved. Use this for replay or download. |
| `error`       | JSON `{ "error": "message" }` | Something went wrong. |

Chunk payloads are base64-encoded WAV (sample rate from the engine, typically 24 kHz for Chatterbox-Turbo). For a single continuous WAV or raw PCM stream, a separate endpoint and `media_type` would be used.

---

## Frontend migration steps

### 1. Chat returns text only; audio_url is always null

Call **text-chat** or **voice-chat** as usual. You get **`reply_text`**, **`correction`**, **`score`**, and **`conversation_id`**. **`audio_url` is always `null`** in this response. Do not use it for playback.

### 2. Get audio by calling the streaming endpoint

After you have the reply text, call **`POST /api/ai/tts/stream`** with **reply_text only** (do not include correction) to generate and stream audio. The **audio URL is only provided in the `audio_ready` SSE event** from this endpoint, not in the initial chat response.

```javascript
// Use reply_text only for TTS (do not include correction)
const response = await fetch('/api/ai/tts/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: data.reply_text,
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
async function consumeTtsStream(response, onChunk, onDone, onAudioReady, onError) {
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
        } else if (eventType === 'audio_ready') {
          try {
            onAudioReady(JSON.parse(data));
          } catch (_) {
            onAudioReady({ audio_url: null });
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
const textForTts = data.reply_text;

const response = await fetch('/api/ai/tts/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: textForTts, response_language: 'en' }),
});

await consumeTtsStream(
  response,
  (base64Wav) => queueChunk(base64Wav),
  (payload) => {
    // done: streaming finished (payload.audio_url is null; saving_in_background may be true)
    console.log('Stream done, saving in background:', payload.saving_in_background);
  },
  (payload) => {
    // audio_ready: full MP3 URL for replay
    console.log('Full audio URL (for replay):', payload.audio_url);
  },
  (err) => console.error('TTS stream error:', err)
);
```

### 6. Fallback and replay URL

- **Replay URL:** Use the **`audio_ready`** event's **`audio_url`** for replay or download. Do not rely on **`done`** for a URL; it is sent immediately and may have `audio_url: null` and `saving_in_background: true`.
- If you still get **`audio_url`** from text-chat/voice-chat, use that as fallback when streaming is disabled, or prefer **`audio_ready`** when using the stream.


---

## Summary

| Aspect | Chat response | TTS stream |
|--------|----------------|------------|
| **Endpoint** | `POST /api/ai/text-chat` or `voice-chat` | `POST /api/ai/tts/stream` with `{ text, response_language }` |
| **Response** | JSON with text and metadata; **`audio_url` is always `null`** | SSE stream: `audio_chunk` → `done` → `audio_ready` (with URL) |
| **Audio URL** | Not provided | Only in **`audio_ready`** event after background stitch |
| **Playback** | N/A | Decode base64 WAV chunks and play as they arrive; use `audio_ready` URL for replay |

The initial chat response never includes an audio URL; the client must call `/tts/stream` and use the **`audio_ready`** SSE event to get the replay URL.
