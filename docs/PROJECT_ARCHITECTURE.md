# AI English Practice App - Backend Architecture

This document provides a comprehensive overview of the backend structure, data flow, features, and current working architecture of the AI English Practice App.

## 1. Project Structure

The project follows a standard FastAPI structure, separated into modular components:

```text
backend/
├── app/
│   ├── api/
│   │   └── ai.py               # Main API endpoints (Text/Voice Chat, Streaming)
│   ├── core/
│   │   ├── config.py           # Application settings (Pydantic, environment variables)
│   │   └── prompts.py          # LLM system prompts and JSON response schemas
│   ├── models/
│   │   └── usage.py            # SQLAlchemy database models (Usage, Conversation, Message)
│   ├── schemas/
│   │   └── ai.py               # Pydantic validation schemas for API requests/responses
│   ├── services/
│   │   ├── cache.py            # Redis / In-memory caching layer
│   │   ├── llm.py              # LLM Service (Google Gemini integration)
│   │   ├── stt.py              # Speech-to-Text orchestrator
│   │   ├── stt_backends/       # Specific STT implementations (Faster Whisper, Groq, OpenAI)
│   │   └── tts.py              # Text-to-Speech orchestrator (Chatterbox, IndicF5, Gemini TTS)
│   ├── utils/
│   │   ├── audio.py            # Audio validation and manipulation utilities
│   │   ├── device.py           # Hardware device resolution (CUDA, MPS, CPU)
│   │   └── language.py         # Language and script detection utilities
│   ├── database.py             # Database connection and session management
│   └── main.py                 # FastAPI application entry point
├── data/                       # Local SQLite database storage (if used)
├── docs/                       # Project documentation
├── audio_storage/              # Local storage for generated audio files
├── requirements.txt            # Python dependencies
└── .env                        # Environment configuration
```

## 2. Core Features

- **Text & Voice Chat Interfaces:** Users can practice English by typing or speaking.
- **Real-Time Streaming:** Sentence-level pipelined streaming (STT -> LLM -> TTS) using Server-Sent Events (SSE) for low latency.
- **Speech-to-Text (STT):** Configurable transcription using local models (`faster-whisper`, `openai/whisper-large-v3`) or cloud APIs (Groq Whisper API).
- **Text-to-Speech (TTS):** 
  - English: Local `Chatterbox-Turbo` with voice cloning.
  - Indic Languages (Hindi, Malayalam, Tamil, etc.): Local `IndicF5`.
  - Fallback: Cloud-based `Gemini TTS`.
- **Intelligent Feedback:** The AI acts as a tutor, providing conversation replies, grammatical corrections, and a performance score (0-100).
- **Conversation Memory:** Tracks conversation history and long-term context (`learner_context` like "preparing for IELTS") to maintain continuity.
- **Caching & Storage:** Redis/memory caching for TTS/LLM responses. Audio files are saved locally or in AWS S3 with presigned URLs.
- **Usage Tracking:** Monitors daily request counts and minutes used per user.

## 3. Current Working Architecture

The architecture is designed to handle synchronous blocks through threading (`asyncio.to_thread`) and provides a highly modular service layer.

### A. Environment & Configuration (`app/core/config.py`)
- Configured via environment variables (`.env`).
- Supports running in `dev` (CPU, relaxed) and `prod` (GPU, strict) modes.
- Toggles for local models (`STT_WHISPER_LOCAL_ENABLED`, `TTS_CHATTERBOX_ENABLED`, `TTS_INDICF5_ENABLED`) to gracefully degrade to cloud APIs if hardware is limited.

### B. Database Layer (`app/database.py`, `app/models/usage.py`)
- Uses **SQLAlchemy** with support for SQLite (dev) and PostgreSQL (prod).
- **Models:**
  - `Conversation`: Tracks unique sessions and `long_term_context`.
  - `Message`: Stores the user message, AI reply, correction, score, and timestamps.
  - `Usage`: Tracks daily `minutes_used` and `request_count` per user.

### C. Services Layer

#### 1. Large Language Model (LLM) - `app/services/llm.py`
- Powered by **Google Gemini** (`gemini-2.5-flash`).
- **Prompting:** The AI acts as a tutor, instructed via `app/core/prompts.py` to return JSON containing `reply_text` (spoken), `correction` (displayed), and `score` (0-100).
- **Context Window Management:** Drops oldest messages dynamically when exceeding the `llm_context_token_budget` (16,384 tokens).
- **Streaming:** Yields tokens dynamically for the streaming pipeline (`stream_gemini_tokens`).

#### 2. Speech-to-Text (STT) - `app/services/stt.py`
- Transcribes WAV audio files and auto-detects language.
- **Backends:**
  - `faster_whisper` (medium/large): Fast local inference using CTranslate2.
  - `openai_whisper_large_v3`: HuggingFace Transformers implementation.
  - `groq_whisper_api`: Cloud API fallback (faster, cheaper) when local models are disabled.

#### 3. Text-to-Speech (TTS) - `app/services/tts.py`
- Converts AI text replies back into audio.
- **Backends:**
  - **Chatterbox-Turbo (English):** Requires a reference WAV (`tts_audio_prompt_path`) to perform high-quality zero-shot voice cloning locally.
  - **IndicF5 (Indic Languages):** Dedicated local model for languages like Hindi, Malayalam, Tamil, etc., requiring reference audios per language.
  - **Gemini TTS (Fallback):** Cloud API generation if local models are disabled or fail.
- **Storage:** Audio is synthesized as WAV bytes, converted to MP3 (via FFmpeg), and saved either to `audio_storage/` or uploaded to AWS S3 (returning a presigned GET URL).

## 4. Database Structure and Data Storage

The application uses SQLAlchemy as its ORM, supporting both SQLite (for local development) and PostgreSQL (for production). The data models are defined in `app/models/usage.py`.

### Schema Details

#### `Conversation` Table (`conversations`)
Tracks individual chat sessions and long-term user context.
- `id` (String, Primary Key): Unique UUID for the conversation.
- `user_id` (String, Index): The identifier for the user.
- `created_at` (DateTime): Timestamp of creation.
- `updated_at` (DateTime): Timestamp of last update.
- `long_term_context` (Text, Nullable): Stores persistent context about the user (e.g., "preparing for IELTS") which is injected into the LLM system prompt on every turn.

#### `Message` Table (`messages`)
Stores individual turns (exchanges) within a conversation.
- `id` (Integer, Primary Key): Auto-incrementing identifier.
- `conversation_id` (String, ForeignKey): Links back to the `Conversation` table.
- `user_message` (Text): The text input from the user (or transcribed text if voice was used).
- `ai_reply` (Text): The text response generated by the AI tutor.
- `correction` (Text, Nullable): Grammatical corrections provided by the AI for the user's message.
- `hinglish_explanation` (Text, Nullable): Reserved for future language-specific explanations.
- `score` (Integer, Nullable): A performance score (0-100) assigned by the AI for the user's utterance.
- `created_at` (DateTime, Index): Timestamp of the message.

#### `Usage` Table (`usage`)
Tracks daily consumption metrics per user.
- `id` (Integer, Primary Key): Auto-incrementing identifier.
- `user_id` (String, Index): The identifier for the user.
- `date` (Date, Index): The specific date of the usage record.
- `minutes_used` (Float): Approximate audio minutes processed (useful for billing/quotas).
- `request_count` (Integer): Total number of interactions the user had on that date.

### Storage Strategy
- **Relational Data:** Managed via SQLAlchemy. History is retrieved dynamically and truncated based on the `llm_context_token_budget` and `llm_history_max_exchanges` settings before being sent to the LLM.
- **Audio Files:** Generated AI voice responses (MP3s) are stored either locally in the `audio_storage/` directory or in an AWS S3 bucket. A unique MD5 hash of the text is used as the filename to allow caching and deduplication.

## 5. Request Flow Patterns

### Standard Text/Voice Request Flow (`/text-chat`, `/voice-chat`)
1. **Input Reception:** Receive JSON text or a Multipart WAV audio upload.
2. **Transcription (Voice only):** Pass audio bytes to the STT service to get transcribed text and detected language.
3. **Context Retrieval:** Fetch the user's conversation history and `learner_context` from the DB.
4. **LLM Generation:** Send history, new message, and context to Gemini. The model returns a structured JSON (reply, correction, score).
5. **Database Update:** Save the interaction (`Message`) and update daily `Usage`.
6. **Response:** Return the structured data to the client (TTS is generated lazily by the client requesting the audio URL or handled separately depending on the client).

### Streaming Request Flow (`/chat/stream`, `/voice-chat/stream`)
The app utilizes a sophisticated **Sentence-Level Streaming Pipeline** (`_llm_tts_streaming_pipeline`) to minimize time-to-first-audio:
1. **Input:** User sends text or audio (transcribed via STT).
2. **LLM Token Stream:** Gemini streams text tokens back to the server.
3. **Sentence Buffering:** A buffer consumer accumulates tokens and splits them at sentence boundaries (`.`, `?`, `\n`, `\u0964`).
4. **TTS Worker:** As soon as a complete sentence is buffered, it is pushed to a TTS worker queue. The TTS service generates WAV bytes for that specific sentence immediately.
5. **Server-Sent Events (SSE):** The pipeline yields events back to the client continuously:
   - `stt_result`: (Voice only) The transcribed text.
   - `text_chunk`: A complete sentence text from the LLM.
   - `audio_chunk`: Base64 encoded WAV audio bytes for that sentence.
   - `metadata`: Full correction, score, and conversation ID (sent after the LLM finishes).
   - `done` / `audio_ready`: Signals completion.
6. **Background Concatenation:** While the client plays the streamed chunks, the server concatenates all WAV chunks in the background, converts them to a single MP3, uploads it to S3/local storage, and saves the final interaction to the database.

## 6. Caching & Concurrency
- **Caching:** Redis (or in-memory dictionary fallback) is used to cache LLM responses (by hashing the context and user message) and TTS audio URLs (by hashing the text and language). This reduces redundant GPU/API calls.
- **Concurrency:** FastAPI's async loops handle web requests, while heavy inference tasks (STT, LLM generation, TTS synthesis) are offloaded to background threads (`asyncio.to_thread`). 
- **GPU Locks:** A threading lock (`_inference_lock`) is used around local TTS generation to prevent out-of-memory (OOM) errors on a single GPU.

## 7. API Endpoints
- **`POST /api/v1/ai/init-models`**: Warms up and pre-loads STT/TTS models into GPU memory on startup to avoid cold-start latency.
- **`POST /api/v1/ai/text-chat`**: Standard synchronous text-in, JSON-out chat.
- **`POST /api/v1/ai/voice-chat`**: Standard synchronous audio-in, JSON-out chat.
- **`POST /api/v1/ai/chat/stream`**: SSE endpoint for text-in, streaming audio/text chunks out.
- **`POST /api/v1/ai/voice-chat/stream`**: SSE endpoint for audio-in, streaming audio/text chunks out.
- **`POST /api/v1/ai/tts/stream`**: Standalone TTS streaming endpoint.
- **`GET /health`**: Standard application health check.
