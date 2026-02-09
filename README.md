# AI English Practice Backend

A FastAPI backend service for an AI-powered English speaking practice app focused on Indian learners. The service provides text and voice chat capabilities with AI-powered corrections, Hinglish explanations, and text-to-speech audio responses.

## Features

- **Text Chat**: Send text messages and receive AI replies with corrections
- **Voice Chat**: Send audio (WAV) files and receive transcribed, corrected responses
- **AI Corrections**: Gentle, one-mistake-per-response corrections
- **Hinglish Explanations**: Explanations in a mix of Hindi and English
- **TTS Audio**: Text-to-speech audio responses for listening practice
- **Usage Tracking**: Basic usage statistics per user
- **Conversation History**: Maintain conversation context across messages
- **Caching**: Redis caching for LLM responses and TTS audio

## Tech Stack

- Python 3.11
- FastAPI
- Google Gemini (LLM)
- faster-whisper (STT)
- Chatterbox-Turbo / IndicF5 (TTS)
- PostgreSQL
- Redis (optional)
- Docker

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL
- Redis (optional, for caching)
- Gemini API key (required in prod)

### Local Development

1. **Clone and navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Set up database migrations:**
   ```bash
   # Make sure PostgreSQL is running
   # Run migrations to create database tables
   alembic upgrade head
   ```

6. **Run the application:**
   ```bash
   uvicorn app.main:app --reload
   ```

   The API will be available at `http://localhost:8000`

## Environment Variables

See `.env.example` for all available configuration options. Key variables:

- `APP_ENV`: `dev` or `prod` — controls GPU usage, cache defaults, and strictness (see DEV vs PROD below).
- `GEMINI_API_KEY`: Google Gemini API key; **required when APP_ENV=prod**.
- `DATABASE_URL`: PostgreSQL connection string.
- `REDIS_URL`: Redis connection string (optional; cache degrades gracefully if unavailable).
- `CACHE_ENABLED`: Override cache; in prod defaults to true when unset.
- `LLM_TIMEOUT_SECONDS`, `STT_TIMEOUT_SECONDS`, `TTS_TIMEOUT_SECONDS`: Timeouts for inference (defaults 60, 30, 45).
- `AWS_*`, `S3_BUCKET_NAME`: S3 credentials and bucket for cloud audio storage (optional; when set, TTS audio is stored in S3 and the API returns **presigned GET URLs**). Falls back to local storage if unset or on error.
- `S3_PRESIGNED_EXPIRY_SECONDS`: Expiry in seconds for presigned GET URLs (default: 3600).

## DEV vs PROD

The app supports two modes via `APP_ENV`:

| | **DEV** (`APP_ENV=dev`) | **PROD** (`APP_ENV=prod`) |
|---|------------------------|---------------------------|
| **Device** | CPU only (STT/TTS) | GPU when CUDA available (e.g. RunPod RTX 3090) |
| **Cache** | Default off | Default on (set `CACHE_ENABLED=false` to disable) |
| **GEMINI_API_KEY** | Optional (LLM will fail without it) | **Required** — app will not start without it |
| **Redis / S3** | Optional | Recommended for latency and scalability |

**Production (e.g. RunPod):** Set `APP_ENV=prod`, provide `GEMINI_API_KEY`, and configure Redis and S3. STT uses faster-whisper large-v3 on GPU when available; TTS uses Chatterbox-Turbo and IndicF5 on GPU. All inference runs in a thread pool with configurable timeouts; slow requests return 504 with a user-safe message.

### Production (RunPod) — GPU Docker and Gunicorn

For RunPod (single GPU, e.g. RTX 3090), use the production Docker image and Gunicorn:

- **Image:** Build with `Dockerfile.prod` (CUDA 12.x, Python 3.11, ffmpeg). No Conda.
- **Run:** The image runs **Gunicorn** with **UvicornWorker**: `--workers 1`, `--threads 4`, `--timeout 120`. One worker avoids loading duplicate GPU models; four threads allow concurrent requests; 120s timeout covers long voice pipelines.
- **Config:** Set `APP_ENV=prod` and required env vars (e.g. `GEMINI_API_KEY`, `DATABASE_URL`, `REDIS_URL`, S3 if used). All settings are via environment variables.
- **GPU:** STT, Turbo TTS, and IndicF5 share a single GPU. Inference locks ensure only one STT and one TTS inference run at a time per process, avoiding OOM and keeping the app stable under concurrent voice requests.

Build and run (example):

```bash
docker build -f Dockerfile.prod -t ai-english-backend:prod .
docker run --gpus all -p 8000:8000 --env-file .env ai-english-backend:prod
```

Local dev continues to use `uvicorn app.main:app --reload` (see Setup).

## API Endpoints

### Health Check

```bash
GET /health
```

### Initialize Models (warmup)

Load all models (STT, LLM client, TTS) so the first user request does not hit cold-start latency. Optional; call after deployment or before traffic.

```bash
POST /api/v1/ai/init-models
```

**Response (200):**
```json
{
  "stt": {"status": "loaded", "mode": "faster_whisper_large"},
  "llm": {"status": "loaded"},
  "tts": {"turbo": "loaded", "indicf5": "loaded"}
}
```

Returns 504 if initialization exceeds 5 minutes.

### Text Chat

```bash
POST /api/v1/ai/text-chat
Content-Type: application/json

{
  "user_id": "user-123",
  "message": "I am go to market",
  "conversation_id": "optional-conversation-id"
}
```

**Response:**
```json
{
  "reply_text": "Hello! I understand you want to go to the market.",
  "correction": "I am go to market → I am going to the market",
  "hinglish_explanation": "Yahan 'going' use karna chahiye kyunki yeh present continuous tense hai.",
  "score": 75,
  "audio_url": "http://localhost:8000/audio/abc123.mp3",
  "conversation_id": "conv-123"
}
```

### Voice Chat

```bash
POST /api/v1/ai/voice-chat
Content-Type: multipart/form-data

user_id: user-123
conversation_id: optional-conversation-id
audio_file: [WAV file]
```

**Response:** Same as text chat endpoint.

## Database Migrations

This project uses Alembic for database migrations.

### Create a new migration

After modifying models in `app/models/`, create a new migration:

```bash
alembic revision --autogenerate -m "Description of changes"
```

### Apply migrations

```bash
# Apply all pending migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# View current migration status
alembic current

# View migration history
alembic history
```

### Initial setup

For a fresh database, run:

```bash
alembic upgrade head
```

This will create all tables defined in your models.

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Docker

- **Dev (CPU):** `Dockerfile` — uvicorn, no GPU. Use for local or CI.
- **Prod (GPU/RunPod):** `Dockerfile.prod` — CUDA 12.x, Gunicorn (workers=1, threads=4, timeout=120). See "Production (RunPod)" above.

Inside a container, `localhost` is the container itself, not your host machine. So if Postgres and Redis run on the host (or in other containers with published ports), the app must not use `localhost` for `DATABASE_URL` / `REDIS_URL` when it runs in Docker.

### Option A: Run app in Docker, Postgres/Redis on host (or other containers)

Use the host’s address from inside Docker:

- **Docker Desktop (Mac/Windows):** use `host.docker.internal` as the hostname.

Override the URLs when running the container:

```bash
docker build -t ai-english-backend .
docker run -p 8000:8000 --env-file .env \
  -e DATABASE_URL="postgresql://user:password@host.docker.internal:5432/english_practice" \
  -e REDIS_URL="redis://host.docker.internal:6379/0" \
  ai-english-backend
```

Replace `user`, `password`, and `english_practice` with your real DB credentials. Ensure Postgres and Redis are listening on 5432 and 6379 (and that those ports are exposed if they run in other containers).

### Option B: Run everything with Docker Compose (recommended)

Run the backend, Postgres, and Redis in the same Docker network so the app can use service names:

```bash
docker compose up --build
```

Compose uses `docker-compose.yml`, which starts Postgres and Redis and sets `DATABASE_URL` and `REDIS_URL` to `postgres` and `redis` (so no `localhost` inside the app). Your `.env` is still loaded for `GEMINI_API_KEY`, etc. Default DB credentials in Compose are `user` / `password` / `english_practice`; adjust in `docker-compose.yml` if needed.

After first start, run migrations against the Compose Postgres (from your host, with port 5432 published):

```bash
alembic upgrade head
```

When S3 is configured via environment variables (`AWS_*`, `S3_BUCKET_NAME`), generated TTS audio is stored in S3 and the API returns presigned GET URLs in `audio_url` (expiry set by `S3_PRESIGNED_EXPIRY_SECONDS`). The frontend uses the same `audio_url` field; no frontend change is required.

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app
│   ├── api/                 # API routes
│   ├── core/                # Configuration and prompts
│   ├── services/            # Business logic (LLM, STT, TTS, cache)
│   ├── models/              # Database models
│   ├── schemas/             # Pydantic schemas
│   ├── database.py          # DB connection
│   └── utils/               # Utilities (audio, language, device)
├── tests/                   # Tests
├── requirements.txt         # Dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # App + Postgres + Redis on same network
└── README.md               # This file
```

## Notes

- No authentication is implemented (MVP phase)
- Audio files are stored locally by default, or in S3 if configured
- Conversation history is maintained per `conversation_id`
- Usage statistics are tracked daily per user
- Redis caching is optional - service degrades gracefully if unavailable

## License

Proprietary - All rights reserved
