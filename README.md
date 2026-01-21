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
- OpenAI (LLM, Whisper STT, TTS)
- PostgreSQL
- Redis
- Docker

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL
- Redis (optional, for caching)
- OpenAI API key

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

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string (optional)
- `AWS_*`: S3 credentials for cloud audio storage (optional, falls back to local)

## API Endpoints

### Health Check

```bash
GET /health
```

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

Build and run with Docker:

```bash
# Build image
docker build -t ai-english-backend .

# Run container
docker run -p 8000:8000 --env-file .env ai-english-backend
```

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
│   └── utils/               # Utilities
├── tests/                   # Tests
├── requirements.txt         # Dependencies
├── Dockerfile              # Docker configuration
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
