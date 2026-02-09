FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (g++ for pkuseg Cython, git for pip, ffmpeg for pydub)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    ffmpeg \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
# Install numpy first so pkuseg (chatterbox-tts dep) can build; its setup.py imports numpy
RUN pip install --no-cache-dir "numpy<1.26.0,>=1.24.0"
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install git+https://github.com/ai4bharat/IndicF5.git
# Copy application code and test page
COPY app/ ./app/
COPY test_voice_chat.html ./

# Create audio storage; copy ref WAVs from build context (e.g. chirp3-hd-puck.wav for Turbo)
RUN mkdir -p /app/audio_storage
COPY audio_storage/ /app/audio_storage/
ENV AUDIO_STORAGE_PATH=/app/audio_storage

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
