"""Basic integration tests for AI endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)


@patch('app.services.tts._generate_presigned_url')
@patch('app.services.tts.get')
def test_text_to_speech_cache_hit_s3_returns_presigned_url(mock_get, mock_presigned):
    """When cache returns s3: key, text_to_speech returns a fresh presigned URL."""
    from app.services.tts import text_to_speech

    mock_get.return_value = "s3:audio/abc123.mp3"
    mock_presigned.return_value = "https://bucket.s3.region.amazonaws.com/audio/abc123.mp3?X-Amz-..."

    result = text_to_speech("Hello world", "en")

    mock_presigned.assert_called_once_with("audio/abc123.mp3")
    assert result == "https://bucket.s3.region.amazonaws.com/audio/abc123.mp3?X-Amz-..."


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@patch('app.services.llm.generate_reply')
@patch('app.database.get_db')
def test_text_chat_happy_path(mock_db, mock_llm):
    """Test text chat endpoint happy path; audio_url is always null (use /tts/stream for audio)."""
    # Mock database session
    from app.models.usage import Conversation, Message, Usage
    from sqlalchemy.orm import Session
    
    mock_session = MagicMock(spec=Session)
    mock_conversation = Conversation(id="test-conv-123", user_id="user-123")
    mock_session.query.return_value.filter.return_value.first.return_value = None
    mock_session.add = MagicMock()
    mock_session.commit = MagicMock()
    mock_session.refresh = MagicMock()
    
    def mock_get_db():
        yield mock_session
    
    mock_db.return_value = mock_get_db()
    
    # Mock LLM response
    mock_llm.return_value = {
        "reply_text": "Hello! How are you today?",
        "correction": "",
        "hinglish_explanation": "",
        "hinglish_explanation_show": "",
        "score": 85
    }
    
    # Make request
    response = client.post(
        "/api/v1/ai/text-chat",
        json={
            "user_id": "user-123",
            "message": "Hi there!"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "reply_text" in data
    assert data["audio_url"] is None
    assert "score" in data


def test_text_chat_invalid_input():
    """Test text chat with invalid input."""
    # Missing required fields
    response = client.post(
        "/api/v1/ai/text-chat",
        json={}
    )
    assert response.status_code == 422  # Validation error
    
    # Empty message
    response = client.post(
        "/api/v1/ai/text-chat",
        json={
            "user_id": "user-123",
            "message": ""
        }
    )
    assert response.status_code == 422


@patch('app.services.stt.transcribe_audio')
@patch('app.services.llm.generate_reply')
@patch('app.database.get_db')
def test_voice_chat_invalid_file(mock_db, mock_llm, mock_stt):
    """Test voice chat with invalid file."""
    # Mock database
    from sqlalchemy.orm import Session
    mock_session = MagicMock(spec=Session)
    mock_session.query.return_value.filter.return_value.first.return_value = None
    mock_session.add = MagicMock()
    mock_session.commit = MagicMock()
    
    def mock_get_db():
        yield mock_session
    
    mock_db.return_value = mock_get_db()
    
    # Mock STT to raise validation error
    mock_stt.side_effect = ValueError("Invalid audio file")
    
    # Make request with invalid file
    response = client.post(
        "/api/v1/ai/voice-chat",
        data={"user_id": "user-123"},
        files={"audio_file": ("test.txt", b"not audio", "text/plain")}
    )
    
    # Should return error
    assert response.status_code in [400, 422]
