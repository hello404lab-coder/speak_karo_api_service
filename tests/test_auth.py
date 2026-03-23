"""Tests for auth endpoints: OAuth login, refresh, and me."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.core.security import create_access_token, create_refresh_token
from app.models.user import User

client = TestClient(app)


def _mock_db_with_user(user: User):
    """Return a mock get_db generator that yields a session with user resolved."""
    mock_session = MagicMock()
    mock_session.query.return_value.filter.return_value.first.return_value = user
    mock_session.add = MagicMock()
    mock_session.commit = MagicMock()
    mock_session.refresh = MagicMock(side_effect=lambda x: None)

    def get_db():
        yield mock_session

    return get_db


@patch("app.api.auth.get_or_create_user")
@patch("app.api.auth.verify_google_token")
@patch("app.database.get_db")
def test_oauth_login_success(mock_get_db, mock_verify_google, mock_get_or_create):
    """OAuth login with valid Google ID token returns tokens and user."""
    mock_verify_google.return_value = {"sub": "google-123", "email": "u@example.com", "name": "Test User"}
    user = User(
        id="user-uuid-1",
        email="u@example.com",
        name="Test User",
        provider="google",
        provider_id="google-123",
    )
    mock_get_or_create.return_value = user
    mock_get_db.return_value = _mock_db_with_user(user)

    response = client.post(
        "/api/v1/auth/oauth",
        json={"provider": "google", "id_token": "valid-google-id-token"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["id"] == "user-uuid-1"
    assert data["user"]["email"] == "u@example.com"
    assert data["user"]["name"] == "Test User"
    mock_verify_google.assert_called_once_with("valid-google-id-token")
    mock_get_or_create.assert_called_once()


@patch("app.api.auth.verify_google_token")
def test_oauth_login_invalid_token(mock_verify_google):
    """OAuth login with invalid ID token returns 401."""
    mock_verify_google.side_effect = ValueError("Invalid Google token")

    response = client.post(
        "/api/v1/auth/oauth",
        json={"provider": "google", "id_token": "bad-token"},
    )

    assert response.status_code == 401
    mock_verify_google.assert_called_once_with("bad-token")


@patch("app.api.auth.verify_apple_token")
def test_oauth_login_apple_invalid_token(mock_verify_apple):
    """OAuth login with invalid Apple ID token returns 401."""
    mock_verify_apple.side_effect = ValueError("Invalid or expired Apple token")

    response = client.post(
        "/api/v1/auth/oauth",
        json={"provider": "apple", "id_token": "bad-apple-token"},
    )

    assert response.status_code == 401


def test_oauth_login_missing_body():
    """OAuth login with missing body returns 422."""
    response = client.post("/api/v1/auth/oauth", json={})
    assert response.status_code == 422


def test_refresh_token_flow():
    """Valid refresh token returns new access token."""
    user_id = "test-user-refresh-1"
    refresh_token = create_refresh_token(subject=user_id)

    response = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_refresh_token_invalid():
    """Invalid or expired refresh token returns 401."""
    response = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": "invalid-refresh-token"},
    )
    assert response.status_code == 401


def test_me_without_token():
    """GET /me without Authorization header returns 401."""
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401


@patch("app.database.get_db")
def test_me_with_valid_token(mock_get_db):
    """GET /me with valid Bearer token returns current user."""
    user = User(
        id="user-me-1",
        email="me@example.com",
        name="Me User",
        provider="google",
        provider_id="google-me",
    )
    mock_get_db.return_value = _mock_db_with_user(user)

    access_token = create_access_token(subject=user.id)
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "user-me-1"
    assert data["email"] == "me@example.com"
    assert data["name"] == "Me User"


@patch("app.database.get_db")
def test_me_with_expired_token(mock_get_db):
    """GET /me with expired access token returns 401."""
    from datetime import datetime, timezone, timedelta
    from jose import jwt
    from app.core.config import settings

    # Create an expired access token
    expire = datetime.now(timezone.utc) - timedelta(minutes=1)
    payload = {"sub": "user-123", "exp": expire, "type": "access"}
    expired_token = jwt.encode(
        payload,
        settings.jwt_secret,
        algorithm="HS256",
    )

    mock_session = MagicMock()
    mock_session.query.return_value.filter.return_value.first.return_value = None
    def get_db():
        yield mock_session
    mock_get_db.return_value = get_db()

    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )

    assert response.status_code == 401
