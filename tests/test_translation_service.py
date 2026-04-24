"""Unit tests for Google Cloud assistant reply translation."""
from unittest.mock import MagicMock, patch

from app.services import translation as translation_service


def _reset_translation_state() -> None:
    translation_service._translation_credentials = None
    translation_service._translation_project_id = None
    translation_service._translation_init_error_logged = False


def test_translate_reply_text_returns_none_for_same_source_and_target():
    """No translation request should happen when the display language matches the reply language."""
    _reset_translation_state()
    with patch.object(translation_service.settings, "translation_api_key", "test-key"):
        with patch("app.services.translation.google_auth_default") as mock_auth:
            result = translation_service.translate_reply_text("Hello there", "en", "en")
    assert result is None
    mock_auth.assert_not_called()


def test_translate_reply_text_uses_cache_hit_without_google_call():
    """Translation cache should short-circuit the Google API call."""
    _reset_translation_state()
    with patch.object(translation_service.settings, "translation_api_key", "test-key"):
        with patch("app.services.translation.get_json", return_value={"translated_reply_text": "സുഖമാണോ?"}) as mock_get:
            with patch("app.services.translation.google_auth_default") as mock_auth:
                result = translation_service.translate_reply_text("How are you?", "en", "ml")
    assert result == "സുഖമാണോ?"
    mock_get.assert_called_once()
    mock_auth.assert_not_called()


def test_translate_reply_text_uses_api_key_basic_endpoint_and_caches_result():
    """When TRANSLATION_API_KEY is configured, the service should use Basic v2 with API-key auth."""
    _reset_translation_state()
    fake_response = MagicMock()
    fake_response.json.return_value = {"data": {"translations": [{"translatedText": "നമസ്കാരം"}]}}
    fake_response.raise_for_status = MagicMock()

    with patch.object(translation_service.settings, "translation_api_key", "test-key"):
        with patch("app.services.translation.get_json", return_value=None):
            with patch("app.services.translation.set_json") as mock_set_json:
                with patch("app.services.translation.requests.post", return_value=fake_response) as mock_post:
                    with patch("app.services.translation.google_auth_default") as mock_auth:
                        result = translation_service.translate_reply_text("Hello", "en", "ml")

    assert result == "നമസ്കാരം"
    mock_auth.assert_not_called()
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["params"] == {"key": "test-key"}
    assert kwargs["json"]["q"] == ["Hello"]
    assert kwargs["json"]["source"] == "en"
    assert kwargs["json"]["target"] == "ml"
    assert kwargs["json"]["format"] == "text"
    assert kwargs["json"]["model"] == "nmt"
    mock_set_json.assert_called_once()


def test_translate_reply_text_falls_back_to_adc_when_no_api_key():
    """ADC/service-account auth remains available when TRANSLATION_API_KEY is not set."""
    _reset_translation_state()
    fake_credentials = MagicMock()
    fake_response = MagicMock()
    fake_response.json.return_value = {"translations": [{"translatedText": "നമസ്കാരം"}]}
    fake_response.raise_for_status = MagicMock()
    fake_session = MagicMock()
    fake_session.post.return_value = fake_response

    with patch.object(translation_service.settings, "translation_api_key", None):
        with patch("app.services.translation.get_json", return_value=None):
            with patch("app.services.translation.set_json") as mock_set_json:
                with patch("app.services.translation.google_auth_default", return_value=(fake_credentials, "demo-project")) as mock_auth:
                    with patch("app.services.translation.AuthorizedSession", return_value=fake_session) as mock_session_cls:
                        result = translation_service.translate_reply_text("Hello", "en", "ml")

    assert result == "നമസ്കാരം"
    mock_auth.assert_called_once()
    mock_session_cls.assert_called_once_with(fake_credentials)
    fake_session.post.assert_called_once()
    _, kwargs = fake_session.post.call_args
    assert kwargs["json"]["contents"] == ["Hello"]
    assert kwargs["json"]["sourceLanguageCode"] == "en"
    assert kwargs["json"]["targetLanguageCode"] == "ml"
    assert kwargs["json"]["mimeType"] == "text/plain"
    assert kwargs["headers"] == {"x-goog-user-project": "demo-project"}
    mock_set_json.assert_called_once()


def test_translate_reply_text_returns_none_on_api_key_failure():
    """Translation failures must not break the turn response."""
    _reset_translation_state()

    with patch.object(translation_service.settings, "translation_api_key", "test-key"):
        with patch("app.services.translation.get_json", return_value=None):
            with patch("app.services.translation.set_json") as mock_set_json:
                with patch("app.services.translation.requests.post", side_effect=RuntimeError("network down")):
                    result = translation_service.translate_reply_text("Hello", "en", "ml")

    assert result is None
    mock_set_json.assert_not_called()
