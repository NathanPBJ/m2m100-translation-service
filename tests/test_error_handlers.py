"""Focused tests for safe and consistent centralized API errors."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.exceptions import (
    DeviceConfigurationError,
    InputTextTooLargeError,
    InputTooLongError,
    InvalidLanguageDetectionInputError,
    InvalidTranslationInputError,
    LanguageDetectionInferenceError,
    LanguageDetectionUncertainError,
    LanguageDetectorLoadError,
    LanguageDetectorNotLoadedError,
    ModelLoadError,
    ModelNotLoadedError,
    TextChunkingFailedError,
    TooManyChunksError,
    TranslationInferenceError,
    TranslationOutputTruncatedError,
    UnsupportedLanguageError,
)
from app.main import create_app
from tests.conftest import FakeLanguageDetectionService, FakeTranslationService


@pytest.mark.parametrize(
    ("exception", "status_code", "error_code"),
    [
        (
            InvalidTranslationInputError("Invalid translation input."),
            422,
            "invalid_translation_input",
        ),
        (
            UnsupportedLanguageError("Language code 'xx' is not supported."),
            422,
            "unsupported_language",
        ),
        (
            InputTextTooLargeError(actual_characters=30, maximum_characters=20),
            413,
            "text_too_large",
        ),
        (
            InputTooLongError(actual_tokens=700, maximum_tokens=512),
            413,
            "input_too_long",
        ),
        (
            TooManyChunksError(actual_chunks=5, maximum_chunks=2),
            413,
            "too_many_chunks",
        ),
        (
            TextChunkingFailedError("raw private chunk TOP_SECRET"),
            500,
            "text_chunking_failed",
        ),
        (
            TranslationOutputTruncatedError("raw partial translation TOP_SECRET"),
            500,
            "translation_output_truncated",
        ),
        (
            ModelNotLoadedError("Translation model is unavailable."),
            503,
            "model_not_loaded",
        ),
        (
            ModelLoadError("raw model path C:/secret/model"),
            503,
            "model_load_failed",
        ),
        (
            DeviceConfigurationError("CUDA is unavailable."),
            503,
            "device_configuration_error",
        ),
        (
            InvalidLanguageDetectionInputError("raw invalid detection input"),
            422,
            "invalid_detection_input",
        ),
        (
            LanguageDetectionUncertainError("raw uncertain result"),
            422,
            "language_detection_uncertain",
        ),
        (
            LanguageDetectorNotLoadedError("raw detector state"),
            503,
            "language_detector_not_loaded",
        ),
        (
            LanguageDetectorLoadError("raw Rust load failure TOP_SECRET"),
            503,
            "language_detector_load_failed",
        ),
        (
            LanguageDetectionInferenceError("raw Rust inference TOP_SECRET"),
            500,
            "language_detection_failed",
        ),
        (
            TranslationInferenceError("raw PyTorch TOP_SECRET failure"),
            500,
            "translation_failed",
        ),
        (
            RuntimeError("unexpected TOP_SECRET model failure"),
            500,
            "internal_server_error",
        ),
    ],
)
def test_exception_mapping_has_consistent_safe_envelope(
    api_settings: Settings,
    exception: Exception,
    status_code: int,
    error_code: str,
) -> None:
    service = FakeTranslationService()
    detection_service = FakeLanguageDetectionService()
    application = create_app(
        translation_service=service,  # type: ignore[arg-type]
        language_detection_service=detection_service,  # type: ignore[arg-type]
        settings=api_settings,
    )

    @application.get("/raise-test-error")
    async def raise_test_error() -> None:
        raise exception

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get("/raise-test-error")

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == error_code
    assert isinstance(response.json()["error"]["message"], str)
    assert "TOP_SECRET" not in response.text
    assert "C:/secret" not in response.text
    assert "Traceback" not in response.text


def test_request_validation_details_exclude_input_values(
    client: TestClient,
) -> None:
    sensitive_text = "PRIVATE_POST_CONTENT"

    response = client.post(
        "/translate",
        json={
            "text": sensitive_text,
            "source_language": "en",
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "request_validation_error",
            "message": "Request validation failed.",
            "details": [
                {
                    "field": "target_language",
                    "message": "Field required",
                    "type": "missing",
                }
            ],
        }
    }
    assert sensitive_text not in response.text
    assert "input" not in response.json()["error"]["details"][0]
