"""Integration tests for the language detection API."""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from tests.conftest import FakeLanguageDetectionService, FakeTranslationService


@pytest.mark.parametrize(
    ("text", "expected_language"),
    [
        ("Сегодня хорошая погода", "ru"),
        ("今日はいい天気です", "ja"),
        ("今天天气很好", "zh"),
        ("Good morning, how are you today?", "en"),
        ("Selamat pagi, apa kabar hari ini?", "id"),
    ],
)
def test_detection_success_for_supported_samples(
    client: TestClient,
    fake_language_detection_service: FakeLanguageDetectionService,
    text: str,
    expected_language: str,
) -> None:
    response = client.post("/detect-language", json={"text": text})

    assert response.status_code == 200
    body = response.json()
    assert body["language"] == expected_language
    assert body["confidence"] == 0.96
    assert body["confidence_margin"] == 0.90
    assert body["detector"] == "lingua"
    assert body["status"] == "detected"
    assert body["candidates"][0]["language"] == expected_language
    assert [candidate["confidence"] for candidate in body["candidates"]] == sorted(
        [candidate["confidence"] for candidate in body["candidates"]],
        reverse=True,
    )
    assert fake_language_detection_service.detect_calls == [text]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "   "},
        {"text": 123},
    ],
)
def test_detection_request_validation_returns_422(
    client: TestClient,
    payload: dict[str, Any],
) -> None:
    response = client.post("/detect-language", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_error"


def test_detection_character_limit_returns_413(
    client: TestClient,
    fake_language_detection_service: FakeLanguageDetectionService,
) -> None:
    response = client.post("/detect-language", json={"text": "x" * 10_001})

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "text_too_large"
    assert fake_language_detection_service.detect_calls == []


@pytest.mark.parametrize("text", ["🔥🔥🔥", "https://example.com", "@username"])
def test_invalid_detection_content_returns_422(
    client: TestClient,
    fake_language_detection_service: FakeLanguageDetectionService,
    text: str,
) -> None:
    fake_language_detection_service.error_mode = "invalid"

    response = client.post("/detect-language", json={"text": text})

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "invalid_detection_input",
            "message": "The text does not contain enough detectable language content.",
        }
    }


def test_uncertain_detection_returns_422(
    client: TestClient,
    fake_language_detection_service: FakeLanguageDetectionService,
) -> None:
    fake_language_detection_service.error_mode = "uncertain"

    response = client.post("/detect-language", json={"text": "Ambiguous words"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "language_detection_uncertain"


def test_detector_not_loaded_returns_503(
    client: TestClient,
    fake_language_detection_service: FakeLanguageDetectionService,
) -> None:
    fake_language_detection_service.is_loaded = False

    response = client.post("/detect-language", json={"text": "Good morning"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "language_detector_not_loaded"
    assert fake_language_detection_service.detect_calls == []


def test_internal_detection_failure_is_generic_and_safe(
    client: TestClient,
    fake_language_detection_service: FakeLanguageDetectionService,
) -> None:
    fake_language_detection_service.error_mode = "internal"

    response = client.post("/detect-language", json={"text": "Good morning"})

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "language_detection_failed",
            "message": "Language detection could not be completed.",
        }
    }
    assert "Lingua" not in response.text
    assert "TOP_SECRET" not in response.text
    assert "Traceback" not in response.text


def test_detection_runs_once_in_threadpool_without_translation(
    monkeypatch: MonkeyPatch,
    test_app: FastAPI,
    fake_language_detection_service: FakeLanguageDetectionService,
    fake_translation_service: FakeTranslationService,
) -> None:
    threadpool_calls: list[tuple[object, tuple[object, ...]]] = []

    async def fake_run_in_threadpool(function: object, *arguments: object) -> object:
        threadpool_calls.append((function, arguments))
        return function(*arguments)  # type: ignore[operator]

    monkeypatch.setattr(
        "app.api.routes.detection.run_in_threadpool",
        fake_run_in_threadpool,
    )

    with TestClient(test_app) as test_client:
        response = test_client.post(
            "/detect-language",
            json={"text": "Good morning"},
        )

    assert response.status_code == 200
    assert threadpool_calls == [(fake_language_detection_service.detect, ("Good morning",))]
    assert fake_language_detection_service.detect_calls == ["Good morning"]
    assert fake_translation_service.translate_calls == []


def test_detection_endpoint_is_documented(test_app: FastAPI) -> None:
    schema = test_app.openapi()

    assert "/detect-language" in schema["paths"]
    operation = schema["paths"]["/detect-language"]["post"]
    assert {"200", "413", "422", "500", "503"} <= set(operation["responses"])
