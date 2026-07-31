"""Integration tests for manual and automatic translation API flows."""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from tests.conftest import FakeLanguageDetectionService, FakeTranslationService


@pytest.mark.parametrize(
    ("payload", "expected_translation"),
    [
        (
            {
                "text": "Good morning",
                "source_language": "en",
                "target_language": "id",
            },
            "Selamat pagi",
        ),
        (
            {
                "text": "Selamat pagi",
                "source_language": "id",
                "target_language": "en",
            },
            "Good Morning",
        ),
    ],
)
def test_manual_translation_remains_backward_compatible(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
    fake_language_detection_service: FakeLanguageDetectionService,
    payload: dict[str, str],
    expected_translation: str,
) -> None:
    response = client.post("/translate", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "original_text": payload["text"],
        "translated_text": expected_translation,
        "source_language": payload["source_language"],
        "source_language_mode": "manual",
        "detected_language": None,
        "detection_confidence": None,
        "detection_confidence_margin": None,
        "target_language": payload["target_language"],
        "model_name": "fake/m2m100_418M",
        "device": "cpu",
        "status": "translated",
    }
    assert fake_translation_service.translate_calls == [
        (
            payload["text"],
            payload["source_language"],
            payload["target_language"],
        )
    ]
    assert fake_language_detection_service.detect_calls == []


@pytest.mark.parametrize(
    ("payload", "expected_source", "expected_translation"),
    [
        (
            {"text": "Сегодня хорошая погода", "target_language": "id"},
            "ru",
            "Cuacanya bagus hari ini",
        ),
        (
            {
                "text": "今日はいい天気です",
                "source_language": "auto",
                "target_language": "en",
            },
            "ja",
            "The weather is nice today",
        ),
        (
            {"text": "今天天气很好", "target_language": "id"},
            "zh",
            "Cuacanya sangat bagus hari ini",
        ),
        (
            {"text": "Good morning", "target_language": "id"},
            "en",
            "Selamat pagi",
        ),
        (
            {"text": "Selamat pagi", "target_language": "en"},
            "id",
            "Good Morning",
        ),
    ],
)
def test_automatic_translation_detects_source_once(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
    fake_language_detection_service: FakeLanguageDetectionService,
    payload: dict[str, str],
    expected_source: str,
    expected_translation: str,
) -> None:
    response = client.post("/translate", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["original_text"] == payload["text"]
    assert body["translated_text"] == expected_translation
    assert body["source_language"] == expected_source
    assert body["source_language_mode"] == "auto"
    assert body["detected_language"] == expected_source
    assert body["detection_confidence"] == 0.96
    assert body["detection_confidence_margin"] == 0.90
    assert body["target_language"] == payload["target_language"]
    assert body["status"] == "translated"
    assert fake_language_detection_service.detect_calls == [payload["text"]]
    assert fake_translation_service.translate_calls == [
        (payload["text"], expected_source, payload["target_language"])
    ]


def test_auto_same_language_returns_unchanged_without_semaphore(
    test_app: FastAPI,
    fake_translation_service: FakeTranslationService,
    fake_language_detection_service: FakeLanguageDetectionService,
) -> None:
    class FailingSemaphore:
        async def __aenter__(self) -> None:
            raise AssertionError("Same-language translation acquired the semaphore.")

        async def __aexit__(self, *_: object) -> None:
            return None

    with TestClient(test_app) as client:
        test_app.state.translation_semaphore = FailingSemaphore()
        response = client.post(
            "/translate",
            json={
                "text": "Selamat pagi, apa kabar hari ini?",
                "target_language": "id",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "unchanged"
    assert response.json()["translated_text"] == "Selamat pagi, apa kabar hari ini?"
    assert response.json()["source_language_mode"] == "auto"
    assert fake_language_detection_service.detect_calls == ["Selamat pagi, apa kabar hari ini?"]
    assert fake_translation_service.translate_calls == [
        ("Selamat pagi, apa kabar hari ini?", "id", "id")
    ]


def test_original_text_is_not_replaced_by_detection_cleaning(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
    fake_language_detection_service: FakeLanguageDetectionService,
) -> None:
    original_text = "🔥 @user Good morning #Jakarta https://example.com"

    response = client.post(
        "/translate",
        json={"text": original_text, "target_language": "id"},
    )

    assert response.status_code == 200
    assert fake_language_detection_service.detect_calls == [original_text]
    assert fake_translation_service.translate_calls == [(original_text, "en", "id")]
    assert response.json()["original_text"] == original_text


def test_manual_same_language_preserves_original_text(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
) -> None:
    original_text = "  Selamat pagi  "

    response = client.post(
        "/translate",
        json={
            "text": original_text,
            "source_language": " ID ",
            "target_language": "id",
        },
    )

    assert response.status_code == 200
    assert response.json()["original_text"] == original_text
    assert response.json()["translated_text"] == original_text
    assert response.json()["status"] == "unchanged"
    assert response.json()["source_language_mode"] == "manual"
    assert fake_translation_service.translate_calls == [(original_text, "id", "id")]


def test_manual_language_codes_are_trimmed_and_normalized(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
) -> None:
    response = client.post(
        "/translate",
        json={
            "text": "Good morning",
            "source_language": " EN ",
            "target_language": " ID ",
        },
    )

    assert response.status_code == 200
    assert response.json()["source_language"] == "en"
    assert response.json()["target_language"] == "id"
    assert fake_translation_service.translate_calls == [("Good morning", "en", "id")]


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "", "source_language": "en", "target_language": "id"},
        {"text": "   ", "source_language": "en", "target_language": "id"},
        {"source_language": "en", "target_language": "id"},
        {"text": "Good morning", "source_language": "en"},
        {"text": 123, "source_language": "en", "target_language": "id"},
        {"text": "Good morning", "source_language": "", "target_language": "id"},
        {"text": "Good morning", "source_language": "en", "target_language": "auto"},
    ],
)
def test_invalid_request_payload_returns_consistent_422(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
    payload: dict[str, Any],
) -> None:
    response = client.post("/translate", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_error"
    assert response.json()["error"]["message"] == "Request validation failed."
    assert isinstance(response.json()["error"]["details"], list)
    assert fake_translation_service.translate_calls == []


@pytest.mark.parametrize(
    ("source_language", "target_language", "unsupported_code"),
    [
        ("xx", "id", "xx"),
        ("en", "yy", "yy"),
    ],
)
def test_unsupported_manual_language_returns_422(
    client: TestClient,
    source_language: str,
    target_language: str,
    unsupported_code: str,
) -> None:
    response = client.post(
        "/translate",
        json={
            "text": "Test",
            "source_language": source_language,
            "target_language": target_language,
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_language"
    assert unsupported_code in response.json()["error"]["message"]


def test_unsupported_detected_language_does_not_call_translation(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
    fake_language_detection_service: FakeLanguageDetectionService,
) -> None:
    fake_language_detection_service.forced_language = "de"

    response = client.post(
        "/translate",
        json={"text": "Guten Morgen", "target_language": "id"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_language"
    assert fake_translation_service.translate_calls == []


def test_detection_failure_does_not_call_translation(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
    fake_language_detection_service: FakeLanguageDetectionService,
) -> None:
    fake_language_detection_service.error_mode = "uncertain"

    response = client.post(
        "/translate",
        json={"text": "Ambiguous words", "target_language": "id"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "language_detection_uncertain"
    assert fake_translation_service.translate_calls == []


def test_character_limit_is_checked_before_detection_and_translation(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
    fake_language_detection_service: FakeLanguageDetectionService,
) -> None:
    response = client.post(
        "/translate",
        json={"text": "x" * 101, "target_language": "id"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "text_too_large"
    assert fake_language_detection_service.detect_calls == []
    assert fake_translation_service.translate_calls == []


def test_engine_token_limit_still_returns_413_after_detection(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
) -> None:
    fake_translation_service.error_mode = "token_limit"

    response = client.post(
        "/translate",
        json={"text": "Token test", "target_language": "id"},
    )

    assert response.status_code == 413
    assert response.json()["error"] == {
        "code": "input_too_long",
        "message": "Input exceeds the maximum token limit.",
        "details": {"actual_tokens": 700, "maximum_tokens": 512},
    }


def test_model_not_loaded_returns_503(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
) -> None:
    fake_translation_service.is_loaded = False

    response = client.post(
        "/translate",
        json={
            "text": "Test",
            "source_language": "en",
            "target_language": "id",
        },
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_loaded"


def test_inference_failure_is_generic_and_safe(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
) -> None:
    fake_translation_service.error_mode = "inference"

    response = client.post(
        "/translate",
        json={
            "text": "Test",
            "source_language": "en",
            "target_language": "id",
        },
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "translation_failed"
    assert "PyTorch" not in response.text
    assert "TOP_SECRET" not in response.text
    assert "Traceback" not in response.text


def test_unexpected_failure_is_generic_and_safe(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
) -> None:
    fake_translation_service.error_mode = "unexpected"

    response = client.post(
        "/translate",
        json={
            "text": "Test",
            "source_language": "en",
            "target_language": "id",
        },
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_server_error"
    assert "TOP_SECRET" not in response.text
    assert ".py" not in response.text


def test_auto_detection_occurs_before_translation_semaphore(
    monkeypatch: MonkeyPatch,
    test_app: FastAPI,
    fake_language_detection_service: FakeLanguageDetectionService,
    fake_translation_service: FakeTranslationService,
) -> None:
    events: list[str] = []

    class RecordingSemaphore:
        async def __aenter__(self) -> None:
            events.append("semaphore_enter")

        async def __aexit__(self, *_: object) -> None:
            events.append("semaphore_exit")

    async def fake_run_in_threadpool(function: object, *arguments: object) -> object:
        if function == fake_language_detection_service.detect:
            events.append("detect")
        if function == fake_translation_service.translate:
            events.append("translate")
        return function(*arguments)  # type: ignore[operator]

    monkeypatch.setattr(
        "app.api.routes.translation.run_in_threadpool",
        fake_run_in_threadpool,
    )

    with TestClient(test_app) as client:
        test_app.state.translation_semaphore = RecordingSemaphore()
        response = client.post(
            "/translate",
            json={"text": "Good morning", "target_language": "id"},
        )

    assert response.status_code == 200
    assert events == ["detect", "semaphore_enter", "translate", "semaphore_exit"]


def test_languages_response_distinguishes_translation_and_detection_sets(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
    fake_language_detection_service: FakeLanguageDetectionService,
) -> None:
    response = client.get("/languages")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "fake/m2m100_418M",
        "count": 5,
        "languages": ["en", "id", "ja", "ru", "zh"],
        "language_detector": "lingua",
        "auto_detectable_count": 5,
        "auto_detectable_languages": ["en", "id", "ja", "ru", "zh"],
    }
    assert set(response.json()["auto_detectable_languages"]) <= set(response.json()["languages"])
    assert fake_translation_service.translate_calls == []
    assert fake_language_detection_service.detect_calls == []


def test_languages_requires_loaded_services(
    client: TestClient,
    fake_language_detection_service: FakeLanguageDetectionService,
) -> None:
    fake_language_detection_service.is_loaded = False

    response = client.get("/languages")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "language_detector_not_loaded"


def test_openapi_contains_auto_translation_contract(test_app: FastAPI) -> None:
    schema = test_app.openapi()

    assert {"/", "/health", "/languages", "/detect-language", "/translate"} <= set(schema["paths"])
    translate_operation = schema["paths"]["/translate"]["post"]
    assert {"200", "413", "422", "500", "503"} <= set(translate_operation["responses"])
    request_schema = schema["components"]["schemas"]["TranslationRequest"]
    assert request_schema["properties"]["source_language"]["default"] == "auto"
    assert "target_language" in request_schema["required"]
    assert "source_language" not in request_schema["required"]
