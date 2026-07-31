"""Integration tests for translation and supported-language API routes."""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from tests.conftest import FakeTranslationService


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
def test_translation_success(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
    payload: dict[str, str],
    expected_translation: str,
) -> None:
    response = client.post("/translate", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "original_text": payload["text"],
        "translated_text": expected_translation,
        "source_language": payload["source_language"],
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


def test_same_language_preserves_original_text(
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
    assert fake_translation_service.translate_calls == [(original_text, "id", "id")]


def test_language_codes_are_trimmed_and_normalized(
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
        {"text": "Good morning", "target_language": "id"},
        {"text": "Good morning", "source_language": "en"},
        {"text": 123, "source_language": "en", "target_language": "id"},
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


def test_auto_source_language_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/translate",
        json={
            "text": "Good morning",
            "source_language": "auto",
            "target_language": "id",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_error"


@pytest.mark.parametrize(
    ("source_language", "target_language", "unsupported_code"),
    [
        ("xx", "id", "xx"),
        ("en", "yy", "yy"),
    ],
)
def test_unsupported_language_returns_422(
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


def test_character_limit_is_checked_before_service_call(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
) -> None:
    response = client.post(
        "/translate",
        json={
            "text": "x" * 21,
            "source_language": "en",
            "target_language": "id",
        },
    )

    assert response.status_code == 413
    assert response.json() == {
        "error": {
            "code": "text_too_large",
            "message": "Input contains 21 characters; the maximum is 20.",
            "details": {
                "actual_characters": 21,
                "maximum_characters": 20,
            },
        }
    }
    assert fake_translation_service.translate_calls == []


def test_engine_token_limit_returns_413(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
) -> None:
    fake_translation_service.error_mode = "token_limit"

    response = client.post(
        "/translate",
        json={"text": "Token test", "source_language": "en", "target_language": "id"},
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
        json={"text": "Test", "source_language": "en", "target_language": "id"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_loaded"
    assert fake_translation_service.translate_calls == []


def test_inference_failure_is_generic_and_safe(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
) -> None:
    fake_translation_service.error_mode = "inference"

    response = client.post(
        "/translate",
        json={"text": "Test", "source_language": "en", "target_language": "id"},
    )
    response_text = response.text

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "translation_failed",
            "message": "Translation could not be completed.",
        }
    }
    assert "PyTorch" not in response_text
    assert "TOP_SECRET" not in response_text
    assert "Traceback" not in response_text


def test_unexpected_failure_is_generic_and_safe(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
) -> None:
    fake_translation_service.error_mode = "unexpected"

    response = client.post(
        "/translate",
        json={"text": "Test", "source_language": "en", "target_language": "id"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_server_error",
            "message": "An unexpected internal error occurred.",
        }
    }
    assert "TOP_SECRET" not in response.text
    assert ".py" not in response.text
    assert "model object" not in response.text


def test_translation_runs_through_threadpool(
    monkeypatch: MonkeyPatch,
    test_app: FastAPI,
    fake_translation_service: FakeTranslationService,
) -> None:
    threadpool_calls: list[tuple[object, tuple[object, ...]]] = []

    async def fake_run_in_threadpool(function: object, *arguments: object) -> object:
        threadpool_calls.append((function, arguments))
        return function(*arguments)  # type: ignore[operator]

    monkeypatch.setattr(
        "app.api.routes.translation.run_in_threadpool",
        fake_run_in_threadpool,
    )

    with TestClient(test_app) as test_client:
        response = test_client.post(
            "/translate",
            json={
                "text": "Good morning",
                "source_language": "en",
                "target_language": "id",
            },
        )

    assert response.status_code == 200
    assert threadpool_calls == [
        (
            fake_translation_service.translate,
            ("Good morning", "en", "id"),
        )
    ]


def test_languages_response_uses_loaded_service(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
) -> None:
    response = client.get("/languages")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "fake/m2m100_418M",
        "count": 5,
        "languages": ["en", "id", "ja", "ru", "zh"],
    }
    assert response.json()["languages"] == sorted(set(response.json()["languages"]))
    assert fake_translation_service.supported_language_calls == 1
    assert fake_translation_service.translate_calls == []
    assert fake_translation_service.load_calls == 1


def test_languages_requires_loaded_model(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
) -> None:
    fake_translation_service.is_loaded = False

    response = client.get("/languages")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_loaded"
    assert fake_translation_service.supported_language_calls == 0


def test_openapi_contains_translation_and_language_contracts(test_app: FastAPI) -> None:
    schema = test_app.openapi()

    assert {"/", "/health", "/languages", "/translate"} <= set(schema["paths"])
    translate_operation = schema["paths"]["/translate"]["post"]
    assert "200" in translate_operation["responses"]
    assert {"413", "422", "500", "503"} <= set(translate_operation["responses"])
    assert "requestBody" in translate_operation
    assert "get" in schema["paths"]["/languages"]
