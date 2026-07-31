"""Tests for application lifespan, health, and existing endpoints."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.exceptions import LanguageDetectorLoadError, ModelLoadError
from app.main import app as module_app
from app.main import create_app
from tests.conftest import FakeLanguageDetectionService, FakeTranslationService


def test_importing_module_application_loads_neither_service() -> None:
    assert not hasattr(module_app.state, "translation_service")
    assert not hasattr(module_app.state, "language_detection_service")


def test_lifespan_loads_in_order_stores_state_and_unloads() -> None:
    translation_service = FakeTranslationService()
    detection_service = FakeLanguageDetectionService()
    settings = Settings(
        model_device="cpu",
        model_local_files_only=True,
        translation_max_concurrency=3,
    )
    application = create_app(
        translation_service=translation_service,  # type: ignore[arg-type]
        language_detection_service=detection_service,  # type: ignore[arg-type]
        settings=settings,
    )

    assert translation_service.load_calls == 0
    assert detection_service.load_calls == 0
    with TestClient(application) as client:
        assert translation_service.load_calls == 1
        assert detection_service.load_calls == 1
        assert detection_service.load_languages == [translation_service.languages]
        assert application.state.translation_service is translation_service
        assert application.state.language_detection_service is detection_service
        assert isinstance(application.state.translation_semaphore, asyncio.Semaphore)
        assert application.state.translation_semaphore._value == 3

        client.get("/")
        client.get("/health")
        client.get("/languages")
        client.post("/detect-language", json={"text": "Good morning"})
        assert translation_service.load_calls == 1
        assert detection_service.load_calls == 1

    assert detection_service.unload_calls == 1
    assert translation_service.unload_calls == 1
    assert detection_service.is_loaded is False
    assert translation_service.is_loaded is False


def test_model_startup_failure_is_not_hidden(api_settings: Settings) -> None:
    translation_service = FakeTranslationService()
    detection_service = FakeLanguageDetectionService()
    translation_service.load_failure = True
    application = create_app(
        translation_service=translation_service,  # type: ignore[arg-type]
        language_detection_service=detection_service,  # type: ignore[arg-type]
        settings=api_settings,
    )

    with pytest.raises(ModelLoadError, match="Fake startup model failure"):
        with TestClient(application):
            pass

    assert translation_service.load_calls == 1
    assert detection_service.load_calls == 0
    assert detection_service.unload_calls == 1
    assert translation_service.unload_calls == 1


def test_detector_startup_failure_cleans_translation_model(
    api_settings: Settings,
) -> None:
    translation_service = FakeTranslationService()
    detection_service = FakeLanguageDetectionService()
    detection_service.load_failure = True
    application = create_app(
        translation_service=translation_service,  # type: ignore[arg-type]
        language_detection_service=detection_service,  # type: ignore[arg-type]
        settings=api_settings,
    )

    with pytest.raises(LanguageDetectorLoadError, match="Fake detector startup failure"):
        with TestClient(application):
            pass

    assert translation_service.load_calls == 1
    assert detection_service.load_calls == 1
    assert detection_service.unload_calls == 1
    assert translation_service.unload_calls == 1
    assert translation_service.is_loaded is False


def test_root_health_and_documentation_remain_available(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
    fake_language_detection_service: FakeLanguageDetectionService,
) -> None:
    root_response = client.get("/")
    health_response = client.get("/health")

    assert root_response.status_code == 200
    assert root_response.json() == {
        "service": "M2M100 Translation Service",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }
    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "healthy",
        "service": "M2M100 Translation Service",
        "version": "0.1.0",
        "environment": "development",
        "model_loaded": True,
        "model_name": "fake/m2m100_418M",
        "model_device": "cpu",
        "language_detector_loaded": True,
        "language_detector_name": "lingua",
        "auto_detectable_language_count": 5,
        "long_text_chunking_enabled": True,
        "long_text_chunk_token_limit": 400,
        "long_text_max_chunks": 64,
    }
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    assert fake_translation_service.translate_calls == []
    assert fake_language_detection_service.detect_calls == []


def test_health_requires_both_loaded_services(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
    fake_language_detection_service: FakeLanguageDetectionService,
) -> None:
    fake_language_detection_service.is_loaded = False

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "language_detector_not_loaded"

    fake_language_detection_service.is_loaded = True
    fake_translation_service.is_loaded = False
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_loaded"
