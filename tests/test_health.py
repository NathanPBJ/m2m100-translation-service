"""Tests for application lifespan and existing service endpoints."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.exceptions import ModelLoadError
from app.main import app as module_app
from app.main import create_app
from tests.conftest import FakeTranslationService


def test_importing_module_application_does_not_load_model() -> None:
    assert not hasattr(module_app.state, "translation_service")


def test_lifespan_loads_once_stores_state_and_unloads() -> None:
    service = FakeTranslationService()
    settings = Settings(
        model_device="cpu",
        model_local_files_only=True,
        translation_max_concurrency=3,
    )
    application = create_app(
        translation_service=service,  # type: ignore[arg-type]
        settings=settings,
    )

    assert service.load_calls == 0
    with TestClient(application) as client:
        assert service.load_calls == 1
        assert service.unload_calls == 0
        assert application.state.translation_service is service
        assert isinstance(application.state.translation_semaphore, asyncio.Semaphore)
        assert application.state.translation_semaphore._value == 3

        client.get("/")
        client.get("/health")
        client.get("/languages")
        assert service.load_calls == 1

    assert service.unload_calls == 1
    assert service.is_loaded is False


def test_startup_failure_is_not_hidden(api_settings: Settings) -> None:
    service = FakeTranslationService()
    service.load_failure = True
    application = create_app(
        translation_service=service,  # type: ignore[arg-type]
        settings=api_settings,
    )

    with pytest.raises(ModelLoadError, match="Fake startup model failure"):
        with TestClient(application):
            pass

    assert service.load_calls == 1
    assert service.unload_calls == 1


def test_root_health_and_documentation_remain_available(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
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
    }
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    assert fake_translation_service.translate_calls == []


def test_health_returns_service_unavailable_when_model_is_not_loaded(
    client: TestClient,
    fake_translation_service: FakeTranslationService,
) -> None:
    fake_translation_service.is_loaded = False

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_loaded"
