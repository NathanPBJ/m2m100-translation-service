"""Shared fixtures and a lightweight translation service fake for API tests."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.exceptions import (
    InputTooLongError,
    InvalidTranslationInputError,
    ModelLoadError,
    ModelNotLoadedError,
    TranslationInferenceError,
    UnsupportedLanguageError,
)
from app.domain.translation import TranslationResult
from app.main import create_app


class FakeTranslationService:
    """Configurable service fake that never imports or downloads model files."""

    def __init__(self) -> None:
        self.is_loaded = False
        self.model_name = "fake/m2m100_418M"
        self.device = "cpu"
        self.load_calls = 0
        self.unload_calls = 0
        self.translate_calls: list[tuple[str, str, str]] = []
        self.supported_language_calls = 0
        self.load_failure = False
        self.error_mode: str | None = None
        self.languages = ("en", "id", "ja", "ru", "zh")

    def load_model(self) -> None:
        self.load_calls += 1
        if self.load_failure:
            raise ModelLoadError("Fake startup model failure.")
        self.is_loaded = True

    def unload_model(self) -> None:
        self.unload_calls += 1
        self.is_loaded = False

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        self.translate_calls.append((text, source_language, target_language))
        if not self.is_loaded:
            raise ModelNotLoadedError("The fake translation model is not loaded.")
        if self.error_mode == "invalid":
            raise InvalidTranslationInputError("Translation text is invalid.")
        if self.error_mode == "token_limit":
            raise InputTooLongError(actual_tokens=700, maximum_tokens=512)
        if self.error_mode == "inference":
            raise TranslationInferenceError("raw PyTorch failure with TOP_SECRET")
        if self.error_mode == "unexpected":
            raise RuntimeError("raw internal model object TOP_SECRET")
        if source_language == "auto":
            raise UnsupportedLanguageError("Language code 'auto' is not supported.")
        if source_language not in self.languages:
            raise UnsupportedLanguageError(
                f"Source language code '{source_language}' is not supported."
            )
        if target_language not in self.languages:
            raise UnsupportedLanguageError(
                f"Target language code '{target_language}' is not supported."
            )

        if source_language == target_language:
            translated_text = text
            status = "unchanged"
        else:
            translated_text = {
                ("Good morning", "en", "id"): "Selamat pagi",
                ("Selamat pagi", "id", "en"): "Good Morning",
                ("Thank you", "en", "ja"): "ありがとう",
            }.get((text, source_language, target_language), f"Translated: {text}")
            status = "translated"

        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language,
            model_name=self.model_name,
            device=self.device,
            status=status,
        )

    def get_supported_languages(self) -> tuple[str, ...]:
        self.supported_language_calls += 1
        if not self.is_loaded:
            raise ModelNotLoadedError("The fake translation model is not loaded.")
        return self.languages


@pytest.fixture
def api_settings() -> Settings:
    """Return API settings with small limits that are easy to test."""
    return Settings(
        model_device="cpu",
        model_local_files_only=True,
        api_max_text_characters=20,
        translation_max_concurrency=2,
    )


@pytest.fixture
def fake_translation_service() -> FakeTranslationService:
    return FakeTranslationService()


@pytest.fixture
def test_app(
    fake_translation_service: FakeTranslationService,
    api_settings: Settings,
) -> FastAPI:
    return create_app(
        translation_service=fake_translation_service,  # type: ignore[arg-type]
        settings=api_settings,
    )


@pytest.fixture
def client(test_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(test_app, raise_server_exceptions=False) as test_client:
        yield test_client
