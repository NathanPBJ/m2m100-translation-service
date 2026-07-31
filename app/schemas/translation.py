"""Pydantic contracts for translation API endpoints."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class TranslationRequest(BaseModel):
    """Text and an automatic or manually supplied source language."""

    text: str
    source_language: str = "auto"
    target_language: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Сегодня хорошая погода",
                    "target_language": "id",
                }
            ]
        }
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Reject empty text without changing the original value."""
        if not value.strip():
            raise ValueError("Text must not be empty or contain only whitespace.")
        return value

    @field_validator("source_language")
    @classmethod
    def normalize_source_language(cls, value: str) -> str:
        normalized_value = value.strip().lower()
        if not normalized_value:
            raise ValueError("Source language code must not be empty.")
        return normalized_value

    @field_validator("target_language")
    @classmethod
    def normalize_target_language(cls, value: str) -> str:
        normalized_value = value.strip().lower()
        if not normalized_value:
            raise ValueError("Target language code must not be empty.")
        if normalized_value == "auto":
            raise ValueError("Target language code cannot be 'auto'.")
        return normalized_value


class TranslationResponse(BaseModel):
    """Structured result returned after translation."""

    original_text: str
    translated_text: str
    source_language: str
    source_language_mode: Literal["auto", "manual"]
    detected_language: str | None
    detection_confidence: float | None
    detection_confidence_margin: float | None
    target_language: str
    model_name: str
    device: str
    status: Literal["translated", "unchanged"]
    chunked: bool
    chunk_count: int
    chunk_token_limit: int

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    "original_text": "Сегодня хорошая погода",
                    "translated_text": "Cuacanya bagus hari ini",
                    "source_language": "ru",
                    "source_language_mode": "auto",
                    "detected_language": "ru",
                    "detection_confidence": 0.99,
                    "detection_confidence_margin": 0.98,
                    "target_language": "id",
                    "model_name": "facebook/m2m100_418M",
                    "device": "cpu",
                    "status": "translated",
                    "chunked": False,
                    "chunk_count": 1,
                    "chunk_token_limit": 400,
                }
            ]
        },
    )


class SupportedLanguagesResponse(BaseModel):
    """Language codes exposed by the currently loaded tokenizer."""

    model_name: str
    count: int
    languages: list[str]
    language_detector: str
    auto_detectable_count: int
    auto_detectable_languages: list[str]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model_name": "facebook/m2m100_418M",
                    "count": 5,
                    "languages": ["en", "id", "ja", "ru", "zh"],
                    "language_detector": "lingua",
                    "auto_detectable_count": 5,
                    "auto_detectable_languages": ["en", "id", "ja", "ru", "zh"],
                }
            ]
        }
    )
