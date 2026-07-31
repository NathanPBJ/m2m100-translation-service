"""Pydantic contracts for translation API endpoints."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class TranslationRequest(BaseModel):
    """Text and explicit language pair supplied by an API client."""

    text: str
    source_language: str
    target_language: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Good morning",
                    "source_language": "en",
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

    @field_validator("source_language", "target_language")
    @classmethod
    def normalize_language_code(cls, value: str) -> str:
        """Trim and lowercase language codes while rejecting empty values."""
        normalized_value = value.strip().lower()
        if not normalized_value:
            raise ValueError("Language code must not be empty.")
        if normalized_value == "auto":
            raise ValueError(
                "Language code 'auto' is not supported because automatic detection "
                "is not implemented."
            )
        return normalized_value


class TranslationResponse(BaseModel):
    """Structured result returned after translation."""

    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    model_name: str
    device: str
    status: Literal["translated", "unchanged"]

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    "original_text": "Good morning",
                    "translated_text": "Selamat pagi",
                    "source_language": "en",
                    "target_language": "id",
                    "model_name": "facebook/m2m100_418M",
                    "device": "cpu",
                    "status": "translated",
                }
            ]
        },
    )


class SupportedLanguagesResponse(BaseModel):
    """Language codes exposed by the currently loaded tokenizer."""

    model_name: str
    count: int
    languages: list[str]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model_name": "facebook/m2m100_418M",
                    "count": 5,
                    "languages": ["en", "id", "ja", "ru", "zh"],
                }
            ]
        }
    )
