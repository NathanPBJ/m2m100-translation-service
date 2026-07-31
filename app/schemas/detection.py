"""Pydantic contracts for language detection."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class LanguageDetectionRequest(BaseModel):
    """Text whose dominant language should be detected."""

    text: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Сегодня хорошая погода",
                }
            ]
        }
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be empty or contain only whitespace.")
        return value


class LanguageCandidateResponse(BaseModel):
    """One candidate returned by the local detector."""

    language: str
    confidence: float

    model_config = ConfigDict(from_attributes=True)


class LanguageDetectionResponse(BaseModel):
    """Successful language detection response."""

    language: str
    confidence: float
    confidence_margin: float
    detector: str
    status: Literal["detected"]
    candidates: list[LanguageCandidateResponse]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "language": "ru",
                    "confidence": 0.99,
                    "confidence_margin": 0.98,
                    "detector": "lingua",
                    "status": "detected",
                    "candidates": [
                        {"language": "ru", "confidence": 0.99},
                        {"language": "uk", "confidence": 0.01},
                    ],
                }
            ]
        }
    )
