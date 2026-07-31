"""Immutable domain objects produced by language detection."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class LanguageCandidate:
    """One language candidate and its normalized confidence."""

    language: str
    confidence: float


@dataclass(frozen=True, slots=True)
class LanguageDetectionResult:
    """Successful dominant-language detection result."""

    language: str
    confidence: float
    confidence_margin: float
    candidates: tuple[LanguageCandidate, ...]
    detector_name: str
    status: Literal["detected"]
