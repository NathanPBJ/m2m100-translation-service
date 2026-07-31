"""Domain types returned by the translation engine."""

from dataclasses import dataclass
from typing import Literal

TranslationStatus = Literal["translated", "unchanged"]


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """Immutable result produced by a translation operation."""

    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    model_name: str
    device: str
    status: TranslationStatus
