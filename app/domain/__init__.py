"""Domain objects used by the application service layer."""

from app.domain.language_detection import LanguageCandidate, LanguageDetectionResult
from app.domain.text_chunking import TextChunk, TextChunkingResult
from app.domain.translation import TranslationResult, TranslationStatus

__all__ = [
    "LanguageCandidate",
    "LanguageDetectionResult",
    "TextChunk",
    "TextChunkingResult",
    "TranslationResult",
    "TranslationStatus",
]
