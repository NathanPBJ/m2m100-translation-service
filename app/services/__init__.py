"""Application service layer."""

from app.services.language_detection import LinguaLanguageDetectionService
from app.services.translation import M2M100TranslationService

__all__ = ["LinguaLanguageDetectionService", "M2M100TranslationService"]
