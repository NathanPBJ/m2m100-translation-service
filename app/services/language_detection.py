"""Offline dominant-language detection backed by Lingua."""

import logging
import re
from collections.abc import Collection
from typing import Any

from lingua import Language, LanguageDetectorBuilder

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    InvalidLanguageDetectionInputError,
    LanguageDetectionError,
    LanguageDetectionInferenceError,
    LanguageDetectionUncertainError,
    LanguageDetectorLoadError,
    LanguageDetectorNotLoadedError,
)
from app.domain.language_detection import LanguageCandidate, LanguageDetectionResult

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+", flags=re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")

# Lingua distinguishes both Norwegian written standards, while M2M100 exposes
# one verified Norwegian code.
LANGUAGE_CODE_ALIASES = {
    "nb": "no",
    "nn": "no",
}


class LinguaLanguageDetectionService:
    """Detect one dominant language from the M2M100-compatible Lingua subset."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._detector: Any | None = None
        self._language_to_code: dict[Language, str] = {}
        self._supported_languages: tuple[str, ...] = ()

    @property
    def is_loaded(self) -> bool:
        """Return whether the reusable Lingua detector is ready."""
        return self._detector is not None

    @property
    def detector_name(self) -> str:
        """Return the stable detector identifier exposed by the API."""
        return "lingua"

    def load_detector(
        self,
        supported_translation_languages: Collection[str],
    ) -> None:
        """Build one detector from the intersection of Lingua and M2M100."""
        if self.is_loaded:
            return

        translation_codes = {
            code.strip().lower()
            for code in supported_translation_languages
            if isinstance(code, str) and code.strip()
        }
        language_to_code: dict[Language, str] = {}
        for language in Language.all():
            try:
                iso_code = language.iso_code_639_1.name.lower()
            except (AttributeError, ValueError):
                continue
            mapped_code = (
                iso_code if iso_code in translation_codes else LANGUAGE_CODE_ALIASES.get(iso_code)
            )
            if mapped_code in translation_codes:
                language_to_code[language] = mapped_code

        supported_codes = tuple(sorted(set(language_to_code.values())))
        if len(language_to_code) < 2:
            raise LanguageDetectorLoadError(
                "At least two compatible languages are required to build the detector."
            )
        if self._settings.language_detection_max_candidates > len(supported_codes):
            raise LanguageDetectorLoadError(
                "LANGUAGE_DETECTION_MAX_CANDIDATES cannot exceed the number "
                "of auto-detectable languages."
            )

        logger.info(
            "Building %s detector with %d Lingua languages and %d unique M2M100 codes",
            self.detector_name,
            len(language_to_code),
            len(supported_codes),
        )
        try:
            detector = (
                LanguageDetectorBuilder.from_languages(*language_to_code)
                .with_minimum_relative_distance(
                    self._settings.language_detection_min_relative_distance
                )
                .build()
            )
        except Exception as exc:
            logger.exception("Failed to build the Lingua language detector")
            raise LanguageDetectorLoadError(
                "The local language detector could not be loaded."
            ) from exc

        self._detector = detector
        self._language_to_code = language_to_code
        self._supported_languages = supported_codes
        logger.info(
            "Language detector loaded successfully with %d auto-detectable languages",
            len(supported_codes),
        )

    def unload_detector(self) -> None:
        """Release the detector reference without touching installed package files."""
        self._detector = None
        self._language_to_code = {}
        self._supported_languages = ()
        logger.info("Language detector unloaded successfully")

    def get_supported_languages(self) -> tuple[str, ...]:
        """Return sorted language codes supported by both Lingua and M2M100."""
        self._ensure_loaded()
        return self._supported_languages

    def supports_language(self, language_code: str) -> bool:
        """Return whether a normalized code can be auto-detected."""
        self._ensure_loaded()
        if not isinstance(language_code, str):
            return False
        return language_code.strip().lower() in self._supported_languages

    def detect(self, text: str) -> LanguageDetectionResult:
        """Detect a dominant language with confidence and margin validation."""
        self._ensure_loaded()
        if not isinstance(text, str) or not text.strip():
            raise InvalidLanguageDetectionInputError(
                "Language detection text must be a non-empty string."
            )

        detection_text = self._prepare_detection_text(text)
        alphabetic_count = sum(character.isalpha() for character in detection_text)
        if alphabetic_count < self._settings.language_detection_min_alphabetic_characters:
            raise InvalidLanguageDetectionInputError(
                "The text does not contain enough detectable language content."
            )

        try:
            confidence_values = self._detector.compute_language_confidence_values(detection_text)
            candidates = self._build_candidates(confidence_values)
            if not candidates:
                raise LanguageDetectionUncertainError(
                    "The source language could not be detected reliably."
                )

            top_candidate = candidates[0]
            confidence_margin = (
                top_candidate.confidence - candidates[1].confidence
                if len(candidates) > 1
                else top_candidate.confidence
            )
            if (
                top_candidate.confidence < self._settings.language_detection_min_confidence
                or confidence_margin < self._settings.language_detection_min_relative_distance
            ):
                raise LanguageDetectionUncertainError(
                    "The source language could not be detected reliably."
                )
        except LanguageDetectionError:
            raise
        except Exception as exc:
            logger.exception("Unexpected failure during local language detection")
            raise LanguageDetectionInferenceError(
                "Language detection could not be completed."
            ) from exc

        return LanguageDetectionResult(
            language=top_candidate.language,
            confidence=top_candidate.confidence,
            confidence_margin=confidence_margin,
            candidates=candidates,
            detector_name=self.detector_name,
            status="detected",
        )

    def _build_candidates(
        self, confidence_values: Collection[Any]
    ) -> tuple[LanguageCandidate, ...]:
        best_confidence_by_code: dict[str, float] = {}
        for confidence_value in confidence_values:
            code = self._language_to_code.get(confidence_value.language)
            if code is None:
                continue
            confidence = float(confidence_value.value)
            if not 0.0 <= confidence <= 1.0:
                raise LanguageDetectionInferenceError(
                    "Language detection returned an invalid confidence value."
                )
            best_confidence_by_code[code] = max(
                confidence,
                best_confidence_by_code.get(code, 0.0),
            )

        ordered_candidates = sorted(
            best_confidence_by_code.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return tuple(
            LanguageCandidate(language=code, confidence=confidence)
            for code, confidence in ordered_candidates[
                : self._settings.language_detection_max_candidates
            ]
        )

    @staticmethod
    def _prepare_detection_text(text: str) -> str:
        without_urls = URL_PATTERN.sub(" ", text)
        without_mentions = MENTION_PATTERN.sub(" ", without_urls)
        without_hashtag_symbols = without_mentions.replace("#", "")
        return WHITESPACE_PATTERN.sub(" ", without_hashtag_symbols).strip()

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise LanguageDetectorNotLoadedError(
                "The language detector is not loaded. Call load_detector() first."
            )
