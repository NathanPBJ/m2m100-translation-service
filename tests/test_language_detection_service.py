"""Unit tests for the offline Lingua language detection service."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import pytest
from lingua import Language
from pydantic import ValidationError
from pytest import MonkeyPatch

from app.core.config import Settings
from app.core.exceptions import (
    InvalidLanguageDetectionInputError,
    LanguageDetectionInferenceError,
    LanguageDetectionUncertainError,
    LanguageDetectorLoadError,
    LanguageDetectorNotLoadedError,
)
from app.services.language_detection import LinguaLanguageDetectionService


class FakeDetector:
    """Detector fake with configurable confidence values and failures."""

    def __init__(
        self,
        values: list[object] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.values = values or []
        self.error = error
        self.inputs: list[str] = []

    def compute_language_confidence_values(self, text: str) -> list[object]:
        self.inputs.append(text)
        if self.error is not None:
            raise self.error
        return self.values


def confidence(language: Language, value: float) -> object:
    return SimpleNamespace(language=language, value=value)


def detection_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "model_device": "cpu",
        "model_local_files_only": True,
        "language_detection_min_confidence": 0.30,
        "language_detection_min_relative_distance": 0.05,
        "language_detection_min_alphabetic_characters": 3,
        "language_detection_max_candidates": 3,
    }
    values.update(overrides)
    return Settings(**values)


def install_fake_builder(
    monkeypatch: MonkeyPatch,
    detector: FakeDetector | None = None,
    build_error: Exception | None = None,
) -> tuple[FakeDetector, dict[str, object]]:
    fake_detector = detector or FakeDetector(
        [
            confidence(Language.ENGLISH, 0.90),
            confidence(Language.INDONESIAN, 0.07),
            confidence(Language.RUSSIAN, 0.03),
        ]
    )
    state: dict[str, object] = {
        "from_languages_calls": [],
        "relative_distances": [],
        "build_calls": 0,
    }

    class FakeBuilder:
        @classmethod
        def from_languages(cls, *languages: Language) -> "FakeBuilder":
            state["from_languages_calls"].append(languages)  # type: ignore[union-attr]
            return cls()

        def with_minimum_relative_distance(self, value: float) -> "FakeBuilder":
            state["relative_distances"].append(value)  # type: ignore[union-attr]
            return self

        def build(self) -> FakeDetector:
            state["build_calls"] = int(state["build_calls"]) + 1
            if build_error is not None:
                raise build_error
            return fake_detector

    monkeypatch.setattr(
        "app.services.language_detection.LanguageDetectorBuilder",
        FakeBuilder,
    )
    return fake_detector, state


def load_fake_service(
    monkeypatch: MonkeyPatch,
    *,
    detector: FakeDetector | None = None,
    settings: Settings | None = None,
    supported_languages: set[str] | None = None,
) -> tuple[LinguaLanguageDetectionService, FakeDetector, dict[str, object]]:
    fake_detector, state = install_fake_builder(monkeypatch, detector)
    service = LinguaLanguageDetectionService(settings or detection_settings())
    service.load_detector(supported_languages or {"en", "id", "ru"})
    return service, fake_detector, state


def test_constructor_is_lazy_and_has_stable_name() -> None:
    service = LinguaLanguageDetectionService(detection_settings())

    assert service.is_loaded is False
    assert service.detector_name == "lingua"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("language_detection_min_confidence", -0.01),
        ("language_detection_min_confidence", 1.01),
        ("language_detection_min_relative_distance", -0.01),
        ("language_detection_min_relative_distance", 1.0),
        ("language_detection_min_alphabetic_characters", 0),
        ("language_detection_max_candidates", 0),
    ],
)
def test_detection_settings_validate_ranges(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        detection_settings(**{field: value})


def test_load_uses_only_m2m100_intersection_and_is_idempotent(
    monkeypatch: MonkeyPatch,
) -> None:
    service, _, state = load_fake_service(monkeypatch)

    service.load_detector({"de", "fr", "en", "id", "ru"})

    loaded_languages = state["from_languages_calls"][0]  # type: ignore[index]
    assert set(loaded_languages) == {
        Language.ENGLISH,
        Language.INDONESIAN,
        Language.RUSSIAN,
    }
    assert Language.GERMAN not in loaded_languages
    assert service.get_supported_languages() == ("en", "id", "ru")
    assert state["relative_distances"] == [0.05]
    assert state["build_calls"] == 1


def test_supports_language_normalizes_codes(monkeypatch: MonkeyPatch) -> None:
    service, _, _ = load_fake_service(monkeypatch)

    assert service.supports_language(" EN ") is True
    assert service.supports_language("de") is False
    assert isinstance(service.get_supported_languages(), tuple)


def test_verified_norwegian_aliases_are_deduplicated(
    monkeypatch: MonkeyPatch,
) -> None:
    detector = FakeDetector(
        [
            confidence(Language.NYNORSK, 0.60),
            confidence(Language.BOKMAL, 0.35),
            confidence(Language.ENGLISH, 0.05),
        ]
    )
    service, _, state = load_fake_service(
        monkeypatch,
        detector=detector,
        settings=detection_settings(language_detection_max_candidates=2),
        supported_languages={"en", "no"},
    )

    result = service.detect("Dette er norsk tekst")

    loaded_languages = set(state["from_languages_calls"][0])  # type: ignore[index]
    assert {Language.BOKMAL, Language.NYNORSK, Language.ENGLISH} == loaded_languages
    assert service.get_supported_languages() == ("en", "no")
    assert [candidate.language for candidate in result.candidates] == ["no", "en"]
    assert result.candidates[0].confidence == 0.60


def test_load_failure_is_wrapped(monkeypatch: MonkeyPatch) -> None:
    install_fake_builder(monkeypatch, build_error=RuntimeError("fake Rust failure"))
    service = LinguaLanguageDetectionService(detection_settings())

    with pytest.raises(LanguageDetectorLoadError) as error:
        service.load_detector({"en", "id", "ru"})

    assert isinstance(error.value.__cause__, RuntimeError)
    assert service.is_loaded is False


def test_candidate_setting_cannot_exceed_available_languages(
    monkeypatch: MonkeyPatch,
) -> None:
    install_fake_builder(monkeypatch)
    service = LinguaLanguageDetectionService(detection_settings())

    with pytest.raises(LanguageDetectorLoadError, match="MAX_CANDIDATES"):
        service.load_detector({"en", "id"})


def test_detect_requires_loaded_detector() -> None:
    service = LinguaLanguageDetectionService(detection_settings())

    with pytest.raises(LanguageDetectorNotLoadedError):
        service.detect("Good morning")


@pytest.mark.parametrize("text", [None, 123, "", "   "])
def test_invalid_basic_input_is_rejected(text: object) -> None:
    service = LinguaLanguageDetectionService(detection_settings())
    service._detector = FakeDetector()  # noqa: SLF001

    with pytest.raises(InvalidLanguageDetectionInputError):
        service.detect(text)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "text",
    [
        "🔥🔥🔥",
        "https://example.com",
        "www.example.com/path",
        "@username",
        "12!",
    ],
)
def test_input_without_enough_alphabetic_content_is_rejected(
    monkeypatch: MonkeyPatch,
    text: str,
) -> None:
    service, _, _ = load_fake_service(monkeypatch)

    with pytest.raises(InvalidLanguageDetectionInputError, match="enough detectable"):
        service.detect(text)


def test_detection_cleaning_removes_urls_mentions_and_keeps_hashtag_word(
    monkeypatch: MonkeyPatch,
) -> None:
    service, detector, _ = load_fake_service(monkeypatch)
    original_text = "🔥 @user Hello #World https://example.com"

    service.detect(original_text)

    assert detector.inputs == ["🔥 Hello World"]
    assert original_text == "🔥 @user Hello #World https://example.com"


def test_non_latin_letters_are_counted_as_alphabetic(
    monkeypatch: MonkeyPatch,
) -> None:
    detector = FakeDetector(
        [
            confidence(Language.RUSSIAN, 0.95),
            confidence(Language.ENGLISH, 0.03),
            confidence(Language.INDONESIAN, 0.02),
        ]
    )
    service, _, _ = load_fake_service(monkeypatch, detector=detector)

    result = service.detect("Сегодня")

    assert result.language == "ru"


def test_candidates_are_sorted_limited_and_margin_is_calculated(
    monkeypatch: MonkeyPatch,
) -> None:
    detector = FakeDetector(
        [
            confidence(Language.INDONESIAN, 0.20),
            confidence(Language.RUSSIAN, 0.10),
            confidence(Language.ENGLISH, 0.70),
        ]
    )
    service, _, _ = load_fake_service(
        monkeypatch,
        detector=detector,
        settings=detection_settings(language_detection_max_candidates=2),
    )

    result = service.detect("Good morning everyone")

    assert [candidate.language for candidate in result.candidates] == ["en", "id"]
    assert len(result.candidates) == 2
    assert result.confidence == pytest.approx(0.70)
    assert result.confidence_margin == pytest.approx(0.50)
    assert result.candidates[0].language == result.language


def test_single_candidate_uses_top_confidence_as_margin(
    monkeypatch: MonkeyPatch,
) -> None:
    detector = FakeDetector([confidence(Language.ENGLISH, 0.80)])
    service, _, _ = load_fake_service(monkeypatch, detector=detector)

    result = service.detect("Good morning everyone")

    assert result.confidence_margin == pytest.approx(0.80)


def test_confidence_below_threshold_is_rejected(monkeypatch: MonkeyPatch) -> None:
    detector = FakeDetector(
        [
            confidence(Language.ENGLISH, 0.29),
            confidence(Language.INDONESIAN, 0.20),
        ]
    )
    service, _, _ = load_fake_service(monkeypatch, detector=detector)

    with pytest.raises(LanguageDetectionUncertainError):
        service.detect("Ambiguous text")


def test_margin_below_threshold_is_rejected(monkeypatch: MonkeyPatch) -> None:
    detector = FakeDetector(
        [
            confidence(Language.ENGLISH, 0.50),
            confidence(Language.INDONESIAN, 0.47),
        ]
    )
    service, _, _ = load_fake_service(monkeypatch, detector=detector)

    with pytest.raises(LanguageDetectionUncertainError):
        service.detect("Ambiguous words")


@pytest.mark.parametrize(
    "values",
    [
        [],
        [confidence(Language.GERMAN, 0.99)],
    ],
)
def test_empty_or_unmapped_candidates_are_rejected(
    monkeypatch: MonkeyPatch,
    values: list[object],
) -> None:
    service, _, _ = load_fake_service(
        monkeypatch,
        detector=FakeDetector(values),
    )

    with pytest.raises(LanguageDetectionUncertainError):
        service.detect("Detectable text")


def test_success_result_is_immutable(monkeypatch: MonkeyPatch) -> None:
    service, _, _ = load_fake_service(monkeypatch)

    result = service.detect("Good morning everyone")

    assert result.status == "detected"
    assert result.detector_name == "lingua"
    with pytest.raises(FrozenInstanceError):
        result.language = "id"  # type: ignore[misc]


def test_unload_is_idempotent_and_detection_requires_reload(
    monkeypatch: MonkeyPatch,
) -> None:
    service, _, _ = load_fake_service(monkeypatch)

    service.unload_detector()
    service.unload_detector()

    assert service.is_loaded is False
    with pytest.raises(LanguageDetectorNotLoadedError):
        service.detect("Good morning")


def test_internal_detector_error_is_wrapped(monkeypatch: MonkeyPatch) -> None:
    detector = FakeDetector(error=RuntimeError("fake detector failure"))
    service, _, _ = load_fake_service(monkeypatch, detector=detector)

    with pytest.raises(LanguageDetectionInferenceError) as error:
        service.detect("Good morning")

    assert isinstance(error.value.__cause__, RuntimeError)


def test_invalid_confidence_value_is_rejected(monkeypatch: MonkeyPatch) -> None:
    detector = FakeDetector([confidence(Language.ENGLISH, 1.1)])
    service, _, _ = load_fake_service(monkeypatch, detector=detector)

    with pytest.raises(LanguageDetectionInferenceError, match="invalid confidence"):
        service.detect("Good morning")


def test_real_lingua_builder_does_not_need_network(monkeypatch: MonkeyPatch) -> None:
    def fail_network(*_: object, **__: object) -> None:
        raise AssertionError("Language detection attempted a network connection.")

    monkeypatch.setattr("socket.create_connection", fail_network)
    service = LinguaLanguageDetectionService(detection_settings())

    service.load_detector({"en", "id", "ru"})
    result = service.detect("Good morning, how are you today?")

    assert result.language == "en"
    assert service.is_loaded is True
