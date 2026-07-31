"""Unit tests for the local M2M100 translation engine."""

from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from app.core.config import Settings
from app.core.exceptions import (
    DeviceConfigurationError,
    InputTooLongError,
    InvalidTranslationInputError,
    ModelLoadError,
    ModelNotLoadedError,
    TranslationInferenceError,
    UnsupportedLanguageError,
)
from app.services.translation import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    M2M100TranslationService,
)


class FakeTensor:
    """Small tensor substitute that records device movement."""

    def __init__(self, sequence_length: int) -> None:
        self.shape = (1, sequence_length)
        self.moved_to: str | None = None

    def to(self, device: str) -> "FakeTensor":
        self.moved_to = device
        return self


class FakeTokenizer:
    """Tokenizer substitute with the M2M100 methods used by the service."""

    def __init__(self, sequence_length: int = 3, decoded_text: str = "Selamat pagi") -> None:
        self.lang_code_to_id = {"ja": 3, "en": 1, "id": 2}
        self.src_lang: str | None = None
        self.sequence_length = sequence_length
        self.decoded_text = decoded_text
        self.call_count = 0
        self.call_options: dict[str, Any] = {}
        self.last_inputs: dict[str, FakeTensor] = {}
        self.decode_options: dict[str, Any] = {}

    def __call__(self, text: str, **options: Any) -> dict[str, FakeTensor]:
        self.call_count += 1
        self.call_options = {"text": text, **options}
        self.last_inputs = {
            "input_ids": FakeTensor(self.sequence_length),
            "attention_mask": FakeTensor(self.sequence_length),
        }
        return self.last_inputs

    def get_lang_id(self, language_code: str) -> int:
        return self.lang_code_to_id[language_code]

    def batch_decode(self, generated_tokens: object, **options: Any) -> list[str]:
        self.decode_options = {"generated_tokens": generated_tokens, **options}
        return [self.decoded_text]


class FakeModel:
    """Model substitute that records setup and generation calls."""

    def __init__(self, generation_error: Exception | None = None) -> None:
        self.generation_error = generation_error
        self.to_calls: list[str] = []
        self.eval_calls = 0
        self.generate_calls: list[dict[str, Any]] = []
        self.generated_tokens = [[10, 20, 30]]

    def to(self, device: str) -> "FakeModel":
        self.to_calls.append(device)
        return self

    def eval(self) -> "FakeModel":
        self.eval_calls += 1
        return self

    def generate(self, **options: Any) -> object:
        self.generate_calls.append(options)
        if self.generation_error is not None:
            raise self.generation_error
        return self.generated_tokens


def make_settings(**overrides: Any) -> Settings:
    """Create deterministic settings for service tests."""
    values: dict[str, Any] = {
        "model_name": "facebook/m2m100_418M",
        "model_cache_dir": Path("test-model-cache"),
        "model_device": "cpu",
        "model_local_files_only": True,
        "model_max_input_tokens": 512,
        "model_max_new_tokens": 32,
        "model_num_beams": 3,
    }
    values.update(overrides)
    return Settings(**values)


def install_fake_loaders(
    monkeypatch: MonkeyPatch,
    tokenizer: FakeTokenizer | None = None,
    model: FakeModel | None = None,
) -> tuple[FakeTokenizer, FakeModel, list[tuple[str, str, dict[str, Any]]]]:
    """Patch Transformers loaders and return their fake instances and calls."""
    fake_tokenizer = tokenizer or FakeTokenizer()
    fake_model = model or FakeModel()
    load_calls: list[tuple[str, str, dict[str, Any]]] = []

    def load_tokenizer(model_name: str, **options: Any) -> FakeTokenizer:
        load_calls.append(("tokenizer", model_name, options))
        return fake_tokenizer

    def load_model(model_name: str, **options: Any) -> FakeModel:
        load_calls.append(("model", model_name, options))
        return fake_model

    monkeypatch.setattr(
        M2M100Tokenizer,
        "from_pretrained",
        staticmethod(load_tokenizer),
    )
    monkeypatch.setattr(
        M2M100ForConditionalGeneration,
        "from_pretrained",
        staticmethod(load_model),
    )
    return fake_tokenizer, fake_model, load_calls


def create_loaded_service(
    monkeypatch: MonkeyPatch,
    *,
    settings: Settings | None = None,
    tokenizer: FakeTokenizer | None = None,
    model: FakeModel | None = None,
) -> tuple[M2M100TranslationService, FakeTokenizer, FakeModel]:
    """Create a service and load patched dependencies."""
    fake_tokenizer, fake_model, _ = install_fake_loaders(monkeypatch, tokenizer, model)
    service = M2M100TranslationService(settings or make_settings())
    service.load_model()
    return service, fake_tokenizer, fake_model


def test_constructor_is_lazy_and_uses_settings(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.translation.torch.cuda.is_available", lambda: False)

    service = M2M100TranslationService(make_settings(model_device="auto"))

    assert service.is_loaded is False
    assert service.model_name == "facebook/m2m100_418M"
    assert service.device == "cpu"


def test_auto_device_selects_cuda_when_available(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.translation.torch.cuda.is_available", lambda: True)

    service = M2M100TranslationService(make_settings(model_device="auto"))

    assert service.device == "cuda"


def test_explicit_cuda_is_rejected_when_unavailable(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.translation.torch.cuda.is_available", lambda: False)

    with pytest.raises(DeviceConfigurationError, match="CUDA was requested"):
        M2M100TranslationService(make_settings(model_device="cuda"))


def test_load_model_uses_configuration_and_is_idempotent(monkeypatch: MonkeyPatch) -> None:
    tokenizer, model, load_calls = install_fake_loaders(monkeypatch)
    service = M2M100TranslationService(make_settings())

    service.load_model()
    service.load_model()

    expected_options = {
        "cache_dir": "test-model-cache",
        "local_files_only": True,
    }
    assert load_calls == [
        ("tokenizer", "facebook/m2m100_418M", expected_options),
        ("model", "facebook/m2m100_418M", expected_options),
    ]
    assert service.is_loaded is True
    assert model.to_calls == ["cpu"]
    assert model.eval_calls == 1
    assert tokenizer.call_count == 0


def test_load_failure_is_wrapped(monkeypatch: MonkeyPatch) -> None:
    def fail_loading(*_: Any, **__: Any) -> None:
        raise OSError("local test failure")

    monkeypatch.setattr(
        M2M100Tokenizer,
        "from_pretrained",
        staticmethod(fail_loading),
    )
    service = M2M100TranslationService(make_settings())

    with pytest.raises(ModelLoadError) as error:
        service.load_model()

    assert isinstance(error.value.__cause__, OSError)
    assert service.is_loaded is False


def test_unload_model_is_idempotent_and_translate_requires_reload(
    monkeypatch: MonkeyPatch,
) -> None:
    service, _, _ = create_loaded_service(monkeypatch)
    empty_cache_calls: list[bool] = []
    monkeypatch.setattr(
        "app.services.translation.torch.cuda.empty_cache",
        lambda: empty_cache_calls.append(True),
    )

    service.unload_model()
    service.unload_model()

    assert service.is_loaded is False
    assert empty_cache_calls == []
    with pytest.raises(ModelNotLoadedError):
        service.translate("Good morning", "en", "id")


def test_supported_languages_are_sorted_and_immutable(monkeypatch: MonkeyPatch) -> None:
    service, _, _ = create_loaded_service(monkeypatch)

    assert service.get_supported_languages() == ("en", "id", "ja")
    assert service.supports_language(" EN ") is True
    assert service.supports_language("en-US") is False


@pytest.mark.parametrize("language_code", ["auto", "fr", "en-US"])
def test_unsupported_source_languages_are_rejected(
    monkeypatch: MonkeyPatch,
    language_code: str,
) -> None:
    service, _, _ = create_loaded_service(monkeypatch)

    with pytest.raises(UnsupportedLanguageError, match=language_code.lower()):
        service.translate("Good morning", language_code, "id")


@pytest.mark.parametrize("text", ["", "   ", None, 123])
def test_invalid_text_is_rejected(monkeypatch: MonkeyPatch, text: object) -> None:
    service, _, _ = create_loaded_service(monkeypatch)

    with pytest.raises(InvalidTranslationInputError):
        service.translate(text, "en", "id")  # type: ignore[arg-type]


def test_translate_requires_loaded_model() -> None:
    service = M2M100TranslationService(make_settings())

    with pytest.raises(ModelNotLoadedError, match=r"load_model\(\)"):
        service.translate("Good morning", "en", "id")


def test_same_language_returns_original_without_inference(monkeypatch: MonkeyPatch) -> None:
    service, tokenizer, model = create_loaded_service(monkeypatch)
    original_text = "  Keep This Text  "

    result = service.translate(original_text, " EN ", "en")

    assert result.original_text == original_text
    assert result.translated_text == original_text
    assert result.source_language == "en"
    assert result.target_language == "en"
    assert result.status == "unchanged"
    assert tokenizer.call_count == 0
    assert model.generate_calls == []


def test_translation_runs_direct_inference_with_expected_options(
    monkeypatch: MonkeyPatch,
) -> None:
    service, tokenizer, model = create_loaded_service(monkeypatch)

    result = service.translate("Good morning", " EN ", " ID ")

    assert tokenizer.src_lang == "en"
    assert tokenizer.call_options == {
        "text": "Good morning",
        "return_tensors": "pt",
        "truncation": False,
    }
    assert all(tensor.moved_to == "cpu" for tensor in tokenizer.last_inputs.values())
    assert model.generate_calls == [
        {
            **tokenizer.last_inputs,
            "forced_bos_token_id": 2,
            "max_new_tokens": 32,
            "num_beams": 3,
        }
    ]
    assert tokenizer.decode_options == {
        "generated_tokens": model.generated_tokens,
        "skip_special_tokens": True,
    }
    assert result.translated_text == "Selamat pagi"
    assert result.source_language == "en"
    assert result.target_language == "id"
    assert result.model_name == "facebook/m2m100_418M"
    assert result.device == "cpu"
    assert result.status == "translated"


def test_input_over_token_limit_is_rejected_without_generation(
    monkeypatch: MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer(sequence_length=6)
    service, _, model = create_loaded_service(
        monkeypatch,
        settings=make_settings(model_max_input_tokens=5),
        tokenizer=tokenizer,
    )

    with pytest.raises(InputTooLongError, match="6 tokens; the maximum is 5"):
        service.translate("Long input", "en", "id")

    assert model.generate_calls == []


def test_generation_error_is_wrapped(monkeypatch: MonkeyPatch) -> None:
    model = FakeModel(generation_error=RuntimeError("test generation failure"))
    service, _, _ = create_loaded_service(monkeypatch, model=model)

    with pytest.raises(TranslationInferenceError) as error:
        service.translate("Good morning", "en", "id")

    assert isinstance(error.value.__cause__, RuntimeError)


def test_empty_decoded_result_is_rejected(monkeypatch: MonkeyPatch) -> None:
    service, _, _ = create_loaded_service(
        monkeypatch,
        tokenizer=FakeTokenizer(decoded_text="   "),
    )

    with pytest.raises(TranslationInferenceError, match="empty result"):
        service.translate("Good morning", "en", "id")
