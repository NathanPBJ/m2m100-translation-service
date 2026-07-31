"""Unit tests for sequential long-text translation with fake model objects."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

from app.core.config import Settings
from app.core.exceptions import (
    InputTooLongError,
    TextChunkingFailedError,
    TooManyChunksError,
    TranslationInferenceError,
    TranslationOutputTruncatedError,
)
from app.services.text_chunking import TokenAwareTextChunkingService
from app.services.translation import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    M2M100TranslationService,
)
from tests.test_translation_service import FakeTensor


class ChunkTokenizer:
    """Whitespace tokenizer fake that includes two special tokens."""

    def __init__(self) -> None:
        self.lang_code_to_id = {"en": 1, "id": 2, "ja": 3}
        self.src_lang: str | None = None
        self.eos_token_id = 99
        self.calls: list[tuple[str, str | None, dict[str, Any]]] = []
        self.decode_calls: list[tuple[object, dict[str, Any]]] = []

    def __call__(self, text: str, **options: Any) -> dict[str, FakeTensor]:
        self.calls.append((text, self.src_lang, options))
        sequence_length = len(text.split()) + 2
        return {
            "input_ids": FakeTensor(sequence_length),
            "attention_mask": FakeTensor(sequence_length),
        }

    def get_lang_id(self, language_code: str) -> int:
        return self.lang_code_to_id[language_code]

    def batch_decode(self, generated_tokens: object, **options: Any) -> list[str]:
        self.decode_calls.append((generated_tokens, options))
        sequence = generated_tokens[0]  # type: ignore[index]
        return [f"<translated-{sequence[0]}>"]


class SequentialModel:
    """Generation fake with ordering, optional failure, and configurable tokens."""

    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
        generated_sequences: list[list[int]] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.config = SimpleNamespace(eos_token_id=99)
        self.fail_on_call = fail_on_call
        self.generated_sequences = generated_sequences
        self.delay = delay
        self.to_calls: list[str] = []
        self.eval_calls = 0
        self.generate_calls: list[dict[str, Any]] = []

    def to(self, device: str) -> "SequentialModel":
        self.to_calls.append(device)
        return self

    def eval(self) -> "SequentialModel":
        self.eval_calls += 1
        return self

    def generate(self, **options: Any) -> list[list[int]]:
        call_index = len(self.generate_calls)
        self.generate_calls.append(options)
        if self.delay:
            time.sleep(self.delay)
        if self.fail_on_call == call_index:
            raise RuntimeError("PRIVATE_MIDDLE_CHUNK")
        if self.generated_sequences is not None:
            return [self.generated_sequences[call_index]]
        return [[call_index, 99]]


class RecordingChunker(TokenAwareTextChunkingService):
    """Chunker that records calls while retaining the real implementation."""

    def __init__(self, *, fail: Exception | None = None) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.fail = fail

    def chunk_text(
        self,
        text: str,
        token_counter: Any,
        maximum_tokens: int,
        maximum_chunks: int,
    ):
        self.calls.append((text, maximum_tokens, maximum_chunks))
        if self.fail is not None:
            raise self.fail
        return super().chunk_text(
            text,
            token_counter,
            maximum_tokens,
            maximum_chunks,
        )


def long_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "model_name": "facebook/m2m100_418M",
        "model_cache_dir": Path("test-model-cache"),
        "model_device": "cpu",
        "model_local_files_only": True,
        "model_max_input_tokens": 12,
        "model_max_new_tokens": 8,
        "model_num_beams": 3,
        "long_text_chunking_enabled": True,
        "long_text_chunk_max_tokens": 5,
        "long_text_max_chunks": 20,
    }
    values.update(overrides)
    return Settings(**values)


def loaded_long_service(
    monkeypatch: MonkeyPatch,
    *,
    settings: Settings | None = None,
    tokenizer: ChunkTokenizer | None = None,
    model: SequentialModel | None = None,
    chunker: TokenAwareTextChunkingService | None = None,
) -> tuple[M2M100TranslationService, ChunkTokenizer, SequentialModel]:
    fake_tokenizer = tokenizer or ChunkTokenizer()
    fake_model = model or SequentialModel()
    monkeypatch.setattr(
        M2M100Tokenizer,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: fake_tokenizer),
    )
    monkeypatch.setattr(
        M2M100ForConditionalGeneration,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: fake_model),
    )
    service = M2M100TranslationService(settings or long_settings(), chunker)
    service.load_model()
    return service, fake_tokenizer, fake_model


def test_chunk_settings_validate_positive_values_and_hard_limit() -> None:
    for overrides in (
        {"long_text_chunk_max_tokens": 0},
        {"long_text_max_chunks": 0},
        {"model_max_new_tokens": 0},
        {"long_text_chunk_max_tokens": 13},
    ):
        with pytest.raises(ValidationError):
            long_settings(**overrides)


def test_default_generation_limit_is_512() -> None:
    assert Settings().model_max_new_tokens == 512


def test_short_text_runs_one_inference_with_chunk_metadata(
    monkeypatch: MonkeyPatch,
) -> None:
    service, tokenizer, model = loaded_long_service(monkeypatch)

    result = service.translate_long_text("one two", "en", "id")

    assert result.chunked is False
    assert result.chunk_count == 1
    assert result.chunk_token_limit == 5
    assert result.translated_text == "<translated-0>"
    assert len(model.generate_calls) == 1
    assert tokenizer.calls[-1][0] == "one two"


def test_long_text_translates_chunks_in_order_with_same_language_pair(
    monkeypatch: MonkeyPatch,
) -> None:
    service, tokenizer, model = loaded_long_service(monkeypatch)

    result = service.translate_long_text(
        "one two three. four five six. seven eight nine.",
        "en",
        "id",
    )

    assert result.chunked is True
    assert result.chunk_count == 3
    assert result.translated_text == "<translated-0> <translated-1> <translated-2>"
    assert len(model.generate_calls) == 3
    inference_calls = [
        call for call in tokenizer.calls if call[2] == {"return_tensors": "pt", "truncation": False}
    ][-3:]
    assert [call[0] for call in inference_calls] == [
        "one two three.",
        "four five six.",
        "seven eight nine.",
    ]
    assert {call[1] for call in inference_calls} == {"en"}
    assert all(call["forced_bos_token_id"] == 2 for call in model.generate_calls)
    assert all(call["max_new_tokens"] == 8 for call in model.generate_calls)
    assert all(call["num_beams"] == 3 for call in model.generate_calls)


def test_paragraph_and_blank_line_separators_are_merged_exactly(
    monkeypatch: MonkeyPatch,
) -> None:
    service, _, _ = loaded_long_service(monkeypatch)

    result = service.translate_long_text(
        "one two three.\r\n\r\n  four five six.\n\nseven eight nine.",
        "en",
        "id",
    )

    assert result.translated_text == ("<translated-0>\r\n\r\n  <translated-1>\n\n<translated-2>")


def test_same_language_skips_chunker_tokenizer_and_generation(
    monkeypatch: MonkeyPatch,
) -> None:
    chunker = RecordingChunker(fail=AssertionError("Chunker must not run."))
    service, tokenizer, model = loaded_long_service(monkeypatch, chunker=chunker)
    original = "one two three " * 100

    result = service.translate_long_text(original, "en", "en")

    assert result.translated_text == original
    assert result.status == "unchanged"
    assert result.chunked is False
    assert result.chunk_count == 0
    assert tokenizer.calls == []
    assert model.generate_calls == []
    assert chunker.calls == []


def test_chunking_disabled_keeps_strict_hard_limit(
    monkeypatch: MonkeyPatch,
) -> None:
    service, _, model = loaded_long_service(
        monkeypatch,
        settings=long_settings(
            long_text_chunking_enabled=False,
            model_max_input_tokens=6,
            long_text_chunk_max_tokens=5,
        ),
    )

    with pytest.raises(InputTooLongError):
        service.translate_long_text("one two three four five", "en", "id")

    assert model.generate_calls == []


def test_chunking_disabled_accepts_one_hard_limit_input(
    monkeypatch: MonkeyPatch,
) -> None:
    service, _, model = loaded_long_service(
        monkeypatch,
        settings=long_settings(
            long_text_chunking_enabled=False,
            model_max_input_tokens=7,
            long_text_chunk_max_tokens=5,
        ),
    )

    result = service.translate_long_text("one two three four five", "en", "id")

    assert result.chunked is False
    assert result.chunk_count == 1
    assert len(model.generate_calls) == 1


def test_too_many_chunks_propagates_without_generation(
    monkeypatch: MonkeyPatch,
) -> None:
    service, _, model = loaded_long_service(
        monkeypatch,
        settings=long_settings(long_text_max_chunks=2),
    )

    with pytest.raises(TooManyChunksError):
        service.translate_long_text(
            "one two three. four five six. seven eight nine.",
            "en",
            "id",
        )

    assert model.generate_calls == []


def test_middle_chunk_failure_returns_no_result_and_hides_source(
    monkeypatch: MonkeyPatch,
) -> None:
    model = SequentialModel(fail_on_call=1)
    service, _, _ = loaded_long_service(monkeypatch, model=model)
    private_text = "secret one two. hidden three four. private five six."

    with pytest.raises(TranslationInferenceError) as error:
        service.translate_long_text(private_text, "en", "id")

    assert len(model.generate_calls) == 2
    assert private_text not in str(error.value)
    assert "PRIVATE_MIDDLE_CHUNK" not in str(error.value)


def test_empty_decoded_middle_chunk_is_rejected(
    monkeypatch: MonkeyPatch,
) -> None:
    tokenizer = ChunkTokenizer()
    original_decode = tokenizer.batch_decode

    def decode_with_empty_second(tokens: object, **options: Any) -> list[str]:
        if tokens[0][0] == 1:  # type: ignore[index]
            return ["   "]
        return original_decode(tokens, **options)

    tokenizer.batch_decode = decode_with_empty_second  # type: ignore[method-assign]
    service, _, model = loaded_long_service(monkeypatch, tokenizer=tokenizer)

    with pytest.raises(TranslationInferenceError, match="empty result"):
        service.translate_long_text(
            "one two three. four five six. seven eight nine.",
            "en",
            "id",
        )

    assert len(model.generate_calls) == 2


def test_output_at_limit_without_eos_is_rejected(
    monkeypatch: MonkeyPatch,
) -> None:
    model = SequentialModel(generated_sequences=[[1, 2, 3, 4]])
    service, _, _ = loaded_long_service(
        monkeypatch,
        settings=long_settings(model_max_new_tokens=4),
        model=model,
    )

    with pytest.raises(TranslationOutputTruncatedError) as error:
        service.translate_long_text("one two", "en", "id")

    assert "one two" not in str(error.value)


def test_output_at_limit_with_eos_is_accepted(monkeypatch: MonkeyPatch) -> None:
    model = SequentialModel(generated_sequences=[[1, 2, 3, 99]])
    service, _, _ = loaded_long_service(
        monkeypatch,
        settings=long_settings(model_max_new_tokens=4),
        model=model,
    )

    result = service.translate_long_text("one two", "en", "id")

    assert result.translated_text == "<translated-1>"


def test_output_below_limit_without_eos_is_accepted(monkeypatch: MonkeyPatch) -> None:
    model = SequentialModel(generated_sequences=[[1, 2, 3]])
    service, _, _ = loaded_long_service(
        monkeypatch,
        settings=long_settings(model_max_new_tokens=4),
        model=model,
    )

    assert service.translate_long_text("one two", "en", "id").status == "translated"


def test_chunker_failure_propagates_without_generation(
    monkeypatch: MonkeyPatch,
) -> None:
    chunker = RecordingChunker(fail=TextChunkingFailedError("safe failure"))
    service, _, model = loaded_long_service(monkeypatch, chunker=chunker)

    with pytest.raises(TextChunkingFailedError):
        service.translate_long_text("one two three four", "en", "id")

    assert model.generate_calls == []


def test_public_strict_translate_still_rejects_oversized_input(
    monkeypatch: MonkeyPatch,
) -> None:
    service, _, model = loaded_long_service(
        monkeypatch,
        settings=long_settings(model_max_input_tokens=5),
    )

    with pytest.raises(InputTooLongError):
        service.translate("one two three four", "en", "id")

    assert model.generate_calls == []


def test_count_and_prepare_helpers_use_loaded_tokenizer(
    monkeypatch: MonkeyPatch,
) -> None:
    service, _, model = loaded_long_service(monkeypatch)
    text = "one two three. four five six."

    assert service.count_tokens(text, "en") == 8
    chunks = service.prepare_text_chunks(text, "en")

    assert chunks.chunk_count == 2
    assert chunks.reconstruct_source() == text
    assert model.generate_calls == []


def test_state_lock_prevents_source_language_interleaving(
    monkeypatch: MonkeyPatch,
) -> None:
    model = SequentialModel(delay=0.005)
    service, tokenizer, _ = loaded_long_service(monkeypatch, model=model)
    text = "one two three. four five six. seven eight nine."

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(service.translate_long_text, text, "en", "id"),
            executor.submit(service.translate_long_text, text, "ja", "id"),
        ]
        [future.result(timeout=5) for future in futures]

    source_runs: list[str] = []
    for _, source_language, _ in tokenizer.calls:
        if source_language and (not source_runs or source_runs[-1] != source_language):
            source_runs.append(source_language)
    assert source_runs in (["en", "ja"], ["ja", "en"])


def test_load_and_unload_remain_idempotent_with_lock(
    monkeypatch: MonkeyPatch,
) -> None:
    service, _, model = loaded_long_service(monkeypatch)

    service.load_model()
    service.unload_model()
    service.unload_model()

    assert model.to_calls == ["cpu"]
    assert service.is_loaded is False
