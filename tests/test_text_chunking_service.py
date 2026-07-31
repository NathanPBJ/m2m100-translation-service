"""Unit tests for lossless token-aware text chunking."""

from dataclasses import FrozenInstanceError

import pytest
from pytest import MonkeyPatch

from app.core.exceptions import (
    InvalidTranslationInputError,
    TextChunkingFailedError,
    TooManyChunksError,
)
from app.domain.text_chunking import TextChunk, TextChunkingResult
from app.services.text_chunking import TokenAwareTextChunkingService


def character_tokens(text: str) -> int:
    """Deterministic fake counter with one token per Unicode code point."""
    return len(text)


def chunk(
    text: str,
    *,
    limit: int = 20,
    maximum_chunks: int = 64,
):
    return TokenAwareTextChunkingService().chunk_text(
        text,
        character_tokens,
        limit,
        maximum_chunks,
    )


def test_short_text_is_one_unchanged_chunk() -> None:
    result = chunk("Short text.", limit=20)

    assert result.chunked is False
    assert result.chunk_count == 1
    assert result.chunks[0].text == "Short text."
    assert result.reconstruct_source() == "Short text."


def test_exact_token_boundary_is_accepted() -> None:
    result = chunk("12345", limit=5)

    assert result.chunk_count == 1
    assert result.chunks[0].token_count == 5


def test_one_token_over_boundary_is_split() -> None:
    result = chunk("123456", limit=5)

    assert result.chunk_count == 2
    assert [item.text for item in result.chunks] == ["12345", "6"]


def test_every_chunk_stays_within_budget_and_has_ordered_index() -> None:
    result = chunk("One sentence is here. Another sentence follows. Last one.", limit=18)

    assert all(item.token_count <= 18 for item in result.chunks)
    assert [item.index for item in result.chunks] == list(range(result.chunk_count))


def test_no_chunk_is_empty_or_whitespace_only() -> None:
    result = chunk("  First sentence.   Second sentence.\n\nThird sentence.  ", limit=18)

    assert all(item.text and item.text.strip() for item in result.chunks)


@pytest.mark.parametrize(
    "text",
    [
        "  Leading space is retained. Another sentence follows.",
        "Trailing space is retained. Another sentence follows.   ",
        "First line.\nSecond line with more words.",
        "First line.\r\nSecond line with more words.",
        "First.\n\n\nSecond.\n\nThird.",
        "First paragraph stays first.\n\nSecond stays second.\n\nThird stays third.",
        "Keep punctuation! Does this stay? Yes; it does.",
        "Emoji 🔥 remains here. Another sentence follows.",
        "Visit https://example.com/a.b for details. Keep the URL intact.",
        "Ask @fictional_user about this. Keep the mention intact.",
        "A #LongText hashtag remains. Then another sentence.",
        "The measured value was 3.14 exactly. It was not three separate parts.",
        "Write to test.person@example.com today. The email remains intact.",
        "最初の文です。次の文も残ります！最後です。",
        "这是一个没有空格的中文长句，需要安全地拆分而且不能丢失任何字符。" * 3,
        "هذه فقرة عربية طويلة للاختبار. يجب أن تبقى الحروف بالترتيب الصحيح!" * 2,
        "Supercalifragilisticexpialidocious" * 5,
        "https://example.com/" + "very-long-path-" * 20,
        "This is one oversized sentence without useful punctuation " * 8,
    ],
)
def test_lossless_reconstruction_for_diverse_text(text: str) -> None:
    result = chunk(text, limit=24)

    assert result.reconstruct_source() == text
    assert all(item.token_count <= 24 for item in result.chunks)


def test_multiple_blank_lines_are_kept_as_separator() -> None:
    text = "First paragraph is long enough.\n\n \n\t\nSecond paragraph is also long."
    result = chunk(text, limit=18)

    assert "\n\n \n\t\n" in "".join(item.separator_after for item in result.chunks)
    assert result.reconstruct_source() == text


def test_sentence_punctuation_remains_attached_to_source_spans() -> None:
    text = "Alpha! Beta? Gamma… Delta。 Epsilon！ Zeta？ Eta; Theta： Iota；"
    result = chunk(text, limit=12)

    reconstructed_translatable = "".join(item.text + item.separator_after for item in result.chunks)
    assert reconstructed_translatable == text
    assert all(symbol in reconstructed_translatable for symbol in "!?…。！？;：；")


def test_decimal_email_and_url_are_not_corrupted() -> None:
    text = (
        "Version 3.14 is listed at https://example.com/v3.14. "
        "Email test.person@example.com for details."
    )
    result = chunk(text, limit=25)

    assert "3.14" in result.reconstruct_source()
    assert "https://example.com/v3.14" in result.reconstruct_source()
    assert "test.person@example.com" in result.reconstruct_source()


def test_common_abbreviations_are_not_corrupted() -> None:
    text = "Dr. Example measured it, e.g. with a small tool. The result was stable."
    result = chunk(text, limit=22)

    assert result.reconstruct_source() == text
    assert "Dr." in result.reconstruct_source()
    assert "e.g." in result.reconstruct_source()


def test_greedy_sentence_packing_uses_available_budget() -> None:
    result = chunk("1234. 5678. 90.", limit=12)

    assert result.chunks[0].text == "1234. 5678."
    assert result.chunks[0].token_count == 11


def test_maximum_chunk_count_is_accepted_exactly() -> None:
    result = chunk("abcdefghij", limit=5, maximum_chunks=2)

    assert result.chunk_count == 2


def test_too_many_chunks_is_rejected() -> None:
    with pytest.raises(TooManyChunksError) as error:
        chunk("abcdefghijk", limit=5, maximum_chunks=2)

    assert error.value.actual_chunks == 3
    assert error.value.maximum_chunks == 2


@pytest.mark.parametrize(
    "bad_counter",
    [
        lambda _text: 0,
        lambda _text: -1,
        lambda _text: 1.5,
        lambda _text: True,
        lambda _text: None,
    ],
)
def test_invalid_token_counter_result_is_wrapped(bad_counter: object) -> None:
    with pytest.raises(TextChunkingFailedError, match="invalid token count"):
        TokenAwareTextChunkingService().chunk_text(
            "Valid text",
            bad_counter,  # type: ignore[arg-type]
            5,
            10,
        )


def test_token_counter_exception_is_safely_chained() -> None:
    def fail_counter(_text: str) -> int:
        raise RuntimeError("PRIVATE_SOURCE_CONTENT")

    with pytest.raises(TextChunkingFailedError) as error:
        TokenAwareTextChunkingService().chunk_text("Valid text", fail_counter, 5, 10)

    assert isinstance(error.value.__cause__, RuntimeError)
    assert "PRIVATE_SOURCE_CONTENT" not in str(error.value)


def test_explicit_reconstruction_failure_is_rejected() -> None:
    with pytest.raises(TextChunkingFailedError, match="reconstruction"):
        TokenAwareTextChunkingService._validate_result(
            original_text="original",
            prefix_separator="",
            chunks=(TextChunk(index=0, text="changed", token_count=7),),
            maximum_tokens=10,
            maximum_chunks=2,
        )


@pytest.mark.parametrize("text", [None, 123, object(), "", "   ", "\r\n"])
def test_non_string_or_empty_input_is_rejected(text: object) -> None:
    with pytest.raises(InvalidTranslationInputError):
        TokenAwareTextChunkingService().chunk_text(
            text,  # type: ignore[arg-type]
            character_tokens,
            10,
            10,
        )


@pytest.mark.parametrize(
    ("maximum_tokens", "maximum_chunks"),
    [(0, 1), (-1, 1), (True, 1), (1, 0), (1, -1), (1, False)],
)
def test_invalid_limits_are_rejected(maximum_tokens: int, maximum_chunks: int) -> None:
    with pytest.raises(TextChunkingFailedError):
        TokenAwareTextChunkingService().chunk_text(
            "Valid",
            character_tokens,
            maximum_tokens,
            maximum_chunks,
        )


def test_chunking_result_and_chunk_are_immutable() -> None:
    result = chunk("This needs more than one immutable chunk.", limit=12)

    with pytest.raises(FrozenInstanceError):
        result.chunked = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.chunks[0].text = "changed"  # type: ignore[misc]


def test_chunk_count_domain_invariant_is_enforced() -> None:
    with pytest.raises(ValueError, match="Chunk count"):
        TextChunkingResult(
            chunks=(TextChunk(index=0, text="valid", token_count=5),),
            chunked=True,
            chunk_count=2,
            maximum_chunk_tokens=5,
        )


def test_chunker_does_not_use_network(monkeypatch: MonkeyPatch) -> None:
    def fail_network(*_: object, **__: object) -> None:
        raise AssertionError("Chunker attempted a network request.")

    monkeypatch.setattr("socket.create_connection", fail_network)

    assert chunk("Network-free chunking works. Another sentence.", limit=18).chunk_count > 1


@pytest.mark.parametrize(
    "text",
    [
        "a" * 101,
        ("word " * 80).strip(),
        "。".join(["中文片段"] * 30),
        "\n\n".join(f"Paragraph {index} has unique content." for index in range(20)),
        "🙂" * 80 + " alphabetic ending",
    ],
)
def test_edge_cases_finish_with_progress_and_exact_order(text: str) -> None:
    result = chunk(text, limit=15, maximum_chunks=128)

    assert result.reconstruct_source() == text
    assert result.chunk_count <= 128
    assert all(item.text for item in result.chunks)
