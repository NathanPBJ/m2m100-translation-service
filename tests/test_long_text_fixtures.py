"""Validation and mocked chunk integration tests for original long-text fixtures."""

import re
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from app.services.text_chunking import TokenAwareTextChunkingService
from tests.test_long_text_translation import loaded_long_service, long_settings

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "long_text"
FIXTURE_RULES = {
    "general_opinion_en.txt": (1_500, 2_500, 3),
    "product_review_en.txt": (3_000, 5_000, 6),
    "chronology_spill_en.txt": (7_500, 9_500, 12),
}


def fixture_token_counter(text: str) -> int:
    """Approximate subword counts deterministically without loading a tokenizer."""
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)) + 2


def paragraph_count(text: str) -> int:
    return len([part for part in re.split(r"(?:\r?\n){2,}", text) if part.strip()])


@pytest.mark.parametrize(
    ("filename", "minimum_characters", "maximum_characters", "minimum_paragraphs"),
    [
        (filename, minimum, maximum, paragraphs)
        for filename, (minimum, maximum, paragraphs) in FIXTURE_RULES.items()
    ],
)
def test_fixture_is_original_utf8_text_with_expected_size(
    filename: str,
    minimum_characters: int,
    maximum_characters: int,
    minimum_paragraphs: int,
) -> None:
    fixture_path = FIXTURE_ROOT / filename
    raw_content = fixture_path.read_bytes()
    text = raw_content.decode("utf-8")

    assert fixture_path.is_file()
    assert minimum_characters <= len(text) <= maximum_characters
    assert paragraph_count(text) >= minimum_paragraphs
    assert text.strip()
    assert b"\x00" not in raw_content


def test_fixture_lengths_are_strictly_ordered_and_chronology_is_fictional() -> None:
    opinion = (FIXTURE_ROOT / "general_opinion_en.txt").read_text(encoding="utf-8")
    review = (FIXTURE_ROOT / "product_review_en.txt").read_text(encoding="utf-8")
    chronology = (FIXTURE_ROOT / "chronology_spill_en.txt").read_text(encoding="utf-8")

    assert len(opinion) < len(review) < len(chronology)
    assert "FICTIONAL TEST FIXTURE" in chronology[:200]
    assert {"09:15", "13:40", "18:20", "@ProjectHelper", "#FictionalProjectSpill"} <= {
        marker
        for marker in ("09:15", "13:40", "18:20", "@ProjectHelper", "#FictionalProjectSpill")
        if marker in chronology
    }
    assert "“I thought you were correcting my work, not the file path.”" in chronology


def test_three_fixtures_chunk_losslessly_in_increasing_counts() -> None:
    service = TokenAwareTextChunkingService()
    results = []
    for filename in FIXTURE_RULES:
        text = (FIXTURE_ROOT / filename).read_text(encoding="utf-8")
        result = service.chunk_text(
            text,
            fixture_token_counter,
            maximum_tokens=100,
            maximum_chunks=64,
        )
        assert result.reconstruct_source() == text
        assert all(item.token_count <= 100 for item in result.chunks)
        assert result.chunked is True
        results.append(result)

    opinion, review, chronology = results
    assert opinion.chunk_count >= 2
    assert review.chunk_count > opinion.chunk_count
    assert chronology.chunk_count > review.chunk_count
    assert chronology.chunk_count >= 5


def test_product_review_source_markers_survive_reconstruction() -> None:
    text = (FIXTURE_ROOT / "product_review_en.txt").read_text(encoding="utf-8")
    result = TokenAwareTextChunkingService().chunk_text(
        text,
        fixture_token_counter,
        maximum_tokens=100,
        maximum_chunks=64,
    )

    reconstructed = result.reconstruct_source()
    assert "https://example.com/product" in reconstructed
    assert "#AuroraX1Review" in reconstructed
    assert reconstructed.count("\n\n") == text.count("\n\n")


def test_chronology_source_order_and_structure_survive_reconstruction() -> None:
    text = (FIXTURE_ROOT / "chronology_spill_en.txt").read_text(encoding="utf-8")
    result = TokenAwareTextChunkingService().chunk_text(
        text,
        fixture_token_counter,
        maximum_tokens=100,
        maximum_chunks=64,
    )

    reconstructed = result.reconstruct_source()
    ordered_markers = ["09:15", "13:40", "18:20", "14:20", "17:55", "18:08"]
    positions = [reconstructed.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)
    assert "@ProjectHelper" in reconstructed
    assert "#FictionalProjectSpill" in reconstructed
    assert "“I thought you were correcting my work, not the file path.”" in reconstructed
    assert reconstructed.count("\n\n") == text.count("\n\n")


@pytest.mark.parametrize(
    ("filename", "minimum_chunks"),
    [
        ("general_opinion_en.txt", 2),
        ("product_review_en.txt", 3),
        ("chronology_spill_en.txt", 5),
    ],
)
def test_fixture_mock_translation_processes_every_chunk_in_order(
    monkeypatch: MonkeyPatch,
    filename: str,
    minimum_chunks: int,
) -> None:
    text = (FIXTURE_ROOT / filename).read_text(encoding="utf-8")
    service, tokenizer, model = loaded_long_service(
        monkeypatch,
        settings=long_settings(
            model_max_input_tokens=100,
            long_text_chunk_max_tokens=50,
            long_text_max_chunks=64,
        ),
    )
    prepared = service.prepare_text_chunks(text, "en")

    result = service.translate_long_text(text, "en", "id")

    inference_texts = [
        source_text
        for source_text, _, options in tokenizer.calls
        if options == {"return_tensors": "pt", "truncation": False}
    ]
    assert prepared.reconstruct_source() == text
    assert prepared.chunk_count >= minimum_chunks
    assert inference_texts[-prepared.chunk_count :] == [item.text for item in prepared.chunks]
    assert len(model.generate_calls) == prepared.chunk_count
    assert result.chunk_count == prepared.chunk_count
    assert result.chunked is True
    assert [f"<translated-{index}>" for index in range(result.chunk_count)] == [
        marker
        for index in range(result.chunk_count)
        if (marker := f"<translated-{index}>") in result.translated_text
    ]
