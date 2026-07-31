"""Translate one UTF-8 text file with local detection and token-aware chunking."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.exceptions import LanguageDetectionError, TranslationServiceError  # noqa: E402
from app.services.language_detection import LinguaLanguageDetectionService  # noqa: E402
from app.services.translation import M2M100TranslationService  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse the input file and language options."""
    parser = argparse.ArgumentParser(
        description="Translate one long UTF-8 file with the cached local M2M100 model.",
    )
    parser.add_argument("--file", required=True, type=Path, help="UTF-8 source text file.")
    parser.add_argument(
        "--source",
        default="auto",
        help="M2M100 source language code or auto (default: auto).",
    )
    parser.add_argument(
        "--target",
        default="id",
        help="M2M100 target language code (default: id).",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Print the full translation instead of short previews.",
    )
    return parser.parse_args()


def preview(text: str, size: int = 150) -> tuple[str, str]:
    """Return compact start and end previews."""
    normalized = text.replace("\r", "\\r").replace("\n", "\\n")
    return normalized[:size], normalized[-size:]


def main() -> int:
    """Load local services, inspect chunks, and run one translation."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    arguments = parse_arguments()
    service: M2M100TranslationService | None = None
    detector: LinguaLanguageDetectionService | None = None
    started_at = perf_counter()
    try:
        source_path = arguments.file.resolve()
        text = source_path.read_text(encoding="utf-8")
        settings = get_settings()
        service = M2M100TranslationService(settings)
        service.load_model()

        source_mode = "auto" if arguments.source.strip().lower() == "auto" else "manual"
        resolved_source = arguments.source.strip().lower()
        if source_mode == "auto":
            detector = LinguaLanguageDetectionService(settings)
            detector.load_detector(service.get_supported_languages())
            resolved_source = detector.detect(text).language

        whole_token_count = service.count_tokens(text, resolved_source)
        prepared = service.prepare_text_chunks(text, resolved_source)
        result = service.translate_long_text(
            text,
            resolved_source,
            arguments.target.strip().lower(),
        )
        duration = perf_counter() - started_at
    except (OSError, UnicodeError, LanguageDetectionError, TranslationServiceError) as exc:
        print(f"Long-text smoke test failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Long-text smoke test failed unexpectedly: {exc}", file=sys.stderr)
        return 1
    finally:
        if detector is not None:
            detector.unload_detector()
        if service is not None:
            service.unload_model()

    print(f"File: {source_path}")
    print(f"Source language mode: {source_mode}")
    print(f"Resolved source: {resolved_source}")
    print(f"Target language: {result.target_language}")
    print(f"Character count: {len(text)}")
    print(f"Whole input token count: {whole_token_count}")
    print(f"Chunk count: {result.chunk_count}")
    print(f"Maximum chunk token count: {prepared.maximum_chunk_tokens}")
    print(f"Chunk token limit: {result.chunk_token_limit}")
    print(f"Device: {result.device}")
    print(f"Status: {result.status}")
    print(f"Total duration: {duration:.2f} seconds")
    if arguments.show_output:
        print("Translation:")
        print(result.translated_text)
    else:
        start_preview, end_preview = preview(result.translated_text)
        print(f"Translation start: {start_preview}")
        print(f"Translation end: {end_preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
