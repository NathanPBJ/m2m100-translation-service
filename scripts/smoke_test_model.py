"""Run a small manual translation using the real local M2M100 model."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.exceptions import TranslationServiceError  # noqa: E402
from app.services.translation import M2M100TranslationService  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse command-line smoke-test input."""
    parser = argparse.ArgumentParser(
        description="Load facebook/m2m100_418M locally and translate one short text.",
    )
    parser.add_argument("--text", required=True, help="Text to translate.")
    parser.add_argument("--source", required=True, help="M2M100 source language code.")
    parser.add_argument("--target", required=True, help="M2M100 target language code.")
    return parser.parse_args()


def main() -> int:
    """Load the real model and print one translation result."""
    arguments = parse_arguments()
    try:
        service = M2M100TranslationService(get_settings())
        service.load_model()
        result = service.translate(
            arguments.text,
            arguments.source,
            arguments.target,
        )
    except TranslationServiceError as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Smoke test failed unexpectedly: {exc}", file=sys.stderr)
        return 1

    print(f"Source language: {result.source_language}")
    print(f"Target language: {result.target_language}")
    print(f"Device: {result.device}")
    print(f"Status: {result.status}")
    print(f"Translation: {result.translated_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
