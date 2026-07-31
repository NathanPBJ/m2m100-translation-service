"""Run one manual language detection using the local Lingua package."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import M2M100Tokenizer  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.exceptions import LanguageDetectionError  # noqa: E402
from app.services.language_detection import LinguaLanguageDetectionService  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse command-line detection input."""
    parser = argparse.ArgumentParser(
        description="Detect one dominant language locally with Lingua.",
    )
    parser.add_argument("--text", required=True, help="Text whose language will be detected.")
    parser.add_argument(
        "--show-candidates",
        action="store_true",
        help="Display the configured top language candidates.",
    )
    return parser.parse_args()


def main() -> int:
    """Load the cached tokenizer language list and run Lingua detection."""
    arguments = parse_arguments()
    settings = get_settings()
    try:
        tokenizer = M2M100Tokenizer.from_pretrained(
            settings.model_name,
            cache_dir=str(settings.model_cache_dir),
            local_files_only=settings.model_local_files_only,
        )
        service = LinguaLanguageDetectionService(settings)
        service.load_detector(tokenizer.lang_code_to_id)
        result = service.detect(arguments.text)
    except LanguageDetectionError as exc:
        print(f"Detection smoke test failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Detection smoke test failed unexpectedly: {exc}", file=sys.stderr)
        return 1

    print(f"Detected language: {result.language}")
    print(f"Confidence: {result.confidence:.6f}")
    print(f"Confidence margin: {result.confidence_margin:.6f}")
    if arguments.show_candidates:
        print("Candidates:")
        for candidate in result.candidates:
            print(f"  {candidate.language}: {candidate.confidence:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
