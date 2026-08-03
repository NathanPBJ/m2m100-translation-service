"""Run non-destructive repository and live API handover checks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ValidationFailure(Exception):
    """A concise validation failure intended for terminal output."""


REQUIRED_PATHS = (
    "README.md",
    "LICENSE",
    "Dockerfile",
    "compose.yaml",
    ".dockerignore",
    ".env.example",
    "CHANGELOG.md",
    "THIRD_PARTY_NOTICES.md",
    "requirements.txt",
    "pyproject.toml",
    "app/main.py",
    "app/services/translation.py",
    "app/services/language_detection.py",
    "app/services/text_chunking.py",
    "docs/ARCHITECTURE.md",
    "docs/API_REFERENCE.md",
    "docs/DOCKER_HANDOVER.md",
    "docs/OPERATIONS_RUNBOOK.md",
    "docs/SENIOR_HANDOVER.md",
    "docs/RELEASE_NOTES_v0.1.0.md",
    "scripts/smoke_test_docker.py",
    "scripts/final_validation.py",
    "tests/test_handover_documentation.py",
)

DOCUMENTATION_PATHS = (
    "README.md",
    "CHANGELOG.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/ARCHITECTURE.md",
    "docs/API_REFERENCE.md",
    "docs/DOCKER_HANDOVER.md",
    "docs/OPERATIONS_RUNBOOK.md",
    "docs/SENIOR_HANDOVER.md",
    "docs/RELEASE_NOTES_v0.1.0.md",
)

PLACEHOLDERS = ("todo", "tbd", "changeme", "your_name", "<insert")
MODEL_EXTENSIONS = {".bin", ".safetensors", ".pt", ".pth", ".onnx"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".7z"}
FORBIDDEN_TRACKED_PARTS = {".venv", "venv", "models", "cache", "__pycache__"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Live API base URL (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--skip-http",
        action="store_true",
        help="Skip checks that require a running service.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFailure(message)


def read_text(root: Path, relative_path: str) -> str:
    try:
        return (root / relative_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationFailure(f"cannot read {relative_path}: {exc}") from exc


def validate_repository_files(root: Path) -> None:
    require(root.is_dir(), f"repository root does not exist: {root}")
    missing = [path for path in REQUIRED_PATHS if not (root / path).is_file()]
    require(not missing, "missing required files: " + ", ".join(missing))
    require((root / "tests").is_dir(), "tests directory is missing")
    require(any((root / "tests").glob("test_*.py")), "no test modules were found")
    print("PASS repository files")


def validate_documentation(root: Path) -> None:
    contents = {path: read_text(root, path) for path in DOCUMENTATION_PATHS}
    lowered = {path: content.lower() for path, content in contents.items()}

    checks = (
        ("docker compose up --build -d" in lowered["README.md"], "README Docker quick start"),
        (
            "docker compose up --build -d" in lowered["docs/SENIOR_HANDOVER.md"],
            "senior handover main command",
        ),
        ("/translate" in lowered["docs/API_REFERENCE.md"], "API /translate reference"),
        (
            "m2m100" in lowered["docs/ARCHITECTURE.md"]
            and "lingua" in lowered["docs/ARCHITECTURE.md"],
            "architecture M2M100/Lingua description",
        ),
        (
            "docker compose down -v" in lowered["docs/OPERATIONS_RUNBOOK.md"]
            and "warning" in lowered["docs/OPERATIONS_RUNBOOK.md"],
            "operations model-cache warning",
        ),
        (
            "facebook/m2m100_418m" in lowered["THIRD_PARTY_NOTICES.md"],
            "third-party model notice",
        ),
        ("0.1.0" in lowered["CHANGELOG.md"], "0.1.0 changelog entry"),
    )
    failed = [label for condition, label in checks if not condition]
    require(not failed, "documentation checks failed: " + ", ".join(failed))

    placeholder_hits = [
        f"{path}:{placeholder}"
        for path, content in lowered.items()
        for placeholder in PLACEHOLDERS
        if placeholder in content
    ]
    require(not placeholder_hits, "final placeholders found: " + ", ".join(placeholder_hits))
    print("PASS documentation")


def run_git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise ValidationFailure(f"git could not be executed: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown git error"
        raise ValidationFailure(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout.strip()


def validate_git(root: Path) -> None:
    require(shutil.which("git") is not None, "git command is unavailable")
    require((root / ".git").exists(), "repository root is not a Git worktree")
    require(not run_git(root, "status", "--porcelain=v1"), "Git working tree is not clean")
    require(run_git(root, "branch", "--show-current") == "main", "active branch is not main")
    require(bool(run_git(root, "remote", "get-url", "origin")), "origin remote is missing")

    tracked = [line.replace("\\", "/") for line in run_git(root, "ls-files").splitlines()]
    violations: list[str] = []
    for tracked_path in tracked:
        path = Path(tracked_path)
        lowered_parts = {part.lower() for part in path.parts}
        lowered_name = path.name.lower()
        suffix = path.suffix.lower()
        if lowered_name == ".env" or FORBIDDEN_TRACKED_PARTS & lowered_parts:
            violations.append(tracked_path)
        elif (
            suffix in MODEL_EXTENSIONS
            or suffix in ARCHIVE_EXTENSIONS
            or lowered_name.endswith(".log")
        ):
            violations.append(tracked_path)
        else:
            try:
                if (root / tracked_path).stat().st_size > 10 * 1024 * 1024:
                    violations.append(f"{tracked_path} (>10 MiB)")
            except OSError as exc:
                raise ValidationFailure(
                    f"cannot inspect tracked file {tracked_path}: {exc}"
                ) from exc
    require(not violations, "suspicious tracked files: " + ", ".join(violations))
    print("PASS git audit")


def request_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 600.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            require(200 <= response.status < 300, f"{path} returned HTTP {response.status}")
            decoded = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ValidationFailure(f"{path} returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise ValidationFailure(f"{path} is unavailable: {exc.reason}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"{path} returned an unreadable response: {exc}") from exc
    require(isinstance(decoded, dict), f"{path} response is not a JSON object")
    return decoded


def validate_http(base_url: str) -> None:
    health = request_json(base_url, "/health", timeout=30.0)
    require(
        health.get("status") == "healthy"
        and health.get("model_loaded") is True
        and health.get("language_detector_loaded") is True,
        "health response is not ready",
    )
    print("PASS health")

    languages = request_json(base_url, "/languages", timeout=30.0)
    require(
        isinstance(languages.get("languages"), list)
        and {"en", "id", "ru"} <= set(languages["languages"])
        and isinstance(languages.get("auto_detectable_languages"), list),
        "languages response is incomplete",
    )
    print("PASS languages")

    detection = request_json(
        base_url,
        "/detect-language",
        {"text": "Сегодня хорошая погода"},
    )
    require(
        detection.get("language") == "ru"
        and detection.get("status") == "detected"
        and isinstance(detection.get("candidates"), list),
        "language detection response is unexpected",
    )
    print("PASS detection")

    manual = request_json(
        base_url,
        "/translate",
        {"text": "Good morning", "source_language": "en", "target_language": "id"},
    )
    require(
        manual.get("status") == "translated"
        and manual.get("source_language") == "en"
        and manual.get("target_language") == "id"
        and manual.get("source_language_mode") == "manual"
        and bool(manual.get("translated_text")),
        "manual translation response is unexpected",
    )
    print("PASS manual translation")

    automatic = request_json(
        base_url,
        "/translate",
        {"text": "Сегодня хорошая погода", "target_language": "id"},
    )
    require(
        automatic.get("status") == "translated"
        and automatic.get("source_language") == "ru"
        and automatic.get("detected_language") == "ru"
        and automatic.get("source_language_mode") == "auto"
        and bool(automatic.get("translated_text")),
        "automatic translation response is unexpected",
    )
    print("PASS automatic translation")

    original = "Selamat pagi, ini pemeriksaan tanpa perubahan."
    unchanged = request_json(
        base_url,
        "/translate",
        {"text": original, "source_language": "id", "target_language": "id"},
    )
    require(
        unchanged.get("status") == "unchanged"
        and unchanged.get("translated_text") == original
        and unchanged.get("chunk_count") == 0,
        "same-language translation response is unexpected",
    )
    print("PASS unchanged translation")


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    try:
        validate_repository_files(root)
        validate_documentation(root)
        validate_git(root)
        if not args.skip_http:
            validate_http(args.base_url)
    except ValidationFailure as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
