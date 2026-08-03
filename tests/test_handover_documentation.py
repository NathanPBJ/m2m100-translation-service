"""Contract tests for the final handover documentation."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

FINAL_DOCUMENTS = (
    "CHANGELOG.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/ARCHITECTURE.md",
    "docs/API_REFERENCE.md",
    "docs/DOCKER_HANDOVER.md",
    "docs/OPERATIONS_RUNBOOK.md",
    "docs/SENIOR_HANDOVER.md",
    "docs/RELEASE_NOTES_v0.1.0.md",
)

README_LINKS = (
    "docs/SENIOR_HANDOVER.md",
    "docs/API_REFERENCE.md",
    "docs/ARCHITECTURE.md",
    "docs/DOCKER_HANDOVER.md",
    "docs/OPERATIONS_RUNBOOK.md",
)

MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def read(relative_path: str) -> str:
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


@pytest.mark.parametrize("relative_path", FINAL_DOCUMENTS)
def test_final_document_exists(relative_path: str) -> None:
    assert (REPOSITORY_ROOT / relative_path).is_file()


@pytest.mark.parametrize("relative_path", README_LINKS)
def test_readme_links_to_required_handover_document(relative_path: str) -> None:
    readme = read("README.md")
    assert f"]({relative_path})" in readme
    assert (REPOSITORY_ROOT / relative_path).is_file()


def test_main_docker_command_is_documented() -> None:
    assert "docker compose up --build -d" in read("docs/SENIOR_HANDOVER.md")


@pytest.mark.parametrize("endpoint", ("/translate", "/detect-language"))
def test_main_api_endpoint_is_documented(endpoint: str) -> None:
    assert endpoint in read("docs/API_REFERENCE.md")


def test_model_and_detector_are_documented() -> None:
    architecture = read("docs/ARCHITECTURE.md").lower()
    assert "facebook/m2m100_418m" in architecture
    assert "lingua" in architecture


def test_long_text_and_cpu_only_limitations_are_documented() -> None:
    handover = read("docs/SENIOR_HANDOVER.md").lower()
    assert "long text" in handover
    assert "cpu-only" in handover
    assert "sequential" in handover


def test_no_auth_security_warning_is_documented() -> None:
    combined = (read("docs/SENIOR_HANDOVER.md") + read("docs/ARCHITECTURE.md")).lower()
    assert "tidak ada authentication" in combined or "belum mempunyai authentication" in combined
    assert "public internet" in combined


def test_named_volume_and_destructive_cleanup_warnings_are_documented() -> None:
    runbook = read("docs/OPERATIONS_RUNBOOK.md").lower()
    handover = read("docs/SENIOR_HANDOVER.md").lower()
    assert "named volume" in runbook and "named volume" in handover
    assert "docker compose down -v" in runbook
    assert "warning" in runbook and "menghapus" in runbook


def test_performance_is_labeled_as_development_measurement() -> None:
    handover = read("docs/SENIOR_HANDOVER.md").lower()
    assert "development measurements" in handover
    assert re.search(r"bukan\s+sla", handover)


def test_indonesian_malay_ambiguity_is_documented() -> None:
    handover = read("docs/SENIOR_HANDOVER.md").lower()
    assert "indonesian" in handover
    assert "malay" in handover
    assert "selamat pagi" in handover
    assert 'source_language="id"' in handover


def test_final_documentation_has_no_placeholders() -> None:
    placeholders = ("todo", "tbd", "changeme", "your_name", "<insert")
    for relative_path in ("README.md", *FINAL_DOCUMENTS):
        content = read(relative_path).lower()
        assert not any(placeholder in content for placeholder in placeholders), relative_path


def test_third_party_notices_include_core_artifacts() -> None:
    notices = read("THIRD_PARTY_NOTICES.md").lower()
    for name in ("facebook/m2m100_418m", "pytorch", "transformers", "sentencepiece", "lingua"):
        assert name in notices


def test_changelog_contains_release() -> None:
    changelog = read("CHANGELOG.md")
    assert re.search(r"^## \[0\.1\.0\] - 2026-08-03$", changelog, re.MULTILINE)


def test_release_notes_identify_initial_internal_release() -> None:
    release_notes = read("docs/RELEASE_NOTES_v0.1.0.md").lower()
    assert "initial internal release" in release_notes
    assert "v0.1.0" in release_notes


def test_documentation_does_not_claim_full_production_hardening() -> None:
    combined = "\n".join(read(path).lower() for path in ("README.md", *FINAL_DOCUMENTS))
    prohibited_claims = (
        "fully production-hardened service",
        "service is fully production-hardened",
        "enterprise production release",
        "production ready without",
    )
    assert not any(claim in combined for claim in prohibited_claims)


def test_all_relative_markdown_links_resolve() -> None:
    markdown_files = [REPOSITORY_ROOT / "README.md"] + sorted(
        (REPOSITORY_ROOT / "docs").glob("*.md")
    )
    for markdown_file in markdown_files:
        for match in MARKDOWN_LINK.finditer(markdown_file.read_text(encoding="utf-8")):
            raw_target = match.group(1).strip().strip("<>")
            target_without_anchor = unquote(raw_target.split("#", maxsplit=1)[0])
            if not target_without_anchor or "://" in target_without_anchor:
                continue
            target = (markdown_file.parent / target_without_anchor).resolve()
            assert target.is_file(), f"broken link in {markdown_file.name}: {raw_target}"
