"""Static checks for the Docker deployment configuration.

These tests intentionally do not require a Docker CLI or daemon.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE_PATH = ROOT / "Dockerfile"
COMPOSE_PATH = ROOT / "compose.yaml"
DOCKERIGNORE_PATH = ROOT / ".dockerignore"
HANDOVER_PATH = ROOT / "docs" / "DOCKER_HANDOVER.md"
README_PATH = ROOT / "README.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalized_lines(path: Path) -> set[str]:
    return {
        line.strip().rstrip("/")
        for line in read(path).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def runtime_command(dockerfile: str) -> list[str]:
    commands = re.findall(r"(?im)^CMD\s+(\[.*\])\s*$", dockerfile)
    assert commands, "Dockerfile must define an exec-form CMD"
    parsed = json.loads(commands[-1])
    assert isinstance(parsed, list)
    return parsed


def test_required_docker_files_exist() -> None:
    assert DOCKERFILE_PATH.is_file()
    assert COMPOSE_PATH.is_file()
    assert DOCKERIGNORE_PATH.is_file()


def test_dockerfile_uses_pinned_python_slim_base() -> None:
    dockerfile = read(DOCKERFILE_PATH)
    from_match = re.search(r"(?im)^FROM\s+(\S+)", dockerfile)
    assert from_match
    image = from_match.group(1).lower()
    assert image == "python:3.12-slim-bookworm"
    assert "latest" not in image
    assert "tiangolo" not in image and "uvicorn-gunicorn" not in image


def test_dockerfile_sets_workdir_and_non_root_user() -> None:
    dockerfile = read(DOCKERFILE_PATH)
    assert re.search(r"(?im)^WORKDIR\s+/service\s*$", dockerfile)
    assert re.search(r"\b(?:useradd|adduser)\b.*\bappuser\b", dockerfile)
    assert re.search(r"(?im)^USER\s+appuser\s*$", dockerfile)
    assert dockerfile.rfind("USER appuser") < dockerfile.rfind("CMD [")


def test_dockerfile_runtime_command_is_safe_and_single_worker() -> None:
    dockerfile = read(DOCKERFILE_PATH)
    command = runtime_command(dockerfile)
    assert command[:3] == ["python", "-m", "uvicorn"]
    assert "app.main:app" in command
    assert "--workers" in command
    assert command[command.index("--workers") + 1] == "1"
    assert "--reload" not in command
    assert "gunicorn" not in " ".join(command).lower()


def test_dockerfile_installs_the_validated_cpu_only_torch_wheel() -> None:
    dockerfile = read(DOCKERFILE_PATH)
    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert "torch==2.13.0+cpu" in dockerfile
    assert not re.search(r"(?i)(?:cu(?:da)?\d+|nvidia/cuda)", dockerfile)


def test_dockerfile_exposes_internal_port_and_has_real_healthcheck() -> None:
    dockerfile = read(DOCKERFILE_PATH)
    assert re.search(r"(?im)^EXPOSE\s+8000\s*$", dockerfile)
    health_match = re.search(r"(?ims)^HEALTHCHECK\b.*?(?=^\w|\Z)", dockerfile)
    assert health_match
    healthcheck = health_match.group(0)
    assert "http://127.0.0.1:8000/health" in healthcheck
    assert "/translate" not in healthcheck
    assert "model_loaded" in healthcheck
    assert "language_detector_loaded" in healthcheck


def test_dockerfile_does_not_copy_local_state() -> None:
    dockerfile = read(DOCKERFILE_PATH).lower()
    copy_lines = re.findall(r"(?im)^(?:copy|add)\s+.*$", dockerfile)
    copied = "\n".join(copy_lines)
    assert "models" not in copied
    assert ".venv" not in copied
    assert not re.search(r"(?im)^(?:copy|add)\s+\.\s+", dockerfile)


def test_compose_has_only_translation_service() -> None:
    compose = read(COMPOSE_PATH)
    service_block = compose.split("\nvolumes:", maxsplit=1)[0]
    service_names = re.findall(r"(?m)^  ([a-zA-Z0-9_-]+):\s*$", service_block)
    assert service_names == ["translation-service"]


def test_compose_maps_host_port_to_internal_8000() -> None:
    compose = read(COMPOSE_PATH)
    assert re.search(r'(?m)^\s*-\s*["\']?\$\{APP_PORT:-8000\}:8000["\']?\s*$', compose)
    assert "network_mode: host" not in compose.lower()


def test_compose_uses_persistent_named_model_volume() -> None:
    compose = read(COMPOSE_PATH)
    assert re.search(r"(?m)^\s+-\s+model-cache:/models\s*$", compose)
    assert re.search(r"(?m)^volumes:\s*\n\s{2}model-cache:\s*$", compose)


def test_compose_cpu_and_concurrency_defaults() -> None:
    compose = read(COMPOSE_PATH)
    assert re.search(r"MODEL_DEVICE:\s*\$\{MODEL_DEVICE:-cpu\}", compose)
    assert re.search(
        r"TRANSLATION_MAX_CONCURRENCY:\s*\$\{TRANSLATION_MAX_CONCURRENCY:-1\}",
        compose,
    )


def test_compose_has_no_privileged_or_scaled_runtime() -> None:
    compose = read(COMPOSE_PATH).lower()
    assert "privileged:" not in compose
    assert "replicas:" not in compose
    assert "/var/run/docker.sock" not in compose
    assert "network_mode:" not in compose
    assert "env_file:" not in compose


def test_dockerignore_excludes_private_and_large_local_files() -> None:
    ignored = normalized_lines(DOCKERIGNORE_PATH)
    assert ".git" in ignored
    assert ".env" in ignored
    assert ".venv" in ignored
    assert "models" in ignored
    assert "*.safetensors" in ignored
    assert "*.bin" in ignored


def test_dockerignore_keeps_runtime_inputs_available() -> None:
    ignored = normalized_lines(DOCKERIGNORE_PATH)
    for required in {"app", "requirements.txt", "dockerfile", "compose.yaml"}:
        assert required not in {entry.lower() for entry in ignored}


def test_handover_and_readme_document_docker_usage() -> None:
    assert HANDOVER_PATH.is_file()
    readme = read(README_PATH).lower()
    assert "docker compose up --build -d" in readme
    assert "docs/docker_handover.md" in readme


def test_docker_configuration_contains_no_literal_secrets() -> None:
    configuration = "\n".join([read(DOCKERFILE_PATH), read(COMPOSE_PATH), read(DOCKERIGNORE_PATH)])
    secret_assignment = re.compile(
        r"(?im)^\s*(?:password|secret|api[_-]?key|access[_-]?token)"
        r"\s*[:=]\s*[\"']?(?!\$\{|$)[^\s\"']+"
    )
    assert not secret_assignment.search(configuration)
