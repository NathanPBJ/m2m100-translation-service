"""Run a small end-to-end smoke test against the Dockerized service."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class SmokeTestError(RuntimeError):
    """A concise smoke-test failure suitable for terminal output."""


class ApiClient:
    """Minimal JSON client using only the Python standard library."""

    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read()
                if not 200 <= response.status < 300:
                    raise SmokeTestError(f"{endpoint}: HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            detail = exc.read(300).decode("utf-8", errors="replace")
            raise SmokeTestError(f"{endpoint}: HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SmokeTestError(f"{endpoint}: {exc}") from exc

        try:
            parsed = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SmokeTestError(f"{endpoint}: response is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise SmokeTestError(f"{endpoint}: expected a JSON object")
        return parsed

    def get(self, endpoint: str) -> dict[str, Any]:
        return self.request("GET", endpoint)

    def post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", endpoint, payload)


def require(condition: bool, endpoint: str, message: str) -> None:
    """Raise a concise failure when an API contract is not met."""
    if not condition:
        raise SmokeTestError(f"{endpoint}: {message}")


def run_smoke_test(client: ApiClient, include_long_text: bool) -> None:
    health = client.get("/health")
    require(health.get("status") == "healthy", "/health", "status is not healthy")
    require(health.get("model_loaded") is True, "/health", "model is not loaded")
    require(
        health.get("language_detector_loaded") is True,
        "/health",
        "language detector is not loaded",
    )
    require(health.get("model_device") == "cpu", "/health", "model device is not CPU")
    require(
        health.get("long_text_chunking_enabled") is True,
        "/health",
        "long-text chunking is disabled",
    )
    print("PASS health")

    languages = client.get("/languages")
    translation_codes = set(languages.get("languages", []))
    detection_codes = set(languages.get("auto_detectable_languages", []))
    required_codes = {"en", "id", "ja", "ru", "zh"}
    require(
        bool(translation_codes) and required_codes <= translation_codes,
        "/languages",
        "required translation languages are missing",
    )
    require(
        bool(detection_codes) and required_codes <= detection_codes,
        "/languages",
        "required auto-detectable languages are missing",
    )
    print("PASS languages")

    russian_text = "Сегодня хорошая погода"
    detection = client.post("/detect-language", {"text": russian_text})
    require(detection.get("language") == "ru", "/detect-language", "Russian was not detected")
    print("PASS detection")

    manual = client.post(
        "/translate",
        {
            "text": "Good morning",
            "source_language": "en",
            "target_language": "id",
        },
    )
    require(
        manual.get("status") == "translated" and bool(manual.get("translated_text")),
        "/translate",
        "manual translation returned no translated text",
    )
    require(
        manual.get("source_language") == "en" and manual.get("target_language") == "id",
        "/translate",
        "manual translation language metadata is incorrect",
    )
    print("PASS manual translation")

    automatic = client.post(
        "/translate",
        {"text": russian_text, "target_language": "id"},
    )
    require(
        automatic.get("status") == "translated" and bool(automatic.get("translated_text")),
        "/translate",
        "automatic translation returned no translated text",
    )
    require(
        automatic.get("detected_language") == "ru"
        and automatic.get("source_language") == "ru"
        and automatic.get("target_language") == "id",
        "/translate",
        "automatic translation language metadata is incorrect",
    )
    print("PASS automatic translation")

    unchanged_text = "This is a clear English sentence used for a container test."
    unchanged = client.post(
        "/translate",
        {
            "text": unchanged_text,
            "source_language": "en",
            "target_language": "en",
        },
    )
    require(
        unchanged.get("status") == "unchanged"
        and unchanged.get("translated_text") == unchanged_text
        and unchanged.get("chunk_count") == 0,
        "/translate",
        "same-language response was not unchanged",
    )
    print("PASS unchanged translation")

    if include_long_text:
        fixture = (
            Path(__file__).resolve().parents[1]
            / "tests"
            / "fixtures"
            / "long_text"
            / "general_opinion_en.txt"
        )
        long_text = fixture.read_text(encoding="utf-8")
        long_result = client.post(
            "/translate",
            {
                "text": long_text,
                "source_language": "en",
                "target_language": "id",
            },
        )
        require(
            long_result.get("status") == "translated"
            and long_result.get("chunked") is True
            and int(long_result.get("chunk_count", 0)) > 1
            and bool(long_result.get("translated_text")),
            "/translate",
            "long-text translation did not produce multiple translated chunks",
        )
        final_health = client.get("/health")
        require(
            final_health.get("status") == "healthy",
            "/health",
            "service became unhealthy after long-text translation",
        )
        print("PASS long-text translation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--include-long-text", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        print("FAIL configuration: --timeout must be greater than zero", file=sys.stderr)
        return 2
    try:
        run_smoke_test(ApiClient(args.base_url, args.timeout), args.include_long_text)
    except (SmokeTestError, OSError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
