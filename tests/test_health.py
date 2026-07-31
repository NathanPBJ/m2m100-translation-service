"""Tests for the root and health endpoints."""

from fastapi.testclient import TestClient

from app.main import app


def test_root_endpoint() -> None:
    """The root endpoint reports the service as running."""
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "M2M100 Translation Service",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


def test_health_endpoint() -> None:
    """The health endpoint matches the public response schema."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "M2M100 Translation Service",
        "version": "0.1.0",
        "environment": "development",
    }
