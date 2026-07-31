"""Consistent API error response schemas."""

from typing import Any

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Machine-readable and human-readable API error information."""

    code: str
    message: str
    details: dict[str, Any] | list[Any] | None = None


class ErrorResponse(BaseModel):
    """Top-level error envelope returned by the API."""

    error: ErrorDetail
