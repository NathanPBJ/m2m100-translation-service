"""Health endpoint response schema."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Schema returned by the health endpoint."""

    status: str
    service: str
    version: str
    environment: str
    model_loaded: bool
    model_name: str
    model_device: str
    language_detector_loaded: bool
    language_detector_name: str
    auto_detectable_language_count: int
    long_text_chunking_enabled: bool
    long_text_chunk_token_limit: int
    long_text_max_chunks: int
