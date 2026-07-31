"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the service."""

    app_name: str = "M2M100 Translation Service"
    app_version: str = "0.1.0"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    model_name: str = "facebook/m2m100_418M"
    model_cache_dir: Path = Path("./models")
    model_device: Literal["auto", "cpu", "cuda"] = "auto"
    model_local_files_only: bool = False
    model_max_input_tokens: PositiveInt = 512
    model_max_new_tokens: PositiveInt = 512
    model_num_beams: PositiveInt = 4
    api_max_text_characters: PositiveInt = 10_000
    translation_max_concurrency: PositiveInt = 1
    long_text_chunking_enabled: bool = True
    long_text_chunk_max_tokens: PositiveInt = 400
    long_text_max_chunks: PositiveInt = 64
    language_detection_min_confidence: float = Field(default=0.30, ge=0.0, le=1.0)
    language_detection_min_relative_distance: float = Field(
        default=0.05,
        ge=0.0,
        le=0.99,
    )
    language_detection_min_alphabetic_characters: PositiveInt = 3
    language_detection_max_candidates: PositiveInt = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_chunk_token_limit(self) -> "Settings":
        """Keep the chunk budget within the model's hard input limit."""
        if self.long_text_chunk_max_tokens > self.model_max_input_tokens:
            raise ValueError("LONG_TEXT_CHUNK_MAX_TOKENS cannot exceed MODEL_MAX_INPUT_TOKENS.")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance."""
    return Settings()
