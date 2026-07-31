"""FastAPI dependencies backed by application lifespan state."""

import asyncio
from typing import cast

from fastapi import Request

from app.core.config import Settings
from app.core.exceptions import ModelNotLoadedError
from app.services.translation import M2M100TranslationService


def get_translation_service(request: Request) -> M2M100TranslationService:
    """Return the loaded service stored by the application lifespan."""
    service = getattr(request.app.state, "translation_service", None)
    if service is None or not service.is_loaded:
        raise ModelNotLoadedError("The translation model is not currently loaded.")
    return cast(M2M100TranslationService, service)


def get_translation_semaphore(request: Request) -> asyncio.Semaphore:
    """Return the process-local inference concurrency semaphore."""
    semaphore = getattr(request.app.state, "translation_semaphore", None)
    if semaphore is None:
        raise ModelNotLoadedError("Translation concurrency control is not available.")
    return cast(asyncio.Semaphore, semaphore)


def get_app_settings(request: Request) -> Settings:
    """Return settings captured when the application was created."""
    return cast(Settings, request.app.state.settings)
