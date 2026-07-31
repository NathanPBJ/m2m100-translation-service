"""Translation and supported-language API routes."""

import asyncio
import logging
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import (
    get_app_settings,
    get_translation_semaphore,
    get_translation_service,
)
from app.core.config import Settings
from app.core.exceptions import InputTextTooLargeError
from app.schemas.error import ErrorResponse
from app.schemas.translation import (
    SupportedLanguagesResponse,
    TranslationRequest,
    TranslationResponse,
)
from app.services.translation import M2M100TranslationService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["translation"])

ERROR_RESPONSES = {
    413: {"model": ErrorResponse, "description": "Character or token limit exceeded."},
    422: {"model": ErrorResponse, "description": "Invalid input or unsupported language."},
    500: {"model": ErrorResponse, "description": "Translation inference failed."},
    503: {"model": ErrorResponse, "description": "Translation model is unavailable."},
}


@router.post(
    "/translate",
    response_model=TranslationResponse,
    responses=ERROR_RESPONSES,
    summary="Translate text between two M2M100 languages",
    description=(
        "Translate text using the local M2M100 model. The source language must be "
        "provided explicitly; `auto` is not supported. Use `/languages` for valid codes."
    ),
)
async def translate_text(
    request_data: TranslationRequest,
    service: Annotated[
        M2M100TranslationService,
        Depends(get_translation_service),
    ],
    semaphore: Annotated[
        asyncio.Semaphore,
        Depends(get_translation_semaphore),
    ],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> TranslationResponse:
    """Run blocking translation inference in a threadpool with bounded concurrency."""
    character_count = len(request_data.text)
    if character_count > settings.api_max_text_characters:
        raise InputTextTooLargeError(
            actual_characters=character_count,
            maximum_characters=settings.api_max_text_characters,
        )

    logger.info(
        "Translation request started for %s to %s (%d characters)",
        request_data.source_language,
        request_data.target_language,
        character_count,
    )
    started_at = perf_counter()
    async with semaphore:
        result = await run_in_threadpool(
            service.translate,
            request_data.text,
            request_data.source_language,
            request_data.target_language,
        )
    logger.info(
        "Translation request completed for %s to %s with status %s in %.2f seconds",
        request_data.source_language,
        request_data.target_language,
        result.status,
        perf_counter() - started_at,
    )
    return TranslationResponse.model_validate(result)


@router.get(
    "/languages",
    response_model=SupportedLanguagesResponse,
    responses={503: ERROR_RESPONSES[503]},
    summary="List supported M2M100 language codes",
    description=(
        "Return the sorted language codes exposed by the currently loaded M2M100 tokenizer."
    ),
)
async def supported_languages(
    service: Annotated[
        M2M100TranslationService,
        Depends(get_translation_service),
    ],
) -> SupportedLanguagesResponse:
    """Return supported codes without running inference or loading another model."""
    languages = list(service.get_supported_languages())
    return SupportedLanguagesResponse(
        model_name=service.model_name,
        count=len(languages),
        languages=languages,
    )
