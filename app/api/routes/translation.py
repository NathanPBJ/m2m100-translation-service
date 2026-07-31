"""Translation and supported-language API routes."""

import asyncio
import logging
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import (
    get_app_settings,
    get_language_detection_service,
    get_translation_semaphore,
    get_translation_service,
)
from app.core.config import Settings
from app.core.exceptions import InputTextTooLargeError, UnsupportedLanguageError
from app.schemas.error import ErrorResponse
from app.schemas.translation import (
    SupportedLanguagesResponse,
    TranslationRequest,
    TranslationResponse,
)
from app.services.language_detection import LinguaLanguageDetectionService
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
        "Translate text using the local M2M100 model. Source language defaults to "
        "`auto`, or a manual code can be supplied. Detection may reject short or "
        "ambiguous text. Use `/languages` for valid codes."
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
    detection_service: Annotated[
        LinguaLanguageDetectionService,
        Depends(get_language_detection_service),
    ],
) -> TranslationResponse:
    """Run blocking translation inference in a threadpool with bounded concurrency."""
    character_count = len(request_data.text)
    if character_count > settings.api_max_text_characters:
        raise InputTextTooLargeError(
            actual_characters=character_count,
            maximum_characters=settings.api_max_text_characters,
        )

    detection_result = None
    source_language_mode = "auto" if request_data.source_language == "auto" else "manual"
    resolved_source_language = request_data.source_language
    if source_language_mode == "auto":
        detection_result = await run_in_threadpool(
            detection_service.detect,
            request_data.text,
        )
        resolved_source_language = detection_result.language
        logger.info(
            "Auto-detected source language %s with confidence %.4f and margin %.4f",
            detection_result.language,
            detection_result.confidence,
            detection_result.confidence_margin,
        )

    if not service.supports_language(resolved_source_language):
        raise UnsupportedLanguageError(
            f"Source language code '{resolved_source_language}' is not supported."
        )
    if not service.supports_language(request_data.target_language):
        raise UnsupportedLanguageError(
            f"Target language code '{request_data.target_language}' is not supported."
        )

    logger.info(
        "Translation request started in %s mode for %s to %s (%d characters)",
        source_language_mode,
        resolved_source_language,
        request_data.target_language,
        character_count,
    )
    started_at = perf_counter()
    if resolved_source_language == request_data.target_language:
        result = await run_in_threadpool(
            service.translate,
            request_data.text,
            resolved_source_language,
            request_data.target_language,
        )
    else:
        async with semaphore:
            result = await run_in_threadpool(
                service.translate,
                request_data.text,
                resolved_source_language,
                request_data.target_language,
            )
    logger.info(
        "Translation request completed for %s to %s with status %s in %.2f seconds",
        resolved_source_language,
        request_data.target_language,
        result.status,
        perf_counter() - started_at,
    )
    return TranslationResponse(
        original_text=result.original_text,
        translated_text=result.translated_text,
        source_language=result.source_language,
        source_language_mode=source_language_mode,
        detected_language=(detection_result.language if detection_result is not None else None),
        detection_confidence=(
            detection_result.confidence if detection_result is not None else None
        ),
        detection_confidence_margin=(
            detection_result.confidence_margin if detection_result is not None else None
        ),
        target_language=result.target_language,
        model_name=result.model_name,
        device=result.device,
        status=result.status,
    )


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
    detection_service: Annotated[
        LinguaLanguageDetectionService,
        Depends(get_language_detection_service),
    ],
) -> SupportedLanguagesResponse:
    """Return supported codes without running inference or loading another model."""
    languages = list(service.get_supported_languages())
    auto_detectable_languages = list(detection_service.get_supported_languages())
    return SupportedLanguagesResponse(
        model_name=service.model_name,
        count=len(languages),
        languages=languages,
        language_detector=detection_service.detector_name,
        auto_detectable_count=len(auto_detectable_languages),
        auto_detectable_languages=auto_detectable_languages,
    )
