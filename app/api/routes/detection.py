"""Language detection API route."""

from typing import Annotated

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_app_settings, get_language_detection_service
from app.core.config import Settings
from app.core.exceptions import InputTextTooLargeError
from app.schemas.detection import (
    LanguageCandidateResponse,
    LanguageDetectionRequest,
    LanguageDetectionResponse,
)
from app.schemas.error import ErrorResponse
from app.services.language_detection import LinguaLanguageDetectionService

router = APIRouter(tags=["Language Detection"])


@router.post(
    "/detect-language",
    response_model=LanguageDetectionResponse,
    responses={
        413: {"model": ErrorResponse, "description": "Character limit exceeded."},
        422: {
            "model": ErrorResponse,
            "description": "Invalid or uncertain detection input.",
        },
        500: {"model": ErrorResponse, "description": "Language detection failed."},
        503: {"model": ErrorResponse, "description": "Detector is unavailable."},
    },
    summary="Detect the dominant language of text",
    description=(
        "Detect one dominant language locally with Lingua. Only languages that "
        "are also supported by the loaded M2M100 tokenizer are considered."
    ),
)
async def detect_language(
    request_data: LanguageDetectionRequest,
    service: Annotated[
        LinguaLanguageDetectionService,
        Depends(get_language_detection_service),
    ],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> LanguageDetectionResponse:
    """Run synchronous Lingua detection in a threadpool."""
    character_count = len(request_data.text)
    if character_count > settings.api_max_text_characters:
        raise InputTextTooLargeError(
            actual_characters=character_count,
            maximum_characters=settings.api_max_text_characters,
        )

    result = await run_in_threadpool(service.detect, request_data.text)
    return LanguageDetectionResponse(
        language=result.language,
        confidence=result.confidence,
        confidence_margin=result.confidence_margin,
        detector=result.detector_name,
        status=result.status,
        candidates=[
            LanguageCandidateResponse.model_validate(candidate) for candidate in result.candidates
        ],
    )
