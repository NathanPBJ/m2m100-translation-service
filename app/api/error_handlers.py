"""Centralized exception handlers for consistent and safe API errors."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    DeviceConfigurationError,
    InputTextTooLargeError,
    InputTooLongError,
    InvalidLanguageDetectionInputError,
    InvalidTranslationInputError,
    LanguageDetectionInferenceError,
    LanguageDetectionUncertainError,
    LanguageDetectorLoadError,
    LanguageDetectorNotLoadedError,
    ModelLoadError,
    ModelNotLoadedError,
    TranslationInferenceError,
    UnsupportedLanguageError,
)
from app.schemas.error import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)

ExceptionHandler = Callable[[Request, Exception], Awaitable[JSONResponse]]


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | list[Any] | None = None,
) -> JSONResponse:
    response = ErrorResponse(
        error=ErrorDetail(code=code, message=message, details=details),
    )
    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(exclude_none=True),
    )


async def request_validation_error_handler(
    _: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return sanitized Pydantic validation details without request values."""
    validation_details = []
    for error in exc.errors():
        location = [str(part) for part in error["loc"] if part not in {"body"}]
        validation_details.append(
            {
                "field": ".".join(location),
                "message": error["msg"],
                "type": error["type"],
            }
        )
    return _error_response(
        422,
        "request_validation_error",
        "Request validation failed.",
        validation_details,
    )


async def invalid_translation_input_handler(
    _: Request,
    exc: InvalidTranslationInputError,
) -> JSONResponse:
    return _error_response(422, "invalid_translation_input", str(exc))


async def unsupported_language_handler(
    _: Request,
    exc: UnsupportedLanguageError,
) -> JSONResponse:
    return _error_response(422, "unsupported_language", str(exc))


async def text_too_large_handler(
    _: Request,
    exc: InputTextTooLargeError,
) -> JSONResponse:
    return _error_response(
        413,
        "text_too_large",
        str(exc),
        {
            "actual_characters": exc.actual_characters,
            "maximum_characters": exc.maximum_characters,
        },
    )


async def input_too_long_handler(
    _: Request,
    exc: InputTooLongError,
) -> JSONResponse:
    return _error_response(
        413,
        "input_too_long",
        "Input exceeds the maximum token limit.",
        {
            "actual_tokens": exc.actual_tokens,
            "maximum_tokens": exc.maximum_tokens,
        },
    )


async def model_not_loaded_handler(
    _: Request,
    exc: ModelNotLoadedError,
) -> JSONResponse:
    return _error_response(503, "model_not_loaded", str(exc))


async def model_load_error_handler(_: Request, exc: ModelLoadError) -> JSONResponse:
    logger.error(
        "Translation model loading failed",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return _error_response(
        503,
        "model_load_failed",
        "The translation model could not be loaded.",
    )


async def device_configuration_error_handler(
    _: Request,
    exc: DeviceConfigurationError,
) -> JSONResponse:
    return _error_response(503, "device_configuration_error", str(exc))


async def invalid_detection_input_handler(
    _: Request,
    _exc: InvalidLanguageDetectionInputError,
) -> JSONResponse:
    return _error_response(
        422,
        "invalid_detection_input",
        "The text does not contain enough detectable language content.",
    )


async def language_detection_uncertain_handler(
    _: Request,
    _exc: LanguageDetectionUncertainError,
) -> JSONResponse:
    return _error_response(
        422,
        "language_detection_uncertain",
        "The source language could not be detected reliably.",
    )


async def language_detector_not_loaded_handler(
    _: Request,
    _exc: LanguageDetectorNotLoadedError,
) -> JSONResponse:
    return _error_response(
        503,
        "language_detector_not_loaded",
        "The language detector is not currently loaded.",
    )


async def language_detector_load_error_handler(
    _: Request,
    exc: LanguageDetectorLoadError,
) -> JSONResponse:
    logger.error(
        "Language detector loading failed",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return _error_response(
        503,
        "language_detector_load_failed",
        "The language detector could not be loaded.",
    )


async def language_detection_inference_error_handler(
    _: Request,
    exc: LanguageDetectionInferenceError,
) -> JSONResponse:
    logger.error(
        "Language detection request failed",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return _error_response(
        500,
        "language_detection_failed",
        "Language detection could not be completed.",
    )


async def translation_inference_error_handler(
    _: Request,
    exc: TranslationInferenceError,
) -> JSONResponse:
    logger.error(
        "Translation inference request failed",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return _error_response(
        500,
        "translation_failed",
        "Translation could not be completed.",
    )


async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unexpected API error",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return _error_response(
        500,
        "internal_server_error",
        "An unexpected internal error occurred.",
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all application error mappings in one place."""
    handlers: dict[type[Exception], ExceptionHandler] = {
        RequestValidationError: request_validation_error_handler,
        InvalidTranslationInputError: invalid_translation_input_handler,
        UnsupportedLanguageError: unsupported_language_handler,
        InputTextTooLargeError: text_too_large_handler,
        InputTooLongError: input_too_long_handler,
        ModelNotLoadedError: model_not_loaded_handler,
        ModelLoadError: model_load_error_handler,
        DeviceConfigurationError: device_configuration_error_handler,
        InvalidLanguageDetectionInputError: invalid_detection_input_handler,
        LanguageDetectionUncertainError: language_detection_uncertain_handler,
        LanguageDetectorNotLoadedError: language_detector_not_loaded_handler,
        LanguageDetectorLoadError: language_detector_load_error_handler,
        LanguageDetectionInferenceError: language_detection_inference_error_handler,
        TranslationInferenceError: translation_inference_error_handler,
        Exception: unexpected_error_handler,
    }
    for exception_type, handler in handlers.items():
        app.add_exception_handler(exception_type, handler)
