"""FastAPI application factory and module-level ASGI application."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app.api.error_handlers import register_error_handlers
from app.api.routes.detection import router as detection_router
from app.api.routes.health import router as health_router
from app.api.routes.translation import router as translation_router
from app.core.config import Settings, get_settings
from app.services.language_detection import LinguaLanguageDetectionService
from app.services.translation import M2M100TranslationService

logger = logging.getLogger(__name__)


def create_app(
    translation_service: M2M100TranslationService | None = None,
    language_detection_service: LinguaLanguageDetectionService | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create an application whose model service can be replaced in tests."""
    application_settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(
            logging,
            application_settings.log_level.upper(),
            logging.INFO,
        ),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "Starting %s version %s in %s environment",
            application_settings.app_name,
            application_settings.app_version,
            application_settings.app_env,
        )
        service = translation_service or M2M100TranslationService(application_settings)
        detector_service = language_detection_service or LinguaLanguageDetectionService(
            application_settings
        )
        app.state.settings = application_settings
        app.state.translation_service = service
        app.state.language_detection_service = detector_service
        app.state.translation_semaphore = asyncio.Semaphore(
            application_settings.translation_max_concurrency
        )
        try:
            logger.info(
                "Loading translation model %s on %s",
                service.model_name,
                service.device,
            )
            await run_in_threadpool(service.load_model)
            supported_translation_languages = service.get_supported_languages()
            logger.info(
                "Loading language detector %s",
                detector_service.detector_name,
            )
            await run_in_threadpool(
                detector_service.load_detector,
                supported_translation_languages,
            )
            logger.info("Application startup completed successfully")
            yield
        except Exception:
            logger.exception("Application startup or runtime lifecycle failed")
            raise
        finally:
            logger.info("Language detector unload started")
            try:
                await run_in_threadpool(detector_service.unload_detector)
            except Exception:
                logger.exception("Language detector unload failed during shutdown")
            logger.info("Translation model unload started")
            try:
                await run_in_threadpool(service.unload_model)
            except Exception:
                logger.exception("Translation model unload failed during shutdown")
            logger.info("Application shutdown completed")

    application = FastAPI(
        title=application_settings.app_name,
        version=application_settings.app_version,
        description=(
            "Local multilingual translation API powered by facebook/m2m100_418M. "
            "Source language can be supplied manually or detected locally with Lingua. "
            "Token-aware chunking supports long text without silent truncation."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    application.state.settings = application_settings
    register_error_handlers(application)
    application.include_router(health_router)
    application.include_router(detection_router)
    application.include_router(translation_router)

    @application.get("/", summary="Get basic service information")
    async def root() -> dict[str, str]:
        """Return basic service information."""
        return {
            "service": application_settings.app_name,
            "version": application_settings.app_version,
            "status": "running",
            "docs": "/docs",
        }

    return application


app = create_app()
