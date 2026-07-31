"""Health check route."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_app_settings, get_translation_service
from app.core.config import Settings
from app.schemas.health import HealthResponse
from app.services.translation import M2M100TranslationService

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check application and model health",
)
async def health_check(
    settings: Annotated[Settings, Depends(get_app_settings)],
    service: Annotated[
        M2M100TranslationService,
        Depends(get_translation_service),
    ],
) -> HealthResponse:
    """Return the current service health."""
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        model_loaded=service.is_loaded,
        model_name=service.model_name,
        model_device=service.device,
    )
