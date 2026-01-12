"""
HTTP Endpoints.
Health check and information endpoints.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import settings
from ..services.whisper_service import whisper_service
from ..services.openrouter_client import openrouter_client
from ..services.tts_service import tts_service
from ..services.session_manager import session_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    services: Dict[str, Any]


class InfoResponse(BaseModel):
    """Service information response model."""
    name: str
    version: str
    description: str
    endpoints: Dict[str, str]
    configuration: Dict[str, Any]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns status of all services.
    """
    services_status = {
        "whisper": whisper_service.get_model_info(),
        "openrouter": openrouter_client.get_model_info(),
        "tts": tts_service.get_service_info(),
    }

    # Check Redis connection
    try:
        await session_manager.connect()
        active_sessions = await session_manager.get_active_sessions_count()
        services_status["redis"] = {
            "connected": True,
            "active_sessions": active_sessions
        }
    except Exception as e:
        services_status["redis"] = {
            "connected": False,
            "error": str(e)
        }

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services=services_status
    )


@router.get("/info", response_model=InfoResponse)
async def get_info() -> InfoResponse:
    """
    Get service information.
    Returns details about the API and configuration.
    """
    return InfoResponse(
        name="Voice Form Assistant",
        version="1.0.0",
        description="Voice-based form filling assistant for Indian government portals",
        endpoints={
            "websocket": "/ws",
            "health": "/health",
            "info": "/info",
            "static": "/static/*"
        },
        configuration={
            "whisper_model": settings.WHISPER_MODEL,
            "openrouter_model": settings.OPENROUTER_MODEL,
            "tts_engine": settings.TTS_ENGINE,
            "debug": settings.DEBUG
        }
    )


@router.get("/ready")
async def readiness_check() -> Dict[str, bool]:
    """
    Readiness check endpoint.
    Returns whether the service is ready to accept requests.
    """
    checks = {
        "whisper_loaded": whisper_service.model is not None,
        "openrouter_configured": bool(settings.OPENROUTER_API_KEY),
        "redis_connected": False
    }

    try:
        await session_manager.connect()
        checks["redis_connected"] = True
    except Exception:
        pass

    is_ready = all(checks.values())

    if not is_ready:
        raise HTTPException(
            status_code=503,
            detail={"ready": False, "checks": checks}
        )

    return {"ready": True, "checks": checks}


@router.post("/warmup")
async def warmup_services() -> Dict[str, str]:
    """
    Warm up services by loading models.
    Call this on startup to ensure fast first response.
    """
    results = {}

    # Load Whisper model
    try:
        await whisper_service.load_model()
        results["whisper"] = "loaded"
    except Exception as e:
        results["whisper"] = f"error: {str(e)}"

    # Test Redis connection
    try:
        await session_manager.connect()
        results["redis"] = "connected"
    except Exception as e:
        results["redis"] = f"error: {str(e)}"

    # Test TTS
    try:
        test_audio = await tts_service.synthesize("Test")
        results["tts"] = "ready" if test_audio else "no audio"
    except Exception as e:
        results["tts"] = f"error: {str(e)}"

    return results


@router.get("/voices")
async def list_voices(language: str = None) -> Dict[str, Any]:
    """
    List available TTS voices.
    Optional language filter (e.g., 'en', 'hi').
    """
    voices = await tts_service.get_available_voices(language)
    return {
        "count": len(voices),
        "voices": voices
    }
