"""
HTTP Endpoints.
Health check, information, and language support endpoints.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import settings
from ..services.assemblyai_service import assemblyai_service, SUPPORTED_LANGUAGES
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
    supported_languages: Dict[str, Any]


class LanguagesResponse(BaseModel):
    """Supported languages response model."""
    total_count: int
    indic_languages: Dict[str, str]
    english: Dict[str, str]
    engines: Dict[str, List[str]]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns status of all services including AssemblyAI STT.
    """
    # Get AssemblyAI service info
    stt_info = assemblyai_service.get_model_info()
    
    services_status = {
        "stt": {
            "service": stt_info["service"],
            "engine": stt_info["engine"],
            "configured": stt_info["configured"],
            "supported_languages": stt_info["supported_languages"],
            "language_detection": stt_info["language_detection"]
        },
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
        version="2.0.0",
        services=services_status
    )


@router.get("/info", response_model=InfoResponse)
async def get_info() -> InfoResponse:
    """
    Get service information.
    Returns details about the API, configuration, and supported languages.
    """
    # Build languages dict
    languages = {
        "total_count": len(SUPPORTED_LANGUAGES),
        "languages": SUPPORTED_LANGUAGES,
        "codes": list(SUPPORTED_LANGUAGES.keys())
    }
    
    return InfoResponse(
        name="Voice Form Assistant",
        version="2.0.0",
        description="Voice-based form filling assistant for Indian government portals. "
                    "Cloud-based transcription via AssemblyAI with automatic language detection.",
        endpoints={
            "websocket": "/ws",
            "health": "/health",
            "info": "/info",
            "languages": "/languages",
            "ready": "/ready",
            "static": "/static/*"
        },
        configuration={
            "stt_service": "AssemblyAI",
            "assemblyai_configured": bool(settings.ASSEMBLYAI_API_KEY),
            "default_language": settings.DEFAULT_LANGUAGE,
            "openrouter_model": settings.OPENROUTER_MODEL,
            "tts_engine": settings.TTS_ENGINE,
            "debug": settings.DEBUG
        },
        supported_languages=languages
    )


@router.get("/languages", response_model=LanguagesResponse)
async def get_languages() -> LanguagesResponse:
    """
    Get detailed information about supported languages.
    Returns all languages supported by AssemblyAI for this MVP.
    """
    # Separate Indic and English languages
    indic_langs = {k: v for k, v in SUPPORTED_LANGUAGES.items() if k != "en"}
    english = {"en": SUPPORTED_LANGUAGES["en"]}
    
    return LanguagesResponse(
        total_count=len(SUPPORTED_LANGUAGES),
        indic_languages=indic_langs,
        english=english,
        engines={"assemblyai": list(SUPPORTED_LANGUAGES.keys())}
    )


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint.
    Returns whether the service is ready to accept requests.
    """
    stt_info = assemblyai_service.get_model_info()
    
    checks = {
        "assemblyai_configured": stt_info["configured"],
        "openrouter_configured": bool(settings.OPENROUTER_API_KEY),
        "redis_connected": False
    }

    try:
        await session_manager.connect()
        checks["redis_connected"] = True
    except Exception:
        pass

    # For readiness, we need AssemblyAI and OpenRouter configured
    is_ready = checks["assemblyai_configured"] and checks["openrouter_configured"]

    if not is_ready:
        raise HTTPException(
            status_code=503,
            detail={"ready": False, "checks": checks}
        )

    return {"ready": True, "checks": checks}


@router.get("/warmup")
async def warmup_services() -> Dict[str, str]:
    """
    Warm up services - mainly for testing connections.
    AssemblyAI is cloud-based so no model loading needed.
    """
    results = {}

    # Check AssemblyAI configuration
    stt_info = assemblyai_service.get_model_info()
    results["assemblyai"] = "configured" if stt_info["configured"] else "not configured"

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


@router.get("/service-info")
async def get_service_info() -> Dict[str, Any]:
    """
    Get information about the STT service.
    AssemblyAI is cloud-based, so no local device info needed.
    """
    stt_info = assemblyai_service.get_model_info()
    
    return {
        "stt_service": stt_info["service"],
        "engine": stt_info["engine"],
        "configured": stt_info["configured"],
        "supported_languages": stt_info["languages"],
        "language_detection": stt_info["language_detection"],
        "deployment": "cloud-based (no local models)"
    }
