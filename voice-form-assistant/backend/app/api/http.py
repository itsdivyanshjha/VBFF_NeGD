"""
HTTP Endpoints.
Health check, information, and language support endpoints.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import settings
from ..services.hybrid_stt_service import hybrid_stt_service
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
    Returns status of all services including hybrid STT.
    """
    # Get hybrid STT service info
    stt_info = hybrid_stt_service.get_model_info()
    
    services_status = {
        "stt": {
            "mode": settings.STT_MODE,
            "device": settings.effective_device,
            "loaded": stt_info["hybrid_service"]["loaded"],
            "language_detector": stt_info["language_detector"],
            "indic_asr": stt_info["indic_asr"],
            "whisper": stt_info["whisper"]
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
    languages = hybrid_stt_service.get_supported_languages()
    
    return InfoResponse(
        name="Voice Form Assistant",
        version="2.0.0",
        description="Voice-based form filling assistant for Indian government portals. "
                    "Supports 22 Indian languages via IndicConformer + English via Whisper.",
        endpoints={
            "websocket": "/ws",
            "health": "/health",
            "info": "/info",
            "languages": "/languages",
            "ready": "/ready",
            "warmup": "/warmup",
            "static": "/static/*"
        },
        configuration={
            "stt_mode": settings.STT_MODE,
            "ml_device": settings.effective_device,
            "indic_decoder": settings.INDIC_ASR_DECODER,
            "whisper_model": settings.WHISPER_MODEL,
            "openrouter_model": settings.OPENROUTER_MODEL,
            "tts_engine": settings.TTS_ENGINE,
            "lang_detection_threshold": settings.LANG_DETECTION_THRESHOLD,
            "debug": settings.DEBUG
        },
        supported_languages=languages
    )


@router.get("/languages", response_model=LanguagesResponse)
async def get_languages() -> LanguagesResponse:
    """
    Get detailed information about supported languages.
    Returns all languages supported by the hybrid STT system.
    """
    languages = hybrid_stt_service.get_supported_languages()
    
    return LanguagesResponse(
        total_count=languages["total_count"],
        indic_languages=languages["indic"],
        english=languages["english"],
        engines=languages["engines"]
    )


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint.
    Returns whether the service is ready to accept requests.
    """
    stt_info = hybrid_stt_service.get_model_info()
    
    checks = {
        "stt_loaded": stt_info["hybrid_service"]["loaded"],
        "language_detector_loaded": stt_info["language_detector"]["loaded"],
        "indic_asr_loaded": stt_info["indic_asr"]["loaded"],
        "whisper_loaded": stt_info["whisper"]["loaded"],
        "openrouter_configured": bool(settings.OPENROUTER_API_KEY),
        "redis_connected": False
    }

    try:
        await session_manager.connect()
        checks["redis_connected"] = True
    except Exception:
        pass

    # For readiness, we need at least one STT engine loaded
    stt_ready = checks["indic_asr_loaded"] or checks["whisper_loaded"]
    is_ready = stt_ready and checks["openrouter_configured"]

    if not is_ready:
        raise HTTPException(
            status_code=503,
            detail={"ready": False, "checks": checks}
        )

    return {"ready": True, "checks": checks}


@router.post("/warmup")
async def warmup_services() -> Dict[str, str]:
    """
    Warm up services by loading all models.
    Call this on startup to ensure fast first response.
    
    Loads:
    - Language Detection model (SpeechBrain VoxLingua107)
    - IndicConformer model (22 Indian languages)
    - Whisper model (English)
    """
    results = {}

    # Load hybrid STT models (includes all three)
    try:
        await hybrid_stt_service.load_models()
        results["stt"] = "loaded"
        results["language_detector"] = "loaded"
        results["indic_asr"] = "loaded"
        results["whisper"] = "loaded"
    except Exception as e:
        results["stt"] = f"error: {str(e)}"

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


@router.get("/device")
async def get_device_info() -> Dict[str, Any]:
    """
    Get information about the ML device (CPU/GPU).
    Useful for verifying GPU acceleration is working.
    """
    import torch
    
    device_info = {
        "configured_device": settings.ML_DEVICE,
        "effective_device": settings.effective_device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        device_info["cuda_devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
            }
            for i in range(torch.cuda.device_count())
        ]
    
    return device_info
