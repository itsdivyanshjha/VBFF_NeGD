"""
Voice Form Assistant - FastAPI Application.
Main entry point for the backend server.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .api.http import router as http_router
from .api.websocket import websocket_endpoint
from .services.whisper_service import whisper_service
from .services.session_manager import session_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Voice Form Assistant...")

    # Connect to Redis
    try:
        await session_manager.connect()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Sessions will not persist.")

    # Optionally preload Whisper model
    if not settings.DEBUG:
        try:
            logger.info("Loading Whisper model...")
            await whisper_service.load_model()
            logger.info("Whisper model loaded")
        except Exception as e:
            logger.warning(f"Failed to preload Whisper model: {e}")

    logger.info(f"Server ready on {settings.HOST}:{settings.PORT}")

    yield

    # Shutdown
    logger.info("Shutting down Voice Form Assistant...")
    await session_manager.disconnect()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Voice Form Assistant",
    description="Voice-based form filling assistant for Indian government portals",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include HTTP routes
app.include_router(http_router, tags=["API"])


# WebSocket endpoint
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice form filling."""
    await websocket_endpoint(websocket)


# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    logger.info(f"Static files mounted from {static_path}")
else:
    logger.warning(f"Static directory not found at {static_path}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return JSONResponse({
        "name": "Voice Form Assistant API",
        "version": "1.0.0",
        "description": "Voice-based form filling for Indian government portals",
        "endpoints": {
            "websocket": "/ws",
            "health": "/health",
            "info": "/info",
            "widget": "/static/widget.js",
            "embed": "/static/embed.js"
        },
        "documentation": "/docs"
    })


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
