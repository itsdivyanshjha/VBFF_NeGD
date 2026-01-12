"""
Configuration settings for the Voice Form Assistant backend.
All settings can be configured via environment variables or .env file.
"""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field
import json


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # OpenAI Whisper Configuration
    WHISPER_MODEL: str = "base"  # Options: tiny, base, small, medium, large
    WHISPER_DEVICE: str = "cpu"  # Options: cpu, cuda
    # Use "auto" to let Whisper detect language per-utterance (recommended for Indian English/Hindi/Hinglish).
    # If set to a language code like "en" or "hi", it will force decoding in that language.
    WHISPER_LANGUAGE: str = "auto"

    # OpenRouter Configuration
    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API key")
    OPENROUTER_MODEL: str = "meta-llama/llama-3.1-8b-instruct"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_TIMEOUT: int = 30  # seconds
    OPENROUTER_SITE_URL: str = "http://localhost:8000"  # Your site URL for OpenRouter headers
    OPENROUTER_APP_NAME: str = "Voice Form Assistant"  # Your app name for OpenRouter headers

    # TTS Configuration (using Google TTS)
    TTS_LANGUAGE: str = "en"  # Default: en (English), hi (Hindi)
    TTS_TLD: str = "co.in"  # Use Indian domain for Indian English accent

    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    SESSION_TTL: int = 3600  # Session expiry in seconds (1 hour)

    # CORS Configuration
    ALLOWED_ORIGINS: str = '["*"]'  # JSON string of allowed origins

    # Audio Configuration
    AUDIO_SAMPLE_RATE: int = 16000  # Whisper expects 16kHz
    AUDIO_CHANNELS: int = 1  # Mono
    AUDIO_CHUNK_DURATION: float = 2.0  # Seconds of audio to buffer before processing

    # Logging
    LOG_LEVEL: str = "INFO"

    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse ALLOWED_ORIGINS JSON string to list."""
        try:
            return json.loads(self.ALLOWED_ORIGINS)
        except json.JSONDecodeError:
            return ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
