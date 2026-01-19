"""
Configuration settings for the Voice Form Assistant backend.
All settings can be configured via environment variables or .env file.

Supports:
- Hybrid STT: IndicConformer (22 Indian languages) + Whisper (English)
- Language Detection: SpeechBrain VoxLingua107
- GPU acceleration for AWS EC2 deployment
"""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field
import json
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # =================================================================
    # AssemblyAI Configuration (Cloud-based STT)
    # =================================================================
    
    # AssemblyAI API key for authentication
    ASSEMBLYAI_API_KEY: str = Field(default="", description="AssemblyAI API key")
    
    # AssemblyAI API base URL
    ASSEMBLYAI_BASE_URL: str = "https://api.assemblyai.com/v2"
    
    # Request timeout in seconds
    ASSEMBLYAI_TIMEOUT: int = 60
    
    # Polling interval for transcript status (seconds)
    ASSEMBLYAI_POLLING_INTERVAL: float = 0.5
    
    # Maximum retries for failed requests
    ASSEMBLYAI_MAX_RETRIES: int = 3

    # =================================================================
    # Language Support Configuration
    # =================================================================
    
    # Supported languages for MVP (ISO 639-1 codes)
    # Hindi, Bengali, Tamil, Telugu, Marathi, English
    SUPPORTED_LANGUAGES: List[str] = ["hi", "bn", "ta", "te", "mr", "en"]
    
    # Default language when detection fails
    DEFAULT_LANGUAGE: str = "hi"

    # =================================================================
    # OpenRouter Configuration (LLM for value extraction)
    # =================================================================
    
    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API key")
    OPENROUTER_MODEL: str = "meta-llama/llama-3.1-8b-instruct"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_TIMEOUT: int = 30  # seconds
    OPENROUTER_SITE_URL: str = "http://localhost:8000"
    OPENROUTER_APP_NAME: str = "Voice Form Assistant"

    # =================================================================
    # TTS Configuration (Text-to-Speech)
    # =================================================================
    
    # TTS engine: "gtts" (Google TTS) - reliable, supports Indian languages
    TTS_ENGINE: str = "gtts"
    
    # Default TTS language
    TTS_LANGUAGE: str = "en"
    
    # TTS domain (co.in for Indian English accent)
    TTS_TLD: str = "co.in"

    # =================================================================
    # Redis Configuration (Session Storage)
    # =================================================================
    
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    SESSION_TTL: int = 3600  # Session expiry in seconds (1 hour)

    # =================================================================
    # CORS Configuration
    # =================================================================
    
    ALLOWED_ORIGINS: str = '["*"]'  # JSON string of allowed origins

    # =================================================================
    # Audio Configuration
    # =================================================================
    
    AUDIO_SAMPLE_RATE: int = 16000  # 16kHz (standard for ASR models)
    AUDIO_CHANNELS: int = 1  # Mono
    AUDIO_CHUNK_DURATION: float = 2.0  # Seconds of audio to buffer
    
    # Audio preprocessing
    AUDIO_ENABLE_PREPROCESSING: bool = True
    AUDIO_NOISE_REDUCTION: bool = True
    AUDIO_AGC: bool = True  # Auto Gain Control

    # =================================================================
    # Logging Configuration
    # =================================================================
    
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
