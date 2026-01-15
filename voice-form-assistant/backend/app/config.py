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
    # STT Configuration - Hybrid System
    # =================================================================
    
    # STT Mode: "hybrid" (auto language detection), "indic" (force IndicConformer), 
    #           "whisper" (force Whisper), "auto" (same as hybrid)
    STT_MODE: str = "hybrid"
    
    # Device for ML models (auto-detected if not specified)
    # Options: "auto", "cpu", "cuda", "cuda:0", "cuda:1", etc.
    ML_DEVICE: str = "auto"
    
    # Model cache directory for downloaded models
    MODEL_CACHE_DIR: str = Field(
        default_factory=lambda: os.path.expanduser("~/.cache/voice-form-assistant")
    )

    # =================================================================
    # IndicConformer Configuration (Indian Languages ASR)
    # =================================================================
    
    # Decoder mode: "ctc" (faster) or "rnnt" (more accurate)
    INDIC_ASR_DECODER: str = "ctc"
    
    # Default language for IndicConformer when detection is uncertain
    INDIC_DEFAULT_LANGUAGE: str = "hi"

    # =================================================================
    # Whisper Configuration (English ASR + Fallback)
    # =================================================================
    
    # Whisper model size: tiny, base, small, medium, large, large-v2, large-v3
    # For GPU: medium or large recommended
    # For CPU: tiny or base recommended
    WHISPER_MODEL: str = "base"
    
    # Whisper device (overridden by ML_DEVICE if set to "auto")
    WHISPER_DEVICE: str = "auto"
    
    # Force Whisper to specific language (None = auto-detect)
    # Use "en" for English-only, "auto" for auto-detection
    WHISPER_LANGUAGE: str = "auto"

    # =================================================================
    # Language Detection Configuration
    # =================================================================
    
    # Confidence threshold for language detection (0.0 - 1.0)
    # Below this threshold, hybrid transcription is used
    LANG_DETECTION_THRESHOLD: float = 0.6
    
    # Prefer IndicConformer for uncertain languages (if True)
    PREFER_INDIC_ASR: bool = True
    
    # Fallback language when detection fails completely
    FALLBACK_LANGUAGE: str = "hi"

    # =================================================================
    # Supported Languages Configuration
    # =================================================================
    
    # Languages to enable (comma-separated ISO codes)
    # Empty = all supported languages
    # Example: "hi,en,ta,te,bn" for Hindi, English, Tamil, Telugu, Bengali
    ENABLED_LANGUAGES: str = ""
    
    # Form language (language for form field values/output)
    # "source" = same as detected speech language
    # Or specific code like "en", "hi"
    FORM_OUTPUT_LANGUAGE: str = "source"

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

    @property
    def enabled_languages_list(self) -> List[str]:
        """Parse ENABLED_LANGUAGES to list."""
        if not self.ENABLED_LANGUAGES:
            return []
        return [lang.strip() for lang in self.ENABLED_LANGUAGES.split(",")]

    @property
    def effective_device(self) -> str:
        """Get the effective ML device (auto-detect if needed)."""
        if self.ML_DEVICE == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.ML_DEVICE

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Create model cache directory if it doesn't exist
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
