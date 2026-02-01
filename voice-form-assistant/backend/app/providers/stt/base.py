"""
Base interface for Speech-to-Text providers.

This module defines the abstract interface that all STT providers must implement,
enabling plug-and-play swapping of STT services (AssemblyAI, Whisper, Google, Azure, etc.).
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class STTResult:
    """Standardized Speech-to-Text result."""

    text: str
    """Transcribed text"""

    confidence: float
    """Confidence score (0.0 to 1.0)"""

    language: Optional[str] = None
    """Detected language code (e.g., 'en', 'hi')"""

    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    """Alternative transcriptions with confidence scores"""

    entities: List[Dict[str, Any]] = field(default_factory=list)
    """Detected entities (emails, phone numbers, dates, etc.)"""

    words: List[Dict[str, Any]] = field(default_factory=list)
    """Word-level timing and confidence information"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Provider-specific metadata"""


class STTProvider(ABC):
    """
    Abstract base class for Speech-to-Text providers.

    All STT implementations must inherit from this class and implement
    the required methods. This enables the application to swap STT providers
    without changing core business logic.

    Example implementations:
    - AssemblyAIProvider
    - WhisperProvider (OpenAI)
    - GoogleSTTProvider
    - AzureSTTProvider
    """

    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        language_hint: Optional[str] = None,
        field_context: Optional[Dict[str, Any]] = None
    ) -> STTResult:
        """
        Transcribe audio to text.

        Args:
            audio_data: Raw audio bytes (any format supported by provider)
            language_hint: Suggested language code (e.g., 'en', 'hi', 'bn')
            field_context: Optional context about the field being filled:
                - field_type: Type of field (email, phone, aadhaar, etc.)
                - field_name: Name of the field
                - expected_format: Expected format/pattern
                - word_boost: Keywords to boost recognition

        Returns:
            STTResult containing transcription and metadata

        Raises:
            Exception: If transcription fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if provider is healthy and accessible.

        Returns:
            True if provider is healthy, False otherwise
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.

        Returns:
            List of ISO language codes (e.g., ['en', 'hi', 'bn', 'ta'])
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            Provider name (e.g., 'assemblyai', 'whisper', 'google')
        """
        pass
