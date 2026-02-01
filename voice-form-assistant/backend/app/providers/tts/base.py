"""
Base interface for Text-to-Speech providers.

This module defines the abstract interface for TTS providers,
enabling plug-and-play swapping of TTS services.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class TTSResult:
    """Result from Text-to-Speech synthesis."""

    audio_data: str
    """Base64-encoded audio data"""

    format: str
    """Audio format (e.g., 'mp3', 'wav', 'ogg')"""

    sample_rate: int
    """Sample rate in Hz"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Provider-specific metadata"""


class TTSProvider(ABC):
    """
    Abstract base class for Text-to-Speech providers.

    All TTS implementations must inherit from this class and implement
    the required methods. This enables swapping TTS providers without
    changing core business logic.

    Example implementations:
    - GTTSProvider (Google TTS)
    - ElevenLabsProvider
    - AzureTTSProvider
    - PlayHTProvider
    """

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        language: str = 'en',
        voice: Optional[str] = None,
        **kwargs
    ) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            language: Language code (e.g., 'en', 'hi', 'bn')
            voice: Optional voice ID/name
            **kwargs: Provider-specific parameters

        Returns:
            TTSResult containing audio data and metadata

        Raises:
            Exception: If synthesis fails
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
    def get_available_voices(self, language: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get available voices, optionally filtered by language.

        Args:
            language: Optional language filter

        Returns:
            List of voice definitions with 'id', 'name', 'language' keys
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            Provider name (e.g., 'gtts', 'elevenlabs', 'azure')
        """
        pass
