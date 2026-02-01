"""
AssemblyAI Speech-to-Text Provider.

This module wraps the existing AssemblyAIService to implement the STTProvider interface,
enabling plug-and-play provider swapping while maintaining backward compatibility.
"""
import logging
from typing import Optional, Dict, Any, List

from .base import STTProvider, STTResult
from ...services.assemblyai_service import (
    AssemblyAIService,
    TranscriptionResult as LegacyTranscriptionResult,
    SUPPORTED_LANGUAGES
)
from ...core.config import get_config

logger = logging.getLogger(__name__)


class AssemblyAIProvider(STTProvider):
    """
    AssemblyAI STT provider implementation.

    Wraps the existing AssemblyAIService to provide a standardized interface
    while preserving all existing functionality.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AssemblyAI provider.

        Args:
            config: Optional provider configuration.
                   If None, loads from global config.
        """
        if config is None:
            app_config = get_config()
            config = app_config.get_provider_config('stt', 'assemblyai')

        self.config = config
        self.api_key = config.get('api_key', '')
        self.base_url = config.get('base_url', 'https://api.assemblyai.com/v2')
        self.timeout = config.get('timeout', 60)

        # Initialize the underlying AssemblyAI service
        self._service = AssemblyAIService()

        logger.info("AssemblyAI STT provider initialized")

    async def transcribe(
        self,
        audio_data: bytes,
        language_hint: Optional[str] = None,
        field_context: Optional[Dict[str, Any]] = None
    ) -> STTResult:
        """
        Transcribe audio to text using AssemblyAI.

        Args:
            audio_data: Raw audio bytes
            language_hint: Suggested language code (e.g., 'en', 'hi')
            field_context: Optional context about the field:
                - field_type: Type of field (email, phone, aadhaar, etc.)
                - field_name: Name of the field
                - expected_format: Expected format/pattern
                - word_boost: Keywords to boost recognition

        Returns:
            STTResult containing transcription and metadata

        Raises:
            Exception: If transcription fails
        """
        try:
            # Call the legacy service method
            legacy_result: LegacyTranscriptionResult = await self._service.transcribe_raw(
                audio_bytes=audio_data,
                language_hint=language_hint,
                field_info=field_context
            )

            # Convert legacy result to standard STTResult
            stt_result = STTResult(
                text=legacy_result.text,
                confidence=legacy_result.confidence,
                language=legacy_result.language,
                alternatives=[],  # AssemblyAI doesn't provide alternatives in current implementation
                entities=getattr(legacy_result, 'entities', []),
                words=getattr(legacy_result, 'words', []),
                metadata={
                    'engine_used': legacy_result.engine_used,
                    'is_indic': legacy_result.is_indic,
                    'language_confidence': legacy_result.language_confidence,
                    'audio_duration': getattr(legacy_result, 'audio_duration', None),
                }
            )

            logger.debug(
                f"Transcribed {len(audio_data)} bytes: '{stt_result.text}' "
                f"(confidence: {stt_result.confidence:.2f}, language: {stt_result.language})"
            )

            return stt_result

        except Exception as e:
            logger.error(f"AssemblyAI transcription failed: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if AssemblyAI service is healthy.

        Returns:
            True if service is accessible, False otherwise
        """
        try:
            # Simple check: verify we have an API key
            if not self.api_key:
                logger.warning("AssemblyAI API key not configured")
                return False

            # Could add actual API ping here in the future
            return True

        except Exception as e:
            logger.error(f"AssemblyAI health check failed: {e}")
            return False

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.

        Returns:
            List of ISO language codes
        """
        return list(SUPPORTED_LANGUAGES.keys())

    def get_provider_name(self) -> str:
        """
        Get provider name.

        Returns:
            'assemblyai'
        """
        return "assemblyai"
