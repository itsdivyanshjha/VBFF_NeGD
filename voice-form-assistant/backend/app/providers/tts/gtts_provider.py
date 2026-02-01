"""
Google Text-to-Speech Provider.

This module wraps the existing TTSService to implement the TTSProvider interface,
enabling plug-and-play provider swapping while maintaining backward compatibility.
"""
import logging
from typing import Optional, Dict, Any, List

from .base import TTSProvider, TTSResult
from ...services.tts_service import TTSService
from ...core.config import get_config

logger = logging.getLogger(__name__)


# Supported languages for gTTS
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "tld": "co.in"},
    "hi": {"name": "Hindi", "tld": "co.in"},
    "bn": {"name": "Bengali", "tld": "co.in"},
    "ta": {"name": "Tamil", "tld": "co.in"},
    "te": {"name": "Telugu", "tld": "co.in"},
    "mr": {"name": "Marathi", "tld": "co.in"},
    "gu": {"name": "Gujarati", "tld": "co.in"},
    "kn": {"name": "Kannada", "tld": "co.in"},
    "ml": {"name": "Malayalam", "tld": "co.in"},
    "pa": {"name": "Punjabi", "tld": "co.in"},
}


class GTTSProvider(TTSProvider):
    """
    Google Text-to-Speech provider implementation.

    Wraps the existing TTSService to provide a standardized interface
    while preserving all existing functionality.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize gTTS provider.

        Args:
            config: Optional provider configuration.
                   If None, loads from global config.
        """
        if config is None:
            app_config = get_config()
            config = app_config.get_provider_config('tts', 'gtts')

        self.config = config
        self.default_language = config.get('language', 'en')
        self.tld = config.get('tld', 'co.in')
        self.slow = config.get('slow', False)

        # Initialize the underlying TTS service
        self._service = TTSService()

        logger.info("gTTS provider initialized")

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
            language: Language code (e.g., 'en', 'hi')
            voice: Optional voice ID (not used by gTTS)
            **kwargs: Additional parameters

        Returns:
            TTSResult containing base64-encoded audio

        Raises:
            Exception: If synthesis fails
        """
        try:
            # Use default language if not specified
            if not language or language not in SUPPORTED_LANGUAGES:
                language = self.default_language

            # Call the legacy service
            audio_base64 = await self._service.synthesize(
                text=text,
                language=language
            )

            # Create result
            result = TTSResult(
                audio_data=audio_base64,
                format="wav",
                sample_rate=44100,  # gTTS default after conversion
                metadata={
                    'language': language,
                    'tld': self.tld,
                    'text_length': len(text),
                }
            )

            logger.debug(
                f"Synthesized {len(text)} chars in {language}: "
                f"{len(audio_base64)} bytes (base64)"
            )

            return result

        except Exception as e:
            logger.error(f"gTTS synthesis failed: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if gTTS service is healthy.

        Returns:
            True (gTTS is a library, always available if imported)
        """
        try:
            # gTTS is a library, not a service
            # Check if we can import it
            import gtts
            return True

        except ImportError:
            logger.error("gTTS library not installed")
            return False

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.

        Returns:
            List of ISO language codes
        """
        return list(SUPPORTED_LANGUAGES.keys())

    def get_available_voices(self, language: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get available voices.

        Note: gTTS doesn't support multiple voices per language,
        so we return a single default voice for each language.

        Args:
            language: Optional language filter

        Returns:
            List of voice definitions
        """
        voices = []

        languages_to_include = [language] if language and language in SUPPORTED_LANGUAGES else SUPPORTED_LANGUAGES.keys()

        for lang_code in languages_to_include:
            lang_info = SUPPORTED_LANGUAGES[lang_code]
            voices.append({
                'id': f'gtts-{lang_code}',
                'name': f'{lang_info["name"]} (gTTS)',
                'language': lang_code
            })

        return voices

    def get_provider_name(self) -> str:
        """
        Get provider name.

        Returns:
            'gtts'
        """
        return "gtts"
