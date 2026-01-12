"""
Text-to-Speech Service.
Uses Google TTS (gTTS) for reliable voice synthesis.
"""

import logging
import asyncio
import base64
import io
from typing import Optional

from gtts import gTTS
from pydub import AudioSegment

from ..config import settings

logger = logging.getLogger(__name__)


class TTSService:
    """Service for text-to-speech using Google TTS."""

    def __init__(self):
        self.default_lang = "en"
        self.tld = "co.in"  # Indian English accent

    async def synthesize(
        self,
        text: str,
        language: str = "en"
    ) -> str:
        """
        Convert text to speech.

        Args:
            text: Text to synthesize
            language: Language code ('en' or 'hi')

        Returns:
            Base64 encoded WAV audio
        """
        if not text:
            logger.warning("Empty text provided for TTS")
            return ""

        loop = asyncio.get_event_loop()

        def generate():
            # Map language codes
            lang_code = "hi" if language == "hi" else "en"

            # Use Indian TLD for Indian English accent
            tld = "co.in" if lang_code == "en" else "com"

            logger.debug(f"Synthesizing: '{text[:50]}...' (lang={lang_code})")

            tts = gTTS(text=text, lang=lang_code, tld=tld)

            # Generate MP3
            mp3_buffer = io.BytesIO()
            tts.write_to_fp(mp3_buffer)
            mp3_buffer.seek(0)

            # Convert MP3 to WAV for browser compatibility
            audio = AudioSegment.from_mp3(mp3_buffer)
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            # Encode to base64
            return base64.b64encode(wav_buffer.read()).decode("utf-8")

        try:
            result = await loop.run_in_executor(None, generate)
            logger.info(f"TTS generated {len(result)} bytes of audio")
            return result

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return ""

    def get_service_info(self) -> dict:
        """Get information about the TTS service."""
        return {
            "engine": "gtts",
            "default_language": self.default_lang,
            "tld": self.tld
        }


# Global instance
tts_service = TTSService()
