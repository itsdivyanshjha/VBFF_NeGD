"""
IndicConformer ASR Service.
Uses AI4Bharat's IndicConformer models for speech-to-text in 22 Indian languages.
Supports both CTC and RNNT decoding modes.
"""

import logging
import asyncio
from typing import Tuple, Optional, Dict
import numpy as np
import tempfile
import os

from ..config import settings

logger = logging.getLogger(__name__)

# Supported Indian languages with their ISO codes
SUPPORTED_LANGUAGES = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ks": "Kashmiri",
    "kok": "Konkani",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}


class IndicASRService:
    """
    Speech-to-text service using IndicConformer models.
    Supports 22 official Indian languages with high accuracy.
    """

    def __init__(self):
        self._transcriber = None
        self._lock = asyncio.Lock()
        self._decoder_mode = settings.INDIC_ASR_DECODER  # "ctc" or "rnnt"

    async def load_model(self) -> None:
        """Load the IndicConformer model via indic-asr-onnx."""
        async with self._lock:
            if self._transcriber is not None:
                return

            logger.info("Loading IndicConformer ASR model...")

            loop = asyncio.get_event_loop()

            def _load():
                try:
                    from indic_asr_onnx import IndicTranscriber
                    transcriber = IndicTranscriber()
                    logger.info("IndicConformer model loaded successfully")
                    return transcriber
                except ImportError as e:
                    logger.error(f"Failed to import indic_asr_onnx: {e}")
                    logger.error("Please install: pip install indic-asr-onnx")
                    raise
                except Exception as e:
                    logger.error(f"Failed to load IndicConformer model: {e}")
                    raise

            self._transcriber = await loop.run_in_executor(None, _load)
            logger.info("IndicConformer ASR service ready")

    async def transcribe(
        self,
        audio_data: np.ndarray,
        language: str = "hi",
        sample_rate: int = 16000
    ) -> Tuple[str, float]:
        """
        Transcribe audio to text using IndicConformer.

        Args:
            audio_data: Numpy array of audio samples (16kHz, mono, float32)
            language: Language code (e.g., 'hi' for Hindi, 'ta' for Tamil)
            sample_rate: Audio sample rate

        Returns:
            Tuple of (transcription_text, confidence_score)
        """
        if self._transcriber is None:
            await self.load_model()

        # Validate language
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(
                f"Language '{language}' not supported by IndicConformer. "
                f"Falling back to Hindi."
            )
            language = "hi"

        loop = asyncio.get_event_loop()

        def _transcribe():
            # Ensure audio is float32
            if audio_data.dtype != np.float32:
                audio = audio_data.astype(np.float32)
            else:
                audio = audio_data

            # Ensure mono
            if len(audio.shape) > 1:
                audio = audio.flatten()

            # Save to temporary WAV file (indic-asr-onnx requires file path)
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, audio, sample_rate)

            try:
                # Transcribe using configured decoder
                if self._decoder_mode == "rnnt":
                    text = self._transcriber.transcribe_rnnt(temp_path, language)
                else:
                    text = self._transcriber.transcribe_ctc(temp_path, language)

                # Clean up transcription
                text = text.strip() if text else ""

                # Calculate confidence (indic-asr-onnx doesn't provide confidence,
                # so we estimate based on text characteristics)
                confidence = self._estimate_confidence(text, language)

                return text, confidence

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        try:
            text, confidence = await loop.run_in_executor(None, _transcribe)
            logger.info(
                f"IndicConformer transcription [{language}]: '{text}' "
                f"(confidence: {confidence:.2f})"
            )
            return text, confidence

        except Exception as e:
            logger.error(f"IndicConformer transcription error: {e}")
            raise

    def _estimate_confidence(self, text: str, language: str) -> float:
        """
        Estimate confidence score based on transcription characteristics.
        Since indic-asr-onnx doesn't provide confidence, we estimate it.
        """
        if not text:
            return 0.0

        # Base confidence
        confidence = 0.7

        # Longer transcriptions are usually more confident
        word_count = len(text.split())
        if word_count >= 3:
            confidence += 0.1
        elif word_count >= 1:
            confidence += 0.05

        # Check for common transcription artifacts
        # (multiple spaces, repeated characters, etc.)
        if "  " in text:
            confidence -= 0.1
        
        # Check for very short transcriptions (might be noise)
        if len(text) < 3:
            confidence -= 0.2

        return max(0.1, min(1.0, confidence))

    async def transcribe_with_both_decoders(
        self,
        audio_data: np.ndarray,
        language: str = "hi",
        sample_rate: int = 16000
    ) -> Dict[str, Tuple[str, float]]:
        """
        Transcribe using both CTC and RNNT decoders for comparison.
        Useful for getting best result or debugging.

        Returns:
            Dict with 'ctc' and 'rnnt' keys, each containing (text, confidence)
        """
        if self._transcriber is None:
            await self.load_model()

        if language not in SUPPORTED_LANGUAGES:
            language = "hi"

        loop = asyncio.get_event_loop()

        def _transcribe_both():
            import soundfile as sf

            if audio_data.dtype != np.float32:
                audio = audio_data.astype(np.float32)
            else:
                audio = audio_data

            if len(audio.shape) > 1:
                audio = audio.flatten()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, audio, sample_rate)

            try:
                ctc_text = self._transcriber.transcribe_ctc(temp_path, language)
                rnnt_text = self._transcriber.transcribe_rnnt(temp_path, language)

                ctc_text = ctc_text.strip() if ctc_text else ""
                rnnt_text = rnnt_text.strip() if rnnt_text else ""

                return {
                    "ctc": (ctc_text, self._estimate_confidence(ctc_text, language)),
                    "rnnt": (rnnt_text, self._estimate_confidence(rnnt_text, language))
                }

            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        return await loop.run_in_executor(None, _transcribe_both)

    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        return language in SUPPORTED_LANGUAGES

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        return {
            "model_type": "IndicConformer (indic-asr-onnx)",
            "decoder_mode": self._decoder_mode,
            "loaded": self._transcriber is not None,
            "supported_languages": len(SUPPORTED_LANGUAGES),
            "languages": list(SUPPORTED_LANGUAGES.keys())
        }


# Global instance
indic_asr_service = IndicASRService()
