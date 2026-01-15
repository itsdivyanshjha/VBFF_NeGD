"""
Hybrid Speech-to-Text Service.
Combines multiple ASR engines for optimal multilingual support:
- Language Detection: SpeechBrain VoxLingua107 (107 languages)
- Indian Languages: IndicConformer (22 scheduled Indian languages)
- English: OpenAI Whisper (high accuracy for English)

This hybrid approach provides the best accuracy for both Indian languages
and English, with automatic language detection and routing.
"""

import logging
import asyncio
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from ..config import settings
from .language_detector import language_detector, INDIC_LANGUAGES
from .indic_asr_service import indic_asr_service
from .whisper_service import whisper_service

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription."""
    text: str
    confidence: float
    language: str
    language_confidence: float
    engine_used: str  # "indicconformer", "whisper", or "hybrid"
    is_indic: bool


class HybridSTTService:
    """
    Hybrid Speech-to-Text service that automatically routes audio
    to the best ASR engine based on detected language.

    Flow:
    1. Detect language using SpeechBrain VoxLingua107
    2. If Indian language detected -> use IndicConformer
    3. If English detected -> use Whisper
    4. If uncertain -> try both and pick best result
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._loaded = False
        
        # Configuration
        self._lang_confidence_threshold = settings.LANG_DETECTION_THRESHOLD
        self._prefer_indic = settings.PREFER_INDIC_ASR
        self._fallback_language = settings.FALLBACK_LANGUAGE

    async def load_models(self) -> None:
        """Pre-load all ASR models for faster first inference."""
        async with self._lock:
            if self._loaded:
                return

            logger.info("Loading hybrid STT models...")

            # Load all models in parallel
            await asyncio.gather(
                language_detector.load_model(),
                indic_asr_service.load_model(),
                whisper_service.load_model(),
                return_exceptions=True
            )

            self._loaded = True
            logger.info("All STT models loaded successfully")

    async def transcribe(
        self,
        audio_data: np.ndarray,
        language_hint: Optional[str] = None,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """
        Transcribe audio to text using the most appropriate engine.

        Args:
            audio_data: Numpy array of audio samples (16kHz, mono, float32)
            language_hint: Optional language code hint (overrides detection if confident)
            sample_rate: Audio sample rate

        Returns:
            TranscriptionResult with text, confidence, and metadata
        """
        # Ensure models are loaded
        if not self._loaded:
            await self.load_models()

        # Step 1: Detect language
        detected_lang, lang_confidence, is_indic = await language_detector.detect_language(
            audio_data, sample_rate
        )

        logger.info(
            f"Language detection: {detected_lang} "
            f"(confidence: {lang_confidence:.2f}, is_indic: {is_indic})"
        )

        # Use language hint if provided and detection is uncertain
        if language_hint and lang_confidence < self._lang_confidence_threshold:
            detected_lang = language_hint
            is_indic = language_hint in INDIC_LANGUAGES
            logger.info(f"Using language hint: {language_hint}")

        # Step 2: Route to appropriate ASR engine
        if is_indic and lang_confidence >= self._lang_confidence_threshold:
            # High confidence Indian language -> use IndicConformer
            return await self._transcribe_indic(
                audio_data, detected_lang, lang_confidence, sample_rate
            )

        elif not is_indic and lang_confidence >= self._lang_confidence_threshold:
            # High confidence English -> use Whisper
            return await self._transcribe_english(
                audio_data, detected_lang, lang_confidence, sample_rate
            )

        else:
            # Uncertain language -> try hybrid approach
            return await self._transcribe_hybrid(
                audio_data, detected_lang, lang_confidence, is_indic, sample_rate
            )

    async def _transcribe_indic(
        self,
        audio_data: np.ndarray,
        language: str,
        lang_confidence: float,
        sample_rate: int
    ) -> TranscriptionResult:
        """Transcribe using IndicConformer for Indian languages."""
        try:
            text, confidence = await indic_asr_service.transcribe(
                audio_data, language, sample_rate
            )

            return TranscriptionResult(
                text=text,
                confidence=confidence,
                language=language,
                language_confidence=lang_confidence,
                engine_used="indicconformer",
                is_indic=True
            )

        except Exception as e:
            logger.error(f"IndicConformer failed: {e}, falling back to Whisper")
            return await self._transcribe_english(
                audio_data, language, lang_confidence, sample_rate
            )

    async def _transcribe_english(
        self,
        audio_data: np.ndarray,
        language: str,
        lang_confidence: float,
        sample_rate: int
    ) -> TranscriptionResult:
        """Transcribe using Whisper for English."""
        try:
            text, confidence = await whisper_service.transcribe(
                audio_data, language="en"
            )

            return TranscriptionResult(
                text=text,
                confidence=confidence,
                language="en",
                language_confidence=lang_confidence,
                engine_used="whisper",
                is_indic=False
            )

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=language,
                language_confidence=lang_confidence,
                engine_used="error",
                is_indic=False
            )

    async def _transcribe_hybrid(
        self,
        audio_data: np.ndarray,
        detected_lang: str,
        lang_confidence: float,
        is_indic: bool,
        sample_rate: int
    ) -> TranscriptionResult:
        """
        Hybrid transcription when language detection is uncertain.
        Tries both engines and picks the best result.
        """
        logger.info("Uncertain language detection, trying hybrid approach...")

        # Try both engines in parallel
        results = await asyncio.gather(
            self._try_indic(audio_data, detected_lang, sample_rate),
            self._try_whisper(audio_data, sample_rate),
            return_exceptions=True
        )

        indic_result, whisper_result = results

        # Handle errors
        if isinstance(indic_result, Exception):
            indic_result = None
            logger.warning(f"Indic transcription failed: {indic_result}")

        if isinstance(whisper_result, Exception):
            whisper_result = None
            logger.warning(f"Whisper transcription failed: {whisper_result}")

        # Pick best result based on confidence and text quality
        best_result = self._pick_best_result(
            indic_result, whisper_result, detected_lang, lang_confidence, is_indic
        )

        if best_result:
            return best_result

        # Fallback: return empty result
        return TranscriptionResult(
            text="",
            confidence=0.0,
            language=self._fallback_language,
            language_confidence=lang_confidence,
            engine_used="none",
            is_indic=is_indic
        )

    async def _try_indic(
        self,
        audio_data: np.ndarray,
        language: str,
        sample_rate: int
    ) -> Optional[Tuple[str, float]]:
        """Try transcription with IndicConformer."""
        try:
            # Default to Hindi if language not supported
            if not indic_asr_service.is_language_supported(language):
                language = "hi"
            
            text, confidence = await indic_asr_service.transcribe(
                audio_data, language, sample_rate
            )
            return (text, confidence, language)
        except Exception as e:
            logger.debug(f"Indic transcription attempt failed: {e}")
            return None

    async def _try_whisper(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Optional[Tuple[str, float]]:
        """Try transcription with Whisper."""
        try:
            text, confidence = await whisper_service.transcribe(
                audio_data, language=None  # Auto-detect
            )
            return (text, confidence, "en")
        except Exception as e:
            logger.debug(f"Whisper transcription attempt failed: {e}")
            return None

    def _pick_best_result(
        self,
        indic_result: Optional[Tuple[str, float, str]],
        whisper_result: Optional[Tuple[str, float, str]],
        detected_lang: str,
        lang_confidence: float,
        is_indic: bool
    ) -> Optional[TranscriptionResult]:
        """
        Pick the best transcription result from multiple engines.
        Uses heuristics based on confidence, text quality, and language hints.
        """
        candidates = []

        if indic_result:
            text, conf, lang = indic_result
            if text:
                # Boost confidence if detected language is Indic
                boost = 0.1 if is_indic else 0.0
                candidates.append({
                    "text": text,
                    "confidence": conf + boost,
                    "language": lang,
                    "engine": "indicconformer",
                    "is_indic": True
                })

        if whisper_result:
            text, conf, lang = whisper_result
            if text:
                # Boost confidence if detected language is English
                boost = 0.1 if not is_indic else 0.0
                candidates.append({
                    "text": text,
                    "confidence": conf + boost,
                    "language": lang,
                    "engine": "whisper",
                    "is_indic": False
                })

        if not candidates:
            return None

        # Sort by confidence and text length (longer is usually better)
        candidates.sort(
            key=lambda x: (x["confidence"], len(x["text"])),
            reverse=True
        )

        best = candidates[0]

        return TranscriptionResult(
            text=best["text"],
            confidence=best["confidence"],
            language=best["language"],
            language_confidence=lang_confidence,
            engine_used=best["engine"],
            is_indic=best["is_indic"]
        )

    async def transcribe_with_language(
        self,
        audio_data: np.ndarray,
        language: str,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """
        Transcribe audio with a specified language (no auto-detection).
        Useful when you know the language beforehand.

        Args:
            audio_data: Audio samples
            language: Language code (e.g., 'hi', 'en', 'ta')
            sample_rate: Sample rate

        Returns:
            TranscriptionResult
        """
        if not self._loaded:
            await self.load_models()

        is_indic = language in INDIC_LANGUAGES

        if is_indic:
            return await self._transcribe_indic(
                audio_data, language, 1.0, sample_rate
            )
        else:
            return await self._transcribe_english(
                audio_data, language, 1.0, sample_rate
            )

    def get_supported_languages(self) -> Dict[str, Any]:
        """Get all supported languages across all engines."""
        indic_langs = indic_asr_service.get_supported_languages()
        
        return {
            "indic": indic_langs,
            "english": {"en": "English"},
            "total_count": len(indic_langs) + 1,
            "engines": {
                "indicconformer": list(indic_langs.keys()),
                "whisper": ["en", "auto"]
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all loaded models."""
        return {
            "hybrid_service": {
                "loaded": self._loaded,
                "lang_confidence_threshold": self._lang_confidence_threshold,
                "prefer_indic": self._prefer_indic,
                "fallback_language": self._fallback_language
            },
            "language_detector": language_detector.get_model_info(),
            "indic_asr": indic_asr_service.get_model_info(),
            "whisper": whisper_service.get_model_info()
        }


# Global instance
hybrid_stt_service = HybridSTTService()
