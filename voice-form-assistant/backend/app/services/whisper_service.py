"""
Whisper Speech-to-Text Service.
Uses the official OpenAI Whisper package for transcription.
"""

import logging
import asyncio
from typing import Tuple, Optional
import numpy as np
import whisper

from ..config import settings

logger = logging.getLogger(__name__)


class WhisperService:
    """Service for speech-to-text using OpenAI Whisper."""

    def __init__(self):
        self.model: Optional[whisper.Whisper] = None
        self._model_name = settings.WHISPER_MODEL
        self._device = settings.WHISPER_DEVICE
        self._lock = asyncio.Lock()

    async def load_model(self) -> None:
        """Load the Whisper model. Called on startup."""
        async with self._lock:
            if self.model is not None:
                return

            logger.info(f"Loading Whisper model: {self._model_name} on {self._device}")

            # Load model in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: whisper.load_model(self._model_name, device=self._device)
            )

            logger.info(f"Whisper model loaded successfully")

    async def transcribe(
        self,
        audio_data: np.ndarray,
        language: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Numpy array of audio samples (16kHz, mono, float32)
            language: Optional language code (e.g., 'en', 'hi'). Auto-detect if None.

        Returns:
            Tuple of (transcription_text, confidence_score)
        """
        if self.model is None:
            await self.load_model()

        # Ensure audio is float32 and normalized
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Normalize if needed
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()

        logger.debug(f"Transcribing audio: {len(audio_data)} samples")

        # Run transcription in thread pool
        loop = asyncio.get_event_loop()

        transcribe_options = {
            "fp16": False if self._device == "cpu" else True,
            "task": "transcribe",
            # Reduce hallucinations / improve stability for short utterances
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "beam_size": 5,
            "no_speech_threshold": 0.6,
        }

        if language:
            transcribe_options["language"] = language

        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(audio_data, **transcribe_options)
            )

            text = result.get("text", "").strip()

            # Calculate confidence from segments if available
            confidence = self._calculate_confidence(result)

            logger.info(f"Transcription: '{text}' (confidence: {confidence:.2f})")

            return text, confidence

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise

    def _calculate_confidence(self, result: dict) -> float:
        """
        Calculate overall confidence score from Whisper result.

        Args:
            result: Whisper transcription result dict

        Returns:
            Confidence score between 0 and 1
        """
        segments = result.get("segments", [])

        if not segments:
            return 0.5  # Default confidence if no segments

        # Calculate average probability from segments
        total_prob = 0.0
        total_words = 0

        for segment in segments:
            # Whisper provides avg_logprob for segments
            avg_logprob = segment.get("avg_logprob", -1.0)
            # Convert log probability to probability
            prob = np.exp(avg_logprob)
            # Weight by number of words
            words = len(segment.get("text", "").split())
            total_prob += prob * words
            total_words += words

        if total_words == 0:
            return 0.5

        avg_confidence = total_prob / total_words

        # Clamp to [0, 1] range
        return float(np.clip(avg_confidence, 0.0, 1.0))

    async def detect_language(self, audio_data: np.ndarray) -> str:
        """
        Detect the language of the audio.

        Args:
            audio_data: Numpy array of audio samples

        Returns:
            Detected language code (e.g., 'en', 'hi')
        """
        if self.model is None:
            await self.load_model()

        # Use first 30 seconds for language detection
        audio_segment = whisper.pad_or_trim(audio_data)

        loop = asyncio.get_event_loop()

        mel = await loop.run_in_executor(
            None,
            lambda: whisper.log_mel_spectrogram(audio_segment).to(self.model.device)
        )

        _, probs = await loop.run_in_executor(
            None,
            lambda: self.model.detect_language(mel)
        )

        detected_lang = max(probs, key=probs.get)
        logger.info(f"Detected language: {detected_lang}")

        return detected_lang

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self._model_name,
            "device": self._device,
            "loaded": self.model is not None
        }


# Global service instance
whisper_service = WhisperService()
