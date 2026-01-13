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
        # Audio is already heavily preprocessed in `audio_processor` (AGC/NR/compression).
        # Keep any additional processing here conservative to avoid increasing hallucinations.

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

        # Trim leading/trailing near-silence to reduce hallucinations on short clips.
        audio_data = self._trim_silence(audio_data, sample_rate=16000)

        # Normalize if needed (prevent clipping)
        max_abs = np.abs(audio_data).max()
        if max_abs > 1.0:
            audio_data = audio_data / max_abs
            logger.debug(f"Normalized audio (max was {max_abs:.4f})")

        # Optional gentle RMS normalization (audio_processor already does AGC).
        # Only act if extremely quiet to avoid over-amplifying noise.
        rms = float(np.sqrt(np.mean(np.square(audio_data)))) if len(audio_data) else 0.0
        if 0.0005 < rms < 0.03:
            target_rms = 0.10
            gain = min(target_rms / rms, 3.0)
            audio_data = np.clip(audio_data * gain, -1.0, 1.0)
            logger.debug(f"Applied gentle RMS gain (RMS: {rms:.4f}, gain: {gain:.2f}x)")

        logger.debug(f"Transcribing audio: {len(audio_data)} samples")

        # Run transcription in thread pool
        loop = asyncio.get_event_loop()

        transcribe_options = {
            "fp16": False if self._device == "cpu" else True,
            "task": "transcribe",
            # Anti-hallucination / stability for short utterances
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "beam_size": 5,
            # Default is ~0.6. Lower means *more* sensitive but can increase false positives/hallucinations.
            # 0.4 is a good tradeoff for your "short field answer" use-case.
            "no_speech_threshold": 0.4,
            # Default is -1.0. More negative = more permissive (accept lower-confidence text).
            # We revert closer to default to reduce hallucinations on names/short utterances.
            "logprob_threshold": -1.0,
            # Default is 2.4. Keep default.
            "compression_ratio_threshold": 2.4,
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

    def _trim_silence(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Trim leading/trailing near-silence based on a simple amplitude threshold.
        This is intentionally lightweight (no VAD dependency) and helps reduce hallucinations
        on short inputs that contain lots of silence.
        """
        if audio_data is None or len(audio_data) == 0:
            return audio_data

        abs_audio = np.abs(audio_data)
        max_abs = float(abs_audio.max())
        if max_abs < 1e-6:
            return audio_data

        # Threshold: relative to peak, with a small absolute floor.
        thr = max(0.01 * max_abs, 0.002)
        voiced = np.where(abs_audio > thr)[0]
        if voiced.size == 0:
            return audio_data

        start = int(voiced[0])
        end = int(voiced[-1])

        # Keep a bit of padding so we don't clip phonemes.
        pad = int(0.10 * sample_rate)  # 100ms
        start = max(0, start - pad)
        end = min(len(audio_data) - 1, end + pad)

        # If trimming would produce an extremely short clip, keep original.
        if (end - start + 1) < int(0.25 * sample_rate):
            return audio_data

        return audio_data[start : end + 1]

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
