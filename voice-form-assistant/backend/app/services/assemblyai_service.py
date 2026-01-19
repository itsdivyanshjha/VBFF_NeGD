"""
AssemblyAI Speech-to-Text Service.
Cloud-based transcription service supporting 100+ languages including
Hindi, Bengali, Tamil, Telugu, Marathi, and English with automatic language detection.
"""

import logging
import asyncio
import tempfile
import os
import wave
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
import httpx

from ..config import settings

logger = logging.getLogger(__name__)


# Supported languages for MVP
SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "bn": "Bengali", 
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "en": "English"
}


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription."""
    text: str
    confidence: float
    language: str
    language_confidence: float
    engine_used: str = "assemblyai"
    is_indic: bool = False


class AssemblyAIService:
    """
    AssemblyAI transcription service with automatic language detection.
    
    Supports pre-recorded audio transcription with polling for completion.
    Handles audio upload, transcription submission, and result retrieval.
    """

    def __init__(self):
        self.api_key = settings.ASSEMBLYAI_API_KEY
        self.base_url = settings.ASSEMBLYAI_BASE_URL
        self.timeout = settings.ASSEMBLYAI_TIMEOUT
        self.polling_interval = settings.ASSEMBLYAI_POLLING_INTERVAL
        self.max_retries = settings.ASSEMBLYAI_MAX_RETRIES
        
        if not self.api_key:
            logger.warning("AssemblyAI API key not configured")

    def _get_headers(self) -> dict:
        """Get headers for API requests."""
        return {
            "authorization": self.api_key,
            "content-type": "application/json"
        }

    async def upload_audio_raw(self, audio_bytes: bytes) -> str:
        """
        Upload raw audio bytes directly to AssemblyAI (no conversion).
        
        Args:
            audio_bytes: Raw audio data (WebM, WAV, MP3, etc.)
            
        Returns:
            Upload URL for the audio file
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/upload",
                headers={"authorization": self.api_key},
                content=audio_bytes
            )
            response.raise_for_status()
            
            upload_url = response.json()["upload_url"]
            logger.info(f"Audio uploaded to AssemblyAI: {upload_url}")
            return upload_url

    async def upload_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Upload audio to AssemblyAI (converted from numpy array).
        
        Args:
            audio_data: Numpy array of audio samples (mono, float32)
            sample_rate: Audio sample rate
            
        Returns:
            Upload URL for the audio file
        """
        # Save audio to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            
        try:
            # Convert float32 to int16 for WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Write WAV file using Python's wave module
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Upload to AssemblyAI
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                with open(temp_path, "rb") as audio_file:
                    response = await client.post(
                        f"{self.base_url}/upload",
                        headers={"authorization": self.api_key},
                        content=audio_file.read()
                    )
                    response.raise_for_status()
                    
                upload_url = response.json()["upload_url"]
                logger.info(f"Audio uploaded to AssemblyAI: {upload_url}")
                return upload_url
                
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    async def create_transcript(
        self,
        audio_url: str,
        language_hint: Optional[str] = None
    ) -> str:
        """
        Create transcription job.
        
        Args:
            audio_url: URL of uploaded audio
            language_hint: Optional language code hint
            
        Returns:
            Transcript ID
        """
        # Build request payload
        payload = {
            "audio_url": audio_url,
            "language_detection": True,  # Enable automatic language detection
        }
        
        # If language hint provided and supported, use it
        if language_hint and language_hint in SUPPORTED_LANGUAGES:
            payload["language_code"] = language_hint
            logger.info(f"Using language hint: {language_hint}")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/transcript",
                headers=self._get_headers(),
                json=payload
            )
            response.raise_for_status()
            
            transcript_id = response.json()["id"]
            logger.info(f"Transcript created: {transcript_id}")
            return transcript_id

    async def poll_transcript(self, transcript_id: str) -> dict:
        """
        Poll for transcript completion.
        
        Args:
            transcript_id: Transcript ID to poll
            
        Returns:
            Completed transcript data
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while True:
                response = await client.get(
                    f"{self.base_url}/transcript/{transcript_id}",
                    headers=self._get_headers()
                )
                response.raise_for_status()
                
                data = response.json()
                status = data["status"]
                
                if status == "completed":
                    logger.info(f"Transcript completed: {transcript_id}")
                    return data
                elif status == "error":
                    error_msg = data.get("error", "Unknown error")
                    logger.error(f"Transcript failed: {error_msg}")
                    raise Exception(f"Transcription failed: {error_msg}")
                
                # Still processing, wait and retry
                logger.debug(f"Transcript status: {status}, polling again...")
                await asyncio.sleep(self.polling_interval)

    async def transcribe_raw(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe raw audio bytes directly (preserves original quality).
        
        Args:
            audio_bytes: Raw audio data (WebM, WAV, MP3, etc.)
            language_hint: Optional language code hint
            
        Returns:
            TranscriptionResult with text, confidence, and detected language
        """
        if not self.api_key:
            raise ValueError("AssemblyAI API key not configured")
        
        logger.debug(f"Transcribing {len(audio_bytes)} bytes of raw audio")
        
        try:
            # Step 1: Upload raw audio (no conversion!)
            upload_url = await self.upload_audio_raw(audio_bytes)
            
            # Step 2: Create transcript
            transcript_id = await self.create_transcript(upload_url, language_hint)
            
            # Step 3: Poll for completion
            result = await self.poll_transcript(transcript_id)
            
            # Extract results
            text = result.get("text", "").strip()
            confidence = result.get("confidence", 0.0)
            detected_lang = result.get("language_code", "en")
            lang_confidence = result.get("language_confidence", 0.9)
            is_indic = detected_lang in ["hi", "bn", "ta", "te", "mr"]
            
            logger.info(
                f"Transcription: '{text}' "
                f"(confidence: {confidence:.2f}, lang: {detected_lang}, "
                f"lang_confidence: {lang_confidence:.2f})"
            )
            
            return TranscriptionResult(
                text=text,
                confidence=confidence,
                language=detected_lang,
                language_confidence=lang_confidence,
                engine_used="assemblyai",
                is_indic=is_indic
            )
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            raise

    async def transcribe(
        self,
        audio_data: np.ndarray,
        language_hint: Optional[str] = None,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """
        Transcribe audio to text with automatic language detection.
        
        Args:
            audio_data: Numpy array of audio samples (16kHz, mono, float32)
            language_hint: Optional language code hint
            sample_rate: Audio sample rate
            
        Returns:
            TranscriptionResult with text, confidence, and detected language
        """
        if not self.api_key:
            raise ValueError("AssemblyAI API key not configured")
        
        # Ensure audio is float32 and mono
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
        
        logger.debug(f"Transcribing {len(audio_data)} samples ({len(audio_data)/sample_rate:.2f}s)")
        
        try:
            # Step 1: Upload audio
            upload_url = await self.upload_audio(audio_data, sample_rate)
            
            # Step 2: Create transcript
            transcript_id = await self.create_transcript(upload_url, language_hint)
            
            # Step 3: Poll for completion
            result = await self.poll_transcript(transcript_id)
            
            # Extract results
            text = result.get("text", "").strip()
            confidence = result.get("confidence", 0.0)
            
            # Get detected language
            detected_lang = result.get("language_code", "en")
            
            # Language detection confidence
            # AssemblyAI provides language_confidence when language_detection is enabled
            lang_confidence = result.get("language_confidence", 0.9)
            
            # Determine if it's an Indic language
            is_indic = detected_lang in ["hi", "bn", "ta", "te", "mr"]
            
            logger.info(
                f"Transcription: '{text}' "
                f"(confidence: {confidence:.2f}, lang: {detected_lang}, "
                f"lang_confidence: {lang_confidence:.2f})"
            )
            
            return TranscriptionResult(
                text=text,
                confidence=confidence,
                language=detected_lang,
                language_confidence=lang_confidence,
                engine_used="assemblyai",
                is_indic=is_indic
            )
            
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_detail = ""
            try:
                error_detail = e.response.json().get("error", "")
            except:
                error_detail = e.response.text[:200]
            
            if status_code == 401:
                logger.error("AssemblyAI authentication failed - check your API key")
            elif status_code == 429:
                logger.error("AssemblyAI rate limit exceeded")
            elif status_code >= 500:
                logger.error(f"AssemblyAI server error: {status_code}")
            else:
                logger.error(f"AssemblyAI error {status_code}: {error_detail}")
            
            raise Exception(f"AssemblyAI API error: {status_code} - {error_detail}")
            
        except httpx.RequestError as e:
            logger.error(f"AssemblyAI request failed: {e}")
            raise Exception(f"Network error communicating with AssemblyAI: {e}")
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            raise

    def get_supported_languages(self) -> dict:
        """Get dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        return language in SUPPORTED_LANGUAGES

    def get_model_info(self) -> dict:
        """Get information about the service."""
        return {
            "service": "AssemblyAI",
            "engine": "cloud-api",
            "configured": bool(self.api_key),
            "base_url": self.base_url,
            "supported_languages": len(SUPPORTED_LANGUAGES),
            "languages": list(SUPPORTED_LANGUAGES.keys()),
            "language_detection": True
        }


# Global instance
assemblyai_service = AssemblyAIService()
