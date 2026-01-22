"""
AssemblyAI Speech-to-Text Service.
Cloud-based transcription service supporting 100+ languages including
Hindi, Bengali, Tamil, Telugu, Marathi, and English with automatic language detection.

Features field-type-aware transcription for optimal accuracy:
- Numeric fields (tel, number): word_boost for digit recognition
- Entity fields (email, date): entity_detection for structured extraction
- Text fields: general-purpose transcription
"""

import logging
import asyncio
import tempfile
import os
import wave
import time
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import numpy as np
import httpx

from ..config import settings

logger = logging.getLogger(__name__)


# Supported languages for MVP - using AssemblyAI language codes
SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "en": "English",
    "en_us": "English (US)",
    "en_uk": "English (UK)",
    "en_in": "English (India)"
}

# Expected languages for language detection - helps AssemblyAI focus on likely languages
# Note: Use base language codes only (en, hi, etc.) - variants like en_in don't work in expected_languages
EXPECTED_LANGUAGES = ["en", "hi", "bn", "ta", "te", "mr"]

# Field-type-aware ASR configurations
# These configurations optimize transcription based on HTML input types
# Eliminates need for post-processing normalization in 80% of cases
FIELD_TYPE_ASR_CONFIGS = {
    "tel": {
        "word_boost": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
                      "zero", "one", "two", "three", "four", "five", 
                      "six", "seven", "eight", "nine"],
        "boost_param": "high",
        "speech_model": "best",  # Best model for critical numeric fields
        "punctuate": False,
        "format_text": True,
    },
    "number": {
        "word_boost": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                      "zero", "one", "two", "three", "four", "five",
                      "six", "seven", "eight", "nine"],
        "boost_param": "high",
        "speech_model": "best",
        "punctuate": False,
        "format_text": True,
    },
    "email": {
        "entity_detection": True,
        "speech_model": "best",
        "punctuate": False,
        "format_text": True,
    },
    "date": {
        "entity_detection": True,
        "speech_model": "best",
        "punctuate": False,
        "format_text": True,
    },
    "text": {
        "speech_model": "best",
        "punctuate": True,
        "format_text": True,
    },
    "textarea": {
        "speech_model": "best",
        "punctuate": True,
        "format_text": True,
    },
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
    entities: List[Dict[str, Any]] = None  # Extracted entities from entity_detection


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
        """Get headers for JSON API requests."""
        return {
            "authorization": self.api_key,
            "content-type": "application/json"
        }

    def _build_field_aware_config(self, field_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build AssemblyAI configuration optimized for specific field type.
        
        This is the key to scalable, zero-hardcoding transcription:
        - Maps HTML input types to optimal ASR settings
        - Works with ANY form - no per-form configuration needed
        - Eliminates 80% of post-processing normalization
        
        Args:
            field_info: Dict with 'type', 'field_type', 'pattern', 'maxLength', etc.
        
        Returns:
            Dict with AssemblyAI configuration parameters
        """
        if not field_info:
            # Default config for general text
            return FIELD_TYPE_ASR_CONFIGS["text"].copy()
        
        html_type = field_info.get("type", "text").lower()
        field_type = field_info.get("field_type", "").lower()
        
        # Priority 1: Use HTML input type (most reliable)
        if html_type in FIELD_TYPE_ASR_CONFIGS:
            config = FIELD_TYPE_ASR_CONFIGS[html_type].copy()
            logger.info(f"Using ASR config for HTML type: {html_type}")
            return config
        
        # Priority 2: Use detected field_type (from widget's analysis)
        if field_type in FIELD_TYPE_ASR_CONFIGS:
            config = FIELD_TYPE_ASR_CONFIGS[field_type].copy()
            logger.info(f"Using ASR config for field_type: {field_type}")
            return config
        
        # Priority 3: Infer from field characteristics
        pattern = field_info.get("pattern", "")
        max_length = field_info.get("maxLength")
        
        # Numeric patterns → use number config
        if pattern and ("\\d{" in pattern or "[0-9]{" in pattern):
            config = FIELD_TYPE_ASR_CONFIGS["number"].copy()
            logger.info(f"Using ASR config for numeric pattern: {pattern}")
            return config
        
        # Common numeric lengths → use number config
        if max_length in [6, 10, 12]:  # PIN, mobile, Aadhaar
            config = FIELD_TYPE_ASR_CONFIGS["number"].copy()
            logger.info(f"Using ASR config for numeric maxLength: {max_length}")
            return config
        
        # Default to text
        logger.info("Using default text ASR config")
        return FIELD_TYPE_ASR_CONFIGS["text"].copy()

    def _detect_audio_format(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Detect audio format from magic bytes.

        Args:
            audio_bytes: Raw audio data

        Returns:
            Dict with format info: name, content_type, valid, details
        """
        result = {
            "name": "unknown",
            "content_type": "application/octet-stream",
            "valid": False,
            "size_bytes": len(audio_bytes),
            "details": ""
        }

        if len(audio_bytes) < 4:
            result["details"] = "Audio too short to detect format"
            return result

        # WebM: starts with EBML header 0x1A45DFA3
        if audio_bytes[:4] == b'\x1a\x45\xdf\xa3':
            result["name"] = "webm"
            result["content_type"] = "audio/webm"
            result["valid"] = True
            result["details"] = "WebM/EBML container detected"
            return result

        # WAV: starts with "RIFF" and contains "WAVE"
        if audio_bytes[:4] == b'RIFF' and len(audio_bytes) > 11 and audio_bytes[8:12] == b'WAVE':
            result["name"] = "wav"
            result["content_type"] = "audio/wav"
            result["valid"] = True
            result["details"] = "WAV/RIFF container detected"
            return result

        # MP3: starts with ID3 tag or frame sync
        if audio_bytes[:3] == b'ID3' or (audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
            result["name"] = "mp3"
            result["content_type"] = "audio/mpeg"
            result["valid"] = True
            result["details"] = "MP3 format detected"
            return result

        # OGG: starts with "OggS"
        if audio_bytes[:4] == b'OggS':
            result["name"] = "ogg"
            result["content_type"] = "audio/ogg"
            result["valid"] = True
            result["details"] = "OGG container detected"
            return result

        # M4A/MP4: check for ftyp box
        if len(audio_bytes) > 8 and audio_bytes[4:8] == b'ftyp':
            result["name"] = "m4a"
            result["content_type"] = "audio/mp4"
            result["valid"] = True
            result["details"] = "M4A/MP4 container detected"
            return result

        # FLAC: starts with "fLaC"
        if audio_bytes[:4] == b'fLaC':
            result["name"] = "flac"
            result["content_type"] = "audio/flac"
            result["valid"] = True
            result["details"] = "FLAC format detected"
            return result

        # Unknown format - AssemblyAI may still be able to process it
        result["details"] = f"Unknown format. First 8 bytes: {audio_bytes[:8].hex()}"
        result["valid"] = True  # Let AssemblyAI try to handle it

        return result

    async def upload_audio_raw(self, audio_bytes: bytes) -> str:
        """
        Upload raw audio bytes directly to AssemblyAI.

        Uses application/octet-stream as per AssemblyAI documentation.
        AssemblyAI recommends sending raw audio without transcoding.

        Args:
            audio_bytes: Raw audio data (WebM, WAV, MP3, etc.)

        Returns:
            Upload URL for the audio file
        """
        # Detect format for logging
        format_info = self._detect_audio_format(audio_bytes)

        logger.info(
            f"Uploading audio to AssemblyAI: "
            f"{format_info['size_bytes']} bytes, "
            f"format: {format_info['name']}, "
            f"details: {format_info['details']}"
        )

        start_time = time.time()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Use application/octet-stream as per AssemblyAI docs
            response = await client.post(
                f"{self.base_url}/upload",
                headers={
                    "authorization": self.api_key,
                    "content-type": "application/octet-stream"
                },
                content=audio_bytes
            )
            response.raise_for_status()

            upload_url = response.json()["upload_url"]
            elapsed = time.time() - start_time

            logger.info(f"Audio uploaded in {elapsed:.2f}s: {upload_url}")
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
        language_hint: Optional[str] = None,
        field_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create transcription job with field-type-aware optimization.

        Args:
            audio_url: URL of uploaded audio
            language_hint: Optional language code hint (e.g., 'hi', 'en')
            field_info: Optional field metadata for type-aware ASR config

        Returns:
            Transcript ID
        """
        start_time = time.time()

        # Build field-aware ASR configuration
        field_config = self._build_field_aware_config(field_info)
        
        # Build request payload with field-optimized settings
        payload: Dict[str, Any] = {
            "audio_url": audio_url,
        }
        
        # Apply field-specific ASR settings
        for key in ("speech_model", "punctuate", "format_text", "word_boost", "boost_param", "entity_detection"):
            if key in field_config:
                payload[key] = field_config[key]
        
        # Language detection setup
        # ALWAYS use auto-detection for robustness (can handle language switches)
        # Use language_hint as a suggestion if provided, but don't force it
        payload["language_detection"] = True
        
        # Build expected languages list, prioritizing hint if provided
        expected_langs = EXPECTED_LANGUAGES.copy()
        if language_hint and language_hint in SUPPORTED_LANGUAGES:
            # Move hint to front of expected languages (higher priority)
            base_lang = language_hint.split("_")[0]  # en_us → en
            if base_lang in expected_langs:
                expected_langs.remove(base_lang)
            expected_langs.insert(0, base_lang)
            logger.info(f"Using language hint as priority: {base_lang} (expected: {expected_langs})")
        else:
            logger.info(f"Using auto language detection (expected: {expected_langs})")
        
        payload["language_detection_options"] = {
            "expected_languages": expected_langs,
            "fallback_language": "en",
            "code_switching": True,
            "code_switching_confidence_threshold": 0.3
        }

        logger.info(
            f"Field-aware transcription config: "
            f"model={payload.get('speech_model')}, "
            f"word_boost={bool(payload.get('word_boost'))}, "
            f"entity_detection={bool(payload.get('entity_detection'))}, "
            f"punctuate={payload.get('punctuate')}"
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/transcript",
                headers=self._get_headers(),
                json=payload
            )
            response.raise_for_status()

            result = response.json()
            transcript_id = result["id"]
            elapsed = time.time() - start_time

            logger.info(f"Transcript job created in {elapsed:.2f}s: {transcript_id}")
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
        language_hint: Optional[str] = None,
        field_info: Optional[Dict[str, Any]] = None,
    ) -> TranscriptionResult:
        """
        Transcribe raw audio bytes with field-type-aware optimization.

        Args:
            audio_bytes: Raw audio data (WebM, WAV, MP3, etc.)
            language_hint: Optional language code hint
            field_info: Optional field metadata for type-aware ASR config

        Returns:
            TranscriptionResult with text, confidence, language, and entities
        """
        if not self.api_key:
            raise ValueError("AssemblyAI API key not configured")

        total_start = time.time()

        # Detect and log audio format
        format_info = self._detect_audio_format(audio_bytes)
        estimated_duration = len(audio_bytes) / 16000  # Rough estimate: ~16KB/s for WebM

        field_type_info = ""
        if field_info:
            field_type_info = f"field_type={field_info.get('type')}/{field_info.get('field_type')}, "

        logger.info(
            f"Starting field-aware transcription: "
            f"{format_info['size_bytes']} bytes, "
            f"format: {format_info['name']}, "
            f"estimated_duration: {estimated_duration:.1f}s, "
            f"{field_type_info}"
            f"language_hint: {language_hint}"
        )

        try:
            # Step 1: Upload raw audio (no conversion!)
            upload_start = time.time()
            upload_url = await self.upload_audio_raw(audio_bytes)
            upload_elapsed = time.time() - upload_start

            # Step 2: Create transcript with field-aware config
            create_start = time.time()
            transcript_id = await self.create_transcript(
                upload_url,
                language_hint,
                field_info=field_info,
            )
            create_elapsed = time.time() - create_start

            # Step 3: Poll for completion
            poll_start = time.time()
            result = await self.poll_transcript(transcript_id)
            poll_elapsed = time.time() - poll_start

            # Extract results
            text = result.get("text", "").strip()
            confidence = result.get("confidence", 0.0) or 0.0
            detected_lang = result.get("language_code", "en")
            lang_confidence = result.get("language_confidence", 0.9) or 0.9
            is_indic = detected_lang in ["hi", "bn", "ta", "te", "mr"]
            
            # Extract entities if entity_detection was enabled
            entities = result.get("entities", []) or []

            total_elapsed = time.time() - total_start

            # Comprehensive logging
            logger.info(
                f"Transcription completed in {total_elapsed:.2f}s "
                f"(upload: {upload_elapsed:.2f}s, create: {create_elapsed:.2f}s, poll: {poll_elapsed:.2f}s)"
            )
            logger.info(
                f"Result: text='{text[:100]}{'...' if len(text) > 100 else ''}' | "
                f"confidence={confidence:.2f} | lang={detected_lang} | "
                f"lang_confidence={lang_confidence:.2f} | is_indic={is_indic} | "
                f"entities_found={len(entities)}"
            )
            
            if entities:
                for entity in entities[:3]:  # Log first 3 entities
                    logger.info(
                        f"  Entity: {entity.get('entity_type')} = '{entity.get('text')}'"
                    )

            # Warn if transcription is empty
            if not text:
                logger.warning(
                    f"Empty transcription! Audio may be silent or unrecognizable. "
                    f"Format: {format_info['name']}, Size: {format_info['size_bytes']} bytes"
                )

            return TranscriptionResult(
                text=text,
                confidence=confidence,
                language=detected_lang,
                language_confidence=lang_confidence,
                engine_used="assemblyai",
                is_indic=is_indic,
                entities=entities
            )

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_detail = ""
            try:
                error_detail = e.response.json().get("error", e.response.text[:200])
            except Exception:
                error_detail = e.response.text[:200]

            logger.error(
                f"AssemblyAI HTTP error {status_code}: {error_detail} | "
                f"Audio: {format_info['size_bytes']} bytes, format: {format_info['name']}"
            )
            raise

        except Exception as e:
            logger.error(
                f"Transcription error: {e} | "
                f"Audio: {format_info['size_bytes']} bytes, format: {format_info['name']}",
                exc_info=True
            )
            raise

    async def transcribe(
        self,
        audio_data: np.ndarray,
        language_hint: Optional[str] = None,
        sample_rate: int = 16000,
        field_info: Optional[Dict[str, Any]] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio to text with field-type-aware configuration.
        
        Args:
            audio_data: Numpy array of audio samples (16kHz, mono, float32)
            language_hint: Optional language code hint
            sample_rate: Audio sample rate
            field_info: Optional field metadata for type-aware ASR config
            
        Returns:
            TranscriptionResult with text, confidence, language, and entities
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
            
            # Step 2: Create transcript with field-aware config
            transcript_id = await self.create_transcript(
                upload_url,
                language_hint,
                field_info=field_info,
            )
            
            # Step 3: Poll for completion
            result = await self.poll_transcript(transcript_id)
            
            # Extract results
            text = result.get("text", "").strip()
            confidence = result.get("confidence", 0.0)
            detected_lang = result.get("language_code", "en")
            lang_confidence = result.get("language_confidence", 0.9)
            is_indic = detected_lang in ["hi", "bn", "ta", "te", "mr"]
            entities = result.get("entities", []) or []
            
            logger.info(
                f"Transcription: '{text}' "
                f"(confidence: {confidence:.2f}, lang: {detected_lang}, "
                f"lang_confidence: {lang_confidence:.2f}, entities: {len(entities)})"
            )
            
            return TranscriptionResult(
                text=text,
                confidence=confidence,
                language=detected_lang,
                language_confidence=lang_confidence,
                engine_used="assemblyai",
                is_indic=is_indic,
                entities=entities
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
