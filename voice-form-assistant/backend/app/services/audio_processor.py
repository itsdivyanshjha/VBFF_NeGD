"""
Audio Processing Service.
Handles audio format conversions between browser and Whisper.
"""

import logging
import base64
import io
import asyncio
import shutil
import subprocess
from typing import List
import numpy as np
from pydub import AudioSegment
from scipy import signal

from ..config import settings

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Service for audio format conversions."""

    def __init__(self):
        self.target_sample_rate = settings.AUDIO_SAMPLE_RATE
        self.target_channels = settings.AUDIO_CHANNELS
        self._ffmpeg_path = shutil.which("ffmpeg")

    async def decode_audio_chunks(
        self,
        base64_chunks: List[str],
        format: str = "webm"
    ) -> np.ndarray:
        """
        Combine and decode multiple base64 audio chunks to numpy array.

        WebM chunks must be combined as raw bytes before decoding because
        only the first chunk contains the file header.

        Args:
            base64_chunks: List of base64 encoded audio chunks
            format: Audio format (webm, wav, mp3, etc.)

        Returns:
            Numpy array of audio samples (16kHz, mono, float32)
        """
        if not base64_chunks:
            return np.array([], dtype=np.float32)

        loop = asyncio.get_event_loop()

        def process():
            # Combine all chunks as raw bytes first
            combined_bytes = b""
            for chunk in base64_chunks:
                try:
                    combined_bytes += base64.b64decode(chunk)
                except Exception as e:
                    logger.warning(f"Failed to decode base64 chunk: {e}")
                    continue

            if not combined_bytes:
                logger.error("No valid audio data after combining chunks")
                return np.array([], dtype=np.float32)

            logger.info(f"Combined {len(base64_chunks)} chunks into {len(combined_bytes)} bytes")

            samples = self._decode_with_ffmpeg(combined_bytes)
            if samples is None:
                samples = self._decode_with_pydub(combined_bytes, format=format)

            if samples is None or len(samples) == 0:
                return np.array([], dtype=np.float32)

            # Apply audio preprocessing to improve recognition
            samples = self._preprocess_audio(samples)
            
            peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
            rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
            logger.info(f"Audio metrics - peak={peak:.4f}, rms={rms:.4f}, samples={len(samples)}")

            # Very low thresholds to accept quiet speech and different mic sensitivities
            # AGC will boost quiet audio, so we want to be very permissive here
            # Only reject truly silent audio
            if peak < 0.003:
                logger.warning(f"Audio peak too low ({peak:.4f}). Mic may not be capturing voice. Rejecting.")
                return np.array([], dtype=np.float32)
            if rms < 0.001:
                logger.warning(f"Audio RMS too low ({rms:.4f}). No clear speech detected. Rejecting.")
                return np.array([], dtype=np.float32)

            logger.info(f"Decoded audio: {len(samples)} samples, {len(samples)/self.target_sample_rate:.2f}s")
            return samples.astype(np.float32, copy=False)

        return await loop.run_in_executor(None, process)

    async def get_audio_duration(self, audio_data: np.ndarray) -> float:
        """
        Get duration of audio in seconds.

        Args:
            audio_data: Numpy array of audio samples

        Returns:
            Duration in seconds
        """
        if len(audio_data) == 0:
            return 0.0
        return len(audio_data) / self.target_sample_rate

    def is_valid_audio(self, base64_audio: str) -> bool:
        """
        Check if base64 string is valid audio data.

        Args:
            base64_audio: Base64 encoded audio

        Returns:
            True if valid, False otherwise
        """
        try:
            data = base64.b64decode(base64_audio)
            return len(data) > 100  # Minimum reasonable audio size
        except Exception:
            return False

    def _decode_with_ffmpeg(self, data: bytes) -> np.ndarray | None:
        """Decode audio to float32 mono PCM at target sample rate using FFmpeg."""
        if not self._ffmpeg_path:
            return None

        try:
            proc = subprocess.run(
                [
                    self._ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    "pipe:0",
                    "-vn",
                    "-ac",
                    str(self.target_channels),
                    "-ar",
                    str(self.target_sample_rate),
                    "-f",
                    "s16le",
                    "pipe:1",
                ],
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"").decode("utf-8", errors="replace")
            logger.warning(f"ffmpeg decode failed; falling back to pydub. ffmpeg stderr: {stderr.strip()}")
            return None

        pcm16 = np.frombuffer(proc.stdout, dtype=np.int16)
        if pcm16.size == 0:
            return np.array([], dtype=np.float32)

        return (pcm16.astype(np.float32) / 32768.0)

    def _decode_with_pydub(self, data: bytes, format: str = "webm") -> np.ndarray | None:
        """Fallback audio decoder using pydub."""
        audio_io = io.BytesIO(data)

        try:
            audio = AudioSegment.from_file(audio_io, format=format)
        except Exception as e:
            logger.warning(f"Failed to decode as {format}, trying auto-detect: {e}")
            audio_io.seek(0)
            try:
                audio = AudioSegment.from_file(audio_io)
            except Exception as e2:
                logger.error(f"Failed to decode audio: {e2}")
                return None

        if audio.channels > 1:
            audio = audio.set_channels(1)

        if audio.frame_rate != self.target_sample_rate:
            audio = audio.set_frame_rate(self.target_sample_rate)

        raw = np.array(audio.get_array_of_samples())

        if audio.sample_width == 1:
            samples = raw.astype(np.float32) / 128.0 - 1.0
        elif audio.sample_width == 2:
            samples = raw.astype(np.float32) / 32768.0
        elif audio.sample_width == 4:
            samples = raw.astype(np.float32) / 2147483648.0
        else:
            samples = raw.astype(np.float32)

        return samples

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply audio preprocessing to improve speech recognition.
        
        Steps:
        1. High-pass filter to remove low-frequency noise
        2. Noise reduction using spectral subtraction
        3. Auto-gain control (AGC) to normalize volume
        4. Dynamic range compression
        
        Args:
            audio: Input audio array (float32, normalized to [-1, 1])
            
        Returns:
            Preprocessed audio array
        """
        if len(audio) == 0:
            return audio
        
        # Make a copy to avoid modifying original
        processed = audio.copy()
        
        # 1. High-pass filter to remove low-frequency noise (below 80Hz)
        # This removes rumble, wind noise, etc.
        nyquist = self.target_sample_rate / 2
        high_pass_freq = 80.0 / nyquist  # Normalized frequency
        if high_pass_freq < 1.0:
            try:
                b, a = signal.butter(4, high_pass_freq, btype='high')
                processed = signal.filtfilt(b, a, processed)
                logger.debug("Applied high-pass filter (80Hz cutoff)")
            except Exception as e:
                logger.warning(f"High-pass filter failed: {e}")
        
        # 2. Noise reduction using spectral gating
        # Estimate noise floor from first 100ms (assuming silence at start)
        noise_samples = int(0.1 * self.target_sample_rate)  # 100ms
        if len(processed) > noise_samples:
            noise_floor = np.percentile(np.abs(processed[:noise_samples]), 10)
            # Apply gentle noise gate
            noise_gate_threshold = max(noise_floor * 3, 0.001)
            processed = np.where(
                np.abs(processed) > noise_gate_threshold,
                processed,
                processed * 0.1  # Reduce noise by 90%
            )
            logger.debug(f"Applied noise gate (threshold: {noise_gate_threshold:.6f})")
        
        # 3. Auto-gain control (AGC) - normalize to target RMS
        # Target RMS for good speech recognition (not too loud, not too quiet)
        target_rms = 0.15
        current_rms = np.sqrt(np.mean(np.square(processed)))
        
        if current_rms > 0.001:  # Avoid division by zero
            # Calculate gain factor, but limit to avoid distortion
            gain_factor = target_rms / current_rms
            # Limit gain to 10x to avoid amplifying noise too much
            gain_factor = min(gain_factor, 10.0)
            # Only apply gain if audio is too quiet
            if gain_factor > 1.2:  # Only boost if significantly quiet
                processed = processed * gain_factor
                logger.debug(f"Applied AGC (gain: {gain_factor:.2f}x, RMS: {current_rms:.4f} -> {target_rms:.4f})")
        
        # 4. Dynamic range compression
        # Soft compression to even out volume variations
        threshold = 0.3  # Start compressing above this level
        ratio = 3.0  # Compression ratio
        
        # Apply soft-knee compression
        abs_audio = np.abs(processed)
        compressed = np.copy(processed)
        
        # Find samples above threshold
        above_threshold = abs_audio > threshold
        
        if np.any(above_threshold):
            # Calculate compression amount
            excess = abs_audio[above_threshold] - threshold
            compressed_amount = excess / ratio
            new_level = threshold + compressed_amount
            
            # Apply compression while preserving sign
            compressed[above_threshold] = np.sign(processed[above_threshold]) * new_level
            logger.debug(f"Applied compression ({np.sum(above_threshold)} samples above threshold)")
        
        # 5. Final normalization to prevent clipping
        max_val = np.max(np.abs(compressed))
        if max_val > 0.95:  # Prevent clipping
            compressed = compressed * (0.95 / max_val)
            logger.debug(f"Normalized to prevent clipping (max was {max_val:.4f})")
        
        return compressed.astype(np.float32)


# Global instance
audio_processor = AudioProcessor()
