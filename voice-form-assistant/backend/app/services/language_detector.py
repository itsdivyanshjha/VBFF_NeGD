"""
Language Detection Service.
Uses SpeechBrain's VoxLingua107 ECAPA-TDNN model for spoken language identification.
Supports 107 languages including English and major Indian languages.
"""

import logging
import asyncio
from typing import Tuple, Optional, Dict, List
import numpy as np
import torch
import torchaudio

from ..config import settings

logger = logging.getLogger(__name__)

# Mapping of VoxLingua107 language codes to our internal codes
# VoxLingua uses ISO 639-1/639-3 codes
VOXLINGUA_TO_INTERNAL = {
    # English
    "en": "en",
    # Indian languages (VoxLingua supports these)
    "hi": "hi",  # Hindi
    "bn": "bn",  # Bengali
    "ta": "ta",  # Tamil
    "te": "te",  # Telugu
    "mr": "mr",  # Marathi
    "gu": "gu",  # Gujarati
    "kn": "kn",  # Kannada
    "ml": "ml",  # Malayalam
    "pa": "pa",  # Punjabi
    "or": "or",  # Odia
    "as": "as",  # Assamese
    "ur": "ur",  # Urdu
    "ne": "ne",  # Nepali
    "sd": "sd",  # Sindhi
    "sa": "sa",  # Sanskrit
}

# Languages supported by IndicConformer (all 22 scheduled languages)
INDIC_LANGUAGES = {
    "as",   # Assamese
    "bn",   # Bengali
    "brx",  # Bodo
    "doi",  # Dogri
    "gu",   # Gujarati
    "hi",   # Hindi
    "kn",   # Kannada
    "ks",   # Kashmiri
    "kok",  # Konkani
    "mai",  # Maithili
    "ml",   # Malayalam
    "mni",  # Manipuri
    "mr",   # Marathi
    "ne",   # Nepali
    "or",   # Odia
    "pa",   # Punjabi
    "sa",   # Sanskrit
    "sat",  # Santali
    "sd",   # Sindhi
    "ta",   # Tamil
    "te",   # Telugu
    "ur",   # Urdu
}


class LanguageDetector:
    """
    Spoken Language Identification using SpeechBrain's VoxLingua107 model.
    Detects whether audio is English or an Indian language, then routes accordingly.
    """

    def __init__(self):
        self._model = None
        self._device = None
        self._lock = asyncio.Lock()
        self._model_name = "speechbrain/lang-id-voxlingua107-ecapa"

    async def load_model(self) -> None:
        """Load the SpeechBrain language identification model."""
        async with self._lock:
            if self._model is not None:
                return

            logger.info(f"Loading language detection model: {self._model_name}")

            loop = asyncio.get_event_loop()

            def _load():
                from speechbrain.inference.classifiers import EncoderClassifier
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                classifier = EncoderClassifier.from_hparams(
                    source=self._model_name,
                    savedir=settings.MODEL_CACHE_DIR + "/lang-id-voxlingua107",
                    run_opts={"device": device}
                )
                return classifier, device

            self._model, self._device = await loop.run_in_executor(None, _load)
            logger.info(f"Language detection model loaded on {self._device}")

    async def detect_language(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> Tuple[str, float, bool]:
        """
        Detect the spoken language in audio.

        Args:
            audio_data: Numpy array of audio samples (mono, float32)
            sample_rate: Audio sample rate (default 16kHz)

        Returns:
            Tuple of (language_code, confidence, is_indic)
            - language_code: ISO language code (e.g., 'hi', 'en', 'ta')
            - confidence: Confidence score (0-1)
            - is_indic: True if language is Indian (for routing to IndicConformer)
        """
        if self._model is None:
            await self.load_model()

        loop = asyncio.get_event_loop()

        def _detect():
            # Convert numpy to torch tensor
            if audio_data.dtype != np.float32:
                audio = audio_data.astype(np.float32)
            else:
                audio = audio_data

            # Ensure 1D
            if len(audio.shape) > 1:
                audio = audio.flatten()

            # Convert to torch tensor
            waveform = torch.from_numpy(audio).unsqueeze(0)

            # Resample if needed (model expects 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Move to device
            waveform = waveform.to(self._device)

            # Get predictions
            out_prob, score, index, text_lab = self._model.classify_batch(waveform)

            # Get top prediction
            lang_code = text_lab[0]
            confidence = float(score[0])

            return lang_code, confidence

        try:
            lang_code, confidence = await loop.run_in_executor(None, _detect)

            # Map to internal code
            internal_code = VOXLINGUA_TO_INTERNAL.get(lang_code, lang_code)

            # Determine if it's an Indic language
            is_indic = internal_code in INDIC_LANGUAGES

            logger.info(
                f"Language detected: {lang_code} -> {internal_code} "
                f"(confidence: {confidence:.2f}, is_indic: {is_indic})"
            )

            return internal_code, confidence, is_indic

        except Exception as e:
            logger.error(f"Language detection error: {e}")
            # Default to Hindi with low confidence if detection fails
            return "hi", 0.3, True

    async def get_top_languages(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        top_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Get top-k language predictions with scores.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            top_k: Number of top predictions to return

        Returns:
            List of dicts with language_code, confidence, is_indic
        """
        if self._model is None:
            await self.load_model()

        loop = asyncio.get_event_loop()

        def _detect_topk():
            if audio_data.dtype != np.float32:
                audio = audio_data.astype(np.float32)
            else:
                audio = audio_data

            if len(audio.shape) > 1:
                audio = audio.flatten()

            waveform = torch.from_numpy(audio).unsqueeze(0)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            waveform = waveform.to(self._device)

            # Get raw probabilities
            out_prob, score, index, text_lab = self._model.classify_batch(waveform)

            # Get all probabilities
            probs = out_prob[0].cpu().numpy()
            
            # Get label encoder to map indices to language codes
            label_encoder = self._model.hparams.label_encoder
            
            # Get top-k indices
            top_indices = np.argsort(probs)[::-1][:top_k]

            results = []
            for idx in top_indices:
                lang_code = label_encoder.decode_ndim(idx)
                conf = float(probs[idx])
                internal_code = VOXLINGUA_TO_INTERNAL.get(lang_code, lang_code)
                is_indic = internal_code in INDIC_LANGUAGES
                results.append({
                    "language_code": internal_code,
                    "original_code": lang_code,
                    "confidence": conf,
                    "is_indic": is_indic
                })

            return results

        try:
            return await loop.run_in_executor(None, _detect_topk)
        except Exception as e:
            logger.error(f"Top-k language detection error: {e}")
            return [{"language_code": "hi", "confidence": 0.3, "is_indic": True}]

    def get_supported_languages(self) -> Dict[str, List[str]]:
        """Get list of supported languages by category."""
        return {
            "indic": list(INDIC_LANGUAGES),
            "english": ["en"],
            "all_mapped": list(VOXLINGUA_TO_INTERNAL.keys())
        }

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        return {
            "model_name": self._model_name,
            "device": str(self._device) if self._device else "not_loaded",
            "loaded": self._model is not None,
            "supported_languages": len(VOXLINGUA_TO_INTERNAL),
            "indic_languages": len(INDIC_LANGUAGES)
        }


# Global instance
language_detector = LanguageDetector()
