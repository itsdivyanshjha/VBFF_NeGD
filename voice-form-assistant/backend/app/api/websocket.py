"""
WebSocket Handler.
Main communication endpoint for voice form filling.
Supports multilingual input via AssemblyAI with automatic language detection.
"""

import logging
import json
import base64
import re
from typing import Dict, Any, Optional, List
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import numpy as np

from ..services.assemblyai_service import assemblyai_service, TranscriptionResult
from ..services.openrouter_client import openrouter_client
from ..services.tts_service import tts_service
from ..services.audio_processor import audio_processor
from ..services.session_manager import session_manager, ConversationSession
from ..services.question_builder import build_next_field_transition
from ..services.validators import validate_field_value
from ..config import settings

logger = logging.getLogger(__name__)


# Minimum audio size in bytes - WebM at 128kbps is ~16KB/sec
# 3KB minimum = roughly 0.2 seconds, but we want at least 0.5s of speech
MIN_AUDIO_BYTES = 4000  # ~0.25 seconds minimum

# Estimated bytes per second for WebM/Opus at 128kbps
BYTES_PER_SECOND_ESTIMATE = 16000

# Minimum duration in seconds
MIN_AUDIO_DURATION_SECONDS = 0.3


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept and register connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")

    def disconnect(self, session_id: str) -> None:
        """Remove connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")

    async def send_message(self, session_id: str, message: Dict[str, Any]) -> None:
        """Send message to specific connection."""
        ws = self.active_connections.get(session_id)
        if ws:
            await ws.send_json(message)
            logger.debug(f"Sent to {session_id}: {message.get('type')}")


manager = ConnectionManager()


class VoiceFormHandler:
    """Handles voice form filling logic."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.session: Optional[ConversationSession] = None
        self._processing = False
        self._detected_language: Optional[str] = None  # Track detected language for hints
        self._empty_transcription_count = 0  # Track consecutive empty transcriptions

    async def send(self, message: Dict[str, Any]) -> bool:
        """Send message to client. Returns False if client disconnected."""
        try:
            await self.websocket.send_json(message)
            return True
        except Exception:
            return False

    def _normalize_entity_value(self, entity_text: str, entity_type: str) -> str:
        """
        Normalize entity values from spoken text to proper format.
        
        AssemblyAI entity_detection returns literal spoken text:
        - "at the rate" instead of "@"
        - "dot" instead of "."
        
        This method converts spoken patterns to proper symbols.
        """
        if entity_type == "email_address":
            # Email normalization
            normalized = entity_text.lower().strip()
            
            # Use regex for robust replacement (handles various spacing)
            # Order matters: longer patterns first
            
            # "at the rate" variants (with or without spaces)
            normalized = re.sub(r'\s*at\s*the\s*rate\s*', '@', normalized)
            normalized = re.sub(r'\s*@\s*the\s*rate\s*', '@', normalized)  # If @ spoken
            
            # "at" as @ (but not in words like "chat")
            # Only replace standalone "at" surrounded by spaces or at boundaries
            normalized = re.sub(r'\s+at\s+', '@', normalized)
            normalized = re.sub(r'^at\s+', '@', normalized)
            normalized = re.sub(r'\s+at$', '@', normalized)
            
            # "dot" as . (handle all variations)
            # "prashant dot singh" → "prashant.singh"
            # "gmail dot com" → "gmail.com"
            normalized = re.sub(r'\s*dot\s*', '.', normalized)
            normalized = re.sub(r'\s*period\s*', '.', normalized)
            normalized = re.sub(r'\s*point\s*', '.', normalized)
            
            # Other symbols
            normalized = re.sub(r'\s*underscore\s*', '_', normalized)
            normalized = re.sub(r'\s*dash\s*', '-', normalized)
            normalized = re.sub(r'\s*hyphen\s*', '-', normalized)
            
            # Remove any remaining spaces
            normalized = normalized.replace(" ", "")
            
            logger.info(f"Email normalized: '{entity_text}' → '{normalized}'")
            return normalized
        
        elif entity_type == "phone_number":
            # Phone normalization - remove all non-digits
            normalized = re.sub(r'[^\d]', '', entity_text)
            return normalized
        
        else:
            # Default: return as-is
            return entity_text
    
    def _validate_audio(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Validate audio before sending to AssemblyAI.

        Checks:
        - Minimum size (at least MIN_AUDIO_BYTES)
        - Audio format detection (WebM magic bytes)
        - Estimated duration

        Returns:
            Dict with: valid, format, size_bytes, estimated_duration_s, error, error_detail
        """
        result = {
            "valid": False,
            "format": "unknown",
            "size_bytes": len(audio_bytes),
            "estimated_duration_s": 0.0,
            "error": None,
            "error_detail": None
        }

        # Check minimum size
        if len(audio_bytes) < MIN_AUDIO_BYTES:
            result["error"] = "audio_too_short"
            result["error_detail"] = (
                f"Audio is too short ({len(audio_bytes)} bytes). "
                "Please speak for at least 1 second."
            )
            logger.warning(
                f"Audio too short: {len(audio_bytes)} bytes "
                f"(minimum: {MIN_AUDIO_BYTES} bytes)"
            )
            return result

        # Detect format from magic bytes
        if len(audio_bytes) >= 4:
            # WebM: starts with EBML header 0x1A45DFA3
            if audio_bytes[:4] == b'\x1a\x45\xdf\xa3':
                result["format"] = "webm"
            # WAV
            elif audio_bytes[:4] == b'RIFF':
                result["format"] = "wav"
            # OGG
            elif audio_bytes[:4] == b'OggS':
                result["format"] = "ogg"
            # MP3
            elif audio_bytes[:3] == b'ID3' or (audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
                result["format"] = "mp3"
            # M4A/MP4
            elif len(audio_bytes) > 8 and audio_bytes[4:8] == b'ftyp':
                result["format"] = "m4a"

        # Log format detection
        if result["format"] == "unknown":
            logger.warning(
                f"Unknown audio format. First 16 bytes: {audio_bytes[:16].hex()}"
            )

        # Estimate duration based on typical bitrate
        result["estimated_duration_s"] = len(audio_bytes) / BYTES_PER_SECOND_ESTIMATE

        # Check minimum duration
        if result["estimated_duration_s"] < MIN_AUDIO_DURATION_SECONDS:
            result["error"] = "audio_too_short"
            result["error_detail"] = (
                f"Audio duration too short ({result['estimated_duration_s']:.1f}s). "
                "Please speak for at least 1 second."
            )
            return result

        result["valid"] = True
        return result


    async def handle_message(self, data: Dict[str, Any]) -> None:
        """Route incoming messages to handlers."""
        msg_type = data.get("type")

        handlers = {
            "init": self._handle_init,
            "audio_complete": self._handle_audio_complete,
            "user_confirmation": self._handle_confirmation,
            "skip_field": self._handle_skip,
            "restart": self._handle_restart,
            "navigate_field": self._handle_navigate,
        }

        handler = handlers.get(msg_type)
        if handler:
            try:
                await handler(data)
            except Exception as e:
                if "disconnect" in str(e).lower() or "closed" in str(e).lower():
                    return
                logger.error(f"Error handling {msg_type}: {e}", exc_info=True)
                try:
                    await self._send_error(str(e))
                except Exception:
                    pass
        else:
            logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_init(self, data: Dict[str, Any]) -> None:
        """Initialize a new form-filling session."""
        form_schema = data.get("formSchema", {})

        if not form_schema.get("fields"):
            await self._send_error("Invalid form schema: no fields found")
            return

        # Create session
        self.session = await session_manager.create_session(form_schema)
        self.session.state = "filling"

        logger.info(f"Session initialized: {self.session.session_id} with {len(self.session.get_fields())} fields")

        # Generate greeting
        form_name = form_schema.get("name", "the form")
        greeting_text = await openrouter_client.generate_response(
            "greeting",
            form_name=form_name
        )

        # Generate TTS for greeting
        greeting_audio = await tts_service.synthesize(greeting_text)

        self.session.add_to_history("assistant", greeting_text)
        await session_manager.save_session(self.session)

        # Send greeting
        await self.send({
            "type": "greeting",
            "text": greeting_text,
            "audio": greeting_audio,
            "sessionId": self.session.session_id
        })

        # Ask for first field after greeting
        # Note: Frontend will wait for greeting audio to finish before showing record button
        await asyncio.sleep(0.5)  # Brief pause
        await self._ask_current_field()

    async def _handle_audio_complete(self, data: Dict[str, Any]) -> None:
        """Handle complete audio recording from client."""
        session_id = data.get("sessionId")
        audio_data = data.get("audio", "")
        client_language_hint = data.get("languageHint")  # Optional language hint from client

        if not session_id or not audio_data:
            await self._send_error("Missing session or audio data")
            return

        # Load session if needed
        if not self.session or self.session.session_id != session_id:
            self.session = await session_manager.get_session(session_id)
            if not self.session:
                await self._send_error("Session not found")
                return

        if self._processing:
            return

        self._processing = True

        try:
            # Decode base64 audio to raw bytes
            audio_bytes = base64.b64decode(audio_data)

            # Validate audio before sending to AssemblyAI
            audio_info = self._validate_audio(audio_bytes)

            logger.info(
                f"Received audio: {audio_info['size_bytes']} bytes, "
                f"format: {audio_info['format']}, "
                f"estimated_duration: {audio_info['estimated_duration_s']:.2f}s"
            )

            if not audio_info["valid"]:
                await self._send_audio_quality_error(audio_info.get("error_detail", ""))
                return

            # Determine language hint: prefer client hint, then detected language from previous turns
            language_hint = client_language_hint or self._detected_language
            if language_hint:
                logger.info(f"Using language hint: {language_hint}")

            # Get current field for field-type-aware transcription
            current_field = None
            if self.session.state != "confirming":
                current_field = self.session.get_current_field()

            # Pass field metadata to AssemblyAI for optimal ASR configuration
            # This eliminates 80% of normalization needs by getting it right at source
            if current_field:
                logger.info(
                    f"Field context: type={current_field.get('type')}, "
                    f"field_type={current_field.get('field_type')}, "
                    f"label={current_field.get('label')}"
                )

            # Send RAW audio directly to AssemblyAI with field-aware config
            # This preserves the original quality from the browser
            result: TranscriptionResult = await assemblyai_service.transcribe_raw(
                audio_bytes,
                language_hint=language_hint,
                field_info=current_field,
            )

            transcription = result.text
            confidence = result.confidence
            detected_lang = result.language
            engine_used = result.engine_used
            entities = result.entities or []

            logger.info(
                f"Transcription [{engine_used}]: '{transcription}' "
                f"(confidence: {confidence:.2f}, lang: {detected_lang}, "
                f"is_indic: {result.is_indic}, entities: {len(entities)})"
            )

            # Store detected language for future hints (improves consistency)
            self._detected_language = detected_lang

            # Very low confidence threshold (0.1) to accept more transcriptions
            # This is especially important for accented speech (Indian English, Hindi, Hinglish)
            # We prefer false positives over false negatives - users can always say "no" to confirmations
            if not transcription or confidence < 0.1:
                self._empty_transcription_count += 1
                logger.warning(
                    f"Low confidence transcription: '{transcription}' (confidence: {confidence:.2f}) "
                    f"[{self._empty_transcription_count} consecutive]"
                )
                
                # Reset language hint after 2 consecutive empty transcriptions
                # This handles cases where the hint is causing the problem
                if self._empty_transcription_count >= 2 and self._detected_language:
                    logger.warning(
                        f"Resetting language hint (was: {self._detected_language}) "
                        "due to repeated empty transcriptions"
                    )
                    self._detected_language = None
                    self._empty_transcription_count = 0
                
                await self._ask_repeat()
                return
            
            # Reset empty transcription counter on success
            self._empty_transcription_count = 0

            # Record user input with language metadata
            self.session.add_to_history("user", transcription)

            # Store detected language in session for potential use in response generation
            if hasattr(self.session, 'detected_language'):
                self.session.detected_language = detected_lang

            # Process based on session state
            if self.session.state == "confirming":
                await self._process_confirmation_response(transcription)
            else:
                await self._process_field_input(transcription, entities=entities)

        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            await self._send_error("Failed to process audio. Please try again.")

        finally:
            self._processing = False

    async def _process_field_input(self, transcription: str, entities: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Process user input for a field with entity-aware extraction.
        
        For numeric/entity fields: Uses AssemblyAI entities (if available) to bypass LLM
        For text fields: Falls back to LLM extraction
        """
        current_field = self.session.get_current_field()

        if not current_field:
            # All fields might be filled
            if self.session.is_complete():
                await self._complete_form()
            else:
                # Find next unfilled field
                next_field = self.session.get_next_unfilled_field()
                if next_field:
                    await self._ask_current_field()
                else:
                    await self._complete_form()
            return

        field_id = current_field.get("id") or current_field.get("name")
        field_label = current_field.get("label", field_id)
        field_type = current_field.get("field_type", "").lower()
        html_type = current_field.get("type", "text").lower()

        # TIER 1: Try entity extraction first (fastest, most accurate for supported types)
        extracted_value = None
        confidence = 0.0
        extraction_method = "unknown"
        
        if entities:
            # Map field types to entity types
            # NOTE: Excluding date/dob - LLM handles date formatting better (YYYY-MM-DD)
            entity_type_map = {
                "mobile": "phone_number",
                "tel": "phone_number",
                "email": "email_address",
                # "date": "date",  # Skip - entity returns "august 21 1998", LLM formats to "1998-08-21"
                # "dob": "date",   # Skip - same reason
            }
            
            expected_entity_type = entity_type_map.get(field_type) or entity_type_map.get(html_type)
            
            if expected_entity_type:
                # Look for matching entity
                for entity in entities:
                    if entity.get("entity_type") == expected_entity_type:
                        raw_entity = entity.get("text", "").strip()
                        
                        # Normalize entity value (AssemblyAI returns literal speech)
                        normalized = self._normalize_entity_value(raw_entity, expected_entity_type)
                        
                        extracted_value = normalized
                        confidence = 0.95  # High confidence for entity extraction
                        extraction_method = "entity_detection"
                        logger.info(
                            f"ENTITY EXTRACTION: Found {expected_entity_type} = '{raw_entity}' → '{normalized}'"
                        )
                        break
        
        # TIER 2: For numeric fields, use transcription directly (word_boost should handle it)
        if not extracted_value and html_type in ("tel", "number"):
            # Remove spaces and non-digit chars for numeric fields
            cleaned = re.sub(r'[^\d]', '', transcription)
            if cleaned:
                extracted_value = cleaned
                confidence = 0.9  # High confidence for numeric fields with word_boost
                extraction_method = "word_boost_numeric"
                logger.info(
                    f"NUMERIC EXTRACTION: '{transcription}' → '{extracted_value}'"
                )
        
        # TIER 3: Fall back to LLM extraction for complex fields
        if not extracted_value:
            context = {
                "filled_fields": self.session.filled_fields,
                "history": self.session.conversation_history[-5:]
            }

            extraction = await openrouter_client.extract_field_value(
                current_field,
                transcription,
                context
            )

            extracted_value = extraction.get("value")
            confidence = extraction.get("confidence", 0)
            extraction_method = "llm"
        
        needs_confirmation = True  # Always default to confirmation

        # LOG EXTRACTED VALUE FOR DEBUGGING
        logger.info(
            f"EXTRACTION ({extraction_method}): transcription='{transcription}' → "
            f"extracted_value='{extracted_value}' | "
            f"confidence={confidence:.2f} | field={field_label}"
        )

        if not extracted_value:
            # Could not extract value
            await self._ask_repeat()
            return

        # Validate value
        validation = validate_field_value(current_field, extracted_value)

        # LOG VALIDATION RESULT
        logger.info(
            f"VALIDATION: value='{extracted_value}' → "
            f"valid={validation.get('valid')} | "
            f"error={validation.get('error', 'none')}"
        )

        if not validation.get("valid"):
            # Validation error
            error_text = await openrouter_client.generate_response(
                "validation_error",
                field_label=field_label,
                error=validation.get("error", "Invalid value")
            )
            error_audio = await tts_service.synthesize(error_text)

            self.session.add_to_history("assistant", error_text)
            await session_manager.save_session(self.session)

            await self.send({
                "type": "validation_error",
                "text": error_text,
                "audio": error_audio,
                "fieldId": field_id,
                "fieldLabel": field_label,
                "error": validation.get("error")
            })
            return

        # Use formatted value
        formatted_value = validation.get("formatted", extracted_value)

        # Decide if confirmation needed
        critical_types = ["aadhaar", "pan", "mobile", "email"]
        is_critical = current_field.get("field_type") in critical_types

        if needs_confirmation or is_critical or confidence < 0.85:
            # Request confirmation
            self.session.set_pending_confirmation(
                field_id,
                field_label,
                formatted_value,
                confidence
            )

            confirm_text = await openrouter_client.generate_response(
                "confirm_value",
                field_label=field_label,
                value=formatted_value
            )
            confirm_audio = await tts_service.synthesize(confirm_text)

            self.session.add_to_history("assistant", confirm_text)
            await session_manager.save_session(self.session)

            await self.send({
                "type": "confirmation_request",
                "text": confirm_text,
                "audio": confirm_audio,
                "fieldId": field_id,
                "fieldLabel": field_label,
                "value": formatted_value
            })
        else:
            # Auto-fill without confirmation
            await self._fill_and_advance(field_id, formatted_value)

    async def _process_confirmation_response(self, transcription: str) -> None:
        """Process user's yes/no confirmation response."""
        text_lower = transcription.lower().strip()

        # Check for yes/no
        positive_words = ["yes", "yeah", "yep", "correct", "right", "haan", "ha", "okay", "ok", "sure", "confirm"]
        negative_words = ["no", "nope", "wrong", "incorrect", "nahi", "change", "different"]

        is_positive = any(word in text_lower for word in positive_words)
        is_negative = any(word in text_lower for word in negative_words)

        pending = self.session.pending_confirmation

        if not pending:
            # No pending confirmation
            self.session.state = "filling"
            await self._ask_current_field()
            return

        if is_positive and not is_negative:
            # Confirmed - fill the field
            field_id = pending["field_id"]
            value = pending["value"]
            self.session.clear_pending_confirmation()
            await self._fill_and_advance(field_id, value)

        elif is_negative and not is_positive:
            # Rejected - ask again
            field_label = pending["field_label"]
            self.session.clear_pending_confirmation()

            ask_again_text = f"Okay, let me ask again. What is your {field_label}?"
            ask_again_audio = await tts_service.synthesize(ask_again_text)

            self.session.add_to_history("assistant", ask_again_text)
            await session_manager.save_session(self.session)

            await self.send({
                "type": "ask_field",
                "text": ask_again_text,
                "audio": ask_again_audio,
                "fieldId": pending["field_id"]
            })
        else:
            # Unclear response
            await self._ask_yes_no()

    async def _handle_confirmation(self, data: Dict[str, Any]) -> None:
        """Handle button-based confirmation."""
        session_id = data.get("sessionId")
        confirmed = data.get("confirmed", False)

        if not self.session or self.session.session_id != session_id:
            self.session = await session_manager.get_session(session_id)
            if not self.session:
                await self._send_error("Session not found")
                return

        pending = self.session.pending_confirmation

        if not pending:
            return

        if confirmed:
            field_id = pending["field_id"]
            value = pending["value"]
            self.session.clear_pending_confirmation()
            await self._fill_and_advance(field_id, value)
        else:
            field_label = pending["field_label"]
            self.session.clear_pending_confirmation()

            ask_again_text = f"Let me ask again. What is your {field_label}?"
            ask_again_audio = await tts_service.synthesize(ask_again_text)

            self.session.add_to_history("assistant", ask_again_text)
            await session_manager.save_session(self.session)

            await self.send({
                "type": "ask_field",
                "text": ask_again_text,
                "audio": ask_again_audio,
                "fieldId": pending["field_id"]
            })

    async def _handle_skip(self, data: Dict[str, Any]) -> None:
        """Skip current field."""
        session_id = data.get("sessionId")

        if not self.session or self.session.session_id != session_id:
            self.session = await session_manager.get_session(session_id)
            if not self.session:
                return

        current_field = self.session.get_current_field()
        if current_field and not current_field.get("required"):
            self.session.clear_pending_confirmation()
            self.session.advance_field()
            await session_manager.save_session(self.session)
            await self._ask_current_field()
        else:
            # Can't skip required field
            skip_error = "This field is required and cannot be skipped."
            skip_audio = await tts_service.synthesize(skip_error)

            await self.send({
                "type": "error",
                "text": skip_error,
                "audio": skip_audio
            })

    async def _handle_navigate(self, data: Dict[str, Any]) -> None:
        """Handle field navigation (previous/next)."""
        session_id = data.get("sessionId")
        direction = data.get("direction")  # "previous" or "next"

        if not self.session or self.session.session_id != session_id:
            self.session = await session_manager.get_session(session_id)
            if not self.session:
                await self._send_error("Session not found")
                return

        # Clear any pending confirmation
        self.session.clear_pending_confirmation()

        if direction == "previous":
            # Go to previous field
            if self.session.current_field_index > 0:
                self.session.current_field_index -= 1
                
                # Clear the value of the field we're going back to (so user can re-enter)
                current_field = self.session.get_current_field()
                if current_field:
                    field_id = current_field.get("id") or current_field.get("name")
                    if field_id and field_id in self.session.filled_fields:
                        del self.session.filled_fields[field_id]
                        logger.info(f"Cleared field {field_id} for re-entry")

                await session_manager.save_session(self.session)
                await self._ask_current_field()
            else:
                # Already at first field
                await self._ask_current_field()

        elif direction == "next":
            # Skip to next field (same as skip_field but via navigation)
            current_field = self.session.get_current_field()
            if current_field and not current_field.get("required"):
                self.session.advance_field()
                await session_manager.save_session(self.session)
                await self._ask_current_field()
            else:
                # Can't skip required field
                skip_error = "This field is required and cannot be skipped."
                skip_audio = await tts_service.synthesize(skip_error)

                await self.send({
                    "type": "error",
                    "text": skip_error,
                    "audio": skip_audio
                })

    async def _handle_restart(self, data: Dict[str, Any]) -> None:
        """Restart the form filling process."""
        if self.session:
            # Reset session
            self.session.current_field_index = 0
            self.session.filled_fields = {}
            self.session.conversation_history = []
            self.session.state = "filling"
            self.session.pending_confirmation = None

            await session_manager.save_session(self.session)

            # Send restart message and ask first field
            restart_text = "Okay, let's start over from the beginning."
            restart_audio = await tts_service.synthesize(restart_text)

            await self.send({
                "type": "restart",
                "text": restart_text,
                "audio": restart_audio
            })

            await asyncio.sleep(0.5)
            await self._ask_current_field()

    async def _fill_and_advance(self, field_id: str, value: Any) -> None:
        """Fill field and move to next."""
        previous_field = self.session.get_current_field()
        previous_label = previous_field.get("label", field_id) if previous_field else ""

        # Record filled value
        self.session.add_filled_field(field_id, value)

        # Move to next field
        next_field = self.session.advance_field()

        # Get next unfilled field
        next_field = self.session.get_next_unfilled_field()

        await session_manager.save_session(self.session)

        if next_field:
            next_id = next_field.get("id") or next_field.get("name")
            next_label = next_field.get("label", next_id)

            # Deterministic transition (avoid LLM for flow control)
            transition_text = build_next_field_transition(previous_label, next_label)
            transition_audio = await tts_service.synthesize(transition_text)

            self.session.add_to_history("assistant", transition_text)
            await session_manager.save_session(self.session)

            await self.send({
                "type": "fill_field",
                "fieldId": field_id,
                "value": value,
                "nextField": {
                    "id": next_id,
                    "label": next_label
                },
                "text": transition_text,
                "audio": transition_audio
            })
        else:
            # Form complete
            await self.send({
                "type": "fill_field",
                "fieldId": field_id,
                "value": value,
                "nextField": None
            })
            await self._complete_form()

    async def _ask_current_field(self) -> None:
        """Ask for the current field."""
        current_field = self.session.get_next_unfilled_field()

        if not current_field:
            if self.session.is_complete():
                await self._complete_form()
            return

        field_id = current_field.get("id") or current_field.get("name")
        field_label = current_field.get("label", field_id)

        # DYNAMIC question generation using LLM - works with ANY field!
        # The LLM analyzes the field metadata (label, type, pattern, options, etc.)
        # and generates a natural question automatically
        ask_text = await openrouter_client.generate_field_question(current_field)
        ask_audio = await tts_service.synthesize(ask_text)

        self.session.add_to_history("assistant", ask_text)
        await session_manager.save_session(self.session)

        await self.send({
            "type": "ask_field",
            "text": ask_text,
            "audio": ask_audio,
            "fieldId": field_id,
            "fieldLabel": field_label,
            "fieldIndex": self.session.current_field_index,
            "fieldRequired": current_field.get("required", False),
            "progress": self.session.get_progress(),
            "detectedLanguage": self._detected_language  # Send detected language for client hints
        })

    async def _ask_repeat(self) -> None:
        """Ask user to repeat."""
        repeat_text = await openrouter_client.generate_response("repeat")
        repeat_audio = await tts_service.synthesize(repeat_text)

        self.session.add_to_history("assistant", repeat_text)
        await session_manager.save_session(self.session)

        await self.send({
            "type": "repeat",
            "text": repeat_text,
            "audio": repeat_audio
        })

    async def _send_audio_quality_error(self, detail: str = "") -> None:
        """Send error when audio quality is too low or invalid.

        Args:
            detail: Optional specific error detail to include in message
        """
        if detail:
            error_text = f"I couldn't process your audio. {detail}"
        else:
            error_text = (
                "I couldn't hear you clearly. "
                "Please speak louder and closer to your microphone, then try again."
            )

        logger.warning(f"Audio quality error: {detail or 'no detail'}")

        try:
            error_audio = await tts_service.synthesize(error_text)
        except Exception:
            error_audio = ""

        await self.send({
            "type": "audio_quality_error",
            "text": error_text,
            "audio": error_audio
        })

    async def _ask_yes_no(self) -> None:
        """Ask for clear yes/no."""
        text = "Please say yes to confirm or no to change."
        audio = await tts_service.synthesize(text)

        await self.send({
            "type": "clarify",
            "text": text,
            "audio": audio
        })

    async def _complete_form(self) -> None:
        """Handle form completion."""
        self.session.state = "completed"

        form_name = self.session.form_schema.get("name", "the form")
        field_count = len(self.session.filled_fields)

        complete_text = await openrouter_client.generate_response(
            "completion",
            form_name=form_name,
            field_count=field_count
        )
        complete_audio = await tts_service.synthesize(complete_text)

        self.session.add_to_history("assistant", complete_text)
        await session_manager.save_session(self.session)

        await self.send({
            "type": "completion",
            "text": complete_text,
            "audio": complete_audio,
            "filledFields": self.session.filled_fields,
            "progress": self.session.get_progress()
        })

    async def _send_error(self, error: str) -> None:
        """Send error message to client."""
        try:
            error_text = await openrouter_client.generate_response("error", error=error)
        except Exception:
            error_text = "Sorry, something went wrong. Please try again."

        # Try TTS but don't fail if it doesn't work
        try:
            error_audio = await tts_service.synthesize(error_text)
        except Exception:
            error_audio = ""

        await self.send({
            "type": "error",
            "text": error_text,
            "audio": error_audio,
            "error": error
        })


async def websocket_endpoint(websocket: WebSocket) -> None:
    """Main WebSocket endpoint handler."""
    handler = VoiceFormHandler(websocket)

    await websocket.accept()
    logger.info("WebSocket connection accepted")

    try:
        while True:
            data = await websocket.receive_json()
            await handler.handle_message(data)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        if handler.session:
            await session_manager.save_session(handler.session)

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
