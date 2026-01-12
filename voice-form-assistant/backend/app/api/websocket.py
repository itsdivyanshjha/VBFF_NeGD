"""
WebSocket Handler.
Main communication endpoint for voice form filling.
"""

import logging
import json
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import numpy as np

from ..services.whisper_service import whisper_service
from ..services.openrouter_client import openrouter_client
from ..services.tts_service import tts_service
from ..services.audio_processor import audio_processor
from ..services.session_manager import session_manager, ConversationSession
from ..services.validators import validate_field_value
from ..services.question_builder import build_ask_question, build_examples, build_next_field_transition
from ..config import settings

logger = logging.getLogger(__name__)


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

    async def send(self, message: Dict[str, Any]) -> bool:
        """Send message to client. Returns False if client disconnected."""
        try:
            await self.websocket.send_json(message)
            return True
        except Exception:
            return False

    async def handle_message(self, data: Dict[str, Any]) -> None:
        """Route incoming messages to handlers."""
        msg_type = data.get("type")

        handlers = {
            "init": self._handle_init,
            "audio_complete": self._handle_audio_complete,
            "user_confirmation": self._handle_confirmation,
            "skip_field": self._handle_skip,
            "restart": self._handle_restart,
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

        # Ask for first field
        await asyncio.sleep(0.5)  # Brief pause
        await self._ask_current_field()

    async def _handle_audio_complete(self, data: Dict[str, Any]) -> None:
        """Handle complete audio recording from client."""
        session_id = data.get("sessionId")
        audio_data = data.get("audio", "")

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
            # Decode the complete audio
            audio_array = await audio_processor.decode_audio_chunks(
                [audio_data],  # Single complete audio
                format="webm"
            )

            if len(audio_array) == 0:
                await self._send_audio_quality_error()
                return

            # Check duration
            duration = await audio_processor.get_audio_duration(audio_array)
            logger.info(f"Audio duration: {duration:.2f}s")

            if duration < 0.5:
                await self._ask_repeat()
                return

            detected_lang = await whisper_service.detect_language(audio_array)
            logger.info(f"Detected language: {detected_lang}")

            cfg_lang = (settings.WHISPER_LANGUAGE or "auto").strip().lower()
            if cfg_lang in {"auto", "detect", "none", ""}:
                lang_for_transcribe = detected_lang
            else:
                lang_for_transcribe = cfg_lang

            transcription, confidence = await whisper_service.transcribe(audio_array, language=lang_for_transcribe)

            if not transcription or confidence < 0.3:
                await self._ask_repeat()
                return

            logger.info(f"Transcription: '{transcription}' (confidence: {confidence})")

            # Record user input
            self.session.add_to_history("user", transcription)

            # Process based on session state
            if self.session.state == "confirming":
                await self._process_confirmation_response(transcription)
            else:
                await self._process_field_input(transcription)

        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            await self._send_error("Failed to process audio. Please try again.")

        finally:
            self._processing = False

    async def _process_field_input(self, transcription: str) -> None:
        """Process user input for a field."""
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

        # Extract value using LLM
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
        needs_confirmation = extraction.get("needs_confirmation", True)

        if not extracted_value:
            # Could not extract value
            await self._ask_repeat()
            return

        # Validate value
        validation = validate_field_value(current_field, extracted_value)

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
        # Deterministic ask question derived from the field itself (avoid LLM)
        ask_text = build_ask_question(current_field)
        hint = build_examples(current_field)
        if hint:
            ask_text = f"{ask_text} {hint}"
        ask_audio = await tts_service.synthesize(ask_text)

        self.session.add_to_history("assistant", ask_text)
        await session_manager.save_session(self.session)

        await self.send({
            "type": "ask_field",
            "text": ask_text,
            "audio": ask_audio,
            "fieldId": field_id,
            "fieldLabel": field_label,
            "progress": self.session.get_progress()
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

    async def _send_audio_quality_error(self) -> None:
        """Send error when audio quality is too low (no speech detected)."""
        error_text = "I couldn't hear you clearly. Please speak louder and closer to your microphone, then try again."
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
