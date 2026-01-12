"""
Session Manager.
Redis-backed session management for conversation state.
"""

import logging
import json
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import redis.asyncio as redis

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class ConversationSession:
    """Represents a form-filling conversation session."""

    session_id: str
    form_schema: Dict[str, Any]
    current_field_index: int = 0
    filled_fields: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    audio_buffer: List[str] = field(default_factory=list)  # Base64 chunks
    state: str = "initialized"  # initialized, filling, confirming, completed
    pending_confirmation: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def get_fields(self) -> List[Dict[str, Any]]:
        """Get list of form fields."""
        return self.form_schema.get("fields", [])

    def get_current_field(self) -> Optional[Dict[str, Any]]:
        """Get the current field to fill."""
        fields = self.get_fields()
        if 0 <= self.current_field_index < len(fields):
            return fields[self.current_field_index]
        return None

    def get_next_unfilled_field(self) -> Optional[Dict[str, Any]]:
        """Get the next unfilled field."""
        fields = self.get_fields()
        for i, fld in enumerate(fields):
            if i >= self.current_field_index:
                field_id = fld.get("id") or fld.get("name")
                if field_id and field_id not in self.filled_fields:
                    self.current_field_index = i
                    return fld
        return None

    def advance_field(self) -> Optional[Dict[str, Any]]:
        """Move to the next field and return it."""
        self.current_field_index += 1
        self.updated_at = datetime.utcnow().isoformat()
        return self.get_next_unfilled_field()

    def add_filled_field(self, field_id: str, value: Any) -> None:
        """Record a filled field value."""
        self.filled_fields[field_id] = value
        self.updated_at = datetime.utcnow().isoformat()
        logger.info(f"Session {self.session_id}: filled {field_id} = {value}")

    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        # Keep history manageable
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
        self.updated_at = datetime.utcnow().isoformat()

    def set_pending_confirmation(
        self,
        field_id: str,
        field_label: str,
        value: Any,
        confidence: float
    ) -> None:
        """Set a value pending user confirmation."""
        self.pending_confirmation = {
            "field_id": field_id,
            "field_label": field_label,
            "value": value,
            "confidence": confidence
        }
        self.state = "confirming"
        self.updated_at = datetime.utcnow().isoformat()

    def clear_pending_confirmation(self) -> Optional[Dict[str, Any]]:
        """Clear and return pending confirmation."""
        pending = self.pending_confirmation
        self.pending_confirmation = None
        self.state = "filling"
        self.updated_at = datetime.utcnow().isoformat()
        return pending

    def is_complete(self) -> bool:
        """Check if all required fields are filled."""
        fields = self.get_fields()
        for fld in fields:
            if fld.get("required"):
                field_id = fld.get("id") or fld.get("name")
                if field_id not in self.filled_fields:
                    return False
        return True

    def get_progress(self) -> Dict[str, Any]:
        """Get form filling progress."""
        fields = self.get_fields()
        total = len(fields)
        filled = len(self.filled_fields)
        return {
            "total_fields": total,
            "filled_fields": filled,
            "remaining_fields": total - filled,
            "percentage": round((filled / total) * 100) if total > 0 else 0
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """Create session from dictionary."""
        return cls(**data)


class SessionManager:
    """Manages conversation sessions with Redis backend."""

    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._prefix = "voice_form_session:"
        self._ttl = settings.SESSION_TTL

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._redis is None:
            self._redis = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD or None,
                decode_responses=True
            )
            # Test connection
            await self._redis.ping()
            logger.info(f"Connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Disconnected from Redis")

    def _key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"{self._prefix}{session_id}"

    async def create_session(self, form_schema: Dict[str, Any]) -> ConversationSession:
        """
        Create a new conversation session.

        Args:
            form_schema: Schema describing the form fields

        Returns:
            New ConversationSession instance
        """
        await self.connect()

        session_id = str(uuid.uuid4())

        session = ConversationSession(
            session_id=session_id,
            form_schema=form_schema,
            state="initialized"
        )

        # Save to Redis
        await self._redis.setex(
            self._key(session_id),
            self._ttl,
            json.dumps(session.to_dict())
        )

        logger.info(f"Created session: {session_id}")

        return session

    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ConversationSession or None if not found
        """
        await self.connect()

        data = await self._redis.get(self._key(session_id))

        if data is None:
            logger.warning(f"Session not found: {session_id}")
            return None

        try:
            session_dict = json.loads(data)
            return ConversationSession.from_dict(session_dict)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error deserializing session {session_id}: {e}")
            return None

    async def save_session(self, session: ConversationSession) -> bool:
        """
        Save session to Redis.

        Args:
            session: Session to save

        Returns:
            True if successful
        """
        await self.connect()

        try:
            session.updated_at = datetime.utcnow().isoformat()

            await self._redis.setex(
                self._key(session.session_id),
                self._ttl,
                json.dumps(session.to_dict())
            )

            logger.debug(f"Saved session: {session.session_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {e}")
            return False

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted
        """
        await self.connect()

        result = await self._redis.delete(self._key(session_id))

        logger.info(f"Deleted session: {session_id}")

        return result > 0

    async def extend_session(self, session_id: str) -> bool:
        """
        Extend session TTL.

        Args:
            session_id: Session identifier

        Returns:
            True if extended
        """
        await self.connect()

        result = await self._redis.expire(self._key(session_id), self._ttl)

        return result

    async def get_active_sessions_count(self) -> int:
        """Get count of active sessions."""
        await self.connect()

        keys = await self._redis.keys(f"{self._prefix}*")
        return len(keys)


# Global instance
session_manager = SessionManager()
