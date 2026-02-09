"""
Redis Session Storage Provider.

This module wraps the existing SessionManager to implement the SessionStorage interface,
enabling plug-and-play storage backend swapping while maintaining backward compatibility.
"""
import logging
import json
from typing import Optional, Dict, Any

from .base import SessionStorage
from ...services.session_manager import SessionManager, ConversationSession
from ...config import settings

logger = logging.getLogger(__name__)


class RedisStorage(SessionStorage):
    """
    Redis session storage provider implementation.

    Wraps the existing SessionManager to provide a standardized interface
    while preserving all existing functionality.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Redis storage provider.

        Args:
            config: Optional provider configuration.
                   If None, loads from global config.
        """
        if config is None:
            # Use simple settings
            config = {
                'host': settings.REDIS_HOST,
                'port': settings.REDIS_PORT,
                'db': settings.REDIS_DB,
                'password': settings.REDIS_PASSWORD,
                'ttl': settings.SESSION_TTL,
                'prefix': 'voice_assistant:'
            }

        self.config = config
        self.host = config.get('host', settings.REDIS_HOST)
        self.port = config.get('port', settings.REDIS_PORT)
        self.db = config.get('db', settings.REDIS_DB)
        self.password = config.get('password', settings.REDIS_PASSWORD)
        self.ttl = config.get('ttl', settings.SESSION_TTL)
        self.prefix = config.get('prefix', 'voice_assistant:')

        # Initialize the underlying session manager
        self._manager = SessionManager()

        logger.info(f"Redis storage provider initialized (host: {self.host}:{self.port})")

    async def create(self, session_id: str, session_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Create a new session.

        Args:
            session_id: Unique session identifier
            session_data: Session data to store
            ttl: Time-to-live in seconds

        Returns:
            True if created successfully

        Raises:
            Exception: If creation fails
        """
        try:
            # Convert dict to ConversationSession
            session = ConversationSession.from_dict(session_data)
            session.session_id = session_id

            # Save using the underlying manager
            success = await self._manager.save_session(session)

            if success:
                logger.debug(f"Created session: {session_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            raise

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session data dict if found, None otherwise
        """
        try:
            session = await self._manager.get_session(session_id)

            if session:
                return session.to_dict()

            return None

        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None

    async def update(self, session_id: str, session_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Update an existing session.

        Args:
            session_id: Session identifier
            session_data: Updated session data
            ttl: Optional new TTL

        Returns:
            True if updated successfully

        Raises:
            Exception: If update fails
        """
        try:
            # Convert dict to ConversationSession
            session = ConversationSession.from_dict(session_data)
            session.session_id = session_id

            # Save using the underlying manager
            success = await self._manager.save_session(session)

            if success:
                logger.debug(f"Updated session: {session_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            raise

    async def delete(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted successfully
        """
        try:
            success = await self._manager.delete_session(session_id)

            if success:
                logger.debug(f"Deleted session: {session_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    async def exists(self, session_id: str) -> bool:
        """
        Check if a session exists.

        Args:
            session_id: Session identifier

        Returns:
            True if session exists
        """
        try:
            session = await self._manager.get_session(session_id)
            return session is not None

        except Exception as e:
            logger.error(f"Failed to check session existence {session_id}: {e}")
            return False

    async def get_active_session_count(self) -> int:
        """
        Get count of active sessions.

        Returns:
            Number of active sessions
        """
        try:
            return await self._manager.get_active_sessions_count()

        except Exception as e:
            logger.error(f"Failed to get active session count: {e}")
            return 0

    async def cleanup_expired(self) -> int:
        """
        Clean up expired sessions.

        Note: Redis automatically expires keys based on TTL,
        so this is a no-op for Redis storage.

        Returns:
            Number of sessions cleaned up (always 0 for Redis)
        """
        # Redis handles expiration automatically
        return 0

    async def health_check(self) -> bool:
        """
        Check if Redis is healthy and accessible.

        Returns:
            True if Redis is healthy
        """
        try:
            await self._manager.connect()

            # Try a simple ping operation
            if self._manager._redis:
                await self._manager._redis.ping()
                return True

            return False

        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def get_provider_name(self) -> str:
        """
        Get provider name.

        Returns:
            'redis'
        """
        return "redis"
