"""
Base interface for Session Storage providers.

This module defines the abstract interface for session storage backends,
enabling plug-and-play swapping of storage systems.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List


class SessionStorage(ABC):
    """
    Abstract base class for Session Storage providers.

    All storage implementations must inherit from this class and implement
    the required methods. This enables swapping storage backends without
    changing core business logic.

    Example implementations:
    - RedisStorage
    - PostgreSQLStorage
    - MongoDBStorage
    - DynamoDBStorage
    - InMemoryStorage (for testing)
    """

    @abstractmethod
    async def create(self, session_id: str, session_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Create a new session.

        Args:
            session_id: Unique session identifier
            session_data: Session data to store
            ttl: Time-to-live in seconds (None = no expiration)

        Returns:
            True if created successfully, False otherwise

        Raises:
            Exception: If creation fails
        """
        pass

    @abstractmethod
    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session data dict if found, None otherwise
        """
        pass

    @abstractmethod
    async def update(self, session_id: str, session_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Update an existing session.

        Args:
            session_id: Session identifier
            session_data: Updated session data
            ttl: Optional new TTL in seconds

        Returns:
            True if updated successfully, False otherwise

        Raises:
            Exception: If update fails
        """
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted successfully, False otherwise
        """
        pass

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """
        Check if a session exists.

        Args:
            session_id: Session identifier

        Returns:
            True if session exists, False otherwise
        """
        pass

    @abstractmethod
    async def get_active_session_count(self) -> int:
        """
        Get count of active sessions.

        Returns:
            Number of active sessions
        """
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """
        Clean up expired sessions (if applicable).

        Returns:
            Number of sessions cleaned up
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if storage backend is healthy and accessible.

        Returns:
            True if storage is healthy, False otherwise
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            Provider name (e.g., 'redis', 'postgresql', 'mongodb')
        """
        pass
