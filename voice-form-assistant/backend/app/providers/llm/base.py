"""
Base interface for Language Model providers.

This module defines the abstract interface for LLM providers,
enabling plug-and-play swapping of language models.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class MessageRole(str, Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class LLMMessage:
    """A message in the conversation."""

    role: MessageRole
    """Role of the message sender"""

    content: str
    """Message content"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Optional metadata"""


@dataclass
class LLMResult:
    """Result from LLM completion."""

    text: str
    """Generated text"""

    finish_reason: str
    """Reason for completion (stop, length, error, etc.)"""

    tokens_used: int
    """Number of tokens consumed"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Provider-specific metadata"""


class LLMProvider(ABC):
    """
    Abstract base class for Language Model providers.

    All LLM implementations must inherit from this class and implement
    the required methods. This enables swapping LLM providers without
    changing core business logic.

    Example implementations:
    - OpenRouterProvider
    - OpenAIProvider
    - AnthropicProvider
    - LocalLLMProvider (Ollama, llama.cpp)
    """

    @abstractmethod
    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.3,
        max_tokens: int = 500,
        **kwargs
    ) -> LLMResult:
        """
        Generate completion from messages.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            LLMResult containing generated text and metadata

        Raises:
            Exception: If completion fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if provider is healthy and accessible.

        Returns:
            True if provider is healthy, False otherwise
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            Provider name (e.g., 'openrouter', 'openai', 'anthropic')
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the model being used.

        Returns:
            Model name (e.g., 'gpt-4', 'claude-3-5-sonnet')
        """
        pass
