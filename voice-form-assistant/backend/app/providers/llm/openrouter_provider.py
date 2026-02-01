"""
OpenRouter Language Model Provider.

This module wraps the existing OpenRouterClient to implement the LLMProvider interface,
enabling plug-and-play provider swapping while maintaining backward compatibility.
"""
import logging
from typing import Optional, Dict, Any, List

from .base import LLMProvider, LLMMessage, LLMResult, MessageRole
from ...services.openrouter_client import OpenRouterClient
from ...core.config import get_config

logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter LLM provider implementation.

    Wraps the existing OpenRouterClient to provide a standardized interface
    while preserving all existing functionality for field value extraction
    and question generation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenRouter provider.

        Args:
            config: Optional provider configuration.
                   If None, loads from global config.
        """
        if config is None:
            app_config = get_config()
            config = app_config.get_provider_config('llm', 'openrouter')

        self.config = config
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', 'meta-llama/llama-3.1-8b-instruct')
        self.base_url = config.get('base_url', 'https://openrouter.ai/api/v1')
        self.timeout = config.get('timeout', 30)
        self.temperature = config.get('temperature', 0.3)

        # Initialize the underlying OpenRouter client
        self._client = OpenRouterClient()

        logger.info(f"OpenRouter LLM provider initialized (model: {self.model})")

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
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResult containing generated text and metadata

        Raises:
            Exception: If completion fails
        """
        try:
            # Convert LLMMessage objects to dict format expected by client
            messages_dict = [
                {
                    "role": msg.role.value,
                    "content": msg.content
                }
                for msg in messages
            ]

            # Call the underlying client
            response_text = await self._client._make_request(
                messages=messages_dict,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Create result
            result = LLMResult(
                text=response_text,
                finish_reason="stop",  # OpenRouter doesn't provide finish reason in current impl
                tokens_used=len(response_text.split()),  # Rough estimate
                metadata={
                    'model': self.model,
                    'temperature': temperature,
                    'max_tokens': max_tokens
                }
            )

            logger.debug(f"LLM completion: {len(response_text)} chars, ~{result.tokens_used} tokens")

            return result

        except Exception as e:
            logger.error(f"OpenRouter completion failed: {e}")
            raise

    async def extract_field_value(
        self,
        field_info: Dict[str, Any],
        user_input: str,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Extract and format field value from user input.

        This is a domain-specific method that wraps the OpenRouter client's
        field value extraction functionality.

        Args:
            field_info: Field metadata (type, label, pattern, etc.)
            user_input: Raw transcribed text from user
            language: Language code

        Returns:
            Dict containing:
                - value: Extracted value
                - confidence: Confidence score
                - formatted: Formatted value

        Raises:
            Exception: If extraction fails
        """
        try:
            return await self._client.extract_field_value(
                field_info=field_info,
                user_input=user_input,
                language=language
            )

        except Exception as e:
            logger.error(f"Field value extraction failed: {e}")
            raise

    async def generate_question(self, field: Dict[str, Any]) -> str:
        """
        Generate a natural question for a form field.

        This is a domain-specific method that wraps the OpenRouter client's
        question generation functionality.

        Args:
            field: Field metadata (type, label, name, etc.)

        Returns:
            Generated question text

        Raises:
            Exception: If generation fails
        """
        try:
            return await self._client.generate_field_question(field)

        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if OpenRouter service is healthy.

        Returns:
            True if service is accessible, False otherwise
        """
        try:
            # Simple check: verify we have an API key and model
            if not self.api_key:
                logger.warning("OpenRouter API key not configured")
                return False

            if not self.model:
                logger.warning("OpenRouter model not configured")
                return False

            return True

        except Exception as e:
            logger.error(f"OpenRouter health check failed: {e}")
            return False

    def get_provider_name(self) -> str:
        """
        Get provider name.

        Returns:
            'openrouter'
        """
        return "openrouter"

    def get_model_name(self) -> str:
        """
        Get the model being used.

        Returns:
            Model name (e.g., 'meta-llama/llama-3.1-8b-instruct')
        """
        return self.model
