"""
Application initialization.

This module ensures that configuration and registries are loaded
before the application starts processing requests.
"""
import logging
from pathlib import Path

from .config import get_config
from .field_registry import get_field_registry

logger = logging.getLogger(__name__)


def initialize_application():
    """
    Initialize application components.

    This should be called during application startup to:
    1. Load configuration from YAML files
    2. Initialize field type registry
    3. Validate critical settings
    4. Log initialization status

    Returns:
        True if initialization successful, False otherwise
    """
    try:
        logger.info("=" * 60)
        logger.info("Voice Form Assistant - Initializing (Phase 1)")
        logger.info("=" * 60)

        # 1. Load configuration
        logger.info("Loading configuration...")
        config = get_config()

        environment = config.get('server.debug', False)
        env_name = "development" if environment else "production"
        logger.info(f"Environment: {env_name}")

        # Log provider configuration
        stt_provider = config.get('providers.stt.default', 'assemblyai')
        llm_provider = config.get('providers.llm.default', 'openrouter')
        tts_provider = config.get('providers.tts.default', 'gtts')
        storage_provider = config.get('providers.storage.default', 'redis')

        logger.info(f"Providers: STT={stt_provider}, LLM={llm_provider}, TTS={tts_provider}, Storage={storage_provider}")

        # 2. Initialize field type registry
        logger.info("Loading field type registry...")
        registry = get_field_registry()

        field_count = len(registry.list_all())
        categories = registry.get_categories()
        logger.info(f"Loaded {field_count} field types across {len(categories)} categories")
        logger.info(f"Categories: {', '.join(categories)}")

        # List field types by category
        for category in sorted(categories):
            fields = registry.list_by_category(category)
            logger.info(f"  {category}: {', '.join(fields)}")

        # 3. Validate critical settings
        logger.info("Validating configuration...")

        # Check API keys (warn if missing, don't fail)
        assemblyai_key = config.get('providers.stt.assemblyai.api_key', '')
        if not assemblyai_key:
            logger.warning("⚠️  AssemblyAI API key not configured (ASSEMBLYAI_API_KEY)")

        openrouter_key = config.get('providers.llm.openrouter.api_key', '')
        if not openrouter_key:
            logger.warning("⚠️  OpenRouter API key not configured (OPENROUTER_API_KEY)")

        redis_host = config.get('providers.storage.redis.host', 'localhost')
        redis_port = config.get('providers.storage.redis.port', 6379)
        logger.info(f"Redis: {redis_host}:{redis_port}")

        # 4. Feature flags
        features_enabled = []
        if config.is_feature_enabled('rate_limiting'):
            features_enabled.append('rate_limiting')
        if config.is_feature_enabled('circuit_breaker'):
            features_enabled.append('circuit_breaker')
        if config.is_feature_enabled('retry_logic'):
            features_enabled.append('retry_logic')

        if features_enabled:
            logger.info(f"Features enabled: {', '.join(features_enabled)}")
        else:
            logger.info("Features: all disabled (development mode)")

        logger.info("=" * 60)
        logger.info("✅ Initialization complete - Ready to accept connections")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}", exc_info=True)
        return False


def get_app_info() -> dict:
    """
    Get application information for /info endpoint.

    Returns:
        Dictionary with application configuration info
    """
    config = get_config()
    registry = get_field_registry()

    return {
        "name": "Voice Form Assistant",
        "version": "1.0.0-phase1",
        "phase": "Phase 1 - Configuration & Providers",
        "configuration": {
            "providers": {
                "stt": config.get('providers.stt.default'),
                "llm": config.get('providers.llm.default'),
                "tts": config.get('providers.tts.default'),
                "storage": config.get('providers.storage.default'),
            },
            "field_types": {
                "count": len(registry.list_all()),
                "categories": registry.get_categories(),
                "packs": config.get('field_types.packs', []),
            },
            "features": {
                "rate_limiting": config.is_feature_enabled('rate_limiting'),
                "circuit_breaker": config.is_feature_enabled('circuit_breaker'),
                "retry_logic": config.is_feature_enabled('retry_logic'),
            },
            "languages": {
                "supported": config.get('languages.supported', []),
                "default": config.get('languages.default', 'en'),
            }
        },
        "status": "operational"
    }
