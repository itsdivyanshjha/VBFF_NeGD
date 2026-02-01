"""
Configuration management system.

This module provides hierarchical configuration loading with:
- YAML-based configuration files
- Environment variable substitution
- Environment-specific overrides (dev, staging, production)
- Multi-tenant support (optional)
- Type-safe access to configuration values
"""
import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Hierarchical configuration loader.

    Loads configuration in order of precedence:
    1. Default configuration (config/default.yaml)
    2. Environment-specific config (config/environments/{env}.yaml)
    3. Tenant-specific config (config/tenants/{tenant}.yaml) - if multi-tenant
    4. Environment variable substitution

    Example:
        config = ConfigLoader()
        api_key = config.get('providers.stt.assemblyai.api_key')
        timeout = config.get('providers.stt.assemblyai.timeout', default=60)
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize configuration loader.

        Args:
            base_dir: Base directory containing config/ folder
                     If None, uses backend/config/ by default
        """
        if base_dir is None:
            # Default to backend/config directory
            backend_dir = Path(__file__).parent.parent.parent
            base_dir = backend_dir / "config"

        self.base_dir = Path(base_dir)
        self._config: Dict[str, Any] = {}
        self._load_configuration()

    def _load_configuration(self):
        """Load configuration from files in order of precedence."""
        logger.info(f"Loading configuration from {self.base_dir}")

        # 1. Load default config
        default_config_path = self.base_dir / "default.yaml"
        if default_config_path.exists():
            self._config = self._load_yaml(default_config_path)
            logger.info("Loaded default configuration")
        else:
            logger.warning(f"Default config not found at {default_config_path}")
            self._config = {}

        # 2. Load environment-specific config
        env = os.getenv('ENVIRONMENT', os.getenv('ENV', 'development'))
        env_config_path = self.base_dir / "environments" / f"{env}.yaml"
        if env_config_path.exists():
            env_config = self._load_yaml(env_config_path)
            self._config = self._deep_merge(self._config, env_config)
            logger.info(f"Loaded {env} environment configuration")
        else:
            logger.debug(f"No environment config found for {env}")

        # 3. Load tenant-specific config (if multi-tenant enabled)
        tenant_id = os.getenv('TENANT_ID')
        if tenant_id:
            tenant_config_path = self.base_dir / "tenants" / f"{tenant_id}.yaml"
            if tenant_config_path.exists():
                tenant_config = self._load_yaml(tenant_config_path)
                self._config = self._deep_merge(self._config, tenant_config)
                logger.info(f"Loaded tenant configuration for {tenant_id}")

        # 4. Apply environment variable substitution
        self._config = self._substitute_env_vars(self._config)
        logger.info("Configuration loaded successfully")

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """
        Load YAML file safely.

        Args:
            path: Path to YAML file

        Returns:
            Parsed YAML as dictionary

        Raises:
            Exception: If YAML parsing fails
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load YAML from {path}: {e}")
            raise

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Dictionary with override values

        Returns:
            Merged dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.

        Supports ${VAR} and ${VAR:default} syntax.

        Args:
            config: Configuration value (can be dict, list, string, etc.)

        Returns:
            Configuration with substituted values
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._replace_env_var(config)
        return config

    def _replace_env_var(self, value: str) -> str:
        """
        Replace ${VAR:default} patterns with environment variable values.

        Examples:
            ${API_KEY} -> value of API_KEY env var
            ${API_KEY:default_value} -> value of API_KEY or 'default_value'
            ${PORT:8000} -> value of PORT or '8000'

        Args:
            value: String potentially containing env var references

        Returns:
            String with environment variables replaced
        """
        pattern = r'\$\{([^:}]+)(?::([^}]*))?\}'

        def replacer(match):
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default)

        return re.sub(pattern, replacer, value)

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path.

        Args:
            path: Dot-separated path (e.g., 'providers.stt.assemblyai.api_key')
            default: Default value if path not found

        Returns:
            Configuration value or default

        Example:
            api_key = config.get('providers.stt.assemblyai.api_key')
            timeout = config.get('providers.stt.assemblyai.timeout', default=60)
        """
        keys = path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_provider_config(self, provider_type: str, provider_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.

        Args:
            provider_type: Type of provider ('stt', 'llm', 'tts', 'storage')
            provider_name: Name of provider ('assemblyai', 'openrouter', etc.)

        Returns:
            Provider configuration dictionary

        Example:
            assemblyai_config = config.get_provider_config('stt', 'assemblyai')
            api_key = assemblyai_config.get('api_key')
        """
        return self.get(f'providers.{provider_type}.{provider_name}', {})

    def get_field_types_dir(self) -> Path:
        """
        Get path to field types configuration directory.

        Returns:
            Path to field_types directory
        """
        field_types_dir = self.get('field_types.config_dir', './config/field_types')
        if not Path(field_types_dir).is_absolute():
            field_types_dir = self.base_dir / field_types_dir
        return Path(field_types_dir)

    def get_prompts_dir(self) -> Path:
        """
        Get path to prompts templates directory.

        Returns:
            Path to prompts directory
        """
        prompts_dir = self.get('prompts.templates_dir', './config/prompts')
        if not Path(prompts_dir).is_absolute():
            prompts_dir = self.base_dir / prompts_dir
        return Path(prompts_dir)

    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature is enabled.

        Args:
            feature_name: Name of feature (e.g., 'rate_limiting', 'circuit_breaker')

        Returns:
            True if enabled, False otherwise

        Example:
            if config.is_feature_enabled('rate_limiting'):
                # Apply rate limiting
        """
        return self.get(f'features.{feature_name}.enabled', False)

    def reload(self):
        """Reload configuration from files."""
        logger.info("Reloading configuration")
        self._load_configuration()

    def to_dict(self) -> Dict[str, Any]:
        """
        Get full configuration as dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()


# Global configuration instance
# This is initialized once and used throughout the application
_global_config: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """
    Get global configuration instance.

    Returns:
        Global ConfigLoader instance

    Example:
        from app.core.config import get_config

        config = get_config()
        api_key = config.get('providers.stt.assemblyai.api_key')
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader()
    return _global_config


def reload_config():
    """Reload global configuration."""
    global _global_config
    if _global_config:
        _global_config.reload()
