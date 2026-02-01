"""
Field Type Registry.

This module provides a centralized registry for field type definitions,
loading metadata from YAML configuration files. This eliminates hardcoded
field types and enables easy addition of new field types without code changes.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field as dataclass_field
import yaml

logger = logging.getLogger(__name__)


@dataclass
class FieldTypeDefinition:
    """
    Definition of a field type.

    Contains all metadata needed to handle a specific field type including:
    - Detection hints (keywords, patterns)
    - Validation rules
    - Speech processing configuration
    - Prompt templates
    - UI hints
    """

    name: str
    """Unique identifier for this field type (e.g., 'aadhaar', 'pan', 'email')"""

    display_name: str
    """Human-readable name (e.g., 'Aadhaar Number')"""

    description: str
    """Description of what this field type represents"""

    category: str
    """Category (e.g., 'identity', 'contact', 'address', 'date')"""

    validation: Dict[str, Any]
    """Validation rules (pattern, length, type, custom_validator)"""

    detection_hints: Dict[str, List[str]]
    """Hints for detecting this field type (keywords, patterns)"""

    speech_processing: Dict[str, Any]
    """Speech processing configuration (word_boost, entity_type)"""

    prompts: Dict[str, str]
    """Prompt templates (question, confirmation, validation_error)"""

    ui_hints: Dict[str, Any]
    """UI hints (input_mask, placeholder, icon)"""

    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)
    """Additional metadata"""

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> 'FieldTypeDefinition':
        """
        Create FieldTypeDefinition from dictionary.

        Args:
            name: Field type name
            data: Field type data from YAML

        Returns:
            FieldTypeDefinition instance
        """
        return cls(
            name=name,
            display_name=data.get('display_name', name.title()),
            description=data.get('description', ''),
            category=data.get('category', 'general'),
            validation=data.get('validation', {}),
            detection_hints=data.get('detection_hints', {}),
            speech_processing=data.get('speech_processing', {}),
            prompts=data.get('prompts', {}),
            ui_hints=data.get('ui_hints', {}),
            metadata=data.get('metadata', {})
        )


class FieldTypeRegistry:
    """
    Central registry for field type definitions.

    Loads field types from YAML files and provides methods to:
    - Detect field type based on hints
    - Get validation rules
    - Get prompt templates
    - Get speech processing configuration
    """

    def __init__(self, config_dir: Path, packs: Optional[List[str]] = None):
        """
        Initialize field type registry.

        Args:
            config_dir: Directory containing field type YAML files
            packs: List of field type packs to load (e.g., ['india', 'usa'])
                  If None, loads all YAML files in config_dir
        """
        self.config_dir = Path(config_dir)
        self.packs = packs or []
        self._field_types: Dict[str, FieldTypeDefinition] = {}
        self._validators: Dict[str, Callable] = {}
        self._load_field_types()

    def _load_field_types(self):
        """Load field type definitions from YAML files."""
        if not self.config_dir.exists():
            logger.warning(f"Field types directory not found: {self.config_dir}")
            self._load_default_field_types()
            return

        # Load specified packs
        if self.packs:
            for pack in self.packs:
                pack_file = self.config_dir / f"{pack}.yaml"
                if pack_file.exists():
                    self._load_file(pack_file, pack_name=pack)
                else:
                    logger.warning(f"Field type pack not found: {pack_file}")
        else:
            # Load all YAML files
            for yaml_file in self.config_dir.glob("*.yaml"):
                self._load_file(yaml_file)

        # Load custom field types from plugins directory
        plugins_dir = self.config_dir / "plugins"
        if plugins_dir.exists():
            for yaml_file in plugins_dir.glob("*.yaml"):
                self._load_file(yaml_file, is_plugin=True)

        logger.info(f"Loaded {len(self._field_types)} field types")

    def _load_file(self, file_path: Path, pack_name: Optional[str] = None, is_plugin: bool = False):
        """
        Load field types from a YAML file.

        Args:
            file_path: Path to YAML file
            pack_name: Name of the pack (for logging)
            is_plugin: Whether this is a plugin field type
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            field_types_data = data.get('field_types', {})
            count = 0

            for field_name, field_data in field_types_data.items():
                try:
                    field_def = FieldTypeDefinition.from_dict(field_name, field_data)
                    self._field_types[field_name] = field_def
                    count += 1

                    if is_plugin:
                        logger.info(f"Loaded plugin field type: {field_name}")

                except Exception as e:
                    logger.error(f"Failed to load field type '{field_name}' from {file_path}: {e}")

            pack_info = f" from pack '{pack_name}'" if pack_name else ""
            logger.info(f"Loaded {count} field types from {file_path.name}{pack_info}")

        except Exception as e:
            logger.error(f"Failed to load field types from {file_path}: {e}")

    def _load_default_field_types(self):
        """Load minimal default field types as fallback."""
        logger.warning("Loading minimal default field types (config directory not found)")

        # Basic fallback field types
        defaults = {
            'text': {'display_name': 'Text', 'category': 'general'},
            'email': {'display_name': 'Email', 'category': 'contact'},
            'phone': {'display_name': 'Phone', 'category': 'contact'},
            'date': {'display_name': 'Date', 'category': 'date'},
            'number': {'display_name': 'Number', 'category': 'general'},
        }

        for name, data in defaults.items():
            self._field_types[name] = FieldTypeDefinition.from_dict(name, data)

    def get(self, field_type: str) -> Optional[FieldTypeDefinition]:
        """
        Get field type definition.

        Args:
            field_type: Field type name

        Returns:
            FieldTypeDefinition if found, None otherwise
        """
        return self._field_types.get(field_type)

    def exists(self, field_type: str) -> bool:
        """
        Check if field type exists.

        Args:
            field_type: Field type name

        Returns:
            True if field type exists, False otherwise
        """
        return field_type in self._field_types

    def detect_field_type(self, field_info: Dict[str, Any], min_score: int = 5) -> Optional[str]:
        """
        Auto-detect field type based on hints.

        Uses scoring system:
        - Keyword match: +10 points
        - Pattern match: +20 points

        Args:
            field_info: Field metadata (name, label, pattern, type, etc.)
            min_score: Minimum score required for detection

        Returns:
            Field type name if detected, None otherwise
        """
        name_lower = field_info.get('name', '').lower()
        label_lower = field_info.get('label', '').lower()
        pattern = field_info.get('pattern', '')
        html_type = field_info.get('type', '')

        # Combined text for keyword matching
        combined_text = f"{name_lower} {label_lower}"

        # Score each field type
        scores = {}

        for field_type, definition in self._field_types.items():
            score = 0

            # Check keyword matches
            keywords = definition.detection_hints.get('keywords', [])
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    score += 10

            # Check pattern matches
            patterns = definition.detection_hints.get('patterns', [])
            for hint_pattern in patterns:
                if hint_pattern in pattern:
                    score += 20

            # Check HTML type hints
            html_type_hints = definition.detection_hints.get('html_types', [])
            if html_type in html_type_hints:
                score += 15

            if score > 0:
                scores[field_type] = score

        # Return highest scoring field type if above minimum
        if scores:
            best_match = max(scores, key=scores.get)
            if scores[best_match] >= min_score:
                logger.debug(f"Detected field type '{best_match}' with score {scores[best_match]}")
                return best_match

        return None

    def get_validation_rules(self, field_type: str) -> Dict[str, Any]:
        """
        Get validation rules for a field type.

        Args:
            field_type: Field type name

        Returns:
            Validation rules dictionary
        """
        definition = self.get(field_type)
        return definition.validation if definition else {}

    def get_prompt(self, field_type: str, prompt_type: str, **kwargs) -> str:
        """
        Get prompt template for a field type.

        Args:
            field_type: Field type name
            prompt_type: Type of prompt ('question', 'confirmation', 'validation_error')
            **kwargs: Variables to format into template

        Returns:
            Formatted prompt string
        """
        definition = self.get(field_type)
        if not definition:
            return ""

        template = definition.prompts.get(prompt_type, "")
        if template and kwargs:
            try:
                return template.format(**kwargs)
            except KeyError as e:
                logger.warning(f"Missing variable in prompt template: {e}")
                return template
        return template

    def get_speech_config(self, field_type: str) -> Dict[str, Any]:
        """
        Get speech processing configuration for a field type.

        Args:
            field_type: Field type name

        Returns:
            Speech processing config dictionary
        """
        definition = self.get(field_type)
        return definition.speech_processing if definition else {}

    def get_ui_hints(self, field_type: str) -> Dict[str, Any]:
        """
        Get UI hints for a field type.

        Args:
            field_type: Field type name

        Returns:
            UI hints dictionary
        """
        definition = self.get(field_type)
        return definition.ui_hints if definition else {}

    def list_all(self) -> List[str]:
        """
        List all registered field type names.

        Returns:
            List of field type names
        """
        return list(self._field_types.keys())

    def list_by_category(self, category: str) -> List[str]:
        """
        List field types by category.

        Args:
            category: Category name

        Returns:
            List of field type names in the category
        """
        return [
            name for name, definition in self._field_types.items()
            if definition.category == category
        ]

    def get_categories(self) -> List[str]:
        """
        Get all unique categories.

        Returns:
            List of category names
        """
        return list(set(
            definition.category
            for definition in self._field_types.values()
        ))

    def register_validator(self, field_type: str, validator: Callable):
        """
        Register a custom validator function for a field type.

        Args:
            field_type: Field type name
            validator: Validator function (takes value, returns Dict[str, Any])
        """
        self._validators[field_type] = validator
        logger.info(f"Registered custom validator for field type: {field_type}")

    def get_validator(self, field_type: str) -> Optional[Callable]:
        """
        Get registered validator for a field type.

        Args:
            field_type: Field type name

        Returns:
            Validator function if registered, None otherwise
        """
        return self._validators.get(field_type)


# Global registry instance
_global_registry: Optional[FieldTypeRegistry] = None


def get_field_registry() -> FieldTypeRegistry:
    """
    Get global field type registry instance.

    Returns:
        Global FieldTypeRegistry instance

    Example:
        from app.core.field_registry import get_field_registry

        registry = get_field_registry()
        field_type = registry.detect_field_type({'name': 'aadhaar_number'})
    """
    global _global_registry
    if _global_registry is None:
        # Import config to get field types directory
        from app.core.config import get_config
        config = get_config()

        field_types_dir = config.get_field_types_dir()
        packs = config.get('field_types.packs', ['india'])

        _global_registry = FieldTypeRegistry(field_types_dir, packs)

    return _global_registry


def reload_field_registry():
    """Reload global field type registry."""
    global _global_registry
    if _global_registry:
        from app.core.config import get_config
        config = get_config()

        field_types_dir = config.get_field_types_dir()
        packs = config.get('field_types.packs', ['india'])

        _global_registry = FieldTypeRegistry(field_types_dir, packs)
