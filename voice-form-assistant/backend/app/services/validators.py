"""
Field Validators.
Validation functions for Indian government form fields.
"""

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def validate_aadhaar(value: str) -> Dict[str, Any]:
    """
    Validate Aadhaar number.

    Args:
        value: Aadhaar number string

    Returns:
        Dict with 'valid' bool and optional 'error' message
    """
    if not value:
        return {"valid": False, "error": "Aadhaar number is required"}

    # Remove spaces and dashes
    cleaned = re.sub(r"[\s\-]", "", value)

    # Check if 12 digits
    if not re.match(r"^\d{12}$", cleaned):
        return {
            "valid": False,
            "error": "Aadhaar must be exactly 12 digits"
        }

    # Check if starts with valid digit (not 0 or 1)
    if cleaned[0] in "01":
        return {
            "valid": False,
            "error": "Aadhaar cannot start with 0 or 1"
        }

    # Verhoeff algorithm for checksum (simplified check)
    # In production, implement full Verhoeff validation
    return {"valid": True, "formatted": cleaned}


def validate_pan(value: str) -> Dict[str, Any]:
    """
    Validate PAN (Permanent Account Number).

    Args:
        value: PAN string

    Returns:
        Dict with 'valid' bool and optional 'error' message
    """
    if not value:
        return {"valid": False, "error": "PAN is required"}

    # Remove spaces, convert to uppercase
    cleaned = re.sub(r"\s", "", value).upper()

    # PAN format: AAAAA0000A (5 letters, 4 digits, 1 letter)
    if not re.match(r"^[A-Z]{5}\d{4}[A-Z]$", cleaned):
        return {
            "valid": False,
            "error": "PAN must be in format: 5 letters, 4 digits, 1 letter (e.g., ABCDE1234F)"
        }

    # Fourth character indicates entity type
    valid_fourth_chars = "ABCFGHLJPT"  # Valid entity type codes
    if cleaned[3] not in valid_fourth_chars:
        return {
            "valid": False,
            "error": f"Invalid PAN entity type code"
        }

    return {"valid": True, "formatted": cleaned}


def validate_mobile(value: str) -> Dict[str, Any]:
    """
    Validate Indian mobile number.

    Args:
        value: Mobile number string

    Returns:
        Dict with 'valid' bool and optional 'error' message
    """
    if not value:
        return {"valid": False, "error": "Mobile number is required"}

    # Remove spaces, dashes, and country code
    cleaned = re.sub(r"[\s\-\+]", "", value)

    # Remove country code if present
    if cleaned.startswith("91") and len(cleaned) == 12:
        cleaned = cleaned[2:]

    # Check if 10 digits starting with 6-9
    if not re.match(r"^[6-9]\d{9}$", cleaned):
        return {
            "valid": False,
            "error": "Mobile number must be 10 digits starting with 6, 7, 8, or 9"
        }

    return {"valid": True, "formatted": cleaned}


def validate_email(value: str) -> Dict[str, Any]:
    """
    Validate email address.

    Args:
        value: Email string

    Returns:
        Dict with 'valid' bool and optional 'error' message
    """
    if not value:
        # Email might be optional
        return {"valid": True, "formatted": ""}

    # Basic email regex
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    cleaned = value.strip().lower()

    if not re.match(email_pattern, cleaned):
        return {
            "valid": False,
            "error": "Please provide a valid email address"
        }

    return {"valid": True, "formatted": cleaned}


def validate_pincode(value: str) -> Dict[str, Any]:
    """
    Validate Indian PIN code.

    Args:
        value: PIN code string

    Returns:
        Dict with 'valid' bool and optional 'error' message
    """
    if not value:
        return {"valid": False, "error": "PIN code is required"}

    # Remove spaces
    cleaned = re.sub(r"\s", "", value)

    # Check if 6 digits
    if not re.match(r"^\d{6}$", cleaned):
        return {
            "valid": False,
            "error": "PIN code must be exactly 6 digits"
        }

    # First digit cannot be 0
    if cleaned[0] == "0":
        return {
            "valid": False,
            "error": "Invalid PIN code format"
        }

    return {"valid": True, "formatted": cleaned}


def validate_name(value: str) -> Dict[str, Any]:
    """
    Validate name field.

    Args:
        value: Name string

    Returns:
        Dict with 'valid' bool and optional 'error' message
    """
    if not value:
        return {"valid": False, "error": "Name is required"}

    # Remove extra spaces
    cleaned = " ".join(value.split())

    # Check minimum length
    if len(cleaned) < 2:
        return {
            "valid": False,
            "error": "Name must be at least 2 characters"
        }

    # Check for valid characters (letters, spaces, common punctuation)
    if not re.match(r"^[a-zA-Z\s\.\-\']+$", cleaned):
        return {
            "valid": False,
            "error": "Name should contain only letters and spaces"
        }

    # Title case formatting
    formatted = cleaned.title()

    return {"valid": True, "formatted": formatted}


def validate_date(value: str) -> Dict[str, Any]:
    """
    Validate date field (YYYY-MM-DD format).

    Args:
        value: Date string

    Returns:
        Dict with 'valid' bool and optional 'error' message
    """
    if not value:
        return {"valid": False, "error": "Date is required"}

    # Try various date formats
    import datetime

    formats = [
        "%Y-%m-%d",  # 2000-01-15
        "%d-%m-%Y",  # 15-01-2000
        "%d/%m/%Y",  # 15/01/2000
        "%Y/%m/%d",  # 2000/01/15
    ]

    parsed_date = None
    for fmt in formats:
        try:
            parsed_date = datetime.datetime.strptime(value.strip(), fmt)
            break
        except ValueError:
            continue

    if not parsed_date:
        return {
            "valid": False,
            "error": "Please provide date in YYYY-MM-DD format"
        }

    # Format to standard YYYY-MM-DD
    formatted = parsed_date.strftime("%Y-%m-%d")

    return {"valid": True, "formatted": formatted}


def validate_address(value: str) -> Dict[str, Any]:
    """
    Validate address field.

    Args:
        value: Address string

    Returns:
        Dict with 'valid' bool and optional 'error' message
    """
    if not value:
        return {"valid": False, "error": "Address is required"}

    # Remove extra whitespace
    cleaned = " ".join(value.split())

    # Check minimum length
    if len(cleaned) < 10:
        return {
            "valid": False,
            "error": "Please provide a complete address"
        }

    return {"valid": True, "formatted": cleaned}


def validate_select(value: str, options: list) -> Dict[str, Any]:
    """
    Validate select/dropdown field.

    Args:
        value: Selected value
        options: List of valid options

    Returns:
        Dict with 'valid' bool and optional 'error' message
    """
    if not value:
        return {"valid": False, "error": "Please select an option"}

    # Check if value matches any option (case-insensitive)
    value_lower = value.lower().strip()

    for option in options:
        if isinstance(option, dict):
            opt_value = option.get("value", "").lower()
            opt_label = option.get("label", "").lower()
        else:
            opt_value = str(option).lower()
            opt_label = opt_value

        if value_lower == opt_value or value_lower == opt_label:
            return {"valid": True, "formatted": option.get("value", option) if isinstance(option, dict) else option}

    return {
        "valid": False,
        "error": f"Please select from available options: {', '.join(str(o) for o in options)}"
    }


def validate_field_value(field_info: Dict[str, Any], value: str) -> Dict[str, Any]:
    """
    Validate field value based on field type.

    Args:
        field_info: Field metadata including type and validation rules
        value: Value to validate

    Returns:
        Dict with 'valid' bool, optional 'error' message, and 'formatted' value
    """
    field_type = field_info.get("field_type", "").lower()
    html_type = field_info.get("type", "text").lower()

    logger.debug(f"Validating field type '{field_type}' with value '{value}'")

    # Route to appropriate validator
    validators = {
        "aadhaar": validate_aadhaar,
        "pan": validate_pan,
        "mobile": validate_mobile,
        "phone": validate_mobile,
        "tel": validate_mobile,
        "email": validate_email,
        "pincode": validate_pincode,
        "pin": validate_pincode,
        "name": validate_name,
        "fullname": validate_name,
        "date": validate_date,
        "dob": validate_date,
        "address": validate_address,
    }

    validator = validators.get(field_type)

    if validator:
        return validator(value)

    # Handle select/radio types
    if html_type in ["select", "radio"] or field_info.get("options"):
        options = field_info.get("options", [])
        return validate_select(value, options)

    # Default: basic non-empty validation for required fields
    if field_info.get("required") and not value:
        return {
            "valid": False,
            "error": f"{field_info.get('label', 'This field')} is required"
        }

    # Check pattern if provided
    pattern = field_info.get("pattern")
    if pattern and value:
        if not re.match(pattern, value):
            return {
                "valid": False,
                "error": f"Value doesn't match required format"
            }

    # Check maxLength
    max_length = field_info.get("maxLength")
    if max_length and len(value) > int(max_length):
        return {
            "valid": False,
            "error": f"Value must be at most {max_length} characters"
        }

    return {"valid": True, "formatted": value.strip() if value else ""}
