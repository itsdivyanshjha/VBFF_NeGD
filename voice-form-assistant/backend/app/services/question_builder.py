"""
Deterministic question builder for form fields.

Goal: Generate stable, predictable questions derived from the field schema itself.
We avoid using an LLM for asking questions because labels can contain misleading hints
(e.g., "Full Name (as per Aadhaar)") and we want zero ambiguity.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_PARENS_RE = re.compile(r"\s*\([^)]*\)")
_AS_PER_RE = re.compile(r"\s+as\s+per\s+.*$", flags=re.IGNORECASE)


def _clean_label(label: str) -> str:
    # Remove parenthetical hints and trailing "as per ..." clauses.
    s = _PARENS_RE.sub("", label or "").strip()
    s = _AS_PER_RE.sub("", s).strip()
    # Collapse whitespace.
    s = " ".join(s.split())
    return s


def _digits_hint(n: int) -> str:
    return f"Please say the {n}-digit number."


def build_examples(field: Dict[str, Any]) -> str:
    """Short hint string derived from field constraints."""
    ftype = (field.get("field_type") or field.get("type") or "").lower()
    max_len = field.get("maxLength") or field.get("maxlength")
    pattern = field.get("pattern") or ""

    if ftype == "aadhaar":
        return "Example: 1234 5678 9012"
    if ftype == "pan":
        return "Example: ABCDE1234F"
    if ftype in ("mobile", "phone", "tel"):
        return "Example: 9876543210"
    if ftype == "email":
        return "Example: name@example.com"
    if ftype == "pincode":
        return "Example: 110001"
    if ftype in ("date", "dob"):
        return "Example: 2000-01-15"

    # Pattern-based numeric hints
    if isinstance(max_len, int) and max_len in (6, 10, 12):
        return _digits_hint(max_len)
    if isinstance(pattern, str):
        if "\\d{12}" in pattern or "[0-9]{12}" in pattern:
            return _digits_hint(12)
        if "\\d{10}" in pattern or "[0-9]{10}" in pattern:
            return _digits_hint(10)
        if "\\d{6}" in pattern or "[0-9]{6}" in pattern:
            return _digits_hint(6)

    return ""


def build_ask_question(field: Dict[str, Any]) -> str:
    """
    Deterministic, field-derived question.
    Uses field_type when reliable, otherwise falls back to cleaned label.
    """
    field_id = field.get("id") or field.get("name") or "this field"
    raw_label = field.get("label") or str(field_id)
    label = _clean_label(str(raw_label)) or str(field_id)
    ftype = (field.get("field_type") or field.get("type") or "").lower()

    # Canonical questions for common types
    if ftype in ("name", "fullname"):
        # Preserve context in label if it narrows which name (Father/Mother/Spouse)
        lower = label.lower()
        if "father" in lower:
            return "What is your father's name?"
        if "mother" in lower:
            return "What is your mother's name?"
        if "spouse" in lower or "husband" in lower or "wife" in lower:
            return "What is your spouse's name?"
        return "What is your full name?"

    if ftype == "aadhaar":
        return "Please say your 12-digit Aadhaar number. Speak in groups of 4 digits, like: four one two three, five six seven eight, nine zero one two."

    if ftype == "pan":
        return "What is your PAN number?"

    if ftype in ("mobile", "phone", "tel"):
        return "Please say your 10-digit mobile number. You can say it as one continuous number or in groups."

    if ftype == "email":
        return "What is your email address?"

    if ftype in ("date", "dob"):
        return "What is your date of birth?"

    if ftype == "address":
        return "What is your address?"

    if ftype == "pincode":
        return "What is your PIN code?"

    # Select/radio questions can use the label as-is.
    html_type = (field.get("type") or "").lower()
    if html_type in ("select", "radio") or field.get("options"):
        return f"Please choose your {label}."

    # Generic fallback
    return f"What is your {label}?"


def build_next_field_transition(previous_label: str, next_label: str) -> str:
    prev = _clean_label(previous_label or "")
    nxt = _clean_label(next_label or "")
    if prev and nxt:
        return f"Got it. Next, {nxt}."
    if nxt:
        return f"Next, {nxt}."
    return "Let's continue to the next field."

