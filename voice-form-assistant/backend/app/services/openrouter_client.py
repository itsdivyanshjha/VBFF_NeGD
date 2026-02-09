"""
OpenRouter LLM Client.
Handles communication with OpenRouter API for field value extraction and response generation.
Optimized for Llama 3.3 70B Instruct.
"""

import logging
import json
import re
from typing import Dict, Any, Optional, List
import httpx

from ..config import settings

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Client for OpenRouter API - optimized for Llama 3.3 70B."""

    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        self.model = settings.OPENROUTER_MODEL
        self.base_url = settings.OPENROUTER_BASE_URL
        self.timeout = settings.OPENROUTER_TIMEOUT
        self.site_url = settings.OPENROUTER_SITE_URL
        self.app_name = settings.OPENROUTER_APP_NAME
        self._question_cache: Dict[str, str] = {}

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name
        }

    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 200
    ) -> str:
        """Make a request to OpenRouter API."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._get_headers()
                )
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API error {e.response.status_code}")
            raise
        except httpx.RequestError as e:
            logger.error(f"OpenRouter API request failed: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected API response format: {e}")
            raise

    # =========================================================================
    # FIELD VALUE EXTRACTION
    # =========================================================================

    async def extract_field_value(
        self,
        field_info: Dict[str, Any],
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract field value from user's voice input.
        
        Single clean extraction path - let the LLM do the heavy lifting.
        """
        if not user_input or not user_input.strip():
            return {
                "value": None,
                "confidence": 0.0,
                "needs_confirmation": True,
                "reasoning": "Empty input"
            }

        field_type = self._detect_field_type(field_info)
        prompt = self._build_extraction_prompt(user_input.strip(), field_info, field_type)

        try:
            response = await self._make_request(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic for extraction
                max_tokens=150
            )
            
            result = self._parse_extraction_response(response)
            result = self._apply_field_constraints(result, field_info, field_type)
            
            logger.info(f"Extracted '{result.get('value')}' (conf: {result.get('confidence')}) from '{user_input}'")
            return result

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {
                "value": user_input.strip(),
                "confidence": 0.2,
                "needs_confirmation": True,
                "reasoning": f"Extraction error: {str(e)}"
            }

    def _detect_field_type(self, field_info: Dict) -> str:
        """Detect the semantic field type from schema."""
        label = field_info.get('label', '').lower()
        html_type = field_info.get('type', '').lower()
        field_type = field_info.get('field_type', '').lower()
        pattern = field_info.get('pattern', '')
        max_len = field_info.get('maxLength') or field_info.get('maxlength')

        # Check explicit type first
        if field_type:
            return field_type

        # Detect from label keywords
        if any(x in label for x in ['aadhaar', 'aadhar', 'uid']):
            return 'aadhaar'
        if any(x in label for x in ['mobile', 'phone', 'contact']):
            return 'mobile'
        if any(x in label for x in ['pin', 'postal', 'zip']):
            return 'pincode'
        if 'email' in label or html_type == 'email':
            return 'email'
        if 'name' in label:
            return 'name'
        if any(x in label for x in ['date', 'dob', 'birth']):
            return 'date'

        # Detect from pattern/length
        if pattern:
            if max_len == 12 and '[0-9]' in pattern:
                return 'aadhaar'
            if max_len == 10 and '[0-9]' in pattern:
                return 'mobile'
            if max_len == 6 and '[0-9]' in pattern:
                return 'pincode'

        # HTML type fallback
        if html_type in ('tel', 'number'):
            return 'numeric'
        if html_type == 'email':
            return 'email'

        return 'text'

    def _build_extraction_prompt(self, transcription: str, field_info: Dict, field_type: str) -> str:
        """
        Build a focused extraction prompt.
        
        Key principle: Short, structured prompts work better with Llama 3.3.
        """
        label = field_info.get('label', 'field')
        requirements = self._get_field_requirements(field_info)

        # Base instruction - keep it tight
        prompt = f"""TASK: Extract the value for "{label}" from voice input.

INPUT: "{transcription}"
FIELD TYPE: {field_type}
REQUIREMENTS: {requirements}

RULES:
1. Convert ALL number words to digits:
   - English: one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, zero=0
   - Hindi: ek=1, do=2, teen=3, char=4, paanch=5, che=6, saat=7, aath=8, nau=9
   - Transliterated Hindi: फाइव=5, सिक्स=6, टू=2, थ्री=3, फॉर=4, वन=1, सेवन=7, एट=8, नाइन=9, जीरो=0
2. Remove filler words: "my", "is", "the", "number", "मेरा", "है"
3. Handle "double X" = XX, "triple X" = XXX
4. Preserve the EXACT sequence of digits spoken

"""

        # Add type-specific guidance (minimal)
        if field_type == 'email':
            prompt += """EMAIL RULES:
- "at" or "at the rate" = @
- "dot" = .
- "underscore" = _
- Remove all spaces from final email

"""
        elif field_type == 'name':
            prompt += """NAME RULES:
- Use Title Case
- Keep only the name, remove "my name is" etc.

"""
        elif field_type in ('aadhaar', 'mobile', 'pincode', 'numeric'):
            prompt += f"""NUMERIC RULES:
- Output ONLY digits
- Expected length: {field_info.get('maxLength', 'varies')} digits

"""

        prompt += """OUTPUT FORMAT (JSON only, no other text):
{"value": "extracted_value", "confidence": 0.0-1.0}

RESPOND WITH ONLY THE JSON. NO EXPLANATIONS."""

        return prompt

    def _get_field_requirements(self, field_info: Dict) -> str:
        """Get human-readable requirements from field schema."""
        requirements = []

        min_len = field_info.get('minLength') or field_info.get('minlength')
        max_len = field_info.get('maxLength') or field_info.get('maxlength')

        if min_len and max_len:
            if min_len == max_len:
                requirements.append(f"exactly {max_len} characters")
            else:
                requirements.append(f"{min_len}-{max_len} characters")
        elif max_len:
            requirements.append(f"max {max_len} characters")

        pattern = field_info.get('pattern')
        if pattern:
            if '[0-9]' in pattern or r'\d' in pattern:
                requirements.append("digits only")
            if '[A-Z]' in pattern:
                requirements.append("uppercase letters")

        if field_info.get('required'):
            requirements.append("required")

        return ", ".join(requirements) if requirements else "no specific format"

    def _parse_extraction_response(self, response: str) -> Dict:
        """Parse LLM response to extract value and confidence."""
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith('```'):
            response = re.sub(r'^```(?:json)?\n?', '', response)
            response = re.sub(r'\n?```$', '', response)
        response = response.strip()

        # Try direct JSON parse
        try:
            parsed = json.loads(response)
            if 'value' in parsed:
                return {
                    "value": parsed.get('value'),
                    "confidence": float(parsed.get('confidence', 0.5)),
                    "needs_confirmation": parsed.get('confidence', 0.5) < 0.8
                }
        except json.JSONDecodeError:
            pass

        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*"value"[^{}]*\}', response)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return {
                    "value": parsed.get('value'),
                    "confidence": float(parsed.get('confidence', 0.5)),
                    "needs_confirmation": parsed.get('confidence', 0.5) < 0.8
                }
            except json.JSONDecodeError:
                pass

        # Fallback: extract value with regex
        value_match = re.search(r'"value"\s*:\s*"([^"]*)"', response)
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', response)

        if value_match:
            return {
                "value": value_match.group(1),
                "confidence": float(conf_match.group(1)) if conf_match else 0.4,
                "needs_confirmation": True
            }

        # Last resort: return cleaned response as value
        logger.warning(f"Could not parse LLM response: {response[:100]}")
        return {
            "value": response.strip('"').strip(),
            "confidence": 0.3,
            "needs_confirmation": True
        }

    def _apply_field_constraints(self, result: Dict, field_info: Dict, field_type: str) -> Dict:
        """Apply final formatting constraints based on field type."""
        value = result.get('value')
        if not value:
            return result

        value = str(value).strip()

        # Type-specific post-processing
        if field_type in ('aadhaar', 'mobile', 'pincode', 'numeric'):
            # Extract only digits
            digits = re.sub(r'[^\d]', '', value)
            if digits:
                value = digits

        elif field_type == 'email':
            # Ensure email format
            value = value.lower().strip()
            # Final cleanup in case LLM missed something
            value = re.sub(r'\s+at\s+', '@', value, flags=re.IGNORECASE)
            value = re.sub(r'\s+dot\s+', '.', value, flags=re.IGNORECASE)
            value = value.replace(' ', '')

        elif field_type == 'name':
            # Title case for names
            value = ' '.join(word.capitalize() for word in value.split())

        # Validate against pattern if present
        pattern = field_info.get('pattern')
        if pattern and value:
            try:
                if not re.match(f'^{pattern}$', value):
                    result['needs_confirmation'] = True
                    result['confidence'] = min(result.get('confidence', 0.5), 0.6)
            except re.error:
                pass

        result['value'] = value
        return result

    # =========================================================================
    # QUESTION GENERATION
    # =========================================================================

    async def generate_field_question(self, field: Dict[str, Any]) -> str:
        """Generate a natural question for a form field."""
        field_label = field.get("label", "")
        field_type = field.get("field_type", field.get("type", "text"))
        has_options = bool(field.get("options"))

        cache_key = f"{field_label}_{field_type}_{has_options}"
        if cache_key in self._question_cache:
            return self._question_cache[cache_key]

        # Build field context
        field_context = [
            f"Label: {field_label}",
            f"Type: {field_type}",
        ]

        if field.get("required"):
            field_context.append("Required: Yes")
        if field.get("maxLength"):
            field_context.append(f"Max Length: {field.get('maxLength')}")
        if field.get("options"):
            options_preview = ", ".join([
                opt.get("label", opt.get("value", ""))
                for opt in field.get("options", [])[:4]
            ])
            field_context.append(f"Options: {options_preview}")
            field_context.append("NOTE: Do NOT list options in your question!")

        prompt = f"""Generate a short, friendly question to ask for this form field.

FIELD INFO:
{chr(10).join(field_context)}

RULES:
1. Keep it under 15 words
2. Sound natural and conversational
3. For name fields with relationships (father, mother, spouse), ask appropriately
4. For SELECT fields, do NOT list the options
5. Be friendly but concise

EXAMPLES:
- "Full Name" → "What is your full name?"
- "Father's Name" → "What is your father's name?"
- "Mobile Number" → "What is your mobile number?"
- "Gender" (select) → "What is your gender?"
- "Email" → "What is your email address?"

OUTPUT: Write ONLY the question, nothing else."""

        try:
            response = await self._make_request(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50
            )

            question = response.strip().strip('"').strip("'")
            question = re.sub(r'^(Question:|Q:)\s*', '', question, flags=re.IGNORECASE)

            self._question_cache[cache_key] = question
            logger.info(f"Generated question: '{question}' for field '{field_label}'")
            return question

        except Exception as e:
            logger.error(f"Failed to generate question: {e}")
            # Simple fallback
            return f"Please provide your {field_label}."

    # =========================================================================
    # RESPONSE GENERATION
    # =========================================================================

    async def generate_response(self, action: str, **kwargs) -> str:
        """Generate natural language responses for various actions."""
        prompt_builders = {
            "greeting": self._build_greeting_prompt,
            "confirm_value": self._build_confirm_prompt,
            "validation_error": self._build_validation_error_prompt,
            "next_field": self._build_next_field_prompt,
            "completion": self._build_completion_prompt,
            "error": self._build_error_prompt,
            "repeat": self._build_repeat_prompt,
        }

        builder = prompt_builders.get(action)
        if not builder:
            return self._get_fallback_response(action, **kwargs)

        prompt = builder(**kwargs)

        try:
            response = await self._make_request(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=100
            )
            return response.strip().strip('"')
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._get_fallback_response(action, **kwargs)

    def _build_greeting_prompt(self, **kwargs) -> str:
        form_name = kwargs.get("form_name", "the form")
        return f"""Generate a brief, friendly greeting for a voice assistant helping fill "{form_name}".
Keep it under 20 words. Be warm but professional.
OUTPUT: Write ONLY the greeting."""

    def _build_confirm_prompt(self, **kwargs) -> str:
        field_label = kwargs.get("field_label", "field")
        value = kwargs.get("value", "")
        return f"""Generate a brief confirmation question.
Field: {field_label}
Value: {value}
Keep it under 12 words. Example: "I heard {value}. Is that correct?"
OUTPUT: Write ONLY the confirmation question."""

    def _build_validation_error_prompt(self, **kwargs) -> str:
        field_label = kwargs.get("field_label", "field")
        error = kwargs.get("error", "Invalid value")
        return f"""Generate a brief, friendly error message.
Field: {field_label}
Error: {error}
Keep it under 20 words. Be helpful and encouraging.
OUTPUT: Write ONLY the error message."""

    def _build_next_field_prompt(self, **kwargs) -> str:
        previous = kwargs.get("previous_field", "")
        next_field = kwargs.get("next_field", "")
        return f"""Generate a brief transition from "{previous}" to "{next_field}".
Keep it under 10 words. Example: "Got it. Now, [next question]"
OUTPUT: Write ONLY the transition."""

    def _build_completion_prompt(self, **kwargs) -> str:
        form_name = kwargs.get("form_name", "the form")
        return f"""Generate a brief completion message for finishing "{form_name}".
Keep it under 15 words. Be congratulatory.
OUTPUT: Write ONLY the completion message."""

    def _build_error_prompt(self, **kwargs) -> str:
        error = kwargs.get("error", "Something went wrong")
        return f"""Generate a brief, apologetic error message for: {error}
Keep it under 15 words. Ask to try again.
OUTPUT: Write ONLY the error message."""

    def _build_repeat_prompt(self, **kwargs) -> str:
        return """Generate a brief, polite request to repeat.
Keep it under 10 words.
OUTPUT: Write ONLY the request."""

    def _get_fallback_response(self, action: str, **kwargs) -> str:
        """Fallback responses when LLM fails."""
        fallbacks = {
            "greeting": "Hello! I'll help you fill out this form. Let's begin.",
            "confirm_value": f"I heard {kwargs.get('value', 'that')}. Is that correct?",
            "validation_error": f"That doesn't seem right for {kwargs.get('field_label', 'this field')}. Please try again.",
            "next_field": "Okay, moving on.",
            "completion": "Great! The form is complete.",
            "error": "Sorry, something went wrong. Please try again.",
            "repeat": "I didn't catch that. Could you repeat?",
        }
        return fallbacks.get(action, "Please continue.")

    # =========================================================================
    # UTILITY
    # =========================================================================

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the configured model."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "configured": bool(self.api_key)
        }


# Global instance
openrouter_client = OpenRouterClient()