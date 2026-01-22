"""
OpenRouter LLM Client.
Handles communication with OpenRouter API for field value extraction and response generation.
"""

import logging
import json
import re
from typing import Dict, Any, Optional, List
import httpx

from ..config import settings

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Client for OpenRouter API."""

    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        self.model = settings.OPENROUTER_MODEL
        self.base_url = settings.OPENROUTER_BASE_URL
        self.timeout = settings.OPENROUTER_TIMEOUT
        self.site_url = settings.OPENROUTER_SITE_URL
        self.app_name = settings.OPENROUTER_APP_NAME
        # Cache for generated questions (avoid regenerating for same field types)
        self._question_cache: Dict[str, str] = {}

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        # Note: OpenRouter expects HTTP-Referer (not standard Referer)
        # This is their specific requirement for app identification and rankings
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,  # OpenRouter-specific header
            "X-Title": self.app_name  # OpenRouter-specific header
        }

    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 500
    ) -> str:
        """
        Make a request to OpenRouter API with retry logic.

        Args:
            messages: List of message dicts with role and content
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Response text from the model
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                )
                response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                logger.debug(f"OpenRouter API success: {len(content)} chars")
                return content

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                error_detail = ""
                try:
                    error_detail = e.response.json().get("error", {}).get("message", "")
                except:
                    error_detail = e.response.text[:200]
                
                if status_code == 401:
                    logger.error("OpenRouter API authentication failed - check your API key")
                elif status_code == 429:
                    logger.error("OpenRouter API rate limit exceeded")
                elif status_code >= 500:
                    logger.error(f"OpenRouter API server error: {status_code}")
                else:
                    logger.error(f"OpenRouter API error {status_code}: {error_detail}")
                raise
            except httpx.RequestError as e:
                logger.error(f"OpenRouter API request failed: {e}")
                raise
            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected OpenRouter API response format: {e}")
                raise

    async def extract_field_value(
        self,
        field_info: Dict[str, Any],
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract field value from user input using LLM.

        Args:
            field_info: Information about the field (name, type, validation rules)
            user_input: User's spoken input transcription
            context: Optional context (previous fields, conversation history)

        Returns:
            Dict with: value, confidence, needs_confirmation, reasoning
        """
        system_prompt = """You are an expert form-filling assistant specializing in natural speech interpretation. Extract the value from speech and FORMAT it precisely according to field requirements.

CRITICAL RULES:
1. READ field constraints carefully
2. Extract what user said (handle natural speech patterns)
3. FORMAT to match required pattern/format EXACTLY
4. Handle mixed Hindi-English (Hinglish) gracefully

═══════════════════════════════════════════════════════════
NATURAL SPEECH PATTERNS - HANDLE THESE!
═══════════════════════════════════════════════════════════

SPOKEN MULTIPLIERS:
- "triple nine" / "teen nine" / "trip nine" → "999"
- "double zero" / "dub zero" → "00"
- "double five" → "55"
- Pattern: "double X" = XX, "triple X" = XXX

SPOKEN SYMBOLS (Email):
- "at the rate" / "at" → "@"
- "dot" / "period" → "."
- "underscore" → "_"
- "dash" / "hyphen" → "-"
- Remove ALL spaces: "name @ gmail . com" → "name@gmail.com"

ZERO VARIANTS:
- "oh" / "o" (in context of numbers) → "0"
- "zero" → "0"
Example: "nine oh three" → "903"

HINDI-ENGLISH MIXED:
- "सिक्स" = "six" = "6"
- "फाइव" = "five" = "5"
- "जीरो" = "zero" = "0"
- Extract digits regardless of language

═══════════════════════════════════════════════════════════
FIELD-SPECIFIC PROCESSING
═══════════════════════════════════════════════════════════

NUMERIC FIELDS (Aadhaar, Phone, PIN):
1. Convert ALL spoken numbers to digits (English or Hindi)
2. Handle multipliers: "triple nine" → "999"
3. Handle variants: "oh" → "0"
4. Remove ALL formatting: spaces, dashes, periods
5. Match exact length from constraints

Examples:
- "nine eight one zero seven three triple nine five" → "9810739995" ✓
- "981-073-trip-95" → Interpret as "9810739995" ✓
- "Six, five, four, seven..." → "6547..." ✓
- "सिक्स फाइव फॉर" → "654" ✓

DATE FIELDS:
- Parse ANY natural date format
- Output in constraint's EXACT format (usually YYYY-MM-DD)
Examples:
- "august 21 1998" → "1998-08-21" ✓
- "21st august nineteen ninety eight" → "1998-08-21" ✓
- "8/21/98" → "1998-08-21" ✓

EMAIL FIELDS:
- Convert ALL spoken symbols to actual symbols
- Remove all spaces
- Lowercase everything
Examples:
- "john dot smith at gmail dot com" → "john.smith@gmail.com" ✓
- "prashant.singh25 at the rate gmail.com" → "prashant.singh25@gmail.com" ✓

NAME FIELDS:
- Title case each word
- Remove trailing punctuation (commas, periods)
- Keep middle names/initials
Examples:
- "deviant, ja" → "Deviant Ja" ✓
- "RAJESH KUMAR" → "Rajesh Kumar" ✓

═══════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════

Respond with JSON:
{
    "value": "extracted value in correct format or null",
    "confidence": 0.0 to 1.0,
    "needs_confirmation": true/false,
    "reasoning": "brief explanation of transformations made"
}

If you cannot extract a valid value, set "value": null and explain why in reasoning."""

        # Build constraints from HTML field attributes
        field_constraints = []
        field_type = field_info.get('field_type', '').lower()
        html_type = field_info.get('type', 'text').lower()
        
        if field_info.get('required'):
            field_constraints.append("Required field")
        
        # Extract format requirements from HTML attributes
        if html_type == 'date':
            field_constraints.append("OUTPUT FORMAT: YYYY-MM-DD (example: 2003-05-29)")
            field_constraints.append("Parse any spoken date and convert to this format")
        
        if field_info.get('pattern'):
            pattern = field_info.get('pattern')
            field_constraints.append(f"Must match regex: {pattern}")
            
            # Interpret common patterns
            if '\\d{12}' in pattern or r'\d{12}' in pattern:
                field_constraints.append("OUTPUT FORMAT: Exactly 12 digits, no spaces")
            elif '\\d{10}' in pattern or r'\d{10}' in pattern:
                field_constraints.append("OUTPUT FORMAT: Exactly 10 digits, no spaces")
            elif '\\d{6}' in pattern or r'\d{6}' in pattern:
                field_constraints.append("OUTPUT FORMAT: Exactly 6 digits, no spaces")
            elif '[6-9]\\d{9}' in pattern:
                field_constraints.append("OUTPUT FORMAT: 10 digits starting with 6, 7, 8, or 9")
        
        if field_info.get('maxLength'):
            max_len = field_info.get('maxLength')
            field_constraints.append(f"Maximum length: {max_len} characters")
            
            # Infer format from maxLength for common cases
            if max_len == 12 and not any('12' in str(c) for c in field_constraints):
                field_constraints.append("OUTPUT FORMAT: 12 characters (likely Aadhaar)")
            elif max_len == 10 and html_type == 'tel':
                field_constraints.append("OUTPUT FORMAT: 10 digits (mobile number)")
        
        constraints_str = "\n".join(f"- {c}" for c in field_constraints) if field_constraints else "No specific constraints"

        field_description = f"""
Field Name: {field_info.get('label', field_info.get('name', 'Unknown'))}
HTML Type: {html_type}
Detected Type: {field_type}

CONSTRAINTS FROM HTML:
{constraints_str}

IMPORTANT: Output must match the format specified above!
"""

        if field_info.get('options'):
            field_description += f"\nValid options: {', '.join([str(opt.get('label', opt.get('value'))) for opt in field_info.get('options', [])])}"

        context_str = ""
        if context:
            validation_error = context.get('validation_error')
            if validation_error:
                context_str += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED: {validation_error}"
                context_str += "\n→ Pay special attention to length and format constraints!"
                context_str += "\n→ Check for missed digits or incorrect symbols."
            
            filled = context.get('filled_fields', {})
            if filled and len(filled) > 0:
                context_str += f"\n\n📋 Context from other fields: {json.dumps(filled, indent=2)}"

        # Add specific hints for common problematic fields
        field_hints = ""
        if field_type in ["aadhaar", "mobile", "tel", "number", "pincode"]:
            field_hints = """
REMINDER FOR NUMERIC FIELDS:
- Handle "triple nine" → "999"
- Handle "double zero" → "00"  
- Handle "oh" → "0"
- Convert Hindi numbers: "सिक्स" → "6"
- Remove ALL spaces/dashes
- Count the digits carefully to match required length!"""

        user_message = f"""
{field_description}
{context_str}
{field_hints}

═══════════════════════════════════════════════════════════
USER INPUT: "{user_input}"
═══════════════════════════════════════════════════════════

Extract the value, apply ALL transformations, and respond with JSON only."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            response = await self._make_request(messages, temperature=0.0)

            # Parse JSON response
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()

            result = json.loads(response)

            # Ensure required fields exist
            result.setdefault("value", None)
            result.setdefault("confidence", 0.5)
            result.setdefault("needs_confirmation", True)
            result.setdefault("reasoning", "")

            # Post-process: Apply additional normalization if needed
            if result["value"]:
                result["value"] = self._post_process_value(result["value"], field_info)

            # Auto-confirm high confidence non-critical fields
            if result["confidence"] >= 0.9:
                critical_types = ["aadhaar", "pan", "mobile", "email"]
                if field_info.get("field_type") not in critical_types:
                    result["needs_confirmation"] = False

            logger.info(f"Extracted value: {result['value']} (confidence: {result['confidence']})")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {
                "value": user_input,
                "confidence": 0.3,
                "needs_confirmation": True,
                "reasoning": "Failed to parse LLM response, using raw input"
            }
        except Exception as e:
            logger.error(f"Error extracting field value: {e}")
            return {
                "value": None,
                "confidence": 0.0,
                "needs_confirmation": True,
                "reasoning": f"Error: {str(e)}"
            }

    def _post_process_value(self, value: str, field_info: Dict[str, Any]) -> str:
        """
        Final safety net: Apply deterministic normalization rules.
        
        This catches cases where LLM might not perfectly normalize.
        """
        if not value:
            return value
        
        field_type = field_info.get('field_type', '').lower()
        html_type = field_info.get('type', 'text').lower()
        
        # Numeric fields: Ensure only digits
        if field_type in ('aadhaar', 'mobile', 'pincode') or html_type in ('tel', 'number'):
            # Strip everything except digits
            cleaned = re.sub(r'[^\d]', '', value)
            if cleaned:
                return cleaned
        
        # Email fields: Ensure proper format
        elif field_type == 'email' or html_type == 'email':
            # Lowercase
            cleaned = value.lower().strip()
            
            # Robust regex replacements (in case LLM missed some)
            # "at the rate" variants
            cleaned = re.sub(r'\s*at\s*the\s*rate\s*', '@', cleaned)
            
            # Standalone "at" as @
            cleaned = re.sub(r'\s+at\s+', '@', cleaned)
            
            # "dot" variants
            cleaned = re.sub(r'\s*dot\s*', '.', cleaned)
            cleaned = re.sub(r'\s*period\s*', '.', cleaned)
            cleaned = re.sub(r'\s*point\s*', '.', cleaned)
            
            # Other symbols
            cleaned = re.sub(r'\s*underscore\s*', '_', cleaned)
            cleaned = re.sub(r'\s*dash\s*', '-', cleaned)
            cleaned = re.sub(r'\s*hyphen\s*', '-', cleaned)
            
            # Remove remaining spaces
            cleaned = cleaned.replace(' ', '')
            
            return cleaned
        
        # Name fields: Proper title case
        elif field_type == 'name' or 'name' in field_info.get('label', '').lower():
            # Remove trailing punctuation
            cleaned = value.strip().rstrip('.,;')
            # Title case
            return ' '.join(word.capitalize() for word in cleaned.split())
        
        # Default: just strip whitespace
        return value.strip()
    
    async def generate_field_question(self, field: Dict[str, Any]) -> str:
        """
        DYNAMICALLY generate a question for any form field using LLM.
        Works with ANY field - no hardcoding required.

        Args:
            field: Field metadata from DOM (label, type, required, pattern, maxLength, options, etc.)

        Returns:
            Natural language question to ask the user
        """
        # Build cache key from field characteristics
        field_label = field.get("label", "")
        field_type = field.get("field_type", field.get("type", "text"))
        cache_key = f"{field_label}_{field_type}_{field.get('required', False)}"

        # Check cache first
        if cache_key in self._question_cache:
            return self._question_cache[cache_key]

        # Build field context for LLM
        field_info = []
        field_info.append(f"Label: {field_label}")
        field_info.append(f"HTML Type: {field.get('type', 'text')}")
        field_info.append(f"Detected Type: {field_type}")

        if field.get("required"):
            field_info.append("Required: Yes")

        if field.get("pattern"):
            field_info.append(f"Pattern: {field.get('pattern')}")

        if field.get("maxLength"):
            field_info.append(f"Max Length: {field.get('maxLength')}")

        if field.get("placeholder"):
            field_info.append(f"Placeholder: {field.get('placeholder')}")

        if field.get("options"):
            options_text = ", ".join([opt.get("label", opt.get("value", "")) for opt in field.get("options", [])[:5]])
            field_info.append(f"Options: {options_text}")

        prompt = f"""Generate a SHORT, NATURAL question to ask a user for this form field.

FIELD INFORMATION:
{chr(10).join(field_info)}

GUIDELINES:
- Keep it conversational and friendly
- For numeric fields (Aadhaar, phone), instruct user to say digits in groups (e.g., "four one two three, five six seven eight")
- For dates, say they can speak naturally (e.g., "29 May 2003")
- For select/radio fields, mention some options if helpful
- Keep it under 25 words
- Don't repeat the label verbatim - make it natural

Return ONLY the question text, nothing else."""

        try:
            response = await self._make_request(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3  # Low temperature for consistent questions
            )

            question = response.strip().strip('"').strip("'")

            # Clean up any extra formatting
            question = re.sub(r'^(Question:|Ask:)\s*', '', question, flags=re.IGNORECASE)

            # Cache it
            self._question_cache[cache_key] = question

            logger.info(f"[generate_field_question] Generated: '{question}' for {field_type} field")
            return question

        except Exception as e:
            logger.error(f"Failed to generate question for field, using fallback: {e}")
            # Fallback to simple label-based question
            clean_label = re.sub(r'\s*\([^)]*\)', '', field_label).strip()
            return f"What is your {clean_label.lower()}?"

    async def generate_response(
        self,
        action: str,
        **kwargs
    ) -> str:
        """
        Generate a natural language response for various actions.

        Args:
            action: Type of response to generate
            **kwargs: Additional parameters for response generation

        Returns:
            Generated response text
        """
        # NOTE: ask_field is now handled by generate_field_question() instead
        # This method handles other actions like greeting, confirmation, error, etc.
        
        # For other actions, use LLM
        prompts = {
            "greeting": self._greeting_prompt,
            "ask_field": self._ask_field_prompt,  # Won't be used due to above
            "confirm_value": self._confirm_value_prompt,
            "validation_error": self._validation_error_prompt,
            "next_field": self._next_field_prompt,
            "completion": self._completion_prompt,
            "error": self._error_prompt,
            "repeat": self._repeat_prompt
        }

        prompt_func = prompts.get(action, self._default_prompt)
        system_prompt, user_message = prompt_func(**kwargs)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            response = await self._make_request(messages, temperature=0.7, max_tokens=150)
            result = response.strip()
            
            # DEBUG: Log what question is being generated
            logger.info(f"[{action}] Generated text: '{result}'")
            
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._fallback_response(action, **kwargs)

    def _greeting_prompt(self, **kwargs) -> tuple:
        system = """You are a friendly voice assistant helping users fill government forms.
Generate a brief, warm greeting and explain you'll help fill the form step by step.
Keep it under 30 words. Be professional but approachable."""

        form_name = kwargs.get("form_name", "the form")
        user = f"Generate a greeting for helping fill: {form_name}"

        return system, user

    def _ask_field_prompt(self, **kwargs) -> tuple:
        field_label = kwargs.get("field_label", "this field")
        field_type = kwargs.get("field_type", "text")
        examples = kwargs.get("examples", "")

        # Map field types to natural language descriptions
        # DO NOT use the original label - it contains confusing text
        field_description = {
            "name": "full name",
            "fullname": "full name",
            "aadhaar": "Aadhaar number",
            "pan": "PAN number",
            "mobile": "mobile number",
            "phone": "phone number",
            "tel": "phone number",
            "email": "email address",
            "dob": "date of birth",
            "date": "date",
            "address": "address",
            "pincode": "PIN code",
            "zip": "PIN code",
            "state": "state",
            "city": "city"
        }.get(field_type.lower())
        
        # If no mapping, clean the label
        if not field_description:
            clean_label = re.sub(r'\s*\([^)]*\)', '', field_label).strip()
            field_description = clean_label.lower()

        # Generate questions directly based on field type - NO LLM NEEDED
        questions = {
            "full name": "What is your full name?",
            "aadhaar number": "What is your 12-digit Aadhaar number?",
            "pan number": "What is your PAN number?",
            "mobile number": "What is your mobile number?",
            "phone number": "What is your phone number?",
            "email address": "What is your email address?",
            "date of birth": "What is your date of birth?",
            "date": "What is the date?",
            "address": "What is your address?",
            "pin code": "What is your PIN code?",
            "state": "Which state do you live in?",
            "city": "Which city do you live in?"
        }
        
        # Use predefined question or generate simple one
        direct_question = questions.get(field_description, f"What is your {field_description}?")
        
        # DEBUG
        logger.info(f"[ask_field] Type: '{field_type}' → Description: '{field_description}' → Question: '{direct_question}'")
        
        # Return the direct question WITHOUT using LLM
        # This prevents ANY possibility of confusion
        system = "Return the exact question provided."
        user = f"Question: {direct_question}"

        return system, user

    def _confirm_value_prompt(self, **kwargs) -> tuple:
        system = """You are a voice assistant confirming a form field value.
Generate a brief confirmation question that includes the value.
Keep it under 25 words. Be clear and natural."""

        field_label = kwargs.get("field_label", "the field")
        value = kwargs.get("value", "")

        user = f"Confirm this value for {field_label}: '{value}'"

        return system, user

    def _validation_error_prompt(self, **kwargs) -> tuple:
        system = """You are a voice assistant explaining a validation error.
Explain the error simply and ask user to provide correct value.
Keep it under 30 words. Be helpful, not critical."""

        field_label = kwargs.get("field_label", "the field")
        error = kwargs.get("error", "Invalid value")

        user = f"Explain this validation error for {field_label}: {error}"

        return system, user

    def _next_field_prompt(self, **kwargs) -> tuple:
        system = """You are a voice assistant moving to the next form field.
Generate a brief transition acknowledging the previous field and asking for the next.
Keep it under 25 words."""

        previous_field = kwargs.get("previous_field", "")
        next_field = kwargs.get("next_field", "")

        user = f"Moving from '{previous_field}' to asking for '{next_field}'"

        return system, user

    def _completion_prompt(self, **kwargs) -> tuple:
        system = """You are a voice assistant that has finished helping fill a form.
Generate a brief completion message confirming all fields are filled.
Keep it under 30 words. Be congratulatory."""

        form_name = kwargs.get("form_name", "the form")
        field_count = kwargs.get("field_count", 0)

        user = f"Form '{form_name}' is complete. {field_count} fields were filled."

        return system, user

    def _error_prompt(self, **kwargs) -> tuple:
        system = """You are a voice assistant handling an error.
Apologize briefly and ask user to try again.
Keep it under 20 words."""

        error = kwargs.get("error", "Something went wrong")
        user = f"Handle this error: {error}"

        return system, user

    def _repeat_prompt(self, **kwargs) -> tuple:
        system = """You are a voice assistant asking user to repeat.
Ask them to repeat what they said, politely.
Keep it under 15 words."""

        user = "Ask user to repeat what they said"
        return system, user

    def _default_prompt(self, **kwargs) -> tuple:
        system = "You are a helpful voice assistant."
        user = kwargs.get("message", "Say something helpful")
        return system, user

    def _fallback_response(self, action: str, **kwargs) -> str:
        """Fallback responses when LLM fails."""
        # Clean field labels for fallback responses too
        field_label = kwargs.get('field_label', 'answer')
        clean_label = re.sub(r'\s*\([^)]*\)', '', field_label).strip().lower()
        
        # Clean next_field label (can't use regex in f-string directly)
        next_field = kwargs.get('next_field', 'next answer')
        clean_next_field = re.sub(r'\s*\([^)]*\)', '', next_field).strip().lower()
        
        fallbacks = {
            "greeting": "Hello! I'll help you fill this form. Let's start with the first field.",
            "ask_field": f"Please tell me your {clean_label}.",
            "confirm_value": f"I heard {kwargs.get('value', '')}. Is that correct?",
            "validation_error": f"That doesn't seem right. Please try again.",
            "next_field": f"Great! Now, what's your {clean_next_field}?",
            "completion": "All done! Your form is now complete.",
            "error": "Sorry, something went wrong. Please try again.",
            "repeat": "I didn't catch that. Could you please repeat?"
        }
        return fallbacks.get(action, "How can I help you?")

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the configured model."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "configured": bool(self.api_key)
        }


# Global instance
openrouter_client = OpenRouterClient()
