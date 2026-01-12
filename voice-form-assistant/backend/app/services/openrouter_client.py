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
        system_prompt = """You are a form-filling assistant. Extract the value the user spoke for the requested field.

UNIVERSAL RULES:
1. Extract LITERALLY what was said - do not interpret or make assumptions
2. When digits are spoken with spaces between them (e.g., "9 8 1 0"), remove ALL spaces
3. For ANY field containing "at" or "dot", these are email/web patterns - handle accordingly
4. Maintain the EXACT order and content the user provided

DIGIT HANDLING (applies to ANY numeric field):
- "1 2 3 4 5 6 7 8 9 0 1 2" → "123456789012"
- "nine eight one zero seven" → "9810" (convert words, remove spaces)

EMAIL HANDLING (applies to ANY email-like field):
- "jha divyansh 29 at gmail dot com" → "jhadivyansh29@gmail.com"
- "at" or "at the rate" → "@"
- "dot" → "."
- Remove ALL spaces from email addresses

GENERAL TEXT (applies to ANY text field):
- Capitalize names properly
- Preserve special characters mentioned (underscore, dash, etc.)
- Remove filler words like "um", "uh"

Respond with JSON:
{
    "value": "extracted value or null",
    "confidence": 0.0 to 1.0,
    "needs_confirmation": true/false,
    "reasoning": "brief explanation"
}"""

        field_constraints = []
        if field_info.get('required'):
            field_constraints.append("This field is required")
        if field_info.get('pattern'):
            field_constraints.append(f"Must match pattern: {field_info.get('pattern')}")
        if field_info.get('maxLength'):
            field_constraints.append(f"Maximum {field_info.get('maxLength')} characters")
        
        constraints_str = "\n".join(f"- {c}" for c in field_constraints) if field_constraints else "No specific constraints"

        field_description = f"""
Field: {field_info.get('label', field_info.get('name', 'Unknown'))}
Type: {field_info.get('type', 'text')}
{constraints_str}
"""

        if field_info.get('options'):
            field_description += f"\nValid options: {', '.join([str(opt.get('label', opt.get('value'))) for opt in field_info.get('options', [])])}"

        context_str = ""
        if context:
            validation_error = context.get('validation_error')
            if validation_error:
                context_str += f"\n\nPREVIOUS ATTEMPT FAILED: {validation_error}\nPay attention to the constraints above."
            
            filled = context.get('filled_fields', {})
            if filled and len(filled) > 0:
                context_str += f"\n\nContext from other fields: {json.dumps(filled, indent=2)}"

        user_message = f"""
{field_description}
{context_str}

User said: "{user_input}"

Extract the value and respond with JSON only."""

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
        # SPECIAL CASE: For ask_field, use predefined questions to avoid LLM confusion
        if action == "ask_field":
            field_type = kwargs.get("field_type", "text").lower()
            
            # Direct question mapping - NO LLM INVOLVED
            questions = {
                "name": "What is your full name?",
                "fullname": "What is your full name?",
                "aadhaar": "What is your 12-digit Aadhaar number?",
                "pan": "What is your PAN number?",
                "mobile": "What is your mobile number?",
                "phone": "What is your phone number?",
                "tel": "What is your phone number?",
                "email": "What is your email address?",
                "dob": "What is your date of birth?",
                "date": "What is the date?",
                "address": "What is your address?",
                "pincode": "What is your PIN code?",
                "zip": "What is your PIN code?",
                "state": "Which state do you live in?",
                "city": "Which city do you live in?"
            }
            
            result = questions.get(field_type)
            
            if not result:
                # Fallback: use cleaned label
                field_label = kwargs.get("field_label", "this field")
                clean_label = re.sub(r'\s*\([^)]*\)', '', field_label).strip().lower()
                result = f"What is your {clean_label}?"
            
            logger.info(f"[ask_field] DIRECT QUESTION | Type: '{field_type}' → '{result}'")
            return result
        
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
