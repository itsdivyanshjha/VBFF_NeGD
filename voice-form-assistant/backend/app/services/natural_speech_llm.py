"""
Natural Speech LLM - Enhanced prompt for understanding real-world voice input patterns
This handles the specific issue where users say filler words like "and" in number sequences
"""

def build_natural_speech_extraction_prompt(transcription: str, field_info: dict) -> str:
    """
    Build a prompt that focuses on natural speech understanding
    Specifically handles: "five six two four and two three four nine six seven and zero three"
    Should extract: "562423496703" by ignoring filler words
    """
    
    requirements = field_info.get('pattern', '') or field_info.get('maxLength', '') or 'valid input'
    field_label = field_info.get('label', field_info.get('name', 'field'))
    
    prompt = f"""You are an expert at understanding natural speech patterns and extracting form field values.

USER SAID: "{transcription}"
FIELD: {field_label}  
REQUIREMENTS: {requirements}

YOUR CRITICAL TASK:
Extract the intended data sequence from natural speech. Users often include filler words that should be ignored.

NATURAL SPEECH UNDERSTANDING RULES:
1. IGNORE filler words: "and", "uh", "um", "like", "you know", "then", "followed by"
2. FOCUS on the actual DATA SEQUENCE the user intends to communicate
3. Convert number words to digits in sequence
4. Handle corrections and repetitions naturally

SPECIFIC EXAMPLES FOR NUMBER SEQUENCES:

Input: "five six two four and two three four nine six seven and zero three"
Analysis: User intends sequence [5][6][2][4][2][3][4][9][6][7][0][3] - ignore "and" fillers
Output: "562423496703"

Input: "one two three uh four five six seven eight nine zero one two"  
Analysis: User intends sequence [1][2][3][4][5][6][7][8][9][0][1][2] - ignore "uh" filler
Output: "123456789012"

Input: "double zero one two three and four five six seven eight nine"
Analysis: "double zero" = [0][0], ignore "and", sequence = [0][0][1][2][3][4][5][6][7][8][9]
Output: "001234567893"

Input: "my name is uh John Smith"
Analysis: Ignore filler "uh", extract name
Output: "John Smith"

CRITICAL: The user said "{transcription}" - what is the intended data sequence?

RESPONSE FORMAT:
Respond with ONLY valid JSON:
{{"value": "extracted_sequence", "confidence": 0.9}}

No explanations, no markdown, no other text."""

    return prompt