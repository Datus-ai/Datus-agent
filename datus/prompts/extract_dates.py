# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List

DATE_EXTRACTION_SYSTEM_PROMPT = """You are a date extraction specialist. Your task is to identify and extract time
references from natural language queries that might be used in SQL queries.
You must support both English and Chinese temporal expressions.

Extract any temporal expressions including:

English examples:
- Specific dates: "January 15, 2025", "2025-01-15", "15th Jan 2025"
- Relative single points: "yesterday", "tomorrow", "next Friday", "last Monday"
- Relative time ranges: "last week", "next month", "past year", "last 6 months", "from January to March", "Q1 2025",
"from last year to now", "since last month"

Chinese examples:
- Specific dates: "2025年1月15日", "1月15号", "去年12月"
- Relative single points: "昨天", "明天", "前天", "后天", "下周五"
- Relative time ranges: "上周", "下个月", "今年", "最近6个月", "去年全年", "上季度", "从上个月到下个月", "从1月到3月",
"2024年底到现在", "去年到今天"

For each temporal expression found:
1. Extract the original text exactly as it appears (preserve Chinese characters)
2. Classify the date type as:
   - 'specific': Absolute dates (e.g., "2025-01-15", "January 15, 2025")
   - 'range': Time ranges/periods (e.g., "last week", "next 6 months", "from A to B")
   - 'relative': Single relative points (e.g., "yesterday", "tomorrow", "next Friday")
3. Assign a confidence score (0.0-1.0) based on how clear the temporal reference is

IMPORTANT: When you see range expressions like "从A到B" (from A to B), "A到B",
extract the ENTIRE range as ONE expression, not as separate start/end points.

Return your response as a JSON object with a "dates" array containing the extracted temporal expressions.

Example response format:
{
  "dates": [
    {
      "original_text": "最近6个月",
      "date_type": "range",
      "confidence": 0.9
    },
    {
      "original_text": "2025年1月",
      "date_type": "specific",
      "confidence": 1.0
    }
  ]
}

If no temporal expressions are found, return: {"dates": []}"""

DATE_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "dates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original_text": {
                        "type": "string",
                        "description": "The exact temporal expression as it appears in the text",
                    },
                    "date_type": {
                        "type": "string",
                        "enum": ["specific", "range", "relative"],
                        "description": "Classification of the date type",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence score for the extraction",
                    },
                },
                "required": ["original_text", "date_type", "confidence"],
            },
        }
    },
    "required": ["dates"],
}


def get_date_extraction_prompt(question: str) -> str:
    """
    Generate a prompt for extracting dates from a natural language question.

    Args:
        question: The natural language question/task to analyze

    Returns:
        Formatted prompt string for date extraction
    """
    user_prompt = f"""Analyze the following text and extract any temporal expressions:

Text to analyze: "{question}"

Please identify all date, time, or temporal references in the text and return them in the specified JSON format."""

    return f"{DATE_EXTRACTION_SYSTEM_PROMPT}\n\n{user_prompt}"


def parse_date_extraction_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse the LLM response for date extraction.

    Args:
        response: Response dictionary from the LLM

    Returns:
        List of extracted date dictionaries
    """
    try:
        # Validate structure and return dates
        if isinstance(response, dict) and "dates" in response:
            return response["dates"]
        else:
            return []

    except (KeyError, TypeError):
        # Return empty list if parsing fails
        return []
