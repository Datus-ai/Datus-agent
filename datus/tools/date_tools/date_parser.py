from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from datus.models.base import LLMBaseModel
from datus.prompts.extract_dates import get_date_extraction_prompt, parse_date_extraction_response
from datus.schemas.date_parser_node_models import ExtractedDate
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DateParsingTool:
    """Tool for extracting and parsing temporal expressions using LLM."""

    def __init__(self, llm_model: LLMBaseModel):
        """
        Initialize the date parsing tool.

        Args:
            llm_model: The LLM model to use for date extraction
        """
        self.llm_model = llm_model

    def extract_and_parse_dates(self, text: str, current_date: Optional[str] = None) -> List[ExtractedDate]:
        """
        Extract temporal expressions from text and parse them using LLM.
        Support both English and Chinese temporal expressions.

        Args:
            text: The text to analyze for temporal expressions
            current_date: Reference date for relative expressions (YYYY-MM-DD format)

        Returns:
            List of ExtractedDate objects with parsed date information
        """
        try:
            # Step 1: Use LLM to extract temporal expressions
            extraction_prompt = get_date_extraction_prompt(text)
            logger.debug(f"Date extraction prompt: {extraction_prompt}")

            # Get LLM response
            llm_response = self.llm_model.generate_with_json_output(extraction_prompt)
            logger.debug(f"LLM date extraction response: {llm_response}")

            # Parse the response
            extracted_expressions = parse_date_extraction_response(llm_response)
            logger.debug(f"Extracted expressions: {extracted_expressions}")

            if not extracted_expressions:
                logger.info("No temporal expressions found in the text")
                return []

            # Step 2: Parse each expression using LLM
            parsed_dates = []
            reference_date = datetime.strptime(current_date, "%Y-%m-%d")

            for expr in extracted_expressions:
                parsed_date = self._parse_temporal_expression(expr, reference_date)
                if parsed_date:
                    parsed_dates.append(parsed_date)

            logger.info(f"Successfully parsed {len(parsed_dates)} temporal expressions")
            return parsed_dates

        except Exception as e:
            logger.error(f"Error in date extraction and parsing: {str(e)}")
            return []

    def _parse_temporal_expression(
        self, expression: Dict[str, Any], reference_date: datetime
    ) -> Optional[ExtractedDate]:
        """
        Parse temporal expression using LLM.

        Args:
            expression: Dictionary containing the temporal expression info
            reference_date: Reference datetime for relative expressions

        Returns:
            ExtractedDate object or None if parsing fails
        """
        original_text = expression.get("original_text", "")
        date_type = expression.get("date_type", "relative")
        confidence = expression.get("confidence", 1.0)

        logger.debug(f"Parsing '{original_text}' using LLM")

        result = self._parse_with_llm(original_text, reference_date)
        if result:
            start_date, end_date = result
            return self._create_extracted_date(original_text, date_type, confidence, start_date, end_date)

        logger.warning(f"LLM parsing failed for: '{original_text}'")
        return None

    def _parse_with_llm(self, text: str, reference_date: datetime) -> Optional[Tuple[datetime, datetime]]:
        """Parse temporal expressions using LLM."""
        try:
            prompt = f"""Parse the temporal expression "{text}" with reference date
             {reference_date.strftime('%Y-%m-%d')}.

IMPORTANT RULES:
1. The reference date is ALWAYS INCLUDED in the time range
2. For "未来N个月内"/"next N months": start_date = reference_date, end_date = reference_date + N months
3. For "最近N个月"/"last N months": start_date = reference_date - N months, end_date = reference_date
4. For "未来N天内"/"next N days": start_date = reference_date, end_date = reference_date + N days
5. For "最近N天"/"last N days": start_date = reference_date - N days, end_date = reference_date
6. For expressions with "现在"/"now"/"today": treat as the reference date
7. For expressions like "A到现在"/"A to now": end_date should be the reference date
8. Week ranges are from Monday to Sunday (Monday = start, Sunday = end)
9. For "前N周"/"last N weeks": calculate N complete weeks backwards from reference date
10. For "接下来N周"/"next N weeks": start_date = reference_date, end_date = reference_date + (N × 7) days

Return JSON format:
{{
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD"
}}

Examples:
- "未来三个月内" with reference 2025-01-01 → start_date: "2025-01-01", end_date: "2025-04-01"
- "最近三个月" with reference 2025-01-01 → start_date: "2024-10-01", end_date: "2025-01-01"
- "未来30天内" with reference 2025-01-01 → start_date: "2025-01-01", end_date: "2025-01-31"
- "最近7天" with reference 2025-01-01 → start_date: "2024-12-25", end_date: "2025-01-01"
- "从上个月到下个月" with reference 2025-01-15 → start_date: "2024-12-01", end_date: "2025-02-28"
- "2024年底到现在" with reference 2025-01-15 → start_date: "2024-12-31", end_date: "2025-01-15"
- "from last year to now" with reference 2025-01-15 → start_date: "2024-12-31", end_date: "2025-01-15"
- "去年到今天" with reference 2025-01-15 → start_date: "2024-12-31", end_date: "2025-01-15"
- "上周" with reference 2025-01-15 (Wednesday) → start_date: "2025-01-06" (Monday), end_date: "2025-01-12" (Sunday)
- "接下来一周" with reference 2025-01-15 (Wednesday) → start_date: "2025-01-15",
end_date: "2025-01-22" (7 days from reference)
- "前两周" with reference 2025-01-15 (Wednesday) → start_date: "2024-12-30" (Monday 2 weeks ago),
end_date: "2025-01-12" (Sunday last week)
- "最近三周" with reference 2025-01-15 (Wednesday) → start_date: "2024-12-25" (21 days ago), end_date: "2025-01-15"

For single dates, use the same date for both start_date and end_date."""

            response = self.llm_model.generate_with_json_output(prompt)
            logger.debug(f"LLM parsing response: {response}")
            # generate_with_json_output should always return a dict
            if not isinstance(response, dict):
                logger.debug(f"Expected dict from generate_with_json_output, got {type(response)}: {response}")
                return None

            result = response

            start_date = datetime.strptime(result["start_date"], "%Y-%m-%d")
            end_date = datetime.strptime(result["end_date"], "%Y-%m-%d")
            return start_date, end_date

        except Exception as e:
            logger.debug(f"LLM parsing failed for '{text}': {e}")

        return None

    def _create_extracted_date(
        self, original_text: str, date_type: str, confidence: float, start_date: datetime, end_date: datetime
    ) -> ExtractedDate:
        """Create an ExtractedDate object from parsed dates."""
        if start_date == end_date:
            # Single date
            return ExtractedDate(
                original_text=original_text,
                parsed_date=start_date.strftime("%Y-%m-%d"),
                start_date=None,
                end_date=None,
                date_type="specific" if date_type == "range" else date_type,
                confidence=confidence,
            )
        else:
            # Date range
            return ExtractedDate(
                original_text=original_text,
                parsed_date=None,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                date_type="range",
                confidence=confidence,
            )

    def generate_date_context(self, extracted_dates: List[ExtractedDate]) -> str:
        """
        Generate date context for SQL generation prompt.
        This content will be used in the "Parsed Date Ranges:" section.

        Args:
            extracted_dates: List of extracted and parsed dates

        Returns:
            String containing parsed date ranges for SQL prompt
        """
        if not extracted_dates:
            return ""

        context_parts = []

        for date in extracted_dates:
            if date.date_type == "range" and date.start_date and date.end_date:
                context_parts.append(f"- '{date.original_text}' → {date.start_date} to {date.end_date}")
            elif date.parsed_date:
                context_parts.append(f"- '{date.original_text}' → {date.parsed_date}")

        return "\n".join(context_parts)
