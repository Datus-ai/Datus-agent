from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from datus.models.base import LLMBaseModel
from datus.prompts.extract_dates import get_date_extraction_prompt, parse_date_extraction_response
from datus.prompts.prompt_manager import prompt_manager
from datus.schemas.date_parser_node_models import DateParserInput, DateParserResult, ExtractedDate
from datus.tools import BaseTool
from datus.utils.loggings import get_logger
from datus.utils.time_utils import get_default_current_date

logger = get_logger(__name__)


class DateParserTool(BaseTool):
    """Tool for parsing temporal expressions in text using LLM."""

    tool_name = "date_parser_tool"
    tool_description = "Tool for extracting and parsing temporal expressions from natural language"

    def __init__(self, language: str = "en", **kwargs):
        super().__init__(**kwargs)
        self.language = language

    def execute(self, input_param: DateParserInput, model: LLMBaseModel) -> DateParserResult:
        """Execute date parsing operations."""
        return self._parse_temporal_expressions(input_param, model)

    def _parse_temporal_expressions(self, input_param: DateParserInput, model: LLMBaseModel) -> DateParserResult:
        """Core date parsing logic."""
        try:
            # Extract and parse temporal expressions
            extracted_dates = self.extract_and_parse_dates(
                text=input_param.sql_task.task,
                current_date=get_default_current_date(input_param.sql_task.current_date),
                model=model,
            )

            # Generate date context for SQL generation
            date_context = self.generate_date_context(extracted_dates)

            # Create enriched task with date information
            enriched_task_data = input_param.sql_task.model_dump()

            # Store date ranges directly in sql_task.date_ranges
            if date_context:
                enriched_task_data["date_ranges"] = date_context
                # Also add to external knowledge for backward compatibility
                if enriched_task_data.get("external_knowledge"):
                    enriched_task_data["external_knowledge"] += f"\n\n{date_context}"
                else:
                    enriched_task_data["external_knowledge"] = date_context

            from datus.schemas.node_models import SqlTask

            enriched_task = SqlTask.model_validate(enriched_task_data)

            logger.info(f"Date parsing completed: {len(extracted_dates)} expressions found")

            return DateParserResult(
                success=True, extracted_dates=extracted_dates, enriched_task=enriched_task, date_context=date_context
            )

        except Exception as e:
            logger.error(f"Date parsing execution error: {str(e)}")
            return DateParserResult(
                success=False, error=str(e), extracted_dates=[], enriched_task=input_param.sql_task, date_context=""
            )

    def extract_and_parse_dates(
        self, text: str, current_date: Optional[str] = None, model: LLMBaseModel = None
    ) -> List[ExtractedDate]:
        """
        Extract temporal expressions from text and parse them using LLM.
        Support both English and Chinese temporal expressions.

        Args:
            text: The text to analyze for temporal expressions
            current_date: Reference date for relative expressions (YYYY-MM-DD format)
            model: LLM model for parsing

        Returns:
            List of ExtractedDate objects with parsed date information
        """
        try:
            # Step 1: Use LLM to extract temporal expressions
            extraction_prompt = get_date_extraction_prompt(text)
            logger.debug(f"Date extraction prompt: {extraction_prompt}")

            # Get LLM response
            llm_response = model.generate_with_json_output(extraction_prompt)
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
                parsed_date = self.parse_temporal_expression(expr, reference_date, model)
                if parsed_date:
                    parsed_dates.append(parsed_date)

            logger.info(f"Successfully parsed {len(parsed_dates)} temporal expressions")
            return parsed_dates

        except Exception as e:
            logger.error(f"Error in date extraction and parsing: {str(e)}")
            return []

    def parse_temporal_expression(
        self, expression: Dict[str, Any], reference_date: datetime, model: LLMBaseModel
    ) -> Optional[ExtractedDate]:
        """
        Parse temporal expression using LLM.

        Args:
            expression: Dictionary containing the temporal expression info
            reference_date: Reference datetime for relative expressions
            model: LLM model for parsing

        Returns:
            ExtractedDate object or None if parsing fails
        """
        original_text = expression.get("original_text", "")
        date_type = expression.get("date_type", "relative")
        confidence = expression.get("confidence", 1.0)

        logger.debug(f"Parsing '{original_text}' using LLM")

        result = self.parse_with_llm(original_text, reference_date, model)
        if result:
            start_date, end_date = result
            return self.create_extracted_date(original_text, date_type, confidence, start_date, end_date)

        logger.warning(f"LLM parsing failed for: '{original_text}'")
        return None

    def parse_with_llm(
        self, text: str, reference_date: datetime, model: LLMBaseModel
    ) -> Optional[Tuple[datetime, datetime]]:
        """Parse temporal expressions using LLM."""
        response = None
        try:
            prompt = prompt_manager.render_template(
                f"date_parser_{self.language}",
                version="1.0",
                text=text,
                reference_date=reference_date,
            )

            response = model.generate_with_json_output(prompt)
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
            logger.error(f"LLM parsing failed for '{text}': {e}")
            if response is not None:
                logger.error(f"LLM response was: {response}")
                logger.error(f"Response type: {type(response)}")

        return None

    def create_extracted_date(
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
