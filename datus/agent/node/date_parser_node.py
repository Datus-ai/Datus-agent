from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.prompts.extract_dates import get_date_extraction_prompt, parse_date_extraction_response
from datus.prompts.prompt_manager import prompt_manager
from datus.schemas.date_parser_node_models import DateParserInput, DateParserResult, ExtractedDate
from datus.schemas.node_models import SqlTask
from datus.utils.loggings import get_logger
from datus.utils.time_utils import get_default_current_date

logger = get_logger(__name__)


class DateParserNode(Node):
    """Node for parsing temporal expressions in SQL tasks."""

    def _get_language_setting(self) -> str:
        """Get the language setting from agent config."""
        if self.agent_config and hasattr(self.agent_config, "nodes"):
            nodes_config = self.agent_config.nodes
            if "date_parser" in nodes_config:
                date_parser_config = nodes_config["date_parser"]
                # Check if language is in the input attribute of NodeConfig
                if hasattr(date_parser_config, "input") and hasattr(date_parser_config.input, "language"):
                    return date_parser_config.input.language
        return "en"

    def execute(self):
        """Execute date parsing."""
        self.result = self._execute_date_parsing()

    def setup_input(self, workflow: Workflow) -> Dict:
        """Setup input for date parsing node."""
        next_input = DateParserInput(sql_task=workflow.task)
        self.input = next_input
        return {"success": True, "message": "Date parser input setup complete", "suggestions": [next_input]}

    def update_context(self, workflow: Workflow) -> Dict:
        """Update workflow context with parsed date information."""
        result = self.result
        try:
            if result and result.success:
                # Update the workflow task with enriched information
                workflow.task = result.enriched_task

                # Add date context to workflow for later nodes to use
                if not hasattr(workflow, "date_context"):
                    workflow.date_context = result.date_context
                else:
                    # Append to existing context
                    if workflow.date_context:
                        workflow.date_context += "\n\n" + result.date_context
                    else:
                        workflow.date_context = result.date_context

                logger.info(f"Updated workflow with {len(result.extracted_dates)} parsed dates")
                return {
                    "success": True,
                    "message": f"Updated context with {len(result.extracted_dates)} parsed temporal expressions",
                }
            else:
                logger.warning("Date parsing failed, continuing with original task")
                return {"success": True, "message": "Date parsing failed, continuing with original task"}

        except Exception as e:
            logger.error(f"Failed to update date parsing context: {str(e)}")
            return {"success": False, "message": f"Date parsing context update failed: {str(e)}"}

    def _execute_date_parsing(self) -> DateParserResult:
        """Execute date parsing action."""
        if not self.model:
            return DateParserResult(
                success=False,
                error="Date parsing model not provided",
                extracted_dates=[],
                enriched_task=self.input.sql_task,
                date_context="",
            )

        try:
            # Extract and parse temporal expressions
            extracted_dates = self._extract_and_parse_dates(
                text=self.input.sql_task.task, current_date=get_default_current_date(self.input.sql_task.current_date)
            )

            # Generate date context for SQL generation
            date_context = self._generate_date_context(extracted_dates)

            # Create enriched task with date information
            enriched_task_data = self.input.sql_task.model_dump()

            # Store date ranges directly in sql_task.date_ranges
            if date_context:
                enriched_task_data["date_ranges"] = date_context
                # Also add to external knowledge for backward compatibility
                if enriched_task_data.get("external_knowledge"):
                    enriched_task_data["external_knowledge"] += f"\n\n{date_context}"
                else:
                    enriched_task_data["external_knowledge"] = date_context

            enriched_task = SqlTask.model_validate(enriched_task_data)

            logger.info(f"Date parsing completed: {len(extracted_dates)} expressions found")

            return DateParserResult(
                success=True, extracted_dates=extracted_dates, enriched_task=enriched_task, date_context=date_context
            )

        except Exception as e:
            logger.error(f"Date parsing execution error: {str(e)}")
            return DateParserResult(
                success=False, error=str(e), extracted_dates=[], enriched_task=self.input.sql_task, date_context=""
            )

    def _extract_and_parse_dates(self, text: str, current_date: Optional[str] = None) -> List[ExtractedDate]:
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
            llm_response = self.model.generate_with_json_output(extraction_prompt)
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
        response = None
        try:
            prompt = prompt_manager.render_template(
                f"date_parser_{self._get_language_setting()}",
                version="1.0",
                text=text,
                reference_date=reference_date,
            )

            response = self.model.generate_with_json_output(prompt)
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

    def _generate_date_context(self, extracted_dates: List[ExtractedDate]) -> str:
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

    async def execute_stream(self, action_history_manager=None):
        """Empty streaming implementation - not needed for date parsing."""
        return
        yield
