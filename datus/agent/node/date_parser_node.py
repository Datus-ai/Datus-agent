from typing import AsyncGenerator, Dict, Optional

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.date_parser_node_models import DateParserInput, DateParserResult
from datus.schemas.node_models import SqlTask
from datus.tools.date_tools import DateParsingTool
from datus.utils.loggings import get_logger
from datus.utils.time_utils import get_default_current_date

logger = get_logger(__name__)


class DateParserNode(Node):
    """Node for parsing temporal expressions in SQL tasks."""

    def execute(self):
        """Execute date parsing."""
        self.result = self._execute_date_parsing()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute date parsing with streaming support."""
        async for action in self._date_parsing_stream(action_history_manager):
            yield action

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
            # Create date parsing tool
            date_tool = DateParsingTool(self.model)

            # Extract and parse temporal expressions
            extracted_dates = date_tool.extract_and_parse_dates(
                text=self.input.sql_task.task, current_date=get_default_current_date(self.input.sql_task.current_date)
            )

            # Generate date context for SQL generation
            date_context = date_tool.generate_date_context(extracted_dates)

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

    async def _date_parsing_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Date parsing with streaming support and action history tracking."""
        if not self.model:
            logger.error("Model not available for date parsing")
            return

        try:
            # Date parsing preparation action
            prep_action = ActionHistory(
                action_id="date_parsing_prep",
                role=ActionRole.WORKFLOW,
                messages="Preparing date parsing for temporal expressions in the query",
                action_type="date_preparation",
                input={
                    "task": self.input.sql_task.task if hasattr(self.input, "sql_task") else "",
                    "current_date": get_default_current_date(self.input.sql_task.current_date)
                    if hasattr(self.input, "sql_task")
                    else None,
                },
                status=ActionStatus.PROCESSING,
            )
            yield prep_action

            # Date extraction action
            extraction_action = ActionHistory(
                action_id="date_extraction",
                role=ActionRole.WORKFLOW,
                messages="Extracting temporal expressions using LLM",
                action_type="date_extraction",
                input={
                    "query_text": self.input.sql_task.task if hasattr(self.input, "sql_task") else "",
                },
                status=ActionStatus.PROCESSING,
            )
            yield extraction_action

            # Execute date parsing - reuse existing logic
            try:
                result = self._execute_date_parsing()

                # Update preparation action
                prep_action.status = ActionStatus.SUCCESS
                prep_action.output = {
                    "preparation_complete": True,
                    "model_ready": True,
                }

                # Update extraction action
                extraction_action.status = ActionStatus.SUCCESS
                extraction_action.output = {
                    "success": result.success,
                    "extracted_count": len(result.extracted_dates),
                    "has_date_context": bool(result.date_context),
                    "expressions": [date.original_text for date in result.extracted_dates]
                    if result.extracted_dates
                    else [],
                }

                # Store result for later use
                self.result = result

            except Exception as e:
                prep_action.status = ActionStatus.FAILED
                prep_action.output = {"error": str(e)}

                extraction_action.status = ActionStatus.FAILED
                extraction_action.output = {"error": str(e)}
                logger.error(f"Date parsing error: {str(e)}")
                raise

            # Yield the updated actions with final status
            yield extraction_action

        except Exception as e:
            logger.error(f"Date parsing streaming error: {str(e)}")
            raise
