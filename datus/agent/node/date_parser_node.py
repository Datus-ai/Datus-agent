from typing import Dict

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.date_parser_node_models import DateParserInput, DateParserResult
from datus.tools.date_tools.date_parser import DateParserTool
from datus.utils.loggings import get_logger

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
        """Execute date parsing action using DateParserTool."""
        try:
            tool = DateParserTool(language=self._get_language_setting())
            result = tool.execute(self.input, self.model)
            logger.info(f"Date parsing result: {result.success}")
            return result
        except Exception as e:
            logger.error(f"Date parsing tool execution failed: {e}")
            return DateParserResult(
                success=False, error=str(e), extracted_dates=[], enriched_task=self.input.sql_task, date_context=""
            )

    async def execute_stream(self, action_history_manager=None):
        """Empty streaming implementation - not needed for date parsing."""
        return
        yield
