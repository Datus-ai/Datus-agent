# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

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
            from datus.utils.time_utils import get_default_current_date

            # Extract dates using DateParserTool
            tool = DateParserTool(language=self._get_language_setting())
            task_text = self.input.sql_task.task
            current_date = get_default_current_date(self.input.sql_task.current_date)
            extracted_dates = tool.execute(task_text, current_date, self.model)

            # Generate date context for SQL generation
            date_context = tool.generate_date_context(extracted_dates)

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

            from datus.schemas.node_models import SqlTask

            enriched_task = SqlTask.model_validate(enriched_task_data)

            logger.info(f"Date parsing result: success with {len(extracted_dates)} expressions")
            return DateParserResult(
                success=True, extracted_dates=extracted_dates, enriched_task=enriched_task, date_context=date_context
            )
        except Exception as e:
            logger.error(f"Date parsing tool execution failed: {e}")
            return DateParserResult(
                success=False, error=str(e), extracted_dates=[], enriched_task=self.input.sql_task, date_context=""
            )

    async def execute_stream(self, action_history_manager=None):
        """Empty streaming implementation - not needed for date parsing."""
        yield
        return
