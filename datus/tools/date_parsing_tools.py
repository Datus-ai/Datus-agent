# -*- coding: utf-8 -*-
from typing import List, Optional

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.schemas.date_parser_node_models import DateParserInput
from datus.schemas.node_models import SqlTask
from datus.tools.date_tools import DateParserTool
from datus.tools.tools import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DateParsingTools:
    """Function tool wrapper for date parsing operations."""

    def __init__(self, agent_config: AgentConfig, model: LLMBaseModel):
        self.agent_config = agent_config
        self.model = model
        self.date_parser_tool = DateParserTool(language=self._get_language_setting())

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

    def available_tools(self) -> List[Tool]:
        """Get all available date parsing function tools."""
        return [
            trans_to_function_tool(self.parse_temporal_expressions),
        ]

    def parse_temporal_expressions(
        self,
        task_text: str,
        database_type: str = "",
        current_date: Optional[str] = None,
        domain: str = "",
        layer1: str = "",
        layer2: str = "",
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> FuncToolResult:
        """
        Parse temporal expressions from natural language text and enrich SQL task context.

        This tool extracts and parses date-related expressions from natural language queries,
        converting them into structured date information that can be used for SQL generation.

        Use this tool when you need to:
        - Extract dates from natural language queries (e.g., "last month", "Q1 2024", "yesterday")
        - Convert relative dates to absolute date ranges
        - Enrich SQL tasks with temporal context for better query generation
        - Support both English and Chinese temporal expressions

        Args:
            task_text: Natural language text containing temporal expressions (e.g., "sales data from last quarter",
                "user activity in the past 6 months", "revenue trends since January")
            database_type: Type of database for query generation context (e.g., "mysql", "postgresql", "snowflake")
            current_date: Reference date for relative expressions in YYYY-MM-DD format.
                If not provided, current system date will be used.
            domain: Business domain context for the task (e.g., "sales", "marketing", "finance")
            layer1: Primary semantic layer for categorization
            layer2: Secondary semantic layer for fine-grained categorization
            catalog_name: Target catalog name for SQL context
            database_name: Target database name for SQL context
            schema_name: Target schema name for SQL context

        Returns:
            dict: Date parsing results containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if parsing failed
                - 'result' (dict): Parsing results with:
                    - 'extracted_dates' (list): List of parsed temporal expressions with original text,
                        parsed dates, date types, and confidence scores
                    - 'date_context' (str): Formatted date context for SQL generation prompts
                    - 'enriched_task' (dict): Original task enriched with temporal information

        Examples:
            # Parse English temporal expressions
            parse_temporal_expressions("Show sales data from last quarter")

            # Parse Chinese temporal expressions
            parse_temporal_expressions("显示上个月的用户数据")

            # Parse with specific reference date
            parse_temporal_expressions("revenue from last week", current_date="2024-01-15")
        """
        try:
            # Create SQL task from input parameters
            sql_task = SqlTask(
                task=task_text,
                database_type=database_type,
                current_date=current_date,
                domain=domain,
                layer1=layer1,
                layer2=layer2,
                catalog_name=catalog_name,
                database_name=database_name,
                schema_name=schema_name,
            )

            # Create input for date parser tool
            date_parser_input = DateParserInput(sql_task=sql_task)

            # Execute date parsing
            result = self.date_parser_tool.execute(date_parser_input, self.model)

            if result.success:
                return FuncToolResult(
                    success=1,
                    error=None,
                    result={
                        "extracted_dates": [date.model_dump() for date in result.extracted_dates],
                        "date_context": result.date_context,
                        "enriched_task": result.enriched_task.model_dump(),
                    },
                )
            else:
                return FuncToolResult(success=0, error=result.error)

        except Exception as e:
            logger.error(f"Failed to parse temporal expressions for text '{task_text}': {str(e)}")
            return FuncToolResult(success=0, error=str(e))
