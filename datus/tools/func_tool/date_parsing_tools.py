# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
from typing import List, Optional

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.tools.date_tools import DateParserTool
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DateParsingTools:
    """Function tool wrapper for date parsing operations."""

    def __init__(self, agent_config: AgentConfig, model: LLMBaseModel):
        self.agent_config = agent_config
        self.model = model
        self.reference_date: Optional[str] = None
        self.date_parser_tool = DateParserTool(language=self._get_language_setting())

    def set_reference_date(self, reference_date: Optional[str]):
        """
        Set reference date for test datasets.

        When running against test datasets, set this to the dataset's reference date
        so that relative time expressions are evaluated against the correct time point.

        Args:
            reference_date: Reference date in YYYY-MM-DD format.
        """
        self.reference_date = reference_date

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
            trans_to_function_tool(self.get_current_date),
        ]

    def parse_temporal_expressions(
        self,
        task_text: str,
        current_date: Optional[str] = None,
    ) -> FuncToolResult:
        """
        Extract and parse temporal expressions from natural language text.

        Converts relative dates (e.g., "last month", "Q1 2024", "yesterday") into absolute date ranges.
        Supports both English and Chinese temporal expressions.

        Args:
            task_text: Text containing temporal expressions
            current_date: Reference date in YYYY-MM-DD format (defaults to today)

        Returns:
            dict with 'success', 'error', and 'result' containing extracted dates and date context
        """
        try:
            from datus.utils.time_utils import get_default_current_date

            # Extract dates using DateParserTool
            normalized_current_date = get_default_current_date(current_date)
            extracted_dates = self.date_parser_tool.execute(task_text, normalized_current_date, self.model)

            # Generate date context
            date_context = self.date_parser_tool.generate_date_context(extracted_dates)

            return FuncToolResult(
                success=1,
                error=None,
                result={
                    "extracted_dates": [date.model_dump() for date in extracted_dates],
                    "date_context": date_context,
                },
            )

        except Exception as e:
            logger.error(f"Failed to parse temporal expressions for text '{task_text}': {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def get_current_date(self) -> FuncToolResult:
        """
        Get the current date or configured reference date.

        When running against test datasets, returns the configured reference date
        instead of the real current date. This ensures date-relative queries
        (like "last month", "yesterday") are evaluated against the dataset's time point.

        Returns:
            dict with 'success', 'error', and 'result' containing:
            - current_date: The date in YYYY-MM-DD format
            - is_reference_date: True if using configured reference date, False if using real date
        """
        try:
            from datus.utils.time_utils import get_default_current_date

            current_date = get_default_current_date(self.reference_date)

            return FuncToolResult(
                success=1,
                error=None,
                result={
                    "current_date": current_date,
                    "is_reference_date": self.reference_date is not None,
                },
            )

        except Exception as e:
            logger.error(f"Failed to get current date: {str(e)}")
            return FuncToolResult(success=0, error=str(e))
