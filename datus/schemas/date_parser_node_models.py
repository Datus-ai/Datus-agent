# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import SqlTask


class ExtractedDate(BaseModel):
    """Model for extracted date information."""

    original_text: str = Field(description="Original text containing the date reference")
    parsed_date: Optional[str] = Field(description="Parsed absolute date in YYYY-MM-DD format")
    start_date: Optional[str] = Field(description="Start date for date ranges in YYYY-MM-DD format")
    end_date: Optional[str] = Field(description="End date for date ranges in YYYY-MM-DD format")
    date_type: str = Field(description="Type of date: 'specific', 'range', 'relative'")
    confidence: float = Field(default=1.0, description="Confidence score of the parsing (0.0-1.0)")


class DateParserInput(BaseInput):
    """Input model for date parser node."""

    sql_task: SqlTask = Field(description="The SQL task containing the query to parse")
    language: Optional[str] = Field(default="en", description="Language for date parsing (en/cn)")


class DateParserResult(BaseResult):
    """Result model for date parser node."""

    extracted_dates: List[ExtractedDate] = Field(
        default_factory=list, description="List of extracted and parsed dates from the task"
    )
    enriched_task: SqlTask = Field(description="SQL task enriched with parsed date information")
    date_context: str = Field(default="", description="Additional context about parsed dates for SQL generation")

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "error": self.error,
            "extracted_dates": [date.model_dump() for date in self.extracted_dates],
            "enriched_task": self.enriched_task.model_dump(),
            "date_context": self.date_context,
        }
