# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Optional

from pydantic import BaseModel, Field


class ExtractedDate(BaseModel):
    """Extracted and normalized temporal expression."""

    original_text: str = Field(description="Original text containing the date reference")
    parsed_date: Optional[str] = Field(description="Parsed absolute date in YYYY-MM-DD format")
    start_date: Optional[str] = Field(description="Start date for date ranges in YYYY-MM-DD format")
    end_date: Optional[str] = Field(description="End date for date ranges in YYYY-MM-DD format")
    date_type: str = Field(description="Type of date: 'specific', 'range', 'relative'")
    confidence: float = Field(default=1.0, description="Confidence score of the parsing (0.0-1.0)")
