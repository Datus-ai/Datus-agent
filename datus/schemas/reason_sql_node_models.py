# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import List, Optional  # noqa: F401

from pydantic import Field

from datus.schemas.node_models import ExecuteSQLResult, GenerateSQLInput, SQLContext


# Reuse the GenerateSQLInput and ExecuteSQLResult for reasoning node
class ReasoningInput(GenerateSQLInput):
    """
    Input model for reasoning node.
    Validates the input for reasoning.
    """

    max_table_schemas_length: int = Field(default=4000, description="Max table schemas length")
    max_data_details_length: int = Field(default=2000, description="Max data details length")
    max_context_length: int = Field(default=8000, description="Max context length")
    max_value_length: int = Field(default=500, description="Max value length")
    max_sql_return_length: int = Field(default=1000, description="Max SQL return length")
    max_text_mark_length: int = Field(default=16, description="Max text mark length")
    prompt_version: Optional[str] = Field(default=None, description="Version for prompt")
    max_turns: int = Field(default=10, description="Maximum number of conversation turns")


class ReasoningResult(ExecuteSQLResult):
    """
    Result model for reasoning node.
    """

    sql_contexts: List[SQLContext] = Field(default_factory=list, description="The SQL context")
