# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult


class ParallelInput(BaseInput):
    """Input for parallel node execution"""

    child_nodes: List[str] = Field(description="List of child node types to execute in parallel")
    shared_input: Optional[Dict[str, Any]] = Field(default=None, description="Shared input data for all child nodes")


class ParallelResult(BaseResult):
    """Result from parallel node execution"""

    child_results: Dict[str, Any] = Field(description="Results from each child node execution")
    execution_order: List[str] = Field(description="Order in which child nodes completed")


class SelectionInput(BaseInput):
    """Input for selection node"""

    candidate_results: Dict[str, Any] = Field(description="Results from parallel nodes to select from")
    selection_criteria: Optional[str] = Field(default="best_quality", description="Criteria for selection")
    prompt_version: Optional[str] = Field(default=None, description="Version for prompt")


class SelectionResult(BaseResult):
    """Result from selection node"""

    selected_result: Any = Field(description="The selected best result")
    selected_source: str = Field(description="Source node ID of the selected result")
    selection_reason: str = Field(description="Reason for the selection")
    all_candidates: Dict[str, Any] = Field(description="All candidate results for reference")
    score_analysis: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Detailed scoring analysis")
