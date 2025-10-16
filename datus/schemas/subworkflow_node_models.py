# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Optional

from pydantic import Field

from datus.schemas.node_models import BaseInput, BaseResult


class SubworkflowInput(BaseInput):
    """Input model for Subworkflow node"""

    workflow_name: str = Field(..., description="Name of the workflow to execute as a subworkflow")
    node_params: Optional[Dict[str, Dict]] = Field(
        default=None, description="Parameters to override node configurations"
    )
    shared_input: Optional[Dict[str, Any]] = Field(default=None, description="Shared input data for the subworkflow")
    pass_context: bool = Field(
        default=True, description="Whether to pass the parent workflow context to the subworkflow"
    )


class SubworkflowResult(BaseResult):
    """Result model for Subworkflow node"""

    workflow_name: str = Field(..., description="Name of the executed workflow")
    node_results: Dict[str, Any] = Field(default_factory=dict, description="Results of each node in the subworkflow")
    execution_order: List[str] = Field(default_factory=list, description="Order of node execution")

    def compact_result(self) -> str:
        """Generate a compact string representation of the result"""
        if not self.success:
            return f"Subworkflow '{self.workflow_name}' failed: {self.error}"

        return f"Subworkflow '{self.workflow_name}' executed successfully with {len(self.node_results)} nodes"
