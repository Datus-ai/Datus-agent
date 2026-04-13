# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SkillValidatorTarget(BaseModel):
    """Selector describing where a skill-provided validator applies."""

    node: Optional[str] = Field(default=None, description="Target node name")


class SkillValidatorSpec(BaseModel):
    """Machine-readable validator declaration loaded from a skill asset."""

    id: str = Field(..., description="Stable validator id")
    type: str = Field(..., description="Validator implementation type")
    applies_to: SkillValidatorTarget = Field(default_factory=SkillValidatorTarget)
    requires_context: List[str] = Field(default_factory=list, description="Required context keys")
    config: Dict[str, Any] = Field(default_factory=dict, description="Validator-specific config")
