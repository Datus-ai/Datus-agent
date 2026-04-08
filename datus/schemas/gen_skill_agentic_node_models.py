# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Schema models for Skill Creator Agentic Node.

This module defines the input and output models for the SkillCreatorAgenticNode,
providing structured validation for interactive skill creation, editing, and
evaluation workflows.
"""

from typing import Literal, Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult


class SkillCreatorNodeInput(BaseInput):
    """
    Input model for SkillCreatorAgenticNode interactions.
    """

    user_message: str = Field(..., description="User request describing the skill to create or edit")
    storage_location: Optional[Literal["project", "user"]] = Field(
        default=None, description="Storage location: 'project' (./skills/) or 'user' (~/.datus/skills/)"
    )
    target_skill: Optional[str] = Field(default=None, description="Name of existing skill to edit (edit mode)")


class SkillCreatorNodeResult(BaseResult):
    """
    Result model for SkillCreatorAgenticNode interactions.
    """

    response: str = Field(default="", description="Skill creation result summary (plain text)")
    skill_name: Optional[str] = Field(default=None, description="Name of the created or edited skill")
    skill_path: Optional[str] = Field(default=None, description="Full path to the skill directory")
    tokens_used: int = Field(default=0, description="Total tokens used in this interaction")
