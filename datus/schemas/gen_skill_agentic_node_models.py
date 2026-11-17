# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Schema models for Skill Creator Agentic Node.

This module defines the input and output models for the SkillCreatorAgenticNode,
providing structured validation for skill creation, editing, and review workflows.
"""

from typing import Literal, Optional

from pydantic import Field

from datus.schemas.at_context import AtContextInput
from datus.schemas.base import BaseResult


class SkillCreatorNodeInput(AtContextInput):
    """
    Input model for SkillCreatorAgenticNode interactions.
    """

    user_message: str = Field(..., description="User request describing the skill to create, edit, or review")
    storage_location: Optional[Literal["project", "user"]] = Field(
        default=None, description="Storage location: 'project' (./.datus/skills/) or 'user' (~/.datus/skills/)"
    )
    target_skill: Optional[str] = Field(default=None, description="Name of an existing skill to edit or review")


class SkillCreatorNodeResult(BaseResult):
    """
    Result model for SkillCreatorAgenticNode interactions.
    """

    response: str = Field(default="", description="Skill authoring summary or review findings (plain text)")
    skill_name: Optional[str] = Field(default=None, description="Name of the created or edited skill")
    skill_path: Optional[str] = Field(default=None, description="Full path to the skill directory")
    tokens_used: int = Field(default=0, description="Total tokens used in this interaction")
