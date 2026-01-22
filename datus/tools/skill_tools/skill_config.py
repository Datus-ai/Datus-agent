# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Skill configuration models for AgentSkills integration.

Provides:
- SkillConfig: Global skills configuration from agent.yml
- SkillMetadata: Parsed metadata from SKILL.md frontmatter
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SkillConfig(BaseModel):
    """Global skills configuration from agent.yml.

    Configures where to discover skills and global skill behavior.

    Example configuration:
        skills:
          directories:
            - ~/.datus/skills
            - ./skills
            - ~/.claude/skills
          warn_duplicates: true
          whitelist_from_compaction: true

    Attributes:
        directories: List of directories to scan for skills
        warn_duplicates: Warn when duplicate skill names are found
        whitelist_from_compaction: Preserve skill content during session compaction
    """

    directories: List[str] = Field(
        default_factory=lambda: ["~/.datus/skills", "./skills", "~/.claude/skills"],
        description="Directories to scan for SKILL.md files",
    )
    warn_duplicates: bool = Field(default=True, description="Warn on duplicate skill names")
    whitelist_from_compaction: bool = Field(
        default=True, description="Preserve skill responses during session compaction"
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillConfig":
        """Create SkillConfig from dictionary (agent.yml format).

        Args:
            data: Dictionary with skills configuration

        Returns:
            SkillConfig instance
        """
        if not data:
            return cls()

        return cls(
            directories=data.get("directories", cls.model_fields["directories"].default_factory()),
            warn_duplicates=data.get("warn_duplicates", True),
            whitelist_from_compaction=data.get("whitelist_from_compaction", True),
        )


class SkillMetadata(BaseModel):
    """Metadata parsed from SKILL.md frontmatter.

    Represents a skill discovered from the filesystem. The content is lazily loaded
    only when the skill is actually used.

    Example SKILL.md frontmatter:
        ---
        name: sql-optimization
        description: SQL query optimization techniques
        tags: [sql, performance]
        version: 1.0.0
        allowed_commands:
          - "python:scripts/*.py"
          - "sh:*.sh"
        disable_model_invocation: false
        user_invocable: true
        context: fork
        agent: Explore
        ---

    Attributes:
        name: Unique skill name (required)
        description: Human-readable description (required)
        location: Path to the skill directory
        tags: Optional tags for categorization
        version: Optional version string
        allowed_commands: Patterns for allowed script execution (Claude Code compatible)
        disable_model_invocation: If true, only user can invoke via /skill-name
        user_invocable: If false, hidden from menu, only model invokes
        context: "fork" to run in isolated subagent
        agent: Subagent type when context=fork (Explore, Plan, general-purpose)
        content: Full SKILL.md content (lazy loaded)
    """

    name: str = Field(..., description="Unique skill name")
    description: str = Field(..., description="Human-readable description")
    location: Path = Field(..., description="Path to skill directory")
    tags: List[str] = Field(default_factory=list, description="Optional categorization tags")
    version: Optional[str] = Field(default=None, description="Optional version string")

    # Script execution control (Claude Code compatible)
    allowed_commands: List[str] = Field(
        default_factory=list,
        description="Patterns for allowed script execution (e.g., python:scripts/*.py)",
    )

    # Invocation control
    disable_model_invocation: bool = Field(default=False, description="If true, only user can invoke via /skill-name")
    user_invocable: bool = Field(default=True, description="If false, hidden from menu, only model invokes")

    # Subagent execution
    context: Optional[str] = Field(default=None, description="'fork' to run in isolated subagent")
    agent: Optional[str] = Field(default=None, description="Subagent type when context=fork")

    # Content (lazy loaded)
    content: Optional[str] = Field(default=None, description="Full SKILL.md content (lazy loaded)")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_frontmatter(cls, frontmatter: Dict[str, Any], location: Path) -> "SkillMetadata":
        """Create SkillMetadata from parsed YAML frontmatter.

        Args:
            frontmatter: Parsed YAML frontmatter dictionary
            location: Path to the skill directory

        Returns:
            SkillMetadata instance

        Raises:
            ValueError: If required fields (name, description) are missing
        """
        name = frontmatter.get("name")
        description = frontmatter.get("description")

        if not name:
            raise ValueError(f"Skill at {location} missing required 'name' field")
        if not description:
            raise ValueError(f"Skill at {location} missing required 'description' field")

        return cls(
            name=name,
            description=description,
            location=location,
            tags=frontmatter.get("tags", []),
            version=frontmatter.get("version"),
            allowed_commands=frontmatter.get("allowed_commands", []),
            disable_model_invocation=frontmatter.get("disable_model_invocation", False),
            user_invocable=frontmatter.get("user_invocable", True),
            context=frontmatter.get("context"),
            agent=frontmatter.get("agent"),
        )

    def has_scripts(self) -> bool:
        """Check if this skill has script execution capabilities.

        Returns:
            True if allowed_commands is non-empty
        """
        return len(self.allowed_commands) > 0

    def is_model_invocable(self) -> bool:
        """Check if the model can invoke this skill.

        Returns:
            True if model can invoke (disable_model_invocation is False)
        """
        return not self.disable_model_invocation

    def runs_in_subagent(self) -> bool:
        """Check if this skill runs in an isolated subagent.

        Returns:
            True if context is 'fork'
        """
        return self.context == "fork"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation (excluding content for efficiency)
        """
        return {
            "name": self.name,
            "description": self.description,
            "location": str(self.location),
            "tags": self.tags,
            "version": self.version,
            "allowed_commands": self.allowed_commands,
            "disable_model_invocation": self.disable_model_invocation,
            "user_invocable": self.user_invocable,
            "context": self.context,
            "agent": self.agent,
        }
