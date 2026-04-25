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
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from datus.validation.report import TargetFilter


class SkillConfig(BaseModel):
    """Global skills configuration from agent.yml.

    Configures where to discover skills and global skill behavior.

    Example configuration:
        skills:
          directories:
            - ./.datus/skills
            - ~/.datus/skills
          warn_duplicates: true
          whitelist_from_compaction: true

    Attributes:
        directories: List of directories to scan for skills. Project-level
            directories (``./.datus/skills``) take precedence over the global
            fallback (``~/.datus/skills``).
        warn_duplicates: Warn when duplicate skill names are found
        whitelist_from_compaction: Preserve skill content during session compaction
    """

    directories: List[str] = Field(
        default_factory=lambda: ["./.datus/skills", "~/.datus/skills"],
        description="Directories to scan for SKILL.md files (project-level first, global fallback)",
    )
    warn_duplicates: bool = Field(default=True, description="Warn on duplicate skill names")
    whitelist_from_compaction: bool = Field(
        default=True, description="Preserve skill responses during session compaction"
    )
    # Marketplace settings
    marketplace_url: str = Field(default="http://localhost:9000", description="Town backend URL for skill marketplace")
    auto_sync: bool = Field(default=False, description="Auto-sync promoted skills on startup")
    install_dir: str = Field(default="~/.datus/skills", description="Directory for marketplace-installed skills")

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
            marketplace_url=data.get("marketplace_url", "http://localhost:9000"),
            auto_sync=data.get("auto_sync", False),
            install_dir=data.get("install_dir", "~/.datus/skills"),
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
        allowed_agents:
          - gen_dashboard
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
        allowed_agents: Node names (from ``AgenticNode.get_node_name()``) allowed
            to see and load this skill. Empty list means no restriction — every
            agent can see it.
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
    # Agent scoping: empty list == no restriction; non-empty == whitelist of node names
    allowed_agents: List[str] = Field(
        default_factory=list,
        description="Agent node names allowed to see/load this skill; empty = unrestricted",
    )
    # Subagent execution
    context: Optional[str] = Field(default=None, description="'fork' to run in isolated subagent")
    agent: Optional[str] = Field(default=None, description="Subagent type when context=fork")

    # Validator skill extensions (ValidationHook infrastructure)
    # Skills with kind="validator" are NOT injected into the main agent's
    # prompt via SkillFuncTool — they are consumed exclusively by
    # ValidationHook which fires them at run end on matching targets.
    kind: Literal["skill", "validator"] = Field(
        default="skill",
        description="'skill' (default) is loaded by the main agent; 'validator' is driven by ValidationHook",
    )
    severity: Literal["blocking", "advisory", "off"] = Field(
        default="advisory",
        description="Blocking drives retry via on_end final_report; advisory reports only; off disables the validator",
    )
    mode: Literal["llm"] = Field(
        default="llm",
        description="Execution mode for the validator (future: 'declarative'); only 'llm' supported in current PR",
    )
    targets: List[TargetFilter] = Field(
        default_factory=list,
        description="Per-target filters (empty = match all); any matching filter activates the validator",
    )

    # Marketplace metadata
    license: Optional[str] = Field(default=None, description="License identifier (e.g. Apache-2.0)")
    compatibility: Optional[Dict[str, Any]] = Field(default=None, description="Compatibility map")
    source: Optional[str] = Field(default=None, description="'local' or 'marketplace'")
    marketplace_version: Optional[str] = Field(default=None, description="Version from marketplace")

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

        # Validator fields: parsed with pydantic's TargetFilter so YAML "schema"
        # alias flows through to db_schema correctly.
        raw_targets = frontmatter.get("targets", []) or []
        parsed_targets: List[TargetFilter] = []
        for t in raw_targets:
            if isinstance(t, TargetFilter):
                parsed_targets.append(t)
            elif isinstance(t, dict):
                parsed_targets.append(TargetFilter.model_validate(t))
            else:
                from datus.utils.exceptions import DatusException, ErrorCode

                raise DatusException(
                    ErrorCode.SKILL_FRONTMATTER_INVALID,
                    message_args={
                        "location": str(location),
                        "error_message": f"invalid target entry (expected dict): {t!r}",
                    },
                )

        # YAML parses bare ``off`` / ``on`` as booleans (False / True). That
        # collides with our ``severity: off`` spelling — coerce back to string
        # so skill authors can write ``severity: off`` unquoted.
        raw_severity = frontmatter.get("severity", "advisory")
        if raw_severity is False:
            raw_severity = "off"
        elif raw_severity is True:
            raw_severity = "on"  # not a valid enum value — pydantic will flag it

        return cls(
            name=name,
            description=description,
            location=location,
            tags=frontmatter.get("tags", []),
            version=frontmatter.get("version"),
            allowed_commands=frontmatter.get("allowed_commands", []),
            disable_model_invocation=frontmatter.get("disable_model_invocation", False),
            user_invocable=frontmatter.get("user_invocable", True),
            allowed_agents=frontmatter.get("allowed_agents", []),
            context=frontmatter.get("context"),
            agent=frontmatter.get("agent"),
            kind=frontmatter.get("kind", "skill"),
            severity=raw_severity,
            mode=frontmatter.get("mode", "llm"),
            targets=parsed_targets,
            license=frontmatter.get("license"),
            compatibility=frontmatter.get("compatibility"),
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

    def is_allowed_for(self, *node_names: Optional[str]) -> bool:
        """Check whether an agent is allowed to see this skill.

        Empty ``allowed_agents`` means no restriction. A non-empty list is a
        whitelist matched against *any* of the supplied identifiers — callers
        typically pass both the node alias (``get_node_name()``) and the
        canonical class name (``get_node_class_name()``) so that a custom
        subagent alias (e.g. ``my_dashboard`` with ``node_class: gen_dashboard``)
        still matches a whitelist written in terms of the class name.

        Args:
            *node_names: One or more identifiers to test against. ``None`` /
                empty values are ignored.

        Returns:
            True if the skill has no scoping or any provided identifier is
            whitelisted.
        """
        if not self.allowed_agents:
            return True
        return any(name in self.allowed_agents for name in node_names if name)

    def is_validator(self) -> bool:
        """Return True if this skill is a validator driven by ValidationHook.

        Validator skills are excluded from the main agent's available-skills
        list and are invoked exclusively by ``ValidationHook``.
        """
        return self.kind == "validator"

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
            "allowed_agents": self.allowed_agents,
            "context": self.context,
            "agent": self.agent,
            "kind": self.kind,
            "severity": self.severity,
            "mode": self.mode,
            "targets": [t.model_dump(by_alias=True, exclude_none=True) for t in self.targets],
            "license": self.license,
            "compatibility": self.compatibility,
            "source": self.source,
            "marketplace_version": self.marketplace_version,
        }
