# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Permission configuration models for the unified permission system.

Provides:
- PermissionLevel: Enum for allow/deny/ask states
- PermissionRule: Single permission rule with pattern matching
- PermissionConfig: Complete permission configuration
"""

import fnmatch
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PermissionLevel(str, Enum):
    """Permission levels for tool access control.

    - ALLOW: Execute immediately without user intervention
    - DENY: Block execution and hide from available list (LLM never sees it)
    - ASK: Prompt user for confirmation before execution
    """

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionRule(BaseModel):
    """Single permission rule with pattern matching.

    Attributes:
        tool: Tool category (e.g., "db_tools", "mcp", "skills", "*" for all)
        pattern: Pattern within tool category (e.g., "execute_sql", "filesystem_mcp.*", "*")
        permission: Permission level to apply when matched
    """

    tool: str = Field(..., description="Tool category: db_tools, mcp, skills, * for all")
    pattern: str = Field(..., description="Pattern within tool: execute_sql, *, or glob pattern")
    permission: PermissionLevel = Field(..., description="Permission level: allow, deny, or ask")

    class Config:
        use_enum_values = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PermissionRule":
        """Create PermissionRule from dictionary."""
        return cls(
            tool=data.get("tool", "*"),
            pattern=data.get("pattern", "*"),
            permission=PermissionLevel(data.get("permission", "allow")),
        )

    def matches(self, tool_category: str, tool_name: str) -> bool:
        """Check if this rule matches the given tool category and name.

        Uses glob-style pattern matching for both tool category and tool name.

        Args:
            tool_category: Tool category (e.g., "db_tools", "skills", "mcp")
            tool_name: Name of the specific tool (e.g., "execute_sql", "sql-optimization")

        Returns:
            True if this rule matches, False otherwise
        """
        # Check tool category match
        if self.tool != "*" and not fnmatch.fnmatch(tool_category, self.tool):
            return False

        # Check tool name pattern match
        if self.pattern != "*" and not fnmatch.fnmatch(tool_name, self.pattern):
            return False

        return True


class PermissionConfig(BaseModel):
    """Unified permission configuration for all tools, MCP, and skills.

    The permission system evaluates rules in order, with later rules overriding
    earlier ones (last match wins). However, DENY always takes precedence over
    ALLOW at the same specificity level.

    Example configuration in agent.yml:
        permissions:
          default: allow
          rules:
            - tool: db_tools
              pattern: execute_sql
              permission: ask
            - tool: skills
              pattern: dangerous-*
              permission: deny
            - tool: "*"
              pattern: "*"
              permission: allow

    Attributes:
        rules: List of permission rules evaluated in order
        default_permission: Default permission when no rules match
    """

    rules: List[PermissionRule] = Field(default_factory=list, description="Permission rules evaluated in order")
    default_permission: PermissionLevel = Field(
        default=PermissionLevel.ALLOW, description="Default permission when no rules match"
    )

    class Config:
        use_enum_values = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PermissionConfig":
        """Create PermissionConfig from dictionary (agent.yml format).

        Accepts both 'default' and 'default_permission' keys for flexibility.

        Args:
            data: Dictionary with 'default'/'default_permission' and 'rules' keys

        Returns:
            PermissionConfig instance
        """
        if not data:
            return cls()

        # Accept both 'default' and 'default_permission' keys
        default = data.get("default_permission", data.get("default", "allow"))
        rules_data = data.get("rules", [])

        rules = [PermissionRule.from_dict(r) for r in rules_data]

        return cls(
            default_permission=PermissionLevel(default),
            rules=rules,
        )

    def merge_with(self, override: Optional["PermissionConfig"]) -> "PermissionConfig":
        """Merge with another config, with override rules taking precedence.

        Used for combining global config with node-specific overrides.

        Args:
            override: Node-specific permission config to merge

        Returns:
            New PermissionConfig with merged rules
        """
        if not override:
            return self

        # Override rules are appended (evaluated later, thus higher priority)
        merged_rules = self.rules + override.rules

        # Always use override's default_permission when override is provided
        # This allows overriding just the default without adding rules
        return PermissionConfig(
            default_permission=override.default_permission,
            rules=merged_rules,
        )
