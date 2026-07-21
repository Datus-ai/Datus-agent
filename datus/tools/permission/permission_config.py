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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from datus.tools.permission.bash_rules import BashCommandRules


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


# Fine-grained statement kinds (``parse_sql_statement_kind``) mapped to the
# permission classes ``SqlStatementRules`` gates on. Destructive = statements
# that irreversibly mutate or remove existing data/schema; REPLACE deletes
# matched rows before re-inserting, and MERGE carries UPDATE/DELETE semantics
# in its WHEN MATCHED branches, so both count as destructive.
_SQL_KIND_TO_CLASS: Dict[str, str] = {
    "select": "read",
    "metadata": "read",
    "explain": "read",
    "insert": "write",
    "create": "write",
    "ddl": "write",
    "context_set": "write",
    "update": "destructive",
    "delete": "destructive",
    "merge": "destructive",
    "drop": "destructive",
    "truncate": "destructive",
    "alter": "destructive",
    "replace": "destructive",
    "unknown": "unknown",
}


def classify_sql_kind(kind: str) -> str:
    """Map a fine-grained statement kind to its permission class.

    Unmapped kinds fall to ``unknown`` (fail-safe: an unrecognized statement
    must never inherit a permissive class).
    """
    return _SQL_KIND_TO_CLASS.get(kind, "unknown")


class SqlStatementRules(BaseModel):
    """Per-class permission levels for ``db_tools.execute_sql``.

    ``execute_sql`` is the unified SQL entry point, so a single static rule
    cannot express "reads auto-allow, writes ask, destructive statements
    always confirm". This model carries one :class:`PermissionLevel` per
    statement class; the classes are resolved from the concrete statement by
    ``parse_sql_statement_kind`` + :func:`classify_sql_kind`.

    Configured per profile (``profiles.py``) and user-overridable under
    ``permissions.sql_statements`` in agent.yml:

        permissions:
          sql_statements:
            read: allow        # SELECT / SHOW / DESCRIBE / EXPLAIN
            write: ask         # INSERT, CREATE, COMMENT/GRANT/..., USE/SET
            destructive: ask   # UPDATE, DELETE, MERGE, DROP, TRUNCATE, ALTER, REPLACE
            unknown: ask       # unparseable SQL (fail-safe)
    """

    read: PermissionLevel = Field(
        default=PermissionLevel.ALLOW, description="SELECT / SHOW / DESCRIBE / EXPLAIN statements"
    )
    write: PermissionLevel = Field(
        default=PermissionLevel.ASK, description="INSERT, CREATE, benign DDL, and session statements"
    )
    destructive: PermissionLevel = Field(
        default=PermissionLevel.ASK,
        description="Irreversible statements: UPDATE, DELETE, MERGE, DROP, TRUNCATE, ALTER, REPLACE",
    )
    unknown: PermissionLevel = Field(default=PermissionLevel.ASK, description="Unparseable statements (fail-safe)")

    class Config:
        use_enum_values = True

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["SqlStatementRules"]:
        """Build from the ``permissions.sql_statements`` YAML block.

        Returns ``None`` when the block is absent so callers can distinguish
        "not configured" from "configured with defaults". Only keys present in
        the YAML are passed to the constructor so ``model_fields_set`` records
        which levels were explicit — ``merge_with`` relies on that to layer
        user overrides without clobbering profile defaults.
        """
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError(f"permissions.sql_statements must be a mapping, got {type(data).__name__}")
        kwargs: Dict[str, Any] = {}
        for field_name in ("read", "write", "destructive", "unknown"):
            if field_name in data:
                kwargs[field_name] = PermissionLevel(data[field_name])
        return cls(**kwargs)

    def merge_with(self, override: Optional["SqlStatementRules"]) -> "SqlStatementRules":
        """Layer another ruleset on top; explicit override fields win.

        Mirrors ``BashCommandRules.merge_with`` scalar semantics: a level from
        ``override`` replaces this one only when it was explicitly set
        (tracked via ``model_fields_set``).
        """
        if override is None:
            return self
        merged = self.model_copy()
        for field_name in ("read", "write", "destructive", "unknown"):
            if field_name in override.model_fields_set:
                setattr(merged, field_name, getattr(override, field_name))
        return merged

    def level_for_class(self, sql_class: str) -> PermissionLevel:
        """Permission level for a statement class; unmapped classes fail safe.

        ``use_enum_values`` stores fields as plain strings at runtime, so the
        value is coerced back to :class:`PermissionLevel` before returning.
        """
        value = getattr(self, sql_class, None)
        if value is None:
            value = self.unknown
        return PermissionLevel(value)


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
        bash_commands: Optional command-level bash rules (see ``bash_rules.py``);
            consumed by ``PermissionHooks._handle_bash_permission``. When None,
            the coarse ``bash_tools.bash`` rule governs as before.
        sql_statements: Optional per-class statement rules for ``execute_sql``;
            consumed by ``PermissionHooks._handle_sql_permission``. When None,
            the legacy read-only/ASK gating governs as before.
    """

    rules: List[PermissionRule] = Field(default_factory=list, description="Permission rules evaluated in order")
    default_permission: PermissionLevel = Field(
        default=PermissionLevel.ALLOW, description="Default permission when no rules match"
    )
    bash_commands: Optional["BashCommandRules"] = Field(
        default=None, description="Command-level allow/deny/ask rules for the bash tool"
    )
    sql_statements: Optional[SqlStatementRules] = Field(
        default=None, description="Per-class permission levels for execute_sql statements"
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

        # Deferred import: bash_rules imports PermissionLevel from this module.
        from datus.tools.permission.bash_rules import BashCommandRules

        return cls(
            default_permission=PermissionLevel(default),
            rules=rules,
            bash_commands=BashCommandRules.from_dict(data.get("bash_commands")),
            sql_statements=SqlStatementRules.from_dict(data.get("sql_statements")),
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

        # Bash command rules layer: lists concatenate, scalars override only
        # when explicitly set (see BashCommandRules.merge_with).
        if self.bash_commands is not None:
            merged_bash = self.bash_commands.merge_with(override.bash_commands)
        else:
            merged_bash = override.bash_commands

        # SQL statement rules layer: per-class levels override only when
        # explicitly set (see SqlStatementRules.merge_with).
        if self.sql_statements is not None:
            merged_sql = self.sql_statements.merge_with(override.sql_statements)
        else:
            merged_sql = override.sql_statements

        # Always use override's default_permission when override is provided
        # This allows overriding just the default without adding rules
        return PermissionConfig(
            default_permission=override.default_permission,
            rules=merged_rules,
            bash_commands=merged_bash,
            sql_statements=merged_sql,
        )


# Resolve the forward reference to BashCommandRules. bash_rules imports
# PermissionLevel from this module, so whichever module finishes importing
# second completes the rebuild: when this module loads first the import below
# succeeds; when bash_rules loads first the import raises (bash_rules is
# mid-initialization) and bash_rules performs the rebuild at its own bottom.
try:
    from datus.tools.permission.bash_rules import BashCommandRules

    PermissionConfig.model_rebuild()
except ImportError as _exc:  # pragma: no cover - depends on import order
    # Expected only for the circular-import case (bash_rules is
    # mid-initialization and performs the rebuild at its own bottom). Log so
    # a genuine import failure in bash_rules is not silently swallowed.
    from datus.utils.loggings import get_logger

    get_logger(__name__).debug("Deferring PermissionConfig.model_rebuild() to bash_rules: %s", _exc)
