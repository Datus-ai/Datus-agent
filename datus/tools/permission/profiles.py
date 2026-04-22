# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Predefined permission profiles (normal / auto / dangerous).

A Permission Profile is a named base ``PermissionConfig`` that users can
select via ``agent.yml`` (``permissions.profile: <name>``) or switch to
at runtime with ``/profile``. User-defined ``permissions.rules`` are
layered on top via ``PermissionConfig.merge_with`` (last-match-wins).

The three profiles embody three security postures:

* ``normal``:    read-only tools allowed, all writes ASK, named
  destructive tools DENY. Default for new installs.
* ``auto``:      Normal + workspace writes auto-execute, BI/scheduler
  non-trigger writes auto, DB writes still ASK.
* ``dangerous``: everything ALLOW. Filesystem EXTERNAL paths still
  prompt via ``PathZone`` at the hook layer — that gate is orthogonal
  to the rule engine.
"""

from datus.tools.permission.permission_config import (
    PermissionConfig,
    PermissionLevel,
    PermissionRule,
)

PROFILE_NAMES: tuple[str, ...] = ("normal", "auto", "dangerous")


def _rule(tool: str, pattern: str, permission: PermissionLevel) -> PermissionRule:
    return PermissionRule(tool=tool, pattern=pattern, permission=permission)


# --- Normal ------------------------------------------------------------------
# default=ASK + explicit reads ALLOW + named destructives DENY + MCP/skills ASK.
_NORMAL_RULES = [
    # context search / date utilities
    _rule("context_search_tools", "*", PermissionLevel.ALLOW),
    _rule("date_parsing_tools", "*", PermissionLevel.ALLOW),
    # db read
    _rule("db_tools", "read_query", PermissionLevel.ALLOW),
    _rule("db_tools", "list_*", PermissionLevel.ALLOW),
    _rule("db_tools", "describe_*", PermissionLevel.ALLOW),
    _rule("db_tools", "get_*", PermissionLevel.ALLOW),
    # bi read + destructive deny
    _rule("bi_tools", "list_*", PermissionLevel.ALLOW),
    _rule("bi_tools", "get_*", PermissionLevel.ALLOW),
    _rule("bi_tools", "delete_*", PermissionLevel.DENY),
    # semantic read
    _rule("semantic_tools", "list_*", PermissionLevel.ALLOW),
    _rule("semantic_tools", "search_*", PermissionLevel.ALLOW),
    _rule("semantic_tools", "get_*", PermissionLevel.ALLOW),
    _rule("semantic_tools", "query_metrics", PermissionLevel.ALLOW),
    # scheduler read + destructive deny
    _rule("scheduler_tools", "list_*", PermissionLevel.ALLOW),
    _rule("scheduler_tools", "get_*", PermissionLevel.ALLOW),
    _rule("scheduler_tools", "delete_job", PermissionLevel.DENY),
    # filesystem read
    _rule("filesystem_tools", "read_*", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "list_*", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "directory_tree", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "search_files", PermissionLevel.ALLOW),
    # plan read
    _rule("tools", "todo_read", PermissionLevel.ALLOW),
    # mcp + skills: ASK (explicit so future defaults can shift default_permission)
    _rule("mcp.*", "*", PermissionLevel.ASK),
    _rule("skills", "*", PermissionLevel.ASK),
]

NORMAL = PermissionConfig(
    default_permission=PermissionLevel.ASK,
    rules=_NORMAL_RULES,
)

# --- Auto --------------------------------------------------------------------
# Normal's rules + workspace writes + BI create/update + scheduler non-trigger.
# DB writes remain ASK (no env detection in MVP). Named destructives remain DENY.
_AUTO_EXTRA_RULES = [
    # workspace writes (PathZone handles EXTERNAL ASK at hook layer)
    _rule("filesystem_tools", "write_file", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "edit_file", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "create_directory", PermissionLevel.ALLOW),
    _rule("filesystem_tools", "move_file", PermissionLevel.ALLOW),
    # plan writes
    _rule("tools", "todo_write", PermissionLevel.ALLOW),
    _rule("tools", "todo_update", PermissionLevel.ALLOW),
    # generation finalize helpers
    _rule("semantic_tools", "end_*_generation", PermissionLevel.ALLOW),
    _rule("semantic_tools", "validate_semantic", PermissionLevel.ALLOW),
    _rule("semantic_tools", "generate_*_id", PermissionLevel.ALLOW),
    # bi write (excluding delete_*, which stays DENY from NORMAL via earlier rule)
    _rule("bi_tools", "create_*", PermissionLevel.ALLOW),
    _rule("bi_tools", "update_*", PermissionLevel.ALLOW),
    _rule("bi_tools", "add_*", PermissionLevel.ALLOW),
    # scheduler non-trigger writes
    _rule("scheduler_tools", "submit_*", PermissionLevel.ALLOW),
    _rule("scheduler_tools", "update_job", PermissionLevel.ALLOW),
    _rule("scheduler_tools", "pause_job", PermissionLevel.ALLOW),
    _rule("scheduler_tools", "resume_job", PermissionLevel.ALLOW),
    _rule("scheduler_tools", "trigger_*", PermissionLevel.ASK),
    # db writes: always ASK
    _rule("db_tools", "execute_ddl", PermissionLevel.ASK),
    _rule("db_tools", "execute_write", PermissionLevel.ASK),
    _rule("db_tools", "transfer_query_result", PermissionLevel.ASK),
    _rule("db_tools", "write_query", PermissionLevel.ASK),
    # Re-assert deny on bi_tools.delete_* after the create_*/update_*/add_* allows,
    # so the delete rule wins via last-match-wins regardless of declaration order.
    _rule("bi_tools", "delete_*", PermissionLevel.DENY),
]

AUTO = PermissionConfig(
    default_permission=PermissionLevel.ASK,
    rules=_NORMAL_RULES + _AUTO_EXTRA_RULES,
)

# --- Dangerous ---------------------------------------------------------------
# default=ALLOW, no rules. PathZone at hook layer still gates EXTERNAL fs.
DANGEROUS = PermissionConfig(
    default_permission=PermissionLevel.ALLOW,
    rules=[],
)


_PROFILES: dict[str, PermissionConfig] = {
    "normal": NORMAL,
    "auto": AUTO,
    "dangerous": DANGEROUS,
}


def get_profile(name: str) -> PermissionConfig:
    """Return the profile config for ``name``.

    Raises ``ValueError`` with an actionable message if ``name`` is unknown.
    Callers that want to fall back (e.g. ``AgentConfig`` on invalid YAML)
    must catch the exception themselves — this function never silently
    substitutes a default, so bugs that would otherwise mask bad config are
    caught at the call site.
    """
    try:
        return _PROFILES[name]
    except KeyError as e:
        raise ValueError(f"Unknown profile {name!r}. Valid options: {', '.join(PROFILE_NAMES)}") from e
