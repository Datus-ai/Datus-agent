# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Permission hooks for unified permission checking on all tools.

This module provides AgentHooks implementation that intercepts all tool calls
and performs permission checking before execution. It supports:
- Native Tools (db_tools, context_search_tools, filesystem_tools, etc.)
- MCP Tools (mcp.{server}.{tool})
- Skills (skills.{skill_name})

The hooks integrate with the InteractionBroker for async user interactions
when prompting users for permission confirmation.
"""

import asyncio
import json
import logging
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from agents.lifecycle import AgentHooks

from datus.cli.execution_state import InteractionBroker, InteractionCancelled
from datus.tools.func_tool.fs_path_policy import PathZone, classify_path
from datus.tools.permission.permission_config import PermissionLevel
from datus.tools.registry.tool_registry import ToolRegistry

if TYPE_CHECKING:
    from datus.tools.permission.permission_manager import PermissionManager

logger = logging.getLogger(__name__)

# Per-event-loop locks to serialize permission prompts within a single loop.
# A module-level ``asyncio.Lock()`` binds to the loop running on first ``await``
# and then raises ``Lock is bound to a different event loop`` on every
# subsequent ``asyncio.run()`` (the CLI creates a fresh loop per turn via
# ``chat_commands.py``). Key the lock by the running loop so each turn gets its
# own, while still serializing prompts from parallel tool calls inside one loop.
_permission_prompt_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)


def _get_permission_prompt_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _permission_prompt_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _permission_prompt_locks[loop] = lock
    return lock


class PermissionDeniedException(Exception):
    """Exception raised when a tool call is denied by permission rules."""

    def __init__(self, message: str, tool_category: str = "", tool_name: str = ""):
        super().__init__(message)
        self.tool_category = tool_category
        self.tool_name = tool_name


@dataclass(frozen=True)
class FilesystemPolicy:
    """Per-node filesystem policy passed to :class:`PermissionHooks`.

    Carries the information the hook needs to run
    :func:`datus.tools.func_tool.fs_path_policy.classify_path` on every
    filesystem tool call. Leaving this ``None`` on construction keeps the
    old category/tool-level permission behavior (no zone-based overrides).

    ``strict`` mirrors :attr:`FilesystemFuncTool.strict` so the hook and the
    tool agree on what to do with ``EXTERNAL`` paths. When ``True``, the
    hook denies them up front (no broker prompt) because the API / claw
    surfaces have no interactive broker attached — prompting would hang the
    request. The tool-level ``strict`` is still the source of truth, but
    having the same flag in the policy lets the hook fail fast before the
    tool even gets invoked.
    """

    root_path: Path
    current_node: Optional[str]
    datus_home: Optional[Path] = None
    strict: bool = False


class CompositeHooks(AgentHooks):
    """Combines multiple AgentHooks into one.

    This class allows multiple hooks to be applied in sequence,
    enabling composition of permission hooks with other hooks
    (e.g., PlanModeHooks).
    """

    def __init__(self, hooks_list: List[Optional[AgentHooks]]):
        """Initialize with a list of hooks.

        Args:
            hooks_list: List of AgentHooks instances (None values are filtered out)
        """
        self.hooks_list = [h for h in hooks_list if h is not None]

    async def on_start(self, context, agent) -> None:
        """Called when agent starts."""
        for hooks in self.hooks_list:
            if hasattr(hooks, "on_start"):
                await hooks.on_start(context, agent)

    async def on_tool_start(self, context, agent, tool) -> None:
        """Called before a tool is executed."""
        for hooks in self.hooks_list:
            if hasattr(hooks, "on_tool_start"):
                await hooks.on_tool_start(context, agent, tool)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        """Called after a tool completes."""
        for hooks in self.hooks_list:
            if hasattr(hooks, "on_tool_end"):
                await hooks.on_tool_end(context, agent, tool, result)

    async def on_llm_end(self, context, agent, response) -> None:
        """Called when LLM finishes a turn."""
        for hooks in self.hooks_list:
            if hasattr(hooks, "on_llm_end"):
                await hooks.on_llm_end(context, agent, response)

    async def on_end(self, context, agent, output) -> None:
        """Called when agent ends."""
        for hooks in self.hooks_list:
            if hasattr(hooks, "on_end"):
                await hooks.on_end(context, agent, output)


class PermissionHooks(AgentHooks):
    """AgentHooks implementation for unified permission checking on all tools.

    This class intercepts all tool calls and checks permissions before execution.
    It follows the existing tool classification structure:
    - Native Tools: Uses tool_registry to map tool_name -> category
    - MCP Tools: Parses "mcp__{server}__{tool}" format
    - Skills: Uses "skills" category with skill_name as pattern

    Example usage:
        permission_hooks = PermissionHooks(
            broker=interaction_broker,
            permission_manager=manager,
            node_name="chat",
            tool_registry=tool_registry,
        )

        # Use in execution config
        config["hooks"] = CompositeHooks([existing_hooks, permission_hooks])
    """

    def __init__(
        self,
        broker: InteractionBroker,
        permission_manager: "PermissionManager",
        node_name: str,
        tool_registry: ToolRegistry,
        *,
        fs_policy: Optional[FilesystemPolicy] = None,
    ):
        """Initialize the permission hooks.

        Args:
            broker: InteractionBroker for async user interactions
            permission_manager: PermissionManager for checking permissions
            node_name: Name of the current agentic node (e.g., "chat")
            tool_registry: Shared ToolRegistry instance (from AgenticNode)
            fs_policy: Optional per-node filesystem policy. When provided,
                ``filesystem_tools`` calls are routed through
                :func:`classify_path` first so ``EXTERNAL`` paths force a user
                prompt regardless of category rules, and ``HIDDEN`` paths fall
                through silently (the tool itself returns ``File not found``).
                Leaving this ``None`` preserves the old tool/category-level
                behavior for tests and legacy callers.
        """
        self.broker = broker
        self.permission_manager = permission_manager
        self.node_name = node_name
        self.tool_registry = tool_registry
        self.fs_policy = fs_policy

    async def on_tool_start(self, context, agent, tool) -> None:
        """Intercept ALL tool calls for permission checking.

        This method is called before each tool execution. It:
        1. Determines the tool category and pattern name
        2. Checks permission against the PermissionManager
        3. For DENY: raises PermissionDeniedException
        4. For ASK: prompts user via InteractionBroker, handles response
        5. For ALLOW: continues without interruption

        Args:
            context: Tool context with arguments
            agent: The agent instance
            tool: The tool being called

        Raises:
            PermissionDeniedException: If permission is denied or user rejects
        """
        tool_name = getattr(tool, "name", str(tool))

        # Get tool category and pattern name for permission checking
        category, pattern_name = self._get_category_and_pattern(tool_name, context)

        logger.debug(f"Permission check for tool '{tool_name}': category='{category}', pattern='{pattern_name}'")

        # Filesystem tools: zone-based policy overrides rules.
        #   INTERNAL/WHITELIST → bypass, HIDDEN → bypass (tool returns not-found),
        #   EXTERNAL → force ASK with a path-keyed session cache so approving
        #   /Users/foo/secret does not cascade to /Users/foo/other.
        if self.fs_policy is not None and category == "filesystem_tools":
            handled = await self._handle_filesystem_zone(context, tool_name, pattern_name)
            if handled:
                return

        # Check permission
        permission = self.permission_manager.check_permission(category, pattern_name, self.node_name)

        if permission == PermissionLevel.DENY:
            logger.warning(f"Tool '{tool_name}' denied by permission rules")
            raise PermissionDeniedException(
                f"Tool '{tool_name}' is not permitted in node '{self.node_name}'",
                tool_category=category,
                tool_name=pattern_name,
            )

        if permission == PermissionLevel.ASK:
            # Check multiple cache keys (tool_name and pattern_name might differ)
            cache_keys = [
                f"{category}.{pattern_name}",
                f"{category}.{tool_name}",
                f"{category}.*",  # Wildcard approval for category
            ]

            for cache_key in cache_keys:
                if self.permission_manager._session_approvals.get(cache_key):
                    logger.debug(f"Tool '{tool_name}' already approved for session (cache_key: {cache_key})")
                    return

            # Use lock to prevent multiple prompts at once (for parallel tool calls)
            async with _get_permission_prompt_lock():
                # Re-check cache after acquiring lock (another prompt may have approved it)
                for cache_key in cache_keys:
                    if self.permission_manager._session_approvals.get(cache_key):
                        logger.debug(f"Tool '{tool_name}' approved while waiting for lock")
                        return

                # Request user confirmation via InteractionBroker
                approved = await self._request_user_confirmation(category, pattern_name, context, tool_name=tool_name)

                if not approved:
                    logger.info(f"User rejected tool '{tool_name}'")
                    raise PermissionDeniedException(
                        f"User rejected execution of '{tool_name}'",
                        tool_category=category,
                        tool_name=pattern_name,
                    )

                logger.info(f"User approved tool '{tool_name}'")

    async def _handle_filesystem_zone(self, context: Any, tool_name: str, pattern_name: str) -> bool:
        """Zone-based gating for ``filesystem_tools.*`` calls.

        Returns ``True`` when the call has been fully handled (either allowed
        through or rejected) and ``False`` to let the normal category-level
        permission check run. ``EXTERNAL`` zones always land in the ``ASK``
        branch here regardless of what the category rule says — that is the
        whole point of zones, otherwise a session-level ``allow`` on
        ``filesystem_tools`` would silently grant ``/etc/passwd`` access.
        """
        policy = self.fs_policy
        assert policy is not None  # guarded by caller
        args = self._parse_tool_args(context)
        # ``_parse_tool_args`` deliberately returns whatever the JSON decoder
        # produced, so malformed tool_arguments (list, string, number) would
        # otherwise blow up on ``.get()``. Treat non-object payloads as
        # "no path provided" and fall back to the category-level rule check.
        if not isinstance(args, dict):
            logger.debug(
                "Filesystem permission check received non-object tool arguments for %s: %r",
                tool_name,
                args,
            )
            return False
        path_arg = args.get("path", "")
        try:
            resolved = classify_path(
                path_arg,
                root_path=policy.root_path,
                current_node=policy.current_node,
                datus_home=policy.datus_home,
            )
        except Exception as e:
            logger.debug(f"classify_path failed for {tool_name} path={path_arg!r}: {e}")
            return False

        if resolved.zone in (PathZone.INTERNAL, PathZone.WHITELIST):
            logger.debug(
                "Filesystem zone %s: allowing %s on %s without prompt",
                resolved.zone.value,
                tool_name,
                resolved.display,
            )
            return True

        if resolved.zone == PathZone.HIDDEN:
            # Let the tool itself return the uniform ``File not found`` so the
            # LLM cannot distinguish "hidden by policy" from "does not exist".
            logger.debug("Filesystem zone HIDDEN: letting tool return not-found for %s", resolved.display)
            return True

        # EXTERNAL in strict mode → deny up front, no broker prompt. Mirrors
        # FilesystemFuncTool.strict so callers without an interactive broker
        # (API / claw) fail fast instead of hanging waiting for user input.
        if policy.strict:
            logger.info(
                "Filesystem strict mode: rejecting EXTERNAL access to %s (tool=%s)",
                resolved.resolved,
                tool_name,
            )
            raise PermissionDeniedException(
                f"Filesystem strict mode: path outside workspace is not allowed: {resolved.resolved}",
                tool_category="filesystem_tools",
                tool_name=pattern_name,
            )

        # EXTERNAL: force ASK, keyed by absolute path to prevent broad auto-approval.
        cache_key = f"filesystem_tools.external::{resolved.resolved}"
        if self.permission_manager._session_approvals.get(cache_key):
            logger.debug("External path %s already approved for session", resolved.resolved)
            return True

        async with _get_permission_prompt_lock():
            if self.permission_manager._session_approvals.get(cache_key):
                return True

            approved = await self._request_external_confirmation(tool_name, pattern_name, resolved.resolved)
            if not approved:
                logger.info("User rejected external filesystem access to %s", resolved.resolved)
                raise PermissionDeniedException(
                    f"User rejected external filesystem access to {resolved.resolved}",
                    tool_category="filesystem_tools",
                    tool_name=pattern_name,
                )
            logger.info("User approved external filesystem access to %s", resolved.resolved)
            return True

    async def _request_external_confirmation(
        self,
        tool_name: str,
        pattern_name: str,
        abs_path: Path,
    ) -> bool:
        """Prompt the user for an EXTERNAL filesystem access.

        Approval is narrow: the ``a`` (always-allow) choice caches this exact
        absolute path, not the whole tool or category.
        """
        content = (
            "### External Filesystem Access\n\n"
            f"**Tool:** `filesystem_tools.{pattern_name}`\n"
            f"**Path:** `{abs_path}`  _(outside project root)_\n"
        )
        try:
            choice, callback = await self.broker.request(
                contents=[content],
                choices=[
                    {
                        "y": "Allow (once)",
                        "a": "Always allow (this path, session)",
                        "n": "Deny",
                    }
                ],
                default_choices=["n"],
            )

            if choice == "a":
                cache_key = f"external::{abs_path}"
                self.permission_manager.approve_for_session("filesystem_tools", cache_key)
                await callback(f"**{abs_path}** approved for session")
                return True
            if choice == "y":
                await callback("**Approved**")
                return True
            await callback("**Denied**")
            return False
        except InteractionCancelled:
            return False
        except Exception as e:
            logger.error(f"Error in external filesystem confirmation for {tool_name}: {e}")
            return False

    def _get_category_and_pattern(self, tool_name: str, context: Any) -> Tuple[str, str]:
        """Get tool category and pattern name for permission checking.

        This method determines how to classify a tool for permission rules.

        Returns:
            Tuple of (category, pattern_name)

        Examples:
            Native:  ("db_tools", "execute_sql")
            MCP:     ("mcp.filesystem", "read_file")
            Skills:  ("skills", "deep-analysis")  # skill_name from args
        """
        # 1. Skills: load_skill -> extract skill_name as pattern (check BEFORE registry)
        #    This allows permission rules like "skills.admin-*" to match specific skills
        if tool_name == "load_skill":
            args = self._parse_tool_args(context)
            skill_name = args.get("skill_name", "*")
            return ("skills", skill_name)

        # 2. MCP Tools: format "mcp__{server}__{tool}" -> ("mcp.{server}", "{tool}")
        if tool_name.startswith("mcp__"):
            parts = tool_name.split("__")  # ["mcp", "filesystem", "read_file"]
            if len(parts) >= 3:
                server = parts[1]
                method = "__".join(parts[2:])  # Handle multi-part tool names
                return (f"mcp.{server}", method)

        # 3. Check tool registry (Native Tools registered via register_tools())
        category = self.tool_registry.get(tool_name)
        if category is not None:
            return (category, tool_name)

        # 4. Default: unknown category
        logger.debug(f"Tool '{tool_name}' not in registry, using default category 'tools'")
        return ("tools", tool_name)

    def _parse_tool_args(self, context: Any) -> dict:
        """Parse tool arguments from context.

        Args:
            context: Tool context object with tool_arguments attribute

        Returns:
            Dictionary of tool arguments
        """
        try:
            args_str = getattr(context, "tool_arguments", "{}")
            if isinstance(args_str, str):
                return json.loads(args_str)
            elif isinstance(args_str, dict):
                return args_str
            return {}
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"Failed to parse tool arguments: {e}")
            return {}

    async def _request_user_confirmation(
        self,
        category: str,
        pattern_name: str,
        context: Any,
        tool_name: Optional[str] = None,
    ) -> bool:
        """Request user confirmation via InteractionBroker.

        This method uses the async InteractionBroker pattern to prompt
        the user for permission approval.

        Args:
            category: Tool category (e.g., "skills", "mcp.filesystem")
            pattern_name: Specific tool/skill name
            context: Tool context for additional info
            tool_name: Original tool function name (e.g., "load_skill")

        Returns:
            True if user approved, False otherwise
        """
        # Build permission request content (markdown format)
        args = self._parse_tool_args(context)

        content = f"### Permission Request\n\n**Tool:** `{category}.{pattern_name}`\n"

        # Show tool arguments if available (truncate long args)
        if args:
            args_str = json.dumps(args, ensure_ascii=False)
            if len(args_str) > 200:
                args_str = args_str[:197] + "..."
            content += f"\n**Args:** `{args_str}`\n"

        try:
            choice, callback = await self.broker.request(
                contents=[content],
                choices=[
                    {
                        "y": "Allow (once)",
                        "a": "Always allow (session)",
                        "n": "Deny",
                    }
                ],
                default_choices=["n"],
            )

            if choice == "a":
                # Approve for session - all future calls to this tool are auto-approved
                self.permission_manager.approve_for_session(category, pattern_name)
                # Also cache tool-level key so all future calls of the same tool type are auto-approved
                # e.g., load_skill("report-generator") also caches "skills.load_skill"
                # so load_skill("sql-analysis") is auto-approved without a second prompt
                if tool_name and tool_name != pattern_name:
                    self.permission_manager.approve_for_session(category, tool_name)
                await callback(f"**{category}.{pattern_name}** approved for session")
                return True
            elif choice == "y":
                # One-time approval - do NOT cache, will prompt again next time
                await callback("**Approved**")
                return True
            else:
                await callback("**Denied**")
                return False

        except InteractionCancelled:
            return False
        except Exception as e:
            logger.error(f"Error in permission confirmation: {e}")
            return False
