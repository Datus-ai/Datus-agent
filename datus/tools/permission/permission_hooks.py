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
import re
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple

from agents.lifecycle import AgentHooks

from datus.cli.execution_state import InteractionBroker, InteractionCancelled
from datus.schemas.interaction_event import InteractionEvent
from datus.tools.func_tool.fs_path_policy import PathZone, classify_path
from datus.tools.permission.bash_classifier import BashClassifierContext
from datus.tools.permission.bash_rules import (
    BashDecisionSource,
    BashRuleDecision,
    evaluate_bash_command,
)
from datus.tools.permission.permission_config import PermissionLevel
from datus.tools.registry.tool_registry import ToolRegistry
from datus.utils.constants import SQLType
from datus.utils.json_utils import to_pretty_str

if TYPE_CHECKING:
    from datus.tools.permission.bash_classifier import BashCommandClassifier
    from datus.tools.permission.permission_manager import PermissionManager

logger = logging.getLogger(__name__)

# Per-broker locks to serialize permission prompts within a single agent run.
#
# The lock exists so several parallel tool calls in one LLM turn don't fire
# multiple permission prompts at once. Those parallel calls all share the node's
# single ``InteractionBroker``, so the broker is the correct serialization
# scope.
#
# Keying by the running event loop instead (the previous design) is WRONG on a
# long-lived multi-session server: uvicorn runs every request on one shared
# loop, so a per-loop lock serializes prompts across *independent* sessions and
# sub-agents. Because the lock is held across the user-response ``await`` inside
# ``broker.request()``, while session A waits for its answer, sessions B/C/D
# block inside ``on_tool_start`` *before* reaching ``broker.request()`` — they
# emit the TOOL "processing" action (claude_model yields it before the gate) but
# never the INTERACTION event, so their clients hang with no prompt to answer.
#
# An ``asyncio.Lock`` binds to the loop of its first ``await`` and then raises
# "bound to a different event loop" if reused on another loop (the CLI creates a
# fresh loop per turn via ``chat_commands.py``). We therefore cache ``(loop,
# lock)`` per broker and rebuild when the loop changes, so a broker reused
# across CLI turns still gets a loop-correct lock.
_permission_prompt_locks: "weakref.WeakKeyDictionary[Any, Tuple[asyncio.AbstractEventLoop, asyncio.Lock]]" = (
    weakref.WeakKeyDictionary()
)
# Fallback locks for callers without a (weak-referenceable) broker, keyed by the
# running loop to preserve the legacy behavior in that narrow case.
_loop_fallback_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()


def _get_permission_prompt_lock(broker: Any = None) -> asyncio.Lock:
    """Return the prompt-serialization lock for ``broker`` on the running loop.

    Scoped per broker (i.e. per agent run / session) so concurrent sessions and
    sub-agents on a shared event loop never block each other's permission
    prompts. Falls back to a per-loop lock when ``broker`` is missing or not
    weak-referenceable.
    """
    loop = asyncio.get_running_loop()
    if broker is not None:
        try:
            entry = _permission_prompt_locks.get(broker)
            if entry is None or entry[0] is not loop:
                lock = asyncio.Lock()
                _permission_prompt_locks[broker] = (loop, lock)
                return lock
            return entry[1]
        except TypeError:
            # Broker is not weak-referenceable; fall through to a per-loop lock.
            pass
    lock = _loop_fallback_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _loop_fallback_locks[loop] = lock
    return lock


# Scalar strings at or below this length (and free of newlines) render inline on
# the key line; anything longer moves into a fenced code block so the TUI's
# syntax highlighting + scroll/pager handle it instead of cramming it onto one
# line. Nothing is truncated — the InteractionApp already pages long content
# (Shift+Up/Down) and opens it in a pager (``v``).
_INLINE_ARG_MAX = 80


def _format_tool_args_markdown(args: dict) -> str:
    """Render tool arguments as a readable markdown key/value block.

    Replaces the old single-line ``json.dumps`` + hard 200-char truncation,
    which mangled nested structures and cut SQL mid-statement. Rendering rules:

    * Scalars (short strings, ints, floats, bools, ``None``) render inline as
      ``**key:** `value` ``.
    * The ``sql`` argument and any multi-line / long string render in a fenced
      code block — ``sql`` for the SQL argument (so it is highlighted),
      language-less otherwise.
    * ``dict`` / ``list`` values render as pretty (2-space) JSON in a ```json
      block.

    Each argument is emitted as its own top-level paragraph / code block rather
    than nested under a list item, so ``RichMarkdown`` renders the fenced blocks
    unambiguously. Nothing is truncated.

    Returns an empty string for empty ``args`` so callers can skip the section.
    """
    if not args:
        return ""

    parts: List[str] = ["", "**Arguments**", ""]
    for key, value in args.items():
        if isinstance(value, (dict, list)):
            pretty = to_pretty_str(value) or "{}"
            parts.extend([f"**{key}:**", "", "```json", pretty, "```", ""])
        elif isinstance(value, str) and ("\n" in value or len(value) > _INLINE_ARG_MAX):
            lang = "sql" if key == "sql" else ""
            parts.extend([f"**{key}:**", "", f"```{lang}", value, "```", ""])
        else:
            parts.append(f"**{key}:** `{value}`")
    return "\n".join(parts)


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
    hook skips the broker prompt and delegates the denial to the
    filesystem tool, which returns ``FuncToolResult(success=0)`` with a
    "strict mode" error message. This matters for API / gateway surfaces with
    no interactive broker attached — prompting would hang the request,
    while raising would surface as an uncaught exception. The tool-level
    ``strict`` is still the source of truth; having the same flag in the
    policy lets the hook avoid prompting while preserving the normal
    tool-failure payload the caller already knows how to handle.
    """

    root_path: Path
    current_node: Optional[str]
    datus_home: Optional[Path] = None
    strict: bool = False
    # Per-session compact archive directory. When set, ``classify_path`` treats
    # this subtree of ``~/.datus/sessions/{project}/{session_id}/data`` as a
    # read-only WHITELIST anchor so the LLM can ``read_file`` archived tool I/O
    # without prompting. Stays ``None`` outside of agentic sessions (e.g. SaaS
    # request-scoped tools) — those paths then remain EXTERNAL.
    session_data_dir: Optional[Path] = None


class CompositeHooks(AgentHooks):
    """Combines multiple AgentHooks into one.

    This class allows multiple hooks to be applied in sequence,
    enabling composition of permission hooks with other hooks
    (e.g., GenerationHooks).
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
        non_interactive: bool = False,
        proxied_tool_names: Optional[Set[str]] = None,
        project_root: Optional[str] = None,
        bash_classifier: Optional["BashCommandClassifier"] = None,
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
            non_interactive: When ``True``, ``ASK`` permissions and EXTERNAL
                filesystem zones raise :class:`PermissionDeniedException`
                immediately instead of prompting the broker. Used by
                ``execution_mode="workflow"`` flows (``/bootstrap``, scheduler
                subagents, etc.) where there is no human in the loop and any
                ASK/EXTERNAL hit means the tool is outside the active profile's
                scope. ``DENY`` continues to raise as before.
            proxied_tool_names: Optional set of tool names that
                :func:`datus.tools.proxy.proxy_tool.apply_proxy_tools` wrapped
                with stdin-driven proxies. When provided, ``on_tool_start``
                skips ALL permission checks (DENY/ASK/zone) for these tools —
                the external caller (e.g. ``print_mode`` stdin protocol) owns
                secondary confirmation, so the agent must not double-prompt or
                block proxied calls. Passing the same set reference held by the
                node lets late ``apply_proxy_tools`` invocations be observed
                without rebuilding the hook.
            project_root: Workspace root used to resolve a ``.sql`` file
                reference passed to ``execute_sql``. The SQL permission gate
                reads the file (same logic as the tool) so a read-only ``.sql``
                file auto-allows instead of prompting. ``None`` falls back to the
                current working directory. Also used as the write target for
                project-level bash allow grants (``.datus/config.yml``).
            bash_classifier: Optional LLM classifier for bash commands (reserved
                seam, see ``bash_classifier.py``). Consulted only when the
                static bash rules yield ASK with ``safety_forced=False``; a
                high-confidence ALLOW verdict auto-approves, anything else
                falls through to the normal confirmation prompt (fail closed).
        """
        self.broker = broker
        self.permission_manager = permission_manager
        self.node_name = node_name
        self.tool_registry = tool_registry
        self.fs_policy = fs_policy
        self.non_interactive = non_interactive
        self.proxied_tool_names = proxied_tool_names
        self.project_root = project_root
        self.bash_classifier = bash_classifier

    # Plan-mode tooling is always allowed regardless of permission profile:
    # ``confirm_plan`` already runs its own user interaction, and ``todo_*``
    # are local-only state helpers that never touch the filesystem or DB.
    _PLAN_MODE_BYPASS_TOOLS = frozenset({"confirm_plan", "todo_list", "todo_read", "todo_write", "todo_update"})

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

        # Short-circuit plan-mode helpers: they carry their own UX and have
        # no external side effects, so skip the permission profile entirely.
        if tool_name in self._PLAN_MODE_BYPASS_TOOLS:
            return

        # Short-circuit proxied tools: their execution is delegated to the
        # external caller via the stdin proxy channel, which owns secondary
        # confirmation. Re-checking the profile here would either double-prompt
        # (ASK) or block calls the caller already authorised (DENY).
        if self.proxied_tool_names and tool_name in self.proxied_tool_names:
            logger.debug("Tool '%s' is proxied; skipping permission check", tool_name)
            return

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

        # db_tools.execute_sql: statement-type gating overrides rules.
        #   read-only (SELECT/SHOW/DESCRIBE/EXPLAIN) → bypass; writes (INSERT/
        #   UPDATE/DELETE), DDL, and unknown/MERGE → ASK (session cache bucketed
        #   per SQL type), dangerous bypass, non-interactive raise. A ``.sql``
        #   file is resolved first; an unreadable file or unparseable text falls
        #   through to UNKNOWN → ASK (fail safe).
        if category == "db_tools" and tool_name == "execute_sql":
            handled = await self._handle_sql_permission(context, tool_name, pattern_name)
            if handled:
                return

        # bash_tools.bash: command-level gating overrides the coarse rule.
        #   deny patterns → raise; allow patterns / profile whitelist → bypass;
        #   everything else → ASK with a per-command-prefix session bucket and
        #   an optional project-level persist ("allow (project)"). Shell
        #   wrappers / metacharacters force ASK regardless of allow rules.
        #   Ruleset absent (e.g. dangerous profile) → fall through to the
        #   legacy coarse ``bash_tools.bash`` rule check below.
        if category == "bash_tools" and tool_name == "bash":
            handled = await self._handle_bash_permission(context, tool_name, pattern_name)
            if handled:
                return

        # Check permission
        permission = self.permission_manager.check_permission(category, pattern_name, self.node_name)

        if permission == PermissionLevel.DENY:
            logger.warning(f"Tool '{tool_name}' denied by permission rules")
            profile = getattr(self.permission_manager, "active_profile", None) or "unknown"
            raise PermissionDeniedException(
                (
                    f"PERMISSION_DENIED: Tool '{tool_name}' ({category}) is blocked by the "
                    f"'{profile}' permission profile. STOP retrying this tool — different "
                    "parameters will not change the outcome. Return the failure to your "
                    "caller and stop. The user can run /profile to open the "
                    "profile picker and choose an appropriate mode with the "
                    "arrow keys, or add a permission rule under "
                    "`permissions.rules` in agent.yml."
                ),
                tool_category=category,
                tool_name=pattern_name,
            )

        if permission == PermissionLevel.ASK:
            if self.non_interactive:
                profile = getattr(self.permission_manager, "active_profile", None) or "auto"
                logger.warning(
                    "Non-interactive mode: tool '%s' (%s) requires ASK confirmation under "
                    "profile '%s'; raising PermissionDeniedException instead of prompting.",
                    tool_name,
                    category,
                    profile,
                )
                raise PermissionDeniedException(
                    (
                        f"PERMISSION_DENIED: Tool '{tool_name}' ({category}) requires user "
                        f"confirmation but this flow runs non-interactively under the "
                        f"'{profile}' profile. The tool is outside that profile's scope. "
                        f"STOP retrying — different parameters will not change the outcome. "
                        f"Surface the failure to the caller."
                    ),
                    tool_category=category,
                    tool_name=pattern_name,
                )

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

            # Use lock to prevent multiple prompts at once (for parallel tool
            # calls within THIS run). Scoped per broker so concurrent sessions
            # and sub-agents on a shared event loop don't serialize each other's
            # prompts (see ``_get_permission_prompt_lock``).
            async with _get_permission_prompt_lock(self.broker):
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

    # Tool-name set used to distinguish destructive writes from reads.
    # Keep in sync with ``FilesystemFuncTool.available_tools`` — adding a new
    # write-capable filesystem tool (e.g. ``append_file``) requires extending
    # this set so the profile-aware gate treats it as a write. ``delete_file``
    # belongs here too — it mutates the filesystem just as much as a write
    # and should hit the same INTERNAL × write × normal ASK gate.
    _FILESYSTEM_WRITE_TOOLS = frozenset({"write_file", "edit_file", "delete_file"})

    # Subagents that author their own artifact tree (manifest.json,
    # queries/*, render/*.jsx, analysis/*) in one turn — usually 5-10 files
    # per run. Under ``normal`` profile the default ASK gate would prompt
    # the user for every single ``write_file`` / ``edit_file``, which
    # defeats the purpose: the artifact only makes sense as a whole and
    # the user reviews it via the rendered preview, not per-file diffs.
    # Mirror of ``_FS_DEPENDENT_NODES`` in ``datus.tools.proxy.proxy_tool``
    # (which exempts the same nodes from the IDE proxy round-trip) and the
    # ``isAutoConfirmFilePath`` carve-out in ``Datus-saas`` (which silences
    # the per-tool Accept bar in the chat panel). All three layers must
    # agree or one of them keeps gating the user.
    _ARTIFACT_AUTOALLOW_NODES = frozenset({"gen_visual_report", "gen_visual_dashboard"})

    # Relative-to-project-root path prefixes the carve-out applies to.
    # Slug character class is loose on purpose — ``ARTIFACT_SLUG_PATTERN``
    # in ``datus.schemas.artifact_manifest`` is the source of truth and
    # may evolve; matching ``[^/]+`` keeps the carve-out in lockstep
    # without dragging the schema dep into this hook.
    _ARTIFACT_AUTOALLOW_PATH_RE = re.compile(r"^(?:reports|dashboards)/[^/]+/")

    async def _handle_filesystem_zone(self, context: Any, tool_name: str, pattern_name: str) -> bool:
        """Zone × profile × read-vs-write gating for ``filesystem_tools.*`` calls.

        Returns ``True`` when the call has been fully handled (either allowed
        through or rejected) and ``False`` to let the normal category-level
        permission check run.

        Decision matrix (see plan: profile-aware filesystem permission):

        ============  ========================  ==============  ==============  ==============
        operation     zone                      normal          auto            dangerous
        ============  ========================  ==============  ==============  ==============
        read          INTERNAL / WHITELIST      bypass          bypass          bypass
        read          HIDDEN                    tool not-found  tool not-found  tool not-found
        read          EXTERNAL (interactive)    ASK(path bucket) ASK(path bucket) bypass
        read          EXTERNAL (strict)         tool fail       tool fail       tool fail
        read          EXTERNAL (non-interactive) raise          raise           raise
        write         INTERNAL                  rule lookup ASK bypass          bypass
        write         WHITELIST (parent memory) tool reject     tool reject     tool reject
        write         HIDDEN                    tool not-found  tool not-found  tool not-found
        write         EXTERNAL (interactive)    ASK(path bucket) ASK(path bucket) bypass
        write         EXTERNAL (strict)         tool fail       tool fail       tool fail
        write         EXTERNAL (non-interactive) raise          raise           raise
        ============  ========================  ==============  ==============  ==============

        Key invariants:

        * ``non_interactive`` short-circuits ``dangerous`` — workflow flows must
          never silently write outside the project just because they happen to
          run under the dangerous profile.
        * ``policy.strict`` short-circuits everything for EXTERNAL paths;
          callers without an interactive broker rely on the tool-layer fail.
        * WHITELIST writes are left to ``FilesystemFuncTool._read_only_reject``
          so the user doesn't waste an ASK click on a path the tool will refuse.
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
                session_data_dir=policy.session_data_dir,
            )
        except Exception as e:
            logger.debug(f"classify_path failed for {tool_name} path={path_arg!r}: {e}")
            return False

        is_write = tool_name in self._FILESYSTEM_WRITE_TOOLS
        profile = getattr(self.permission_manager, "active_profile", None) or "normal"

        if resolved.zone in (PathZone.INTERNAL, PathZone.WHITELIST):
            # INTERNAL × write × normal: fall through to rule lookup so the
            # category-level ``default=ASK`` (or any explicit ``filesystem_tools.write_file``
            # rule) takes over. ``_NORMAL_RULES`` has no entry for write_file,
            # so this materialises as a per-session ASK prompt on the user.
            # WHITELIST writes are handled by ``FilesystemFuncTool._read_only_reject``
            # at the tool layer (parent-memory inheritance is read-only),
            # so we keep bypass here to avoid a wasted ASK round-trip.
            if is_write and profile == "normal" and resolved.zone == PathZone.INTERNAL:
                # Visual artifact subagents author the entire artifact tree
                # in one turn — bypass the per-file ASK for paths under
                # their own ``reports/<slug>/`` or ``dashboards/<slug>/``
                # directory. See ``_ARTIFACT_AUTOALLOW_NODES`` docstring.
                if self.node_name in self._ARTIFACT_AUTOALLOW_NODES:
                    try:
                        rel = resolved.resolved.relative_to(policy.root_path).as_posix()
                    except ValueError:
                        rel = None
                    if rel and self._ARTIFACT_AUTOALLOW_PATH_RE.match(rel):
                        logger.debug(
                            "Filesystem zone INTERNAL × write × normal: auto-allowing %s on %s for artifact node %r",
                            tool_name,
                            rel,
                            self.node_name,
                        )
                        return True
                logger.debug(
                    "Filesystem zone INTERNAL × write × normal: deferring to rule lookup for %s",
                    resolved.display,
                )
                return False
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

        # EXTERNAL in strict mode → delegate to the tool, which returns
        # FuncToolResult(success=0). We return True here (no broker prompt,
        # no exception) so callers without an interactive broker (API / gateway)
        # still fail fast but surface the denial as a normal tool-failure
        # payload the agent can read, rather than an uncaught exception.
        if policy.strict:
            logger.info(
                "Filesystem strict mode: delegating EXTERNAL access to tool for %s (tool=%s)",
                resolved.resolved,
                tool_name,
            )
            return True

        # EXTERNAL: force ASK, keyed by absolute path to prevent broad auto-approval.
        if self.non_interactive:
            profile = getattr(self.permission_manager, "active_profile", None) or "auto"
            logger.warning(
                "Non-interactive mode: external filesystem path %s requires confirmation under "
                "profile '%s'; raising PermissionDeniedException instead of prompting.",
                resolved.resolved,
                profile,
            )
            raise PermissionDeniedException(
                (
                    f"PERMISSION_DENIED: filesystem path '{resolved.resolved}' is outside the "
                    f"project root and requires user confirmation, but this flow runs "
                    f"non-interactively under the '{profile}' profile. STOP retrying — "
                    f"choose a path inside the project root or surface the failure."
                ),
                tool_category="filesystem_tools",
                tool_name=pattern_name,
            )

        # Dangerous profile in interactive mode: opt out of the EXTERNAL ASK
        # gate entirely. Workflow flows reach the ``non_interactive`` branch
        # above before this point, so the user's foreground ``/profile dangerous``
        # choice is the only thing that can land here.
        if profile == "dangerous":
            logger.debug(
                "Profile=dangerous: bypassing EXTERNAL ASK for %s on %s",
                tool_name,
                resolved.resolved,
            )
            return True

        cache_key = f"filesystem_tools.external::{resolved.resolved}"
        if self.permission_manager._session_approvals.get(cache_key):
            logger.debug("External path %s already approved for session", resolved.resolved)
            return True

        async with _get_permission_prompt_lock(self.broker):
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
            answers = await self.broker.request(
                [
                    InteractionEvent(
                        title="Permission",
                        content=content,
                        choices={"y": "Allow (once)", "a": "Always allow (this path, session)", "n": "Deny"},
                        default_choice="n",
                    )
                ]
            )
            choice = answers[0][0] if answers and answers[0] else ""

            if choice == "a":
                cache_key = f"external::{abs_path}"
                self.permission_manager.approve_for_session("filesystem_tools", cache_key)
                return True
            if choice == "y":
                return True
            return False
        except InteractionCancelled:
            return False
        except Exception as e:
            logger.error(f"Error in external filesystem confirmation for {tool_name}: {e}")
            return False

    # Read-only SQL statement types that auto-allow under ``execute_sql``.
    # Everything else (INSERT/UPDATE/DELETE/DDL/MERGE/UNKNOWN) defers to the
    # normal category-level permission check.
    _SQL_READONLY_TYPES = frozenset({SQLType.SELECT, SQLType.METADATA_SHOW, SQLType.EXPLAIN})

    async def _handle_sql_permission(self, context: Any, tool_name: str, pattern_name: str) -> bool:
        """Statement-type gating for ``db_tools.execute_sql`` calls.

        ``execute_sql`` is the unified SQL entry point, so a single static rule
        cannot express "reads auto-allow, writes ask". This gate inspects the
        statement type instead:

        * Read-only (SELECT/SHOW/DESCRIBE/EXPLAIN) → auto-allow, unless an
          explicit DENY rule blocks ``db_tools.execute_sql`` (then defer so the
          DENY is surfaced).
        * Writes (INSERT/UPDATE/DELETE), DDL, and unknown/MERGE → category-level
          rule check for DENY/ALLOW, then ASK with a session cache **bucketed by
          the concrete SQL type** (``execute_sql.insert`` / ``.update`` /
          ``.delete`` / ``.ddl`` / ``.merge`` / ``.unknown``). Per-type bucketing
          matters because ``execute_sql`` is one tool name: an "Always allow"
          keyed on the bare tool name would let one approved INSERT silently
          green-light every later DELETE / DROP / MERGE, so each statement type
          carries its own session approval. ALLOW under dangerous (rule check
          returns ALLOW), raise when non-interactive.

        A ``.sql`` file reference is resolved (same workspace-relative read as the
        tool) so the gate classifies the real statement — a read-only ``.sql``
        file auto-allows instead of prompting. An unreadable file or unparseable
        text falls through to UNKNOWN → ASK (fail safe).

        Returns ``True`` when the call has been fully handled (read auto-allowed,
        explicit/dangerous ALLOW, or bucketed ASK approved), ``False`` to let the
        normal category-level permission check run (surfaces DENY / the
        standardized non-interactive raise).
        """
        from datus.utils.sql_utils import looks_like_sql_file_ref, parse_sql_type, read_workspace_sql_file

        args = self._parse_tool_args(context)
        if not isinstance(args, dict):
            return False
        sql = args.get("sql", "")
        # Resolve a ``.sql`` file reference to its real (single) statement so the
        # type detection below sees the SQL, not the path. On any read failure,
        # leave ``sql`` as-is → UNKNOWN → ASK (fail safe).
        if isinstance(sql, str) and looks_like_sql_file_ref(sql):
            try:
                sql = read_workspace_sql_file(sql.strip(), self.project_root or ".")
            except Exception as e:
                logger.debug("execute_sql gate: could not resolve .sql file (%s); treating as UNKNOWN", e)
        # No dialect available at the hook layer; the keyword fallback in
        # ``parse_sql_type`` is enough to separate reads from writes/DDL.
        sql_type = parse_sql_type(sql, "") if isinstance(sql, str) else SQLType.UNKNOWN

        if sql_type in self._SQL_READONLY_TYPES:
            # Respect an explicit DENY; otherwise reads auto-allow regardless of
            # profile (there is no static ALLOW rule for execute_sql to rely on).
            if (
                self.permission_manager.check_permission("db_tools", pattern_name, self.node_name)
                == PermissionLevel.DENY
            ):
                return False
            logger.debug("execute_sql read-only (%s): auto-allow", sql_type.value)
            return True

        # Non-read (write / DDL / unknown). Honour explicit rules first.
        permission = self.permission_manager.check_permission("db_tools", pattern_name, self.node_name)
        if permission == PermissionLevel.DENY:
            # Surface the standardized DENY raise via the main flow.
            return False
        if permission == PermissionLevel.ALLOW:
            # Explicit ALLOW rule or a permissive profile (e.g. dangerous).
            return True

        # ASK. Non-interactive flows must not prompt — defer so the main flow
        # raises the standardized non-interactive PermissionDeniedException.
        if self.non_interactive:
            return False

        # Bucket the session approval by the concrete SQL type so an "always
        # allow" only ever covers that one type (e.g. approving an INSERT never
        # green-lights a later DELETE / DROP / MERGE).
        sql_class = sql_type.value
        bucket_pattern = f"execute_sql.{sql_class}"
        # Honour the per-type bucket plus any deliberately broad session approval
        # (a prior un-bucketed ``db_tools.execute_sql`` or a category wildcard).
        # Only the prompt-driven "always allow" below is bucketed per type, so a
        # single approval never cascades across statement types — but an explicit
        # broad approval still covers every type.
        cache_keys = (f"db_tools.{bucket_pattern}", "db_tools.execute_sql", "db_tools.*")

        def _session_approved() -> bool:
            return any(self.permission_manager._session_approvals.get(key) for key in cache_keys)

        if _session_approved():
            logger.debug("execute_sql %s already approved for session", sql_class)
            return True

        async with _get_permission_prompt_lock(self.broker):
            if _session_approved():
                return True
            # Pass the bucketed pattern as the cache key; deliberately omit
            # ``tool_name`` so "Always allow" caches ONLY ``db_tools.execute_sql.<type>``
            # and not the un-bucketed ``db_tools.execute_sql`` (which would cascade
            # across statement types).
            approved = await self._request_user_confirmation("db_tools", bucket_pattern, context)
            if not approved:
                logger.info("User rejected execute_sql (%s)", sql_class)
                raise PermissionDeniedException(
                    "User rejected execution of 'execute_sql'",
                    tool_category="db_tools",
                    tool_name="execute_sql",
                )
            logger.info("User approved execute_sql (%s)", sql_class)
            return True

    async def _handle_bash_permission(self, context: Any, tool_name: str, pattern_name: str) -> bool:
        """Command-level gating for ``bash_tools.bash`` calls.

        The coarse ``bash_tools.bash -> ASK`` rule cannot express "``git log``
        is fine, ``rm`` never is". This gate evaluates the ``command`` argument
        against the effective ``bash_commands`` ruleset (profile whitelist +
        user agent.yml rules + project ``.datus/config.yml`` allows — see
        ``bash_rules.evaluate_bash_command`` for the deny-first decision order).

        * DENY match → raise immediately, naming the matched pattern.
        * ALLOW match → bypass (no prompt).
        * ASK → non-interactive raises; otherwise consult the optional LLM
          classifier (reserved seam — never for ``safety_forced`` decisions),
          then the per-bucket session cache, then prompt with four choices:
          allow once / allow (session) / allow (project) / deny. The project
          choice persists ``<bucket>:*`` to ``.datus/config.yml`` and is only
          offered for plain unmatched commands — never for safety-ceiling asks
          (wrappers, metacharacters) or explicit ask-rule hits (the user asked
          to review those every time; a persisted allow could not beat the ask
          rule at evaluation time anyway).

        Returns ``True`` when fully handled, ``False`` to defer to the normal
        category-level check (explicit category DENY, missing/empty ruleset,
        or unparseable arguments) — deferring keeps behavior byte-compatible
        with the legacy coarse path. Cross-reference: ``BashTool`` keeps its
        own ``allowed_patterns`` matcher as a legacy secondary gate.
        """
        args = self._parse_tool_args(context)
        if not isinstance(args, dict):
            return False
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return False

        # Respect an explicit category-level DENY — defer so the main flow
        # raises the standardized DENY message.
        if self.permission_manager.check_permission("bash_tools", pattern_name, self.node_name) == PermissionLevel.DENY:
            return False

        effective = self.permission_manager.get_effective_config(self.node_name)
        rules = getattr(effective, "bash_commands", None)
        if rules is None or rules.is_empty():
            # No command-level ruleset (e.g. dangerous profile) — legacy path.
            return False

        decision = evaluate_bash_command(command, rules)

        # When no bash rule matched and the ruleset never explicitly set a
        # ``default``, inherit the effective config's posture instead of
        # BashCommandRules' built-in ASK. Under normal/auto this is ASK either
        # way; under dangerous (default_permission=ALLOW) it keeps unmatched
        # commands flowing — a user adding a ``deny`` list to agent.yml must
        # not silently flip dangerous into ask-everything for bash. Safety
        # ceiling decisions are NOT affected (source=SAFETY, not DEFAULT).
        if decision.source == BashDecisionSource.DEFAULT and "default" not in rules.model_fields_set:
            try:
                fallback = PermissionLevel(effective.default_permission)
            except ValueError:
                fallback = PermissionLevel.ASK
            if fallback != decision.level:
                from dataclasses import replace as _dc_replace

                decision = _dc_replace(
                    decision, level=fallback, reason=f"{decision.reason}; inheriting profile default '{fallback.value}'"
                )

        if decision.level == PermissionLevel.DENY:
            profile = getattr(self.permission_manager, "active_profile", None) or "unknown"
            logger.warning("Bash command denied by rule %r: %s", decision.matched_pattern, command)
            raise PermissionDeniedException(
                (
                    f"PERMISSION_DENIED: Bash command blocked by rule "
                    f"'{decision.matched_pattern}' under the '{profile}' permission "
                    f"profile. STOP retrying this command — rewording it will not "
                    f"change the outcome. Return the failure to your caller. The "
                    f"user can adjust `permissions.bash_commands` in agent.yml."
                ),
                tool_category="bash_tools",
                tool_name="bash",
            )

        if decision.level == PermissionLevel.ALLOW:
            logger.debug(
                "Bash command auto-allowed (%s: %r): %s", decision.source.value, decision.matched_pattern, command
            )
            return True

        # ASK. Non-interactive flows must raise here rather than defer: under a
        # permissive coarse rule (or dangerous profile overrides) the main flow
        # could silently allow what the command-level rules said to confirm.
        if self.non_interactive:
            profile = getattr(self.permission_manager, "active_profile", None) or "auto"
            raise PermissionDeniedException(
                (
                    f"PERMISSION_DENIED: Bash command requires user confirmation "
                    f"({decision.reason}) but this flow runs non-interactively under "
                    f"the '{profile}' profile. STOP retrying — surface the failure "
                    f"to the caller."
                ),
                tool_category="bash_tools",
                tool_name="bash",
            )

        # Reserved LLM-classifier seam: only for non-safety asks, fail closed.
        if self.bash_classifier is not None and not decision.safety_forced:
            try:
                verdict = await self.bash_classifier.classify(
                    command,
                    BashClassifierContext(cwd=self.project_root or ".", node_name=self.node_name),
                )
                if (
                    verdict is not None
                    and PermissionLevel(verdict.permission) == PermissionLevel.ALLOW
                    and verdict.confidence >= rules.classifier.confidence_threshold
                ):
                    logger.info("Bash command auto-allowed by classifier (%.2f): %s", verdict.confidence, command)
                    return True
            except Exception as e:
                logger.warning("Bash classifier failed (%s); falling back to confirmation prompt", e)

        # Session cache: the prompt's "always allow" writes ONLY the bucketed
        # key so one approval never cascades past its command prefix; broad
        # keys still honor a deliberate wide approval (e.g. legacy grants).
        cache_keys = (f"bash_tools.bash::{decision.bucket}", "bash_tools.bash", "bash_tools.*")

        def _session_approved() -> bool:
            return any(self.permission_manager._session_approvals.get(key) for key in cache_keys)

        if _session_approved():
            logger.debug("Bash bucket %r already approved for session", decision.bucket)
            return True

        async with _get_permission_prompt_lock(self.broker):
            if _session_approved():
                return True
            offer_project = decision.source == BashDecisionSource.DEFAULT and not decision.safety_forced
            choice = await self._request_bash_confirmation(command, decision, offer_project=offer_project)
            if choice == "y":
                logger.info("User approved bash command (once): %s", command)
                return True
            if choice == "a":
                self.permission_manager.approve_for_session("bash_tools", f"bash::{decision.bucket}")
                logger.info("User approved bash bucket %r for session", decision.bucket)
                return True
            if choice == "p" and offer_project:
                pattern = self._bucket_to_allow_pattern(decision.bucket)
                persisted = self.permission_manager.add_project_bash_allow(pattern, self.project_root)
                # Session bucket too, so later same-bucket calls this session
                # skip the prompt even though global_config already allows them.
                self.permission_manager.approve_for_session("bash_tools", f"bash::{decision.bucket}")
                logger.info(
                    "User granted project-level bash allow %r%s",
                    pattern,
                    "" if persisted else " (disk write failed; session-only)",
                )
                return True
            logger.info("User rejected bash command: %s", command)
            raise PermissionDeniedException(
                "User rejected execution of bash command",
                tool_category="bash_tools",
                tool_name="bash",
            )

    @staticmethod
    def _bucket_to_allow_pattern(bucket: str) -> str:
        """Convert a session bucket into a persistable allow pattern.

        Plain buckets (``git push``, ``ls``) become prefix rules; a bucket
        that is already a rule pattern (contains ``:``) is used verbatim.
        """
        return bucket if ":" in bucket else f"{bucket}:*"

    async def _request_bash_confirmation(
        self,
        command: str,
        decision: BashRuleDecision,
        *,
        offer_project: bool,
    ) -> str:
        """Prompt for a bash command; returns the raw choice key ('' on cancel).

        Unlike ``_request_user_confirmation`` (bool), callers need the concrete
        choice to distinguish session from project grants.
        """
        content = f"### Bash Command Permission\n\n```bash\n{command}\n```\n\n**Reason:** {decision.reason}\n"
        allow_pattern = self._bucket_to_allow_pattern(decision.bucket)
        choices = {
            "y": "Allow (once)",
            "a": f"Allow '{decision.bucket}' (session)",
        }
        if offer_project:
            choices["p"] = f"Allow '{allow_pattern}' (project)"
        choices["n"] = "Deny"

        try:
            answers = await self.broker.request(
                [
                    InteractionEvent(
                        title="Bash Permission",
                        content=content,
                        choices=choices,
                        default_choice="n",
                    )
                ]
            )
            return answers[0][0] if answers and answers[0] else ""
        except InteractionCancelled:
            return ""
        except Exception as e:
            logger.error(f"Error in bash permission confirmation: {e}")
            return ""

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

        # Show tool arguments as a readable key/value block. Long SQL / nested
        # values land in fenced code blocks (never truncated); the TUI pages and
        # opens them in a pager.
        args_md = _format_tool_args_markdown(args)
        if args_md:
            content += args_md + "\n"

        try:
            answers = await self.broker.request(
                [
                    InteractionEvent(
                        title="Permission",
                        content=content,
                        choices={"y": "Allow (once)", "a": "Always allow (session)", "n": "Deny"},
                        default_choice="n",
                    )
                ]
            )
            choice = answers[0][0] if answers and answers[0] else ""

            if choice == "a":
                self.permission_manager.approve_for_session(category, pattern_name)
                if tool_name and tool_name != pattern_name:
                    self.permission_manager.approve_for_session(category, tool_name)
                return True
            elif choice == "y":
                return True
            else:
                return False

        except InteractionCancelled:
            return False
        except Exception as e:
            logger.error(f"Error in permission confirmation: {e}")
            return False
