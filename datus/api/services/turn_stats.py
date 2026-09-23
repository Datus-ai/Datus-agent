"""Tally subagent and tool calls across one chat turn.

Every model backend streams a tool call as a PROCESSING action keyed by its
call id, then a ``complete_<call_id>`` action carrying SUCCESS / FAILED.
Subagent actions are forwarded into the parent's stream at ``depth=1`` with
``parent_action_id`` pointing at the ``task()`` call that spawned them, so a
single pass over the parent's stream sees both levels.

Calls are paired by id rather than counted on completion alone: a call that
started but never completed (turn stopped, permission denied, run crashed) is
reported as ``interrupted`` instead of silently vanishing.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from datus.api.hooks.turn_stats_hooks import (
    SubagentCallStat,
    SubagentEntry,
    SubagentKind,
    ToolCaller,
    ToolCallStat,
    TurnStatus,
)
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.tools.registry.tool_registry import known_tool_category
from datus.utils.constants import RETIRED_SYS_SUB_AGENTS, SYS_SUB_AGENTS

TASK_TOOL_NAME = "task"
_COMPLETE_PREFIX = "complete_"
# Addressable by name like SYS_SUB_AGENTS but kept out of that set.
_EXTRA_BUILTIN_AGENTS = frozenset({"explore", "ask_report", "ask_dashboard"})

SubagentResolver = Callable[[str], Tuple[str, SubagentKind]]


def resolve_subagent(agent_config: Any, name_or_id: str) -> Tuple[str, SubagentKind]:
    """Map a subagent name (or custom DB id) to ``(root_type, kind)``.

    Built-ins report under their own name. A custom subagent reports under the
    ``node_class`` it derives from, so statistics never fan out by user-chosen
    names. Unknown names fall back to ``gen_sql`` — the node factory's default.
    """
    if name_or_id in RETIRED_SYS_SUB_AGENTS:
        return "semantic_modeling", "builtin"
    if name_or_id in SYS_SUB_AGENTS or name_or_id in _EXTRA_BUILTIN_AGENTS:
        return name_or_id, "builtin"

    entry = _find_agentic_node(agent_config, name_or_id)
    node_class = _entry_field(entry, "node_class") or _entry_field(entry, "type")
    if node_class in RETIRED_SYS_SUB_AGENTS:
        return "semantic_modeling", "custom"
    return (node_class or "gen_sql"), "custom"


def _find_agentic_node(agent_config: Any, name_or_id: str) -> Any:
    nodes = getattr(agent_config, "agentic_nodes", None) or {}
    if name_or_id in nodes:
        return nodes[name_or_id]
    for entry in nodes.values():
        if _entry_field(entry, "id") == name_or_id:
            return entry
    return None


def _entry_field(entry: Any, key: str) -> Optional[str]:
    if entry is None:
        return None
    value = entry.get(key) if isinstance(entry, dict) else getattr(entry, key, None)
    return value if isinstance(value, str) and value else None


def _elapsed_ms(start: Optional[datetime], end: Optional[datetime] = None) -> int:
    if start is None:
        return 0
    return max(0, int(((end or datetime.now()) - start).total_seconds() * 1000))


def _task_subagent_name(action: ActionHistory) -> Optional[str]:
    """Read the ``type`` argument off a ``task()`` call's start action."""
    payload = action.input if isinstance(action.input, dict) else {}
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            return None
    if not isinstance(arguments, dict):
        return None
    name = arguments.get("type")
    return name if isinstance(name, str) and name else None


class TurnStatsCollector:
    """Accumulates one turn's calls from the parent node's action stream."""

    def __init__(self, resolve: SubagentResolver):
        self._resolve = resolve
        self._tools: Dict[Tuple[str, ToolCaller], ToolCallStat] = {}
        self._subagents: Dict[Tuple[str, SubagentKind, SubagentEntry], SubagentCallStat] = {}
        # call_id -> (tool name, caller, start time)
        self._pending_tools: Dict[str, Tuple[str, ToolCaller, datetime]] = {}
        # task() call_id -> (root type, kind, start time)
        self._pending_tasks: Dict[str, Tuple[str, SubagentKind, datetime]] = {}
        # task() call_id -> tools its subagent started
        self._task_tool_calls: Dict[str, int] = {}
        self._main_tool_calls = 0

    def observe(self, action: ActionHistory) -> None:
        if action.role != ActionRole.TOOL or not action.action_id:
            return

        if action.action_id.startswith(_COMPLETE_PREFIX):
            self._on_complete(action, action.action_id[len(_COMPLETE_PREFIX) :])
        elif action.status == ActionStatus.PROCESSING:
            self._on_start(action)

    def _on_start(self, action: ActionHistory) -> None:
        call_id = action.action_id
        if call_id in self._pending_tools:
            return

        caller: ToolCaller = "subagent" if action.depth else "main"
        self._pending_tools[call_id] = (action.action_type, caller, action.start_time)
        if caller == "main":
            self._main_tool_calls += 1
        elif action.parent_action_id in self._task_tool_calls:
            self._task_tool_calls[action.parent_action_id] += 1

        if caller == "main" and action.action_type == TASK_TOOL_NAME:
            name = _task_subagent_name(action)
            if name:
                root, kind = self._resolve(name)
                self._pending_tasks[call_id] = (root, kind, action.start_time)
                self._task_tool_calls[call_id] = 0

    def _on_complete(self, action: ActionHistory, call_id: str) -> None:
        pending = self._pending_tools.pop(call_id, None)
        caller: ToolCaller = pending[1] if pending else ("subagent" if action.depth else "main")
        start = pending[2] if pending else action.start_time
        failed = action.status == ActionStatus.FAILED
        duration = _elapsed_ms(start, action.end_time)

        stat = self._tool(action.action_type, caller)
        stat.calls += 1
        stat.duration_ms += duration
        if failed:
            stat.failed += 1
        else:
            stat.success += 1

        task = self._pending_tasks.pop(call_id, None)
        if task:
            root, kind, task_start = task
            sub = self._subagent(root, kind, "dispatch")
            sub.calls += 1
            sub.duration_ms += _elapsed_ms(task_start, action.end_time)
            sub.tool_calls += self._task_tool_calls.pop(call_id, 0)
            if failed:
                sub.failed += 1
            else:
                sub.success += 1

    def finalize(
        self,
        status: TurnStatus,
        duration_ms: int,
        direct: Optional[Tuple[str, SubagentKind]] = None,
    ) -> Tuple[List[SubagentCallStat], List[ToolCallStat]]:
        """Close out the turn: anything still pending was interrupted.

        ``direct`` is the ``(root_type, kind)`` of the subagent the user is
        chatting with directly, recorded as one invocation of that subagent.
        """
        now = datetime.now()
        for name, caller, start in self._pending_tools.values():
            stat = self._tool(name, caller)
            stat.calls += 1
            stat.interrupted += 1
            stat.duration_ms += _elapsed_ms(start, now)
        self._pending_tools.clear()

        for call_id, (root, kind, start) in self._pending_tasks.items():
            sub = self._subagent(root, kind, "dispatch")
            sub.calls += 1
            sub.interrupted += 1
            sub.duration_ms += _elapsed_ms(start, now)
            sub.tool_calls += self._task_tool_calls.get(call_id, 0)
        self._pending_tasks.clear()

        if direct:
            root, kind = direct
            sub = self._subagent(root, kind, "direct")
            sub.calls += 1
            sub.duration_ms += duration_ms
            sub.tool_calls += self._main_tool_calls
            if status == "completed":
                sub.success += 1
            elif status == "error":
                sub.failed += 1
            else:
                sub.interrupted += 1

        # Looked up at the end: subagent nodes register their tools as they run.
        for stat in self._tools.values():
            stat.group = known_tool_category(stat.name) or ""
        return list(self._subagents.values()), list(self._tools.values())

    def _tool(self, name: str, caller: ToolCaller) -> ToolCallStat:
        key = (name, caller)
        if key not in self._tools:
            self._tools[key] = ToolCallStat(name=name, caller=caller)
        return self._tools[key]

    def _subagent(self, name: str, kind: SubagentKind, entry: SubagentEntry) -> SubagentCallStat:
        key = (name, kind, entry)
        if key not in self._subagents:
            self._subagents[key] = SubagentCallStat(name=name, kind=kind, entry=entry)
        return self._subagents[key]
