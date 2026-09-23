"""Per-turn call statistics hook for hosts that embed datus-agent.

When a chat turn finishes — completed, failed, or stopped by the user — the
task manager hands the host a :class:`TurnStatsEvent` summarising which
subagents ran and which tools were called, with their outcomes. A host (e.g.
a SaaS wrapper) persists it for usage dashboards.

The event is emitted from the background task itself, not from the SSE
stream, so a client that disconnects mid-turn does not lose the record.

The hook is optional and **fire-and-forget**: the task manager never awaits
the host on the turn's critical path and swallows its errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from fastapi import Request

from datus.api.models.cli_models import StreamChatInput

TurnStatus = Literal["completed", "error", "cancelled"]
SubagentKind = Literal["builtin", "custom"]
SubagentEntry = Literal["dispatch", "direct"]
ToolCaller = Literal["main", "subagent"]


@dataclass
class SubagentCallStat:
    """Invocations of one subagent root type within a turn.

    ``name`` is the root type (``gen_sql``, ``ask_metrics`` …): a custom
    subagent is reported under the built-in class it derives from, never under
    its user-chosen name. ``entry`` separates a main-agent ``task()`` dispatch
    from a user chatting with the subagent directly.
    """

    name: str
    kind: SubagentKind
    entry: SubagentEntry
    calls: int = 0
    success: int = 0
    failed: int = 0
    interrupted: int = 0
    duration_ms: int = 0
    tool_calls: int = 0


@dataclass
class ToolCallStat:
    """Calls of one tool within a turn, split by who made them.

    ``interrupted`` counts calls that started but never produced a result —
    the turn was stopped, a permission was denied, or the run crashed.
    """

    name: str
    caller: ToolCaller
    calls: int = 0
    success: int = 0
    failed: int = 0
    interrupted: int = 0
    duration_ms: int = 0


@dataclass
class TurnStatsEvent:
    """Summary of one chat turn, emitted once the turn is over.

    ``agent_name`` is ``"chat"`` for the main agent, otherwise the subagent the
    user is talking to directly (whose invocation also appears in
    ``subagents`` with ``entry="direct"``). ``context`` is whatever the host
    returned from :meth:`TurnStatsHook.capture_context` when the turn started.
    """

    session_id: str
    user_id: Optional[str]
    agent_name: str
    status: TurnStatus
    duration_ms: int
    subagents: List[SubagentCallStat] = field(default_factory=list)
    tools: List[ToolCallStat] = field(default_factory=list)
    origin: Optional[str] = None
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TurnStatsHook(Protocol):
    """Extension contract for recording per-turn call statistics."""

    def capture_context(
        self,
        http_request: Request,
        stream_request: StreamChatInput,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Snapshot request-scoped data (trace id, tenant ids …) at turn start.

        The HTTP request is gone by the time the turn ends, so anything the host
        needs later must be captured here. Must not raise.
        """
        ...

    async def on_turn_finished(self, event: TurnStatsEvent) -> None:
        """Record a finished turn. Runs as a background task; must not raise."""
        ...


_turn_stats_hook: Optional[TurnStatsHook] = None


def set_turn_stats_hook(hook: Optional[TurnStatsHook]) -> None:
    """Register (or clear, with ``None``) the active turn-stats hook."""
    global _turn_stats_hook
    _turn_stats_hook = hook


def get_turn_stats_hook() -> Optional[TurnStatsHook]:
    """Return the active turn-stats hook, or ``None`` when unregistered."""
    return _turn_stats_hook
