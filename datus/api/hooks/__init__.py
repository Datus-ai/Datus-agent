"""Public extension points for hosts that embed datus-agent.

A host (e.g. a SaaS wrapper) registers hook implementations during
lifespan startup. The agent's HTTP routes call into these hooks at well
defined points; if no hook is registered, behavior is unchanged.
"""

from datus.api.hooks.chat_hooks import (
    ChatHooks,
    ChatPostUsageContext,
    ChatPreCheckOutcome,
    PostUsageFn,
    PreCheckFn,
    get_chat_hooks,
    make_chat_hooks,
    set_chat_hooks,
)
from datus.api.hooks.metric_hooks import (
    MetricRetrievalEvent,
    MetricRetrievalHook,
    RetrievalFn,
    get_metric_retrieval_hook,
    make_metric_retrieval_hook,
    set_metric_retrieval_hook,
)
from datus.api.hooks.turn_stats_hooks import (
    SubagentCallStat,
    ToolCallStat,
    TurnStatsEvent,
    TurnStatsHook,
    get_turn_stats_hook,
    set_turn_stats_hook,
)

__all__ = [
    "ChatHooks",
    "ChatPostUsageContext",
    "ChatPreCheckOutcome",
    "PostUsageFn",
    "PreCheckFn",
    "get_chat_hooks",
    "make_chat_hooks",
    "set_chat_hooks",
    "MetricRetrievalEvent",
    "MetricRetrievalHook",
    "RetrievalFn",
    "get_metric_retrieval_hook",
    "make_metric_retrieval_hook",
    "set_metric_retrieval_hook",
    "SubagentCallStat",
    "ToolCallStat",
    "TurnStatsEvent",
    "TurnStatsHook",
    "get_turn_stats_hook",
    "set_turn_stats_hook",
]
