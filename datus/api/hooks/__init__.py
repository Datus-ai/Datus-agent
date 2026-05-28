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

__all__ = [
    "ChatHooks",
    "ChatPostUsageContext",
    "ChatPreCheckOutcome",
    "PostUsageFn",
    "PreCheckFn",
    "get_chat_hooks",
    "make_chat_hooks",
    "set_chat_hooks",
]
