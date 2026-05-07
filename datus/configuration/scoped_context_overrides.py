# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Runtime overrides for sub-agent configuration.

Allows ``SubAgentTaskTool`` to install an effective ``SubAgentConfig`` (with a
parent-merged ``scoped_context``) for the duration of a child agent's
execution. Consumers that already look up sub-agent config by name (e.g.
``rag_scope._build_sub_agent_filter``) automatically observe the override
without changes.

Backed by ``contextvars.ContextVar`` so concurrent ``asyncio.Task`` siblings
remain isolated (each Task copies the current context at creation).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterator, Optional

from datus.schemas.agent_models import SubAgentConfig

_OVERRIDES: ContextVar[Optional[Dict[str, SubAgentConfig]]] = ContextVar("datus_subagent_overrides", default=None)


def _current_map() -> Dict[str, SubAgentConfig]:
    return _OVERRIDES.get() or {}


def get_override(sub_agent_name: str) -> Optional[SubAgentConfig]:
    return _current_map().get(sub_agent_name)


@contextmanager
def effective_subagent(sub_agent_name: str, cfg: SubAgentConfig) -> Iterator[None]:
    new_map = {**_current_map(), sub_agent_name: cfg}
    token = _OVERRIDES.set(new_map)
    try:
        yield
    finally:
        _OVERRIDES.reset(token)
