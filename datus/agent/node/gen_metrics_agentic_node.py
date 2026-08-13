# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Compatibility stub for the retired ``gen_metrics`` agent."""

from typing import Any


class GenMetricsAgenticNode:
    """Reject direct legacy agent construction with an actionable replacement."""

    NODE_NAME = "gen_metrics"

    def __init__(self, agent_config: Any, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        from datus.agent.node.semantic_authoring import ensure_semantic_agent_available

        ensure_semantic_agent_available(self.NODE_NAME, agent_config)
        raise RuntimeError("gen_metrics is retired. Use semantic_modeling instead.")
