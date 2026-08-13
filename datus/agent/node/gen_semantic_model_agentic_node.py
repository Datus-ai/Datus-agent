# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Compatibility stub for the retired ``gen_semantic_model`` agent."""

from typing import Any


class GenSemanticModelAgenticNode:
    """Reject direct legacy agent construction with an actionable replacement."""

    NODE_NAME = "gen_semantic_model"

    def __init__(self, agent_config: Any, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        from datus.agent.node.semantic_authoring import ensure_semantic_agent_available

        ensure_semantic_agent_available(self.NODE_NAME, agent_config)
        raise RuntimeError("gen_semantic_model is retired. Use semantic_modeling instead.")
