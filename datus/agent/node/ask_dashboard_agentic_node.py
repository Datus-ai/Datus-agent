# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``ask_dashboard`` follow-up consultant for a visual dashboard artifact.

See :class:`BaseArtifactAskAgenticNode` for the shared lifecycle. This
file only declares the per-kind constants the base class branches on.
"""

from __future__ import annotations

from datus.agent.node.base_artifact_ask_agentic_node import BaseArtifactAskAgenticNode


class AskDashboardAgenticNode(BaseArtifactAskAgenticNode):
    """Follow-up subagent bound to a single ``dashboards/<slug>/`` artifact."""

    NODE_NAME = "ask_dashboard"
    ARTIFACT_KIND = "dashboard"
    ARTIFACT_ROOT_DIR_NAME = "dashboards"
