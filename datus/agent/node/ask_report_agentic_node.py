# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``ask_report`` follow-up consultant for a visual report artifact.

See :class:`BaseArtifactAskAgenticNode` for the shared lifecycle. This
file only declares the per-kind constants the base class branches on.
"""

from __future__ import annotations

from datus.agent.node.base_artifact_ask_agentic_node import BaseArtifactAskAgenticNode


class AskReportAgenticNode(BaseArtifactAskAgenticNode):
    """Follow-up subagent bound to a single ``reports/<slug>/`` artifact."""

    NODE_NAME = "ask_report"
    ARTIFACT_KIND = "report"
    ARTIFACT_ROOT_DIR_NAME = "reports"
    # The SaaS path always seeds an ``artifact_blob`` (latest published
    # version) into the agentic_nodes entry, so missing blob == unpublished
    # report (half-bound state). Fail loud rather than silently dropping to
    # a disk path the backend may not even have access to.
    BLOB_REQUIRED = True
