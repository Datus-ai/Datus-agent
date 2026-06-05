# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Nightly integration tests for GenVisualDashboardAgenticNode (issue #923).

Expands nightly coverage for the data-product ``gen_visual_dashboard`` subagent.

Two tiers:
* A deterministic ``@pytest.mark.nightly`` init test asserting the filesystem +
  DB tool surface — no LLM call.
* A real-LLM ``@pytest.mark.product_e2e`` test driving ``execute_stream`` end to
  end. The LLM authors a parameterized React-JSX dashboard bundle (query
  templates + render/app.jsx + manifest) against california_schools and
  finalizes it with ``validate_render``; the run must end SUCCESS and surface
  the compiled HTML via a ``dashboard_html_path`` action.

Artifacts are written under ``project_root``, which this test redirects to a
per-test tmp dir so the repository tree is never polluted and concurrent
nightly agents cannot collide on the same ``dashboards/`` directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from datus.agent.node.gen_visual_dashboard_agentic_node import GenVisualDashboardAgenticNode
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.gen_visual_dashboard_models import GenVisualDashboardNodeInput
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _make_node(agent_config, execution_mode="workflow"):
    return GenVisualDashboardAgenticNode(
        node_id="gen_visual_dashboard_node",
        description="Nightly visual dashboard node",
        node_type=NodeType.TYPE_GEN_VISUAL_DASHBOARD,
        agent_config=agent_config,
        node_name="gen_visual_dashboard",
        execution_mode=execution_mode,
    )


@pytest.fixture
def dashboard_project_config(nightly_agent_config, tmp_path):
    """nightly_agent_config with project_root redirected to a tmp workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    # ``project_root`` is a read-only property over ``_project_root``; repointing
    # the backing field redirects all dashboard artifacts into the tmp workspace.
    nightly_agent_config._project_root = workspace
    return nightly_agent_config


@pytest.mark.nightly
class TestGenVisualDashboardAgenticInit:
    """Deterministic node-construction coverage (no LLM)."""

    def test_node_initialization(self, nightly_agent_config):
        """Node initializes with the expected filesystem + DB tool surface."""
        node = _make_node(nightly_agent_config)

        assert node.get_node_name() == "gen_visual_dashboard", f"unexpected node name: {node.get_node_name()}"
        assert node.NODE_NAME == "gen_visual_dashboard", f"unexpected NODE_NAME: {node.NODE_NAME}"
        assert node.execution_mode == "workflow", f"unexpected execution mode: {node.execution_mode}"

        tool_names = [tool.name for tool in node.tools]
        assert "read_file" in tool_names, f"missing read_file, got: {tool_names}"
        assert "write_file" in tool_names, f"missing write_file, got: {tool_names}"
        assert "edit_file" in tool_names, f"missing edit_file, got: {tool_names}"
        assert "list_tables" in tool_names, f"missing list_tables, got: {tool_names}"

        logger.info("gen_visual_dashboard node initialized with %d tools: %s", len(node.tools), tool_names)


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set")
class TestGenVisualDashboardAgenticRealLLM:
    """Real-LLM smoke for the gen_visual_dashboard artifact path."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_execute_stream_builds_dashboard(self, dashboard_project_config):
        """LLM authors + validates a dashboard bundle; last action SUCCESS."""
        node = _make_node(dashboard_project_config, execution_mode="workflow")
        node.input = GenVisualDashboardNodeInput(
            user_message=(
                "Build a small dashboard titled 'County Schools Overview'. Author one "
                "parameterized query template that returns the number of schools per county "
                "from the schools table (group by County, order by the count descending). "
                "Declare a '-- @datus-params' header (it may take no parameters). Render a "
                "simple bar chart of county vs school count. Start a new dashboard, save the "
                "query template, author render/app.jsx, then call validate_render to finalize."
            ),
            database="california_schools",
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)
            logger.info("Action: role=%s status=%s type=%s", action.role, action.status, action.action_type)

        assert len(actions) >= 2, f"expected at least 2 actions, got {len(actions)}"
        assert actions[0].role == ActionRole.USER, f"first action role was {actions[0].role}"
        assert actions[0].status == ActionStatus.PROCESSING, f"first action status was {actions[0].status}"

        last = actions[-1]
        assert last.status == ActionStatus.SUCCESS, f"last action should be SUCCESS, got {last.status}: {last.output}"

        result = last.output
        assert isinstance(result, dict), f"final output should be a dict, got {type(result)}"
        assert result["success"] is True, f"dashboard run did not succeed: {result.get('error')}"
        assert result["dashboard_slug"], f"dashboard_slug should be populated, got {result.get('dashboard_slug')}"

        # CLI mode compiles a standalone HTML and surfaces its path in the stream.
        path_actions = [a for a in actions if a.action_type == "dashboard_html_path"]
        assert len(path_actions) == 1, f"expected exactly one dashboard_html_path action, got {len(path_actions)}"
        html_path = path_actions[0].output["html_path"]
        assert Path(html_path).is_file(), f"compiled dashboard HTML missing on disk: {html_path}"
