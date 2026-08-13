# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Integration coverage for the unified Dosi semantic-modeling node."""

import pytest

from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
from datus.utils.path_manager import DatusPathManager

pytestmark = [pytest.mark.nightly, pytest.mark.product_e2e]


def _use_dosi(agent_config, project_root):
    agent_config.semantic_layer_configs = {"dosi": {}}
    agent_config._project_root = project_root.resolve()
    agent_config.path_manager = DatusPathManager(
        agent_config.home,
        project_name=agent_config.project_name,
        project_root=str(project_root),
    )
    return agent_config


def test_semantic_modeling_scopes_share_one_real_project_workspace(nightly_agent_config, tmp_path):
    config = _use_dosi(nightly_agent_config, tmp_path)

    full_node = SemanticModelingAgenticNode(
        agent_config=config,
        execution_mode="workflow",
        authoring_scope="full",
    )
    datasets_node = SemanticModelingAgenticNode(
        agent_config=config,
        execution_mode="workflow",
        authoring_scope="datasets",
    )

    assert full_node.get_node_name() == "semantic_modeling"
    assert full_node.filesystem_func_tool.config.root_path == str(config.path_manager.subject_dir.parent)
    assert config.path_manager.semantic_model_path(config.current_datasource).is_relative_to(
        config.path_manager.subject_dir / "semantic_models"
    )

    full_tools = {tool.name for tool in full_node.tools}
    datasets_tools = {tool.name for tool in datasets_node.tools}
    assert {"upsert_osi_datasets", "delete_osi_datasets", "upsert_osi_metrics", "delete_osi_metrics"} <= full_tools
    assert {"upsert_osi_datasets", "delete_osi_datasets"} <= datasets_tools
    assert {"upsert_osi_metrics", "delete_osi_metrics"}.isdisjoint(datasets_tools)
