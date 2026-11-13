# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""A replaced tool name in a sub-agent's ``tools`` list keeps its capability."""

import pytest

from datus.configuration import legacy_tool_names
from datus.configuration.agent_config import AgentConfig, NodeConfig
from datus.configuration.legacy_tool_names import expand_replaced_tools
from datus.schemas.agent_models import SubAgentConfig

OLD = "semantic_tools.attribution_analyze"
NEW = "semantic_tools.attribute_metrics, semantic_tools.attribute_dimensions"


@pytest.fixture(autouse=True)
def _fresh_warnings():
    legacy_tool_names._reset_warnings_for_tests()
    yield
    legacy_tool_names._reset_warnings_for_tests()


@pytest.fixture
def warnings_logged(monkeypatch):
    messages = []
    monkeypatch.setattr(legacy_tool_names.logger, "warning", lambda message, *a, **k: messages.append(message))
    return messages


@pytest.mark.parametrize(
    "tools, expanded",
    [
        (OLD, NEW),
        (
            f"semantic_tools.query_metrics, {OLD}",
            "semantic_tools.query_metrics, semantic_tools.attribute_metrics, semantic_tools.attribute_dimensions",
        ),
        (
            [OLD, "db_tools.*"],
            ["semantic_tools.attribute_metrics", "semantic_tools.attribute_dimensions", "db_tools.*"],
        ),
        # Already listing a successor: no duplicate.
        (
            f"semantic_tools.attribute_metrics,{OLD}",
            "semantic_tools.attribute_metrics, semantic_tools.attribute_dimensions",
        ),
    ],
)
def test_old_name_becomes_its_successors(tools, expanded, warnings_logged):
    assert expand_replaced_tools("ask_sales", tools) == expanded
    assert warnings_logged == [
        "agentic_nodes.ask_sales.tools: `semantic_tools.attribution_analyze` has been replaced by "
        "`semantic_tools.attribute_metrics` and `semantic_tools.attribute_dimensions`; both are enabled for "
        "this run. Rename it in agent.yml — the old name stops working in the next release."
    ]


@pytest.mark.parametrize(
    "tools", ["semantic_tools.*", "semantic_tools", "db_tools.*, semantic_tools.query_metrics", "", None, 3]
)
def test_other_lists_are_left_as_they_are(tools, warnings_logged):
    assert expand_replaced_tools("chat", tools) is tools
    assert warnings_logged == []


def test_one_warning_per_sub_agent_per_process(warnings_logged):
    for _ in range(3):
        expand_replaced_tools("ask_sales", OLD)
    expand_replaced_tools("ask_finance", OLD)

    assert len(warnings_logged) == 2
    assert "agentic_nodes.ask_sales.tools" in warnings_logged[0]
    assert "agentic_nodes.ask_finance.tools" in warnings_logged[1]


def test_agent_yml_sub_agents_are_expanded_on_load(tmp_path, warnings_logged):
    config = AgentConfig(
        nodes={"test": NodeConfig(model="test-model", input=None)},
        home=str(tmp_path / "h"),
        target="mock",
        models={"mock": {"type": "openai", "api_key": "k", "model": "m"}},
        skip_init_dirs=True,
        agentic_nodes={
            "ask_sales": {"node_class": "ask_metrics", "tools": f"semantic_tools.query_metrics, {OLD}"},
            "chat_plain": {"tools": "semantic_tools.*"},
        },
    )

    assert config.sub_agent_config("ask_sales")["tools"] == (
        "semantic_tools.query_metrics, semantic_tools.attribute_metrics, semantic_tools.attribute_dimensions"
    )
    assert config.sub_agent_config("chat_plain")["tools"] == "semantic_tools.*"
    assert len(warnings_logged) == 1


def test_sub_agent_config_from_the_api_is_expanded(warnings_logged):
    config = SubAgentConfig(system_prompt="ask_sales", tools=[OLD, "semantic_tools.get_metric"])

    assert (
        config.tools == "semantic_tools.attribute_metrics,semantic_tools.attribute_dimensions,semantic_tools.get_metric"
    )
    assert "agentic_nodes.ask_sales.tools" in warnings_logged[0]
