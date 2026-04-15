# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Shared test helpers for built-in agentic node tests (gen_job, migration).

Prefix with `_` so pytest does not collect this file as a test module.
"""

import pytest

from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from tests.unit_tests.mock_llm_model import build_simple_response

# ---------------------------------------------------------------------------
# Init helpers
# ---------------------------------------------------------------------------


def check_node_name(node_cls, agent_config, expected_name):
    node = node_cls(agent_config=agent_config, execution_mode="workflow")
    assert node.NODE_NAME == expected_name
    assert node.get_node_name() == expected_name


def check_inherits_agentic_node(node_cls, agent_config):
    from datus.agent.node.agentic_node import AgenticNode

    node = node_cls(agent_config=agent_config, execution_mode="workflow")
    assert isinstance(node, AgenticNode)


def check_node_id(node_cls, agent_config, expected_id):
    node = node_cls(agent_config=agent_config, execution_mode="workflow")
    assert node.id == expected_id


def check_tools_include(node_cls, agent_config, tool_name):
    node = node_cls(agent_config=agent_config, execution_mode="workflow")
    tool_names = [tool.name for tool in node.tools]
    assert tool_name in tool_names


def check_tools_exclude(node_cls, agent_config, tool_name):
    node = node_cls(agent_config=agent_config, execution_mode="workflow")
    tool_names = [tool.name for tool in node.tools]
    assert tool_name not in tool_names


def check_standard_db_tools(node_cls, agent_config):
    node = node_cls(agent_config=agent_config, execution_mode="workflow")
    tool_names = [tool.name for tool in node.tools]
    assert "list_tables" in tool_names
    assert "describe_table" in tool_names
    assert "read_query" in tool_names
    assert "get_table_ddl" in tool_names


def check_filesystem_tools(node_cls, agent_config):
    node = node_cls(agent_config=agent_config, execution_mode="workflow")
    tool_names = [tool.name for tool in node.tools]
    assert "read_file" in tool_names


def check_max_turns(node_cls, agent_config, expected_turns):
    node = node_cls(agent_config=agent_config, execution_mode="workflow")
    assert node.max_turns == expected_turns


def check_dynamic_db_func_tool(node_cls, agent_config):
    node = node_cls(agent_config=agent_config, execution_mode="workflow")
    assert node.db_func_tool is not None


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


async def check_execute_stream_raises_without_input(node_cls, agent_config):
    from datus.utils.exceptions import DatusException

    node = node_cls(agent_config=agent_config, execution_mode="workflow")
    assert node.input is None

    with pytest.raises(DatusException) as exc_info:
        async for _ in node.execute_stream():
            pass
    assert "input" in str(exc_info.value).lower()


async def check_execute_stream_basic_workflow(node_cls, agent_config, mock_llm, user_message):
    mock_llm.reset(
        responses=[
            build_simple_response("Completed successfully."),
        ]
    )

    node = node_cls(agent_config=agent_config, execution_mode="workflow")
    node.input = SemanticNodeInput(user_message=user_message)

    action_manager = ActionHistoryManager()
    actions = []
    async for action in node.execute_stream(action_manager):
        actions.append(action)

    assert len(actions) >= 2
    assert actions[0].role == ActionRole.USER
    assert actions[0].status == ActionStatus.PROCESSING
    assert actions[-1].status == ActionStatus.SUCCESS


async def check_execute_stream_error_handling(node_cls, agent_config, mock_llm, user_message):
    async def _raise_error(*args, **kwargs):
        raise RuntimeError("LLM connection error")
        yield  # noqa: F841 — makes this an async generator

    node = node_cls(agent_config=agent_config, execution_mode="workflow")
    node.input = SemanticNodeInput(user_message=user_message)
    mock_llm.generate_with_tools_stream = _raise_error

    action_manager = ActionHistoryManager()
    actions = []
    async for action in node.execute_stream(action_manager):
        actions.append(action)

    assert len(actions) >= 2
    last = actions[-1]
    assert last.status == ActionStatus.FAILED
    assert last.action_type == "error"


# ---------------------------------------------------------------------------
# Node type helpers
# ---------------------------------------------------------------------------


def check_node_type_constant(type_attr, expected_value):
    from datus.configuration.node_type import NodeType

    assert hasattr(NodeType, type_attr)
    assert getattr(NodeType, type_attr) == expected_value


def check_node_type_in_action_types(type_attr):
    from datus.configuration.node_type import NodeType

    assert getattr(NodeType, type_attr) in NodeType.ACTION_TYPES


def check_node_factory(node_cls, node_type, agent_config):
    from datus.agent.node.node import Node

    node = Node.new_instance(
        node_id="test_node",
        description="Test factory",
        node_type=node_type,
        input_data=None,
        agent_config=agent_config,
        tools=[],
    )
    assert isinstance(node, node_cls)
    assert node.execution_mode == "workflow"


def check_node_factory_with_input(node_cls, node_type, agent_config, user_message):
    from datus.agent.node.node import Node

    input_data = SemanticNodeInput(user_message=user_message)
    node = Node.new_instance(
        node_id="test_node",
        description="Test factory with input",
        node_type=node_type,
        input_data=input_data,
        agent_config=agent_config,
        tools=[],
    )
    assert isinstance(node, node_cls)
    assert node.input is not None
    assert node.input.user_message == user_message
