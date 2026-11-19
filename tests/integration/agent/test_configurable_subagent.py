# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Small real-model acceptance cases using the existing nightly Agent fixture."""

import json
import os

import pytest

from datus.agent.node.node_factory import create_interactive_node, create_node_input
from datus.schemas.action_history import ActionRole, ActionStatus
from datus.tools.func_tool.sub_agent_task_tool import BUILTIN_SUBAGENT_DESCRIPTIONS


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY is not configured")
@pytest.mark.parametrize("query_database", [False, True], ids=["instructions-json", "sqlite-tool-json"])
async def test_configurable_subagent_real_model(nightly_agent_config, query_database):
    config = nightly_agent_config
    declaration = {
        "node_class": "configurable",
        "instructions": "Follow the user's requested JSON contract exactly. Use only supplied capabilities.",
        "output": {"format": "json"},
    }
    if query_database:
        declaration.update(tools=["db_tools.execute_sql"], scoped_context={"datasource": config.current_datasource})
        prompt = 'Use execute_sql to count all rows in the schools table. Return only {"count": <integer>}.'
    else:
        prompt = 'For the topic ontology, return exactly {"topic": "ontology", "ready": true}.'
    config.agentic_nodes["nightly_configurable"] = declaration
    node = create_interactive_node("nightly_configurable", config, execution_mode="workflow")
    node.max_turns = 3
    node.input = create_node_input(prompt, node)
    try:
        actions = [action async for action in node.execute_stream()]
        assert node.result.success is True, node.result.error
        calls = [action for action in actions if action.role == ActionRole.TOOL]
        if query_database:
            assert any(
                action.function_name() == "execute_sql" and action.status == ActionStatus.SUCCESS for action in calls
            )
            assert node.result.data == {"count": 17686}
        else:
            assert calls == []
            assert node.result.data == {"topic": "ontology", "ready": True}
    finally:
        await node.aclose()


def _routing_chat(config, description):
    config.agentic_nodes["semantic_modeling"] = {
        "node_class": "configurable",
        "description": description,
        "instructions": (
            "You are a routing acceptance delegate. This acceptance run only drafts a proposal, "
            "with no file or data writes. Acknowledge the requested definition changes as a proposal. "
            "Use only the caller-supplied schema and definitions; do not add fields or infer other tables. "
            "Return exactly three JSON keys: status='proposed', proposal_id='modeling-proposal', "
            "and a short summary of the supplied definitions. Do not claim inspection, validation or publication."
        ),
        "scoped_context": {"datasource": config.current_datasource, "tables": "schools"},
        "output": {"format": "json"},
        "max_turns": 2,
    }
    config.agentic_nodes.setdefault("chat", {}).update(
        tools="db_tools.execute_sql,sub_agent_tools.task", subagents="semantic_modeling", max_turns=4
    )
    return create_interactive_node(None, config, execution_mode="workflow")


async def _run_chat(node, message):
    node.input = create_node_input(message, node)
    actions = [action async for action in node.execute_stream()]
    assert node.result.success is True, node.result.error
    return actions


def _task_calls(actions):
    return [
        action
        for action in actions
        if action.role == ActionRole.TOOL and action.action_type == "task" and action.status == ActionStatus.SUCCESS
    ]


def _arguments(call):
    arguments = call.input["arguments"]
    return json.loads(arguments) if isinstance(arguments, str) else arguments


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY is not configured")
async def test_chat_answers_data_question_without_modeling_handoff(nightly_agent_config):
    node = _routing_chat(nightly_agent_config, BUILTIN_SUBAGENT_DESCRIPTIONS["semantic_modeling"])
    try:
        actions = await _run_chat(
            node, "How many rows are in the schools table? Answer from the database without changing definitions."
        )
        assert not any(action.role == ActionRole.TOOL and action.action_type == "task" for action in actions)
        assert any(action.action_type == "execute_sql" and action.status == ActionStatus.SUCCESS for action in actions)
        assert "17686" in node.result.response.replace(",", "")
    finally:
        await node.aclose()


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY is not configured")
async def test_chat_delegates_semantic_proposal_using_capabilities(nightly_agent_config):
    """Exercise real routing, with a definition proposal rather than asset publication."""
    node = _routing_chat(nightly_agent_config, BUILTIN_SUBAGENT_DESCRIPTIONS["semantic_modeling"])
    try:
        actions = await _run_chat(
            node,
            "Use this supplied schema as the complete input: schools(CDSCode TEXT PRIMARY KEY, School TEXT). "
            "Draft a semantic model proposal with a school_count metric defined as COUNT(*). "
            "No live discovery is requested. Return the proposal without writing or publishing assets.",
        )
        calls = _task_calls(actions)
        assert len(calls) == 1
        assert _arguments(calls[0])["type"] == "semantic_modeling"
    finally:
        await node.aclose()


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY is not configured")
async def test_chat_delegates_ontology_proposal_and_resumes_its_owner(nightly_agent_config):
    """The same public name advertises enhanced capabilities; no private runtime is exercised."""
    node = _routing_chat(
        nightly_agent_config,
        "Create and revise semantic models, metrics, ontology concepts and mappings. "
        "Own coordinated modeling proposals and their follow-up changes. "
        "Do not answer ordinary existing-data questions.",
    )
    try:
        actions = await _run_chat(
            node,
            "Use this supplied schema as the complete input: schools(CDSCode TEXT PRIMARY KEY, School TEXT). "
            "Draft a School ontology concept identified by CDSCode and map its name to School, "
            "keeping the semantic model and mapping consistent. No live discovery is requested. "
            "Return the proposal without writing or publishing assets.",
        )
        calls = _task_calls(actions)
        assert len(calls) == 1
        assert _arguments(calls[0])["type"] == "semantic_modeling"
        output = calls[0].output.get("raw_output", calls[0].output)
        session_id = output["result"]["session_id"]
        actions = await _run_chat(
            node, "Revise that same proposal: name the School concept Academy and keep the other decisions."
        )
        calls = _task_calls(actions)
        assert len(calls) == 1
        assert _arguments(calls[0])["type"] == "semantic_modeling"
        assert _arguments(calls[0])["session_id"] == session_id
    finally:
        await node.aclose()
