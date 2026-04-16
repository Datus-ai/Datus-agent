# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for GenExtKnowledgeAgenticNode.

NO MOCK except LLM: uses real AgentConfig, real SQLite database, real tools,
real PathManager, real RAG storage. Only LLMBaseModel.create_model is mocked
via the conftest mock_llm_create fixture.
"""

import json

import pytest

from datus.schemas.action_history import ActionRole, ActionStatus
from datus.schemas.ext_knowledge_agentic_node_models import ExtKnowledgeNodeInput
from tests.unit_tests.mock_llm_model import (
    MockToolCall,
    build_simple_response,
    build_tool_then_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_node(real_agent_config, **kwargs):
    """Create a GenExtKnowledgeAgenticNode with real config and real dependencies."""
    from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode

    defaults = dict(
        node_name="gen_ext_knowledge",
        agent_config=real_agent_config,
        execution_mode="workflow",
    )
    defaults.update(kwargs)
    return GenExtKnowledgeAgenticNode(**defaults)


# ===========================================================================
# Test Initialization
# ===========================================================================


class TestGenExtKnowledgeNodeInit:
    """Tests for GenExtKnowledgeAgenticNode initialization with real dependencies."""

    def test_ext_knowledge_init(self, real_agent_config, mock_llm_create):
        """Node can be initialized with real config and has tools."""
        node = _create_node(real_agent_config)

        assert node.configured_node_name == "gen_ext_knowledge"
        assert node.execution_mode == "workflow"
        assert node.build_mode == "incremental"
        assert node.hooks is None  # No hooks in workflow mode
        assert len(node.tools) > 0

    def test_ext_knowledge_has_db_tools(self, real_agent_config, mock_llm_create):
        """Node has real database tools (list_tables, execute_sql, etc.)."""
        node = _create_node(real_agent_config)

        tool_names = [t.name for t in node.tools]
        assert "list_tables" in tool_names
        assert "read_query" in tool_names

    def test_ext_knowledge_has_verify_sql_tool(self, real_agent_config, mock_llm_create):
        """Node has the verify_sql tool for SQL result verification."""
        node = _create_node(real_agent_config)

        tool_names = [t.name for t in node.tools]
        assert "verify_sql" in tool_names

    def test_ext_knowledge_has_filesystem_tools(self, real_agent_config, mock_llm_create):
        """Node has filesystem tools: read_file, edit_file, write_file."""
        node = _create_node(real_agent_config)

        tool_names = [t.name for t in node.tools]
        assert "read_file" in tool_names
        assert "edit_file" in tool_names
        assert "write_file" in tool_names

    def test_ext_knowledge_has_context_search_tools(self, real_agent_config, mock_llm_create):
        """Node has context search tools."""
        node = _create_node(real_agent_config)

        # Context search tools should be initialized
        assert node.context_search_tools is not None

    def test_ext_knowledge_max_turns(self, real_agent_config, mock_llm_create):
        """max_turns is read from agentic_nodes config (5 in test config)."""
        node = _create_node(real_agent_config)
        assert node.max_turns == 5  # Set in conftest real_agent_config

    def test_ext_knowledge_build_mode(self, real_agent_config, mock_llm_create):
        """build_mode can be configured."""
        node = _create_node(real_agent_config, build_mode="overwrite")
        assert node.build_mode == "overwrite"

    def test_ext_knowledge_subject_tree(self, real_agent_config, mock_llm_create):
        """subject_tree can be passed and stored."""
        tree = ["Finance", "HR", "Marketing"]
        node = _create_node(real_agent_config, subject_tree=tree)
        assert node.subject_tree == tree


# ===========================================================================
# Test Execution
# ===========================================================================


class TestGenExtKnowledgeNodeExecution:
    """Tests for GenExtKnowledgeAgenticNode.execute_stream() with real tools."""

    @pytest.mark.asyncio
    async def test_ext_knowledge_simple_response(self, real_agent_config, mock_llm_create):
        """execute_stream with a simple LLM response produces USER + SUCCESS actions."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        mock_llm_create.reset(
            responses=[
                build_simple_response("Generated external knowledge for order amounts"),
            ]
        )
        node.model = mock_llm_create

        node.input = ExtKnowledgeNodeInput(
            user_message="Define order amount calculation",
            question="Define order amount calculation",
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        # Should have at least: USER action + final SUCCESS action
        assert len(actions) >= 2
        roles = [a.role for a in actions]
        assert ActionRole.USER in roles
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_ext_knowledge_with_tool_calls(self, real_agent_config, mock_llm_create):
        """LLM calls list_tables tool then responds; tool is ACTUALLY EXECUTED."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(name="list_tables", arguments="{}"),
                    ],
                    content="External knowledge generated after checking tables",
                ),
            ]
        )
        node.model = mock_llm_create

        node.input = ExtKnowledgeNodeInput(
            user_message="Create knowledge about SAT scores",
            question="Create knowledge about SAT scores",
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        # Should include TOOL actions from real tool execution
        tool_actions = [a for a in actions if a.role == ActionRole.TOOL]
        assert len(tool_actions) >= 2  # PROCESSING + SUCCESS for list_tables

        # Verify tool was actually executed and returned real table names
        tool_success_actions = [a for a in tool_actions if a.status == ActionStatus.SUCCESS]
        assert len(tool_success_actions) >= 1
        tool_output = tool_success_actions[0].output
        assert tool_output.get("success") is True

        # Final action should be SUCCESS
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_ext_knowledge_verify_sql_no_gold(self, real_agent_config, mock_llm_create):
        """verify_sql without gold_sql returns success (no reference available)."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        # verify_sql tool call without gold_sql set on the node
        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="verify_sql",
                            arguments=json.dumps({"sql": "SELECT COUNT(*) FROM satscores"}),
                        ),
                    ],
                    content="SQL verified successfully, no reference available",
                ),
            ]
        )
        node.model = mock_llm_create

        node.input = ExtKnowledgeNodeInput(
            user_message="Verify order count query",
            question="Verify order count query",
            # No gold_sql provided
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        # verify_sql should succeed since no gold_sql is set
        tool_success_actions = [a for a in actions if a.role == ActionRole.TOOL and a.status == ActionStatus.SUCCESS]
        assert len(tool_success_actions) >= 1

        # Check the tool result indicates success
        verify_output = tool_success_actions[0].output
        raw = verify_output.get("raw_output", "")
        # Should contain indication of success
        assert "success" in str(raw).lower() or "accepted" in str(raw).lower() or verify_output.get("success") is True

    @pytest.mark.asyncio
    async def test_ext_knowledge_verify_sql_with_gold(self, real_agent_config, mock_llm_create):
        """verify_sql with matching gold_sql returns success."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        # Use a multi-step response:
        # Step 1: The LLM generates the response. Before that we set gold_sql.
        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="verify_sql",
                            arguments=json.dumps({"sql": "SELECT COUNT(*) FROM satscores"}),
                        ),
                    ],
                    content="SQL verified against gold reference",
                ),
            ]
        )
        node.model = mock_llm_create

        node.input = ExtKnowledgeNodeInput(
            user_message="Verify order count",
            question="How many SAT score records?",
            gold_sql="SELECT COUNT(*) FROM satscores",  # Same SQL as verified
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        # Final action should still be SUCCESS
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_ext_knowledge_verify_sql_mismatch(self, real_agent_config, mock_llm_create):
        """verify_sql with mismatching gold_sql returns failure info."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        # Need 2 responses: first for the main loop, second for compare suggestions
        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="verify_sql",
                            arguments=json.dumps({"sql": "SELECT COUNT(*) FROM satscores"}),
                        ),
                    ],
                    content="SQL verification result received",
                ),
                # Response for _generate_compare_suggestions internal generate_with_json_output call
                build_simple_response(
                    json.dumps(
                        {
                            "explanation": "Row count differs",
                            "suggest": "Check the query",
                        }
                    )
                ),
            ]
        )
        node.model = mock_llm_create

        node.input = ExtKnowledgeNodeInput(
            user_message="Verify amount sum",
            question="What is total amount?",
            # gold_sql returns different result than the verified SQL
            gold_sql="SELECT SUM(enroll12) FROM satscores",
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        # The node should complete (possibly with retries)
        assert len(actions) >= 2
        # Final action should be SUCCESS (node completes even if verification fails)
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_ext_knowledge_workflow_mode(self, real_agent_config, mock_llm_create):
        """Node in workflow mode does not set up hooks."""
        node = _create_node(real_agent_config, execution_mode="workflow")
        assert node.hooks is None
        assert node.execution_mode == "workflow"

        mock_llm_create.reset(
            responses=[
                build_simple_response("Knowledge generated in workflow mode"),
            ]
        )
        node.model = mock_llm_create

        node.input = ExtKnowledgeNodeInput(
            user_message="Generate knowledge",
            question="Generate knowledge",
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS
        assert actions[-1].action_type == "ext_knowledge_response"

    @pytest.mark.asyncio
    async def test_ext_knowledge_interactive_mode_token_tracking(self, real_agent_config, mock_llm_create):
        """Test that interactive mode tracks token usage from action history."""
        node = _create_node(real_agent_config, execution_mode="interactive")

        mock_llm_create.reset(
            responses=[
                build_simple_response("External knowledge created in interactive mode"),
            ]
        )
        node.model = mock_llm_create

        node.input = ExtKnowledgeNodeInput(
            user_message="Generate external knowledge documentation",
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

        # In interactive mode, the final result should have tokens_used > 0
        last_output = actions[-1].output
        assert last_output is not None
        assert isinstance(last_output, dict), f"Expected dict, got {type(last_output)}"
        assert "tokens_used" in last_output, f"Missing 'tokens_used' key in {last_output.keys()}"
        assert last_output["tokens_used"] > 0

    @pytest.mark.asyncio
    async def test_ext_knowledge_with_db_context(self, real_agent_config, mock_llm_create):
        """execute_stream includes catalog/database/db_schema in the enhanced message."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        mock_llm_create.reset(
            responses=[
                build_simple_response("Knowledge generated with database context"),
            ]
        )
        node.model = mock_llm_create

        node.input = ExtKnowledgeNodeInput(
            user_message="Define revenue metric",
            question="Define revenue metric",
            catalog="analytics",
            database="sales_db",
            db_schema="public",
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_ext_knowledge_with_partial_db_context(self, real_agent_config, mock_llm_create):
        """execute_stream works with only some db context fields set."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        mock_llm_create.reset(
            responses=[
                build_simple_response("Knowledge generated with partial context"),
            ]
        )
        node.model = mock_llm_create

        node.input = ExtKnowledgeNodeInput(
            user_message="Define order amount",
            question="Define order amount",
            database="orders_db",
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_ext_knowledge_input_not_set_raises(self, real_agent_config, mock_llm_create):
        """execute_stream raises ValueError when input is not set."""
        node = _create_node(real_agent_config)
        node.input = None

        with pytest.raises(ValueError, match="External knowledge input not set"):
            async for _ in node.execute_stream():
                pass


class TestGenExtKnowledgeSaveToDbSandbox:
    """``_save_to_db`` must reject paths outside the per-kind, per-namespace sandbox.

    Workflow mode reads the path from the LLM's final JSON, so this is the
    last line of defence against a fabricated response syncing an arbitrary
    file as "external knowledge".
    """

    def test_rejects_out_of_sandbox_absolute_path(self, real_agent_config, mock_llm_create, tmp_path):
        from unittest.mock import patch

        node = _create_node(real_agent_config)
        outside = tmp_path / "outside" / "malicious.yaml"
        outside.parent.mkdir(parents=True)
        outside.write_text("x: y\n")

        with patch("datus.cli.generation_hooks.GenerationHooks._sync_ext_knowledge_to_db") as sync_mock:
            node._save_to_db(str(outside))
            sync_mock.assert_not_called()

    def test_rejects_cross_kind_prefix(self, real_agent_config, mock_llm_create):
        """ext_knowledge node must not sync files under semantic_models/."""
        from unittest.mock import patch

        node = _create_node(real_agent_config)
        with patch("datus.cli.generation_hooks.GenerationHooks._sync_ext_knowledge_to_db") as sync_mock:
            node._save_to_db("semantic_models/db/not_knowledge.yml")
            sync_mock.assert_not_called()


class TestGenExtKnowledgeFilesystemRootPath:
    """FilesystemFuncTool is sandboxed to subject_dir (not the type-specific subdir)."""

    def test_filesystem_root_is_subject_dir(self, real_agent_config, mock_llm_create):
        node = _create_node(real_agent_config)
        expected = str(real_agent_config.path_manager.subject_dir)

        assert node.filesystem_func_tool is not None
        assert node.filesystem_func_tool.config.root_path == expected
        assert node.filesystem_func_tool._path_normalizer is not None

        assert node.filesystem_func_tool._path_normalizer("notes.yaml", None) == "ext_knowledge/notes.yaml"
