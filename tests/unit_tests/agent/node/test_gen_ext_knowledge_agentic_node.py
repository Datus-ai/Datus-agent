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
from pathlib import Path

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

    def test_ext_knowledge_verify_sql_absent_at_init(self, real_agent_config, mock_llm_create):
        """verify_sql is NOT registered at __init__: it is bound lazily inside
        execute_stream only when a non-empty, runnable gold_sql is supplied.
        """
        node = _create_node(real_agent_config)

        tool_names = [t.name for t in node.tools]
        assert "verify_sql" not in tool_names

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


@pytest.mark.acceptance
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
    async def test_ext_knowledge_no_gold_sql_leaves_verify_sql_unregistered(self, real_agent_config, mock_llm_create):
        """Without gold_sql, execute_stream must NOT register verify_sql — the
        Prompt instructs the model to skip PHASE 2 entirely."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        mock_llm_create.reset(
            responses=[
                build_simple_response("Extracted knowledge without reference SQL"),
            ]
        )
        node.model = mock_llm_create

        node.input = ExtKnowledgeNodeInput(
            user_message="Define order amount",
            question="Define order amount",
            # No gold_sql
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        assert actions[-1].status == ActionStatus.SUCCESS
        tool_names = [t.name for t in node.tools]
        assert "verify_sql" not in tool_names

    @pytest.mark.asyncio
    async def test_ext_knowledge_valid_gold_sql_registers_verify_sql(self, real_agent_config, mock_llm_create):
        """A runnable gold_sql passes pre-validation and makes verify_sql available
        to the agent during execute_stream."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        mock_llm_create.reset(
            responses=[
                build_simple_response("Knowledge generated with valid gold SQL"),
            ]
        )
        node.model = mock_llm_create

        node.input = ExtKnowledgeNodeInput(
            user_message="Count SAT score rows",
            question="How many SAT score rows?",
            gold_sql="SELECT COUNT(*) FROM satscores",
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        assert actions[-1].status == ActionStatus.SUCCESS
        tool_names = [t.name for t in node.tools]
        assert "verify_sql" in tool_names

    @pytest.mark.asyncio
    async def test_ext_knowledge_invalid_gold_sql_fails_before_agent_loop(self, real_agent_config, mock_llm_create):
        """Unrunnable gold_sql raises DatusException caught by execute_stream's
        top-level handler and surfaces as a FAILED action without consuming any
        LLM turn. Covers both 'direct' and 'subagent' invocation shapes — the
        subagent wrapper reads the same FAILED action."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        # No LLM response should ever be consumed; seed with a guard response
        # that would make the test obvious if the agent loop did start.
        mock_llm_create.reset(
            responses=[
                build_simple_response("SHOULD NOT BE CALLED"),
            ]
        )
        node.model = mock_llm_create

        node.input = ExtKnowledgeNodeInput(
            user_message="Impossible",
            question="Impossible",
            gold_sql="SELECT * FROM __no_such_table__",
        )

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        # Final action must be FAILED with the gold SQL error message bubbled up.
        assert actions[-1].status == ActionStatus.FAILED
        last_output = actions[-1].output
        assert isinstance(last_output, dict)
        error_text = (last_output.get("error") or "").lower()
        # Error is wrapped by DatusException template which includes the
        # "Gold SQL failed to execute" preamble plus the underlying error.
        assert "gold sql" in error_text or "no such table" in error_text

        # verify_sql must not have been added since validation failed before
        # the enable hook ran.
        tool_names = [t.name for t in node.tools]
        assert "verify_sql" not in tool_names

    @pytest.mark.asyncio
    async def test_ext_knowledge_tool_state_is_idempotent_across_runs(self, real_agent_config, mock_llm_create):
        """Running execute_stream twice with different gold_sql states must not
        leak verify_sql between runs."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        # First run: with gold_sql → verify_sql registered
        mock_llm_create.reset(
            responses=[
                build_simple_response("First run"),
                build_simple_response("Second run"),
            ]
        )
        node.model = mock_llm_create

        node.input = ExtKnowledgeNodeInput(
            user_message="Count SAT",
            question="Count SAT",
            gold_sql="SELECT COUNT(*) FROM satscores",
        )
        async for _ in node.execute_stream():
            pass
        assert "verify_sql" in [t.name for t in node.tools]

        # Second run: no gold_sql → verify_sql must be removed
        node.input = ExtKnowledgeNodeInput(
            user_message="No reference",
            question="No reference",
        )
        async for _ in node.execute_stream():
            pass
        assert "verify_sql" not in [t.name for t in node.tools]

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
    """FilesystemFuncTool now uses project_root; scope enforcement moved to GenerationHooks."""

    def test_filesystem_root_is_project_root(self, real_agent_config, mock_llm_create):
        node = _create_node(real_agent_config)
        expected = str(Path(real_agent_config.project_root).expanduser())

        assert node.filesystem_func_tool is not None
        assert node.filesystem_func_tool.root_path == expected


# ===========================================================================
# Template context wiring
# ===========================================================================


class TestGenExtKnowledgeTemplateContext:
    """has_gold_sql must flow into the rendered prompt to gate PHASE 2."""

    def test_prepare_template_context_has_gold_sql_true(self, real_agent_config, mock_llm_create):
        node = _create_node(real_agent_config)
        user_input = ExtKnowledgeNodeInput(user_message="x", question="x")
        ctx = node._prepare_template_context(user_input, gold_sql="SELECT 1")
        assert ctx["has_gold_sql"] is True

    def test_prepare_template_context_has_gold_sql_false(self, real_agent_config, mock_llm_create):
        node = _create_node(real_agent_config)
        user_input = ExtKnowledgeNodeInput(user_message="x", question="x")
        ctx = node._prepare_template_context(user_input, gold_sql=None)
        assert ctx["has_gold_sql"] is False
        ctx_empty = node._prepare_template_context(user_input, gold_sql="")
        assert ctx_empty["has_gold_sql"] is False


# ===========================================================================
# Gold SQL pre-validation
# ===========================================================================


class TestGenExtKnowledgeValidateGoldSql:
    """``_validate_gold_sql`` rejects any gold SQL the connector can't execute."""

    def test_validate_gold_sql_runnable_passes(self, real_agent_config, mock_llm_create):
        node = _create_node(real_agent_config)
        assert node._validate_gold_sql("SELECT COUNT(*) FROM satscores") is None

    def test_validate_gold_sql_unrunnable_raises(self, real_agent_config, mock_llm_create):
        from datus.utils.exceptions import DatusException, ErrorCode

        node = _create_node(real_agent_config)
        with pytest.raises(DatusException) as exc_info:
            node._validate_gold_sql("SELECT * FROM __no_such_table__")
        assert exc_info.value.code == ErrorCode.NODE_EXT_KNOWLEDGE_GOLD_SQL_INVALID
