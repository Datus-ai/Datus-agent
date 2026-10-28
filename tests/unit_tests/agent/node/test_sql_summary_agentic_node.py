# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SqlSummaryAgenticNode.

NO MOCK except LLM: uses real AgentConfig, real SQLite database, real tools,
real PathManager, real RAG storage. Only LLMBaseModel.create_model is mocked
via the conftest mock_llm_create fixture.
"""

import json

import pytest
from agents.lifecycle import AgentHooksBase

from datus.schemas.action_history import ActionStatus
from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.generation_tools import GenerationTools
from datus.tools.permission.permission_hooks import PermissionHooks
from tests.unit_tests.mock_llm_model import (
    MockToolCall,
    build_simple_response,
    build_tool_then_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_node(real_agent_config, **kwargs):
    """Create a SqlSummaryAgenticNode with real config and real dependencies."""
    from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

    defaults = dict(
        node_name="gen_sql_summary",
        agent_config=real_agent_config,
        execution_mode="workflow",
    )
    defaults.update(kwargs)
    return SqlSummaryAgenticNode(**defaults)


# ===========================================================================
# Test Initialization
# ===========================================================================


class TestSqlSummaryAgenticNodeInit:
    """Tests for SqlSummaryAgenticNode initialization with real dependencies."""

    def test_sql_summary_init(self, real_agent_config, mock_llm_create):
        """Node can be initialized with real config."""
        node = _create_node(real_agent_config)

        assert node.configured_node_name == "gen_sql_summary"
        assert node.execution_mode == "workflow"
        assert node.hooks is None
        assert node.get_node_name() == "gen_sql_summary"

    def test_sql_summary_has_tools(self, real_agent_config, mock_llm_create):
        """Node has generation tools and filesystem tools."""
        node = _create_node(real_agent_config)

        tool_names = [t.name for t in node.tools]

        # Generation tools
        assert "generate_sql_summary_id" in tool_names

        # Filesystem tools
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "edit_file" in tool_names
        assert "glob" in tool_names
        assert "grep" in tool_names

        # Tool instances should be initialized
        assert isinstance(node.filesystem_func_tool, FilesystemFuncTool)
        assert isinstance(node.generation_tools, GenerationTools)

    def test_sql_summary_max_turns(self, real_agent_config, mock_llm_create):
        """max_turns is read from agentic_nodes config (5 in test config)."""
        node = _create_node(real_agent_config)
        assert node.max_turns == 5  # Set in conftest real_agent_config

    def test_sql_summary_subject_tree(self, real_agent_config, mock_llm_create):
        """subject_tree can be passed and stored."""
        tree = ["Analytics", "Reports"]
        node = _create_node(real_agent_config, subject_tree=tree)
        assert node.subject_tree == tree


# ===========================================================================
# Test Execution
# ===========================================================================


SUMMARY_REL_PATH = "subject/sql_summaries/avg_scores_001.yaml"

VALID_SUMMARY_YAML = (
    "id: auto_generated\n"
    'name: "avg_scores"\n'
    "sql: |\n"
    "  SELECT AVG(AvgScrRead) FROM satscores\n"
    "summary: >\n"
    "  Average SAT reading score aggregation\n"
    'search_text: "average sat reading score"\n'
    f"filepath: {SUMMARY_REL_PATH}\n"
    'subject_tree: "Analytics/Scores"\n'
    'tags: "test"\n'
)

# YAML that parses fine but is missing the required non-empty ``sql`` field.
YAML_WITHOUT_SQL = 'name: "no_sql"\nsummary: >\n  Missing sql field\n'

FINAL_RESPONSE = json.dumps({"sql_summary_file": SUMMARY_REL_PATH, "output": "Summary generated"})

SYNC_OK = {"success": True, "message": "Synced reference SQL: avg_scores"}


def _write_file_call(path: str = SUMMARY_REL_PATH, content: str = VALID_SUMMARY_YAML) -> MockToolCall:
    return MockToolCall(name="write_file", arguments=json.dumps({"path": path, "content": content}))


def _patch_reference_sql_sync(**kwargs):
    from unittest.mock import patch

    return patch("datus.storage.reference_sql.artifact_sync.sync_reference_sql_artifact_to_kb", **kwargs)


async def _run_stream(node):
    actions = []
    async for action in node.execute_stream():
        actions.append(action)
    return actions


@pytest.mark.component
@pytest.mark.llm_harness
class TestSqlSummaryAgenticNodeExecution:
    """Tests for SqlSummaryAgenticNode.execute_stream() with real tools.

    The KB sync boundary (``sync_reference_sql_artifact_to_kb``) is mocked;
    its own behavior is covered in tests/unit_tests/storage/reference_sql/.
    Everything else — filesystem tool, mutation tracking, finalizer — is real.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("execution_mode", ["interactive", "workflow"])
    async def test_happy_path_writes_and_syncs_artifact(self, real_agent_config, mock_llm_create, execution_mode):
        """Both execution modes run the same finalizer: write_file + final JSON
        path → node succeeds and the artifact is synced to the KB."""
        if execution_mode == "interactive":
            # The interactive "normal" profile gates INTERNAL writes behind an
            # ASK prompt; there is no broker to answer it in tests, so run the
            # write-permitting "auto" profile instead.
            real_agent_config.active_profile_name = "auto"
        node = _create_node(real_agent_config, execution_mode=execution_mode)

        mock_llm_create.reset(
            responses=[build_tool_then_response(tool_calls=[_write_file_call()], content=FINAL_RESPONSE)]
        )
        node.model = mock_llm_create
        node.input = SqlSummaryNodeInput(
            user_message="Summarize this SQL query",
            sql_query="SELECT AVG(AvgScrRead) FROM satscores",
        )

        with _patch_reference_sql_sync(return_value=SYNC_OK) as sync_mock:
            actions = await _run_stream(node)

        assert actions[-1].status == ActionStatus.SUCCESS
        assert actions[-1].action_type == "gen_sql_summary_response"
        sync_mock.assert_called_once()
        synced_path = sync_mock.call_args.args[0]
        assert synced_path.endswith("subject/sql_summaries/avg_scores_001.yaml")
        assert sync_mock.call_args.args[1] is real_agent_config

        last_output = actions[-1].output
        assert isinstance(last_output, dict)
        # Only interactive mode extracts token usage from the action history.
        assert (last_output["tokens_used"] > 0) == (execution_mode == "interactive")

    @pytest.mark.asyncio
    async def test_reference_template_storage_uses_template_sync(self, real_agent_config, mock_llm_create):
        from unittest.mock import patch

        node = _create_node(real_agent_config, execution_mode="workflow", storage_type="reference_template")

        mock_llm_create.reset(
            responses=[build_tool_then_response(tool_calls=[_write_file_call()], content=FINAL_RESPONSE)]
        )
        node.model = mock_llm_create
        node.input = SqlSummaryNodeInput(
            user_message="Summarize this template",
            sql_query="SELECT * FROM t WHERE x = '{{val}}'",
        )

        with patch(
            "datus.storage.reference_template.artifact_sync.sync_reference_template_artifact_to_kb",
            return_value={"success": True, "message": "ok"},
        ) as sync_mock:
            actions = await _run_stream(node)

        assert actions[-1].status == ActionStatus.SUCCESS
        sync_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_final_path_fails_node(self, real_agent_config, mock_llm_create):
        """A response without sql_summary_file must fail the node instead of
        silently succeeding with nothing synced."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        mock_llm_create.reset(responses=[build_simple_response("SQL summary created for the revenue query")])
        node.model = mock_llm_create
        node.input = SqlSummaryNodeInput(
            user_message="Summarize this SQL query",
            sql_query="SELECT AVG(AvgScrRead) FROM satscores",
        )

        with _patch_reference_sql_sync(return_value=SYNC_OK) as sync_mock:
            actions = await _run_stream(node)

        assert actions[-1].status == ActionStatus.FAILED
        sync_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_file_not_written_this_request_fails_and_survives(self, real_agent_config, mock_llm_create):
        """A pre-existing file the LLM merely references (never wrote) fails
        the finalizer and must NOT be deleted by the rollback."""
        from pathlib import Path

        pre_existing = Path(real_agent_config.project_root) / SUMMARY_REL_PATH
        pre_existing.parent.mkdir(parents=True, exist_ok=True)
        pre_existing.write_text(VALID_SUMMARY_YAML, encoding="utf-8")

        node = _create_node(real_agent_config, execution_mode="workflow")
        mock_llm_create.reset(responses=[build_simple_response(FINAL_RESPONSE)])
        node.model = mock_llm_create
        node.input = SqlSummaryNodeInput(user_message="Summarize", sql_query="SELECT 1")

        with _patch_reference_sql_sync(return_value=SYNC_OK) as sync_mock:
            actions = await _run_stream(node)

        assert actions[-1].status == ActionStatus.FAILED
        sync_mock.assert_not_called()
        assert pre_existing.read_text(encoding="utf-8") == VALID_SUMMARY_YAML

    @pytest.mark.asyncio
    async def test_out_of_sandbox_write_rejected_by_guard(self, real_agent_config, mock_llm_create):
        """The mutation guard rejects writes outside subject/sql_summaries at
        the tool layer, so the file never lands on disk."""
        from pathlib import Path

        node = _create_node(real_agent_config, execution_mode="workflow")

        outside_rel = "outside.yaml"
        final = json.dumps({"sql_summary_file": outside_rel, "output": "x"})
        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[_write_file_call(path=outside_rel, content=VALID_SUMMARY_YAML)],
                    content=final,
                )
            ]
        )
        node.model = mock_llm_create
        node.input = SqlSummaryNodeInput(user_message="Summarize", sql_query="SELECT 1")

        with _patch_reference_sql_sync(return_value=SYNC_OK) as sync_mock:
            actions = await _run_stream(node)

        assert actions[-1].status == ActionStatus.FAILED
        sync_mock.assert_not_called()
        assert not (Path(real_agent_config.project_root) / outside_rel).exists()

    @pytest.mark.asyncio
    async def test_invalid_artifact_fails_and_rolls_back_new_file(self, real_agent_config, mock_llm_create):
        """A written artifact without a usable ``sql`` field fails validation
        and the newly created file is removed on rollback."""
        from pathlib import Path

        node = _create_node(real_agent_config, execution_mode="workflow")
        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[_write_file_call(content=YAML_WITHOUT_SQL)],
                    content=FINAL_RESPONSE,
                )
            ]
        )
        node.model = mock_llm_create
        node.input = SqlSummaryNodeInput(user_message="Summarize", sql_query="SELECT 1")

        with _patch_reference_sql_sync(return_value=SYNC_OK) as sync_mock:
            actions = await _run_stream(node)

        assert actions[-1].status == ActionStatus.FAILED
        sync_mock.assert_not_called()
        assert not (Path(real_agent_config.project_root) / SUMMARY_REL_PATH).exists()

    @pytest.mark.asyncio
    async def test_kb_sync_failure_fails_node_and_rolls_back(self, real_agent_config, mock_llm_create):
        """A failed KB sync fails the node; the half-written YAML is removed so
        no 'YAML saved but KB not updated' state survives."""
        from pathlib import Path

        node = _create_node(real_agent_config, execution_mode="workflow")
        mock_llm_create.reset(
            responses=[build_tool_then_response(tool_calls=[_write_file_call()], content=FINAL_RESPONSE)]
        )
        node.model = mock_llm_create
        node.input = SqlSummaryNodeInput(user_message="Summarize", sql_query="SELECT 1")

        with _patch_reference_sql_sync(return_value={"success": False, "error": "lancedb down"}) as sync_mock:
            actions = await _run_stream(node)

        assert actions[-1].status == ActionStatus.FAILED
        sync_mock.assert_called_once()
        assert not (Path(real_agent_config.project_root) / SUMMARY_REL_PATH).exists()

    @pytest.mark.asyncio
    async def test_rollback_restores_overwritten_file(self, real_agent_config, mock_llm_create):
        """When the request overwrites an existing artifact and then fails,
        the original content is restored."""
        from pathlib import Path

        original = 'id: old\nname: "old"\nsql: |\n  SELECT 0\n'
        pre_existing = Path(real_agent_config.project_root) / SUMMARY_REL_PATH
        pre_existing.parent.mkdir(parents=True, exist_ok=True)
        pre_existing.write_text(original, encoding="utf-8")

        node = _create_node(real_agent_config, execution_mode="workflow")
        mock_llm_create.reset(
            responses=[build_tool_then_response(tool_calls=[_write_file_call()], content=FINAL_RESPONSE)]
        )
        node.model = mock_llm_create
        node.input = SqlSummaryNodeInput(user_message="Summarize", sql_query="SELECT 1")

        with _patch_reference_sql_sync(return_value={"success": False, "error": "boom"}):
            actions = await _run_stream(node)

        assert actions[-1].status == ActionStatus.FAILED
        assert pre_existing.read_text(encoding="utf-8") == original

    @pytest.mark.asyncio
    async def test_rewrite_after_bad_write_syncs_latest_content(self, real_agent_config, mock_llm_create):
        """A bad first write corrected by a second write in the same request
        finalizes successfully with the latest on-disk content."""
        from pathlib import Path

        node = _create_node(real_agent_config, execution_mode="workflow")
        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        _write_file_call(content=YAML_WITHOUT_SQL),
                        _write_file_call(content=VALID_SUMMARY_YAML),
                    ],
                    content=FINAL_RESPONSE,
                )
            ]
        )
        node.model = mock_llm_create
        node.input = SqlSummaryNodeInput(user_message="Summarize", sql_query="SELECT 1")

        with _patch_reference_sql_sync(return_value=SYNC_OK) as sync_mock:
            actions = await _run_stream(node)

        assert actions[-1].status == ActionStatus.SUCCESS
        sync_mock.assert_called_once()
        on_disk = (Path(real_agent_config.project_root) / SUMMARY_REL_PATH).read_text(encoding="utf-8")
        assert on_disk == VALID_SUMMARY_YAML

    @pytest.mark.asyncio
    async def test_mutation_state_reset_between_requests(self, real_agent_config, mock_llm_create):
        """A reused node instance must not treat the previous request's file
        as written by the current request."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        mock_llm_create.reset(
            responses=[build_tool_then_response(tool_calls=[_write_file_call()], content=FINAL_RESPONSE)]
        )
        node.model = mock_llm_create
        node.input = SqlSummaryNodeInput(user_message="Summarize", sql_query="SELECT 1")
        with _patch_reference_sql_sync(return_value=SYNC_OK):
            actions = await _run_stream(node)
        assert actions[-1].status == ActionStatus.SUCCESS

        # Second request references the same file but never writes it.
        mock_llm_create.reset(responses=[build_simple_response(FINAL_RESPONSE)])
        node.input = SqlSummaryNodeInput(user_message="Summarize again", sql_query="SELECT 2")
        with _patch_reference_sql_sync(return_value=SYNC_OK) as sync_mock:
            actions = await _run_stream(node)

        assert actions[-1].status == ActionStatus.FAILED
        sync_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_sql_summary_input_not_set_raises(self, real_agent_config, mock_llm_create):
        """execute_stream raises ValueError when input is not set."""
        node = _create_node(real_agent_config)
        node.input = None

        from datus.utils.exceptions import DatusException

        with pytest.raises(DatusException, match="Missing required field"):
            async for _ in node.execute_stream():
                pass

    @pytest.mark.asyncio
    async def test_sql_summary_with_sql_query_context(self, real_agent_config, mock_llm_create):
        """Input with sql_query and comment enriches the user message."""
        node = _create_node(real_agent_config, execution_mode="workflow")

        mock_llm_create.reset(
            responses=[build_tool_then_response(tool_calls=[_write_file_call()], content=FINAL_RESPONSE)]
        )
        node.model = mock_llm_create

        node.input = SqlSummaryNodeInput(
            user_message="Summarize this query",
            sql_query=(
                "SELECT s.County, AVG(sc.AvgScrRead) FROM satscores sc "
                "JOIN schools s ON sc.cds = s.CDSCode GROUP BY s.County"
            ),
            comment="Average SAT reading score by county",
            database="california_schools",
            db_schema="main",
        )

        with _patch_reference_sql_sync(return_value=SYNC_OK):
            actions = await _run_stream(node)

        # Should complete successfully
        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

        # Verify the model was called with the prompt containing SQL context
        assert len(mock_llm_create.call_history) >= 1
        call = mock_llm_create.call_history[0]
        prompt = call.get("prompt", "")
        # The enhanced message should contain the SQL query
        assert "SELECT s.County, AVG(sc.AvgScrRead)" in prompt


# ===========================================================================
# Test Extract Methods
# ===========================================================================


class TestSqlSummaryExtractMethods:
    """Tests for SqlSummaryAgenticNode extraction utility methods."""

    def test_extract_sql_summary_from_dict_response(self, real_agent_config, mock_llm_create):
        """_extract_sql_summary_and_output_from_response with dict content."""
        node = _create_node(real_agent_config)

        file_name, output = node._extract_sql_summary_and_output_from_response(
            {"content": {"sql_summary_file": "test.yaml", "output": "Done"}}
        )
        assert file_name == "test.yaml"
        assert output == "Done"

    def test_extract_sql_summary_from_json_string(self, real_agent_config, mock_llm_create):
        """_extract_sql_summary_and_output_from_response with JSON string content."""
        node = _create_node(real_agent_config)

        json_content = json.dumps({"sql_summary_file": "summary.yaml", "output": "Generated"})
        file_name, output = node._extract_sql_summary_and_output_from_response({"content": json_content})
        assert file_name == "summary.yaml"
        assert output == "Generated"

    def test_extract_sql_summary_from_fenced_json_after_jinja_text(self, real_agent_config, mock_llm_create):
        """Jinja placeholders before final JSON should not confuse extraction."""
        node = _create_node(real_agent_config)

        content = """
SQL summary file has been created.

The template contains `{{period_start_date}}` and `{{period_end_date}}`.

```json
{
  "sql_summary_file": "subject/sql_summaries/summary.yaml",
  "output": "Generated"
}
```
"""

        file_name, output = node._extract_sql_summary_and_output_from_response({"content": content})
        assert file_name == "subject/sql_summaries/summary.yaml"
        assert output == "Generated"

    def test_extract_sql_summary_from_unfenced_json_after_jinja_text(self, real_agent_config, mock_llm_create):
        """Free-form text may include Jinja braces before an unfenced final JSON object."""
        node = _create_node(real_agent_config)

        content = """
Generated a summary for:
{% if product_tag_id %}
SELECT * FROM campaign WHERE FIND_IN_SET('{{ product_tag_id }}', ac_tags);
{% endif %}

Result:
{"sql_summary_file": "sql_summaries/template_summary.yaml", "output": "Generated"}
"""

        file_name, output = node._extract_sql_summary_and_output_from_response({"content": content})
        assert file_name == "sql_summaries/template_summary.yaml"
        assert output == "Generated"

    def test_extract_sql_summary_from_generic_fenced_block(self, real_agent_config, mock_llm_create):
        """The parser should not require the markdown fence language to be exactly json."""
        node = _create_node(real_agent_config)

        content = """
Done.

```text
{
  "sql_summary_file": "summary_from_text_fence.yaml",
  "output": "Generated"
}
```
"""

        file_name, output = node._extract_sql_summary_and_output_from_response({"content": content})
        assert file_name == "summary_from_text_fence.yaml"
        assert output == "Generated"

    def test_extract_sql_summary_from_escaped_json_object(self, real_agent_config, mock_llm_create):
        """JSON scanning should handle escaped quotes before closing the object."""
        node = _create_node(real_agent_config)

        content = (
            'Done: {"note": "template mentions \\"quoted\\" text and C:\\\\tmp", '
            '"sql_summary_file": "escaped_summary.yaml", "output": "Generated"}'
        )

        file_name, output = node._extract_sql_summary_and_output_from_response({"content": content})
        assert file_name == "escaped_summary.yaml"
        assert output == "Generated"

    def test_extract_sql_summary_reports_missing_keys(self, real_agent_config, mock_llm_create):
        """Relevant-looking JSON without expected keys should not produce a fabricated path."""
        node = _create_node(real_agent_config)

        file_name, output = node._extract_sql_summary_and_output_from_response(
            {"content": '{"output_file": "wrong.yaml"}'}
        )

        assert file_name is None
        assert output is None

    def test_extract_sql_summary_handles_json_parse_failure(self, real_agent_config, mock_llm_create):
        """A malformed relevant payload should fall through cleanly when JSON parsing fails."""
        from unittest.mock import patch

        node = _create_node(real_agent_config)

        with patch("json_repair.loads", side_effect=ValueError("bad json")):
            file_name, output = node._extract_sql_summary_and_output_from_response(
                {"content": "```text\n{'output': 'Generated'}\n```"}
            )

        assert file_name is None
        assert output is None

    def test_extract_sql_summary_uses_regex_fallback(self, real_agent_config, mock_llm_create):
        """If no valid JSON object exists, the legacy sql_summary_file fallback still works."""
        node = _create_node(real_agent_config)

        content = 'Summary saved. "sql_summary_file": "regex_fallback.yaml"'

        file_name, output = node._extract_sql_summary_and_output_from_response({"content": content})
        assert file_name == "regex_fallback.yaml"
        assert output is None

    def test_extract_sql_summary_from_empty(self, real_agent_config, mock_llm_create):
        """_extract_sql_summary_and_output_from_response with empty content returns None."""
        node = _create_node(real_agent_config)

        file_name, output = node._extract_sql_summary_and_output_from_response({"content": ""})
        assert file_name is None
        assert output is None


class TestSqlSummaryFinalizerSandbox:
    """``_finalize_sql_summary_artifact`` must reject paths outside the
    per-kind sandbox.

    These paths come from the LLM's final JSON (not from the write_file tool
    result), so the containment check is the last line of defence against a
    fabricated response syncing an arbitrary file.
    """

    def test_rejects_out_of_sandbox_absolute_path(self, real_agent_config, mock_llm_create, tmp_path):
        from datus.utils.exceptions import DatusException

        node = _create_node(real_agent_config)
        outside = tmp_path / "outside" / "malicious.yaml"
        outside.parent.mkdir(parents=True)
        outside.write_text("x: y\n")

        with _patch_reference_sql_sync(return_value=SYNC_OK) as sync_mock:
            with pytest.raises(DatusException, match="escapes"):
                node._finalize_sql_summary_artifact(str(outside))
            sync_mock.assert_not_called()

    def test_rejects_missing_file(self, real_agent_config, mock_llm_create):
        from datus.utils.exceptions import DatusException

        node = _create_node(real_agent_config)
        with _patch_reference_sql_sync(return_value=SYNC_OK) as sync_mock:
            with pytest.raises(DatusException, match="does not exist"):
                node._finalize_sql_summary_artifact("sql_summaries/other_db/q_001.yaml")
            sync_mock.assert_not_called()


class TestSqlSummaryFilesystemRootPath:
    """FilesystemFuncTool uses project_root; write-scope enforcement lives in the node's mutation guard."""

    def test_filesystem_root_is_project_root(self, real_agent_config, mock_llm_create):
        from pathlib import Path

        node = _create_node(real_agent_config)
        expected = str(Path(real_agent_config.project_root).expanduser())

        assert isinstance(node.filesystem_func_tool, FilesystemFuncTool)
        assert node.filesystem_func_tool.root_path == expected


class TestSqlSummaryNonInteractiveBridge:
    """Workflow mode → ``PermissionHooks.non_interactive=True``.

    SqlSummaryAgenticNode is invoked from ``/bootstrap`` SQL and Template tabs
    via parallel pools. Each per-item node must run non-interactively or the
    pool deadlocks on the first ASK prompt.
    """

    def test_workflow_mode_compose_hooks_is_non_interactive(self, real_agent_config, mock_llm_create):
        node = _create_node(real_agent_config, execution_mode="workflow")
        # Workflow now also wires the token-usage hook (multi-turn history is enabled
        # for all execution_mode values), so _compose_hooks may return a
        # CompositeHooks bundle. Validate the permission gate directly on the
        # node — that's what keeps /bootstrap parallel pools from deadlocking.
        hooks = node._compose_hooks()
        assert isinstance(hooks, AgentHooksBase)
        assert isinstance(node.permission_hooks, PermissionHooks)
        assert node.permission_hooks.non_interactive is True
        assert node.permission_manager.active_profile == "dangerous"
