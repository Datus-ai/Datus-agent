# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for prompt utility modules.

CI-level: zero external deps, zero network, zero API keys.
Mocks prompt_manager.render_template / get_raw_template to avoid template file I/O.
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# TestCompareSqlWithMcp
# ---------------------------------------------------------------------------


class TestGetComparePrompt:
    def test_returns_system_and_user_messages(self):
        from datus.prompts.compare_sql_with_mcp import get_compare_prompt

        sql_task = MagicMock()
        sql_task.database_type = "duckdb"
        sql_task.database_name = "test_db"
        sql_task.task = "Find total sales"
        sql_task.external_knowledge = ""

        with patch("datus.prompts.compare_sql_with_mcp.prompt_manager") as mock_pm:
            mock_pm.get_raw_template.return_value = "system content"
            mock_pm.render_template.return_value = "user content"
            result = get_compare_prompt(sql_task, sql_query="SELECT 1", expectation="result > 0")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[0]["content"] == "system content"
        assert result[1]["content"] == "user content"

    def test_get_compare_system_prompt_returns_string(self):
        from datus.prompts.compare_sql_with_mcp import get_compare_system_prompt

        result = get_compare_system_prompt()
        assert isinstance(result, str)
        assert "SQL" in result


class TestCompareSqlPrompt:
    def test_returns_system_and_user_messages(self):
        from datus.prompts.compare_sql import compare_sql_prompt

        sql_task = MagicMock()
        sql_task.database_type = "sqlite"
        sql_task.database_name = "california_schools"
        sql_task.task = "Count students"
        sql_task.external_knowledge = ""

        with patch("datus.prompts.compare_sql.prompt_manager") as mock_pm:
            mock_pm.get_raw_template.return_value = "system"
            mock_pm.render_template.return_value = "user"
            result = compare_sql_prompt(sql_task, sql_query="SELECT COUNT(*) FROM students")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"


# ---------------------------------------------------------------------------
# TestOutputChecking
# ---------------------------------------------------------------------------


class TestOutputCheckingGenPrompt:
    def test_returns_single_user_message(self):
        from datus.prompts.output_checking import gen_prompt

        with patch("datus.prompts.output_checking.prompt_manager") as mock_pm:
            mock_pm.render_template.return_value = "output checking content"
            result = gen_prompt(
                user_question="What is the total?",
                table_schemas="CREATE TABLE t (id INT)",
                sql_query="SELECT SUM(id) FROM t",
                sql_execution_result="42",
            )

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "output checking content"

    def test_truncates_long_execution_result(self):
        from datus.prompts.output_checking import gen_prompt

        long_result = "x" * 600

        with patch("datus.prompts.output_checking.prompt_manager") as mock_pm:
            mock_pm.render_template.return_value = "content"
            # Capture the call kwargs to verify truncation
            gen_prompt(
                user_question="Q",
                table_schemas="schema",
                sql_query="SELECT 1",
                sql_execution_result=long_result,
            )
            # Verify the template was called with truncated result
            call_kwargs = mock_pm.render_template.call_args[1]
            assert len(call_kwargs["sql_execution_result"]) <= 520  # 500 + len("... (truncated)")

    def test_handles_table_schemas_as_list(self):
        from datus.prompts.output_checking import gen_prompt
        from datus.schemas.node_models import TableSchema

        mock_schema = MagicMock(spec=TableSchema)
        mock_schema.to_prompt.return_value = "table schema text"

        with patch("datus.prompts.output_checking.prompt_manager") as mock_pm:
            mock_pm.render_template.return_value = "content"
            gen_prompt(
                user_question="Q",
                table_schemas=[mock_schema],
                sql_query="SELECT 1",
                sql_execution_result="result",
            )

        mock_schema.to_prompt.assert_called_once()

    def test_handles_metrics(self):
        from datus.prompts.output_checking import gen_prompt
        from datus.schemas.node_models import Metric

        mock_metric = MagicMock(spec=Metric)
        mock_metric.to_prompt.return_value = "metric: revenue = SUM(sales)"

        with patch("datus.prompts.output_checking.prompt_manager") as mock_pm:
            mock_pm.render_template.return_value = "content"
            gen_prompt(
                user_question="Q",
                table_schemas="schema",
                sql_query="SELECT 1",
                sql_execution_result="result",
                metrics=[mock_metric],
            )

        call_kwargs = mock_pm.render_template.call_args[1]
        assert "revenue" in call_kwargs["metrics"]

    def test_handles_external_knowledge(self):
        from datus.prompts.output_checking import gen_prompt

        with patch("datus.prompts.output_checking.prompt_manager") as mock_pm:
            mock_pm.render_template.return_value = "content"
            gen_prompt(
                user_question="Q",
                table_schemas="schema",
                sql_query="SELECT 1",
                sql_execution_result="result",
                external_knowledge="revenue = total sales",
            )

        call_kwargs = mock_pm.render_template.call_args[1]
        assert "revenue = total sales" in call_kwargs["external_knowledge"]


# ---------------------------------------------------------------------------
# TestSchemaLineage
# ---------------------------------------------------------------------------


class TestSchemaLineageGenPrompt:
    def test_returns_empty_list_when_no_metadata(self):
        from datus.prompts.schema_lineage import gen_prompt

        result = gen_prompt(
            dialect="duckdb",
            database_name="test_db",
            user_question="Q",
            table_metadata=[],
        )
        assert result == []

    def test_returns_system_and_user_messages(self):
        from datus.prompts.schema_lineage import gen_prompt

        table_metadata = [{"identifier": "users", "definition": "CREATE TABLE users (id INT)"}]

        with patch("datus.prompts.schema_lineage.prompt_manager") as mock_pm:
            mock_pm.render_template.return_value = "rendered content"
            result = gen_prompt(
                dialect="duckdb",
                database_name="test_db",
                user_question="Find all users",
                table_metadata=table_metadata,
            )

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_gen_summary_prompt_returns_user_message(self):
        from datus.prompts.schema_lineage import gen_summary_prompt

        table_metadata = [
            {
                "schema_name": "main",
                "table_name": "users",
                "schema_text": "CREATE TABLE users",
                "score": 0.9,
                "reasons": "match",
            }
        ]

        with patch("datus.prompts.schema_lineage.prompt_manager") as mock_pm:
            mock_pm.render_template.return_value = "summary content"
            result = gen_summary_prompt(
                dialect="sqlite",
                database_name="test",
                user_question="Q",
                table_metadata=table_metadata,
            )

        assert len(result) == 1
        assert result[0]["role"] == "user"


# ---------------------------------------------------------------------------
# TestSelectionPrompt
# ---------------------------------------------------------------------------


class TestCreateSelectionPrompt:
    def test_returns_prompt_string(self):
        from datus.prompts.selection import create_selection_prompt

        candidates = {"cand_1": {"sql": "SELECT 1", "result": "1"}}

        with patch("datus.prompts.selection.prompt_manager") as mock_pm:
            mock_pm.render_template.return_value = "selection prompt text"
            result = create_selection_prompt(candidates)

        assert result == "selection prompt text"

    def test_truncates_long_error(self):
        from datus.prompts.selection import create_selection_prompt

        long_error = "E" * 600
        candidates = {"cand_1": {"sql": "SELECT 1", "error": long_error}}

        with patch("datus.prompts.selection.prompt_manager") as mock_pm:
            mock_pm.render_template.return_value = "prompt"
            create_selection_prompt(candidates, max_text_length=500)

        call_kwargs = mock_pm.render_template.call_args[1]
        processed = call_kwargs["candidates"]
        assert "truncated" in processed["cand_1"]["error"]

    def test_truncates_long_string_candidate(self):
        from datus.prompts.selection import create_selection_prompt

        long_candidate = "A" * 600
        candidates = {"cand_1": long_candidate}

        with patch("datus.prompts.selection.prompt_manager") as mock_pm:
            mock_pm.render_template.return_value = "prompt"
            create_selection_prompt(candidates, max_text_length=500)

        call_kwargs = mock_pm.render_template.call_args[1]
        processed = call_kwargs["candidates"]
        assert "truncated" in processed["cand_1"]

    def test_passes_version_to_template(self):
        from datus.prompts.selection import create_selection_prompt

        candidates = {"c": {"sql": "SELECT 1"}}

        with patch("datus.prompts.selection.prompt_manager") as mock_pm:
            mock_pm.render_template.return_value = "prompt"
            create_selection_prompt(candidates, prompt_version="v2")

        call_kwargs = mock_pm.render_template.call_args[1]
        assert call_kwargs["version"] == "v2"


# ---------------------------------------------------------------------------
# TestGenMetricsV12TemplateSmokeTest
# ---------------------------------------------------------------------------


class TestGenMetricsV12Template:
    """Smoke tests for gen_metrics_system_1.2.j2 template."""

    def test_v12_template_renders_without_error(self):
        """v1.2 template renders with minimal context and produces non-empty output."""
        from datus.prompts.prompt_manager import PromptManager

        pm = PromptManager()
        # Use only the default_templates_dir (no user templates needed)
        from jinja2 import Environment, FileSystemLoader

        env = Environment(
            loader=FileSystemLoader([str(pm.default_templates_dir)]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template("gen_metrics_system_1.2.j2")

        result = template.render(
            native_tools=["read_file", "write_file", "end_metric_generation"],
            mcp_tools=[],
            has_ask_user_tool=True,
            semantic_model_dir="/tmp/test_models",
            has_subject_tree=False,
            subject_tree=[],
            existing_subject_trees=[],
        )

        assert len(result) > 100, "Template should render substantial content"
        assert "metric" in result.lower()

    def test_v12_template_mentions_skill(self):
        """v1.2 template should reference skills and gen-metrics."""
        from datus.prompts.prompt_manager import PromptManager

        pm = PromptManager()
        from jinja2 import Environment, FileSystemLoader

        env = Environment(
            loader=FileSystemLoader([str(pm.default_templates_dir)]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template("gen_metrics_system_1.2.j2")

        result = template.render(
            native_tools=[],
            mcp_tools=[],
            has_ask_user_tool=False,
            semantic_model_dir="/tmp/models",
            has_subject_tree=False,
            subject_tree=[],
            existing_subject_trees=[],
        )

        assert "gen-metrics" in result
        assert "available_skills" in result or "load_skill" in result

    def test_v12_is_latest_version(self):
        """PromptManager.get_latest_version returns '1.2' for gen_metrics."""
        from datus.prompts.prompt_manager import PromptManager

        pm = PromptManager()
        latest = pm.get_latest_version("gen_metrics_system")
        latest_parts = tuple(int(p) for p in latest.split("."))
        assert latest_parts >= (1, 2), f"Expected latest version >= '1.2', got '{latest}'"
