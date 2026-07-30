# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for the shared SQL modeling preflight."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from datus.configuration.agent_config import DbConfig
from datus.schemas.semantic_agentic_node_models import SourceQueryEvidence
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.sql_modeling_planner import (
    SqlModelingPlan,
    SqlModelingPlanner,
    SqlModelingPlanTools,
    _agent_config_dialect,
    _fingerprint_sources,
)
from datus.utils.exceptions import DatusException


class TestSqlModelingPlanTools:
    @staticmethod
    def _tool(user_message: str, semantic_source_inspector=None):
        evidence = GenerationEvidence()
        accepted = []
        tool = SqlModelingPlanTools(
            agent_config=MagicMock(),
            sub_agent_name="gen_metrics",
            user_message_provider=lambda: user_message,
            generation_evidence=evidence,
            plan_consumer=accepted.append,
            semantic_source_inspector=semantic_source_inspector,
        )
        return tool, evidence, accepted

    def test_no_sql_request_skips_preflight(self):
        tool, evidence, accepted = self._tool("Generate a revenue metric")

        assert tool.request_contains_sql() is False
        result = tool.prepare_sql_modeling_plan([])

        assert result.success == 0
        assert "Skip prepare_sql_modeling_plan" in result.error
        assert evidence.sql_modeling_plan_status == "pending"
        assert accepted == []

    def test_empty_entries_are_rejected_when_request_contains_sql(self):
        sql = "SELECT COUNT(*) AS order_count FROM orders"
        tool, evidence, accepted = self._tool(sql)

        assert tool.request_contains_sql() is True
        result = tool.prepare_sql_modeling_plan([])

        assert result.success == 0
        assert "contains SQL" in result.error
        assert evidence.sql_modeling_plan_status == "unresolved"
        assert accepted == []

    def test_terminal_sql_request_requires_ready_plan(self):
        tool, evidence, _ = self._tool("SELECT COUNT(*) AS order_count FROM orders")

        with pytest.raises(DatusException, match="prepare_sql_modeling_plan"):
            tool.require_plan_for_sql_request()

        evidence.set_sql_modeling_plan("ready", "source")
        assert tool.require_plan_for_sql_request() is True

    def test_all_sql_statements_must_be_submitted_together(self):
        first_sql = "SELECT COUNT(*) AS order_count FROM orders"
        second_sql = "SELECT SUM(amount) AS revenue FROM orders"
        tool, evidence, accepted = self._tool(f"First:\n```sql\n{first_sql}\n```\nSecond:\n```sql\n{second_sql}\n```")

        result = tool.prepare_sql_modeling_plan([{"source_index": 1, "name": "order_count"}])

        assert result.success == 0
        assert "missing source_index=[2]" in result.error
        assert evidence.sql_modeling_plan_status == "unresolved"
        assert accepted == []

    def test_complete_cte_is_copied_and_business_name_is_preserved(self):
        sql = "WITH daily AS (SELECT user_id FROM logins) SELECT COUNT(*) AS users FROM daily;"
        tool, evidence, accepted = self._tool(f"Question: Count active users\n```sql\n{sql}\n```")
        plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            source_queries=[
                SourceQueryEvidence(
                    source_sql_name="active_users",
                    question="Count active users",
                    sql=sql,
                    source_type="prompt",
                )
            ],
            candidate_plan={
                "available": True,
                "metric_requirements": [{"output_id": "active_users:output"}],
                "queryability_contracts": [
                    {
                        "source": "active_users",
                        "dimension_hints": ["user_group"],
                        "dimension_expr_hints": [
                            {
                                "alias": "user_group",
                                "expr": "LOWER(raw_group)",
                            }
                        ],
                    }
                ],
                "dataset_requirements": [
                    {
                        "requirement_id": "query_dataset:active_users",
                        "source_sql_name": "active_users",
                        "sql": sql,
                    }
                ],
            },
        )

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=plan,
        ):
            result = tool.prepare_sql_modeling_plan(
                [{"source_index": 1, "name": "Active Users", "question": "Count active users"}]
            )

        assert result.success == 1
        assert "sql" not in result.result["candidate_plan"]["dataset_requirements"][0]
        assert result.result["candidate_plan"]["dataset_requirements"][0]["source_index"] == 1
        assert evidence.sql_modeling_plan_status == "ready"
        assert evidence.required_metric_output_ids == ["active_users:output"]
        assert evidence.required_query_backed_sql == {"query_dataset:active_users": sql}
        assert evidence.metric_queryability_contracts[0]["dimension_hints"] == ["user_group"]
        assert accepted == [plan]

    def test_reuses_fixed_plan_without_reloading_catalog(self):
        sql = "SELECT COUNT(*) AS order_count FROM orders"
        tool, evidence, accepted = self._tool(sql)
        source = SourceQueryEvidence(
            source_sql_name="order_count",
            question="Count orders",
            sql=sql,
            source_type="prompt",
        )
        fixed_plan = SqlModelingPlan(
            source_fingerprint=_fingerprint_sources([source]),
            metric_catalog_fingerprint="catalog",
            source_queries=[source],
            candidate_plan={"available": True},
        )
        tool._plan = fixed_plan
        evidence.set_sql_modeling_plan("ready", fixed_plan.source_fingerprint)

        with patch("datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan") as planner:
            result = tool.prepare_sql_modeling_plan(
                [{"source_index": 1, "name": "order_count", "question": "Count orders"}]
            )

        assert result.success == 1
        planner.assert_not_called()
        assert accepted == []

    def test_sql_plan_includes_automatic_semantic_source_inspection(self):
        sql = "SELECT SUM(amount) AS revenue FROM orders"
        inspected = {
            "status": "ready",
            "tables": [{"table_name": "orders"}],
            "relationships": [],
        }
        inspector = MagicMock(return_value=inspected)
        tool, _, accepted = self._tool(sql, semantic_source_inspector=inspector)
        plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            source_queries=[SourceQueryEvidence(source_sql_name="revenue", sql=sql)],
            candidate_plan={"available": True},
        )

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=plan,
        ):
            result = tool.prepare_sql_modeling_plan([{"source_index": 1, "name": "revenue"}])

        assert result.success == 1
        assert result.result["semantic_source_evidence"] == inspected
        inspector.assert_called_once_with(plan)
        assert accepted == [plan]

    def test_unknown_source_index_is_rejected(self):
        tool, evidence, accepted = self._tool("SELECT COUNT(*) AS orders FROM orders")

        result = tool.prepare_sql_modeling_plan([{"source_index": 2, "name": "order_count"}])

        assert result.success == 0
        assert result.result["status"] == "unresolved"
        assert evidence.sql_modeling_plan_status == "unresolved"
        assert accepted == []

    def test_tool_preserves_literal_whitespace_and_statement_terminator(self):
        raw_sql = "SELECT 'a  b' AS label;"
        tool, evidence, accepted = self._tool(f"Use this SQL:\n{raw_sql}")
        plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            candidate_plan={"available": True},
        )

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=plan,
        ) as planner:
            result = tool.prepare_sql_modeling_plan([{"source_index": 1, "name": "label_value"}])

        assert result.success == 1
        assert planner.call_args.args[0][0].sql == raw_sql
        assert evidence.sql_modeling_plan_status == "ready"
        assert accepted == [plan]

    def test_generic_sql_index_is_not_a_business_name(self):
        raw_sql = "SELECT COUNT(*) AS order_count FROM orders"
        tool, evidence, accepted = self._tool(raw_sql)

        result = tool.prepare_sql_modeling_plan([{"source_index": 1, "name": "sql_1"}])

        assert result.success == 0
        assert "meaningful English snake_case" in result.error
        assert evidence.sql_modeling_plan_status == "unresolved"
        assert accepted == []


class TestSqlModelingPlanner:
    def test_plan_wraps_the_existing_analyzer_and_adds_fingerprints(self):
        source = SourceQueryEvidence(
            source_sql_name="sql_1",
            sql="SELECT COUNT(*) AS order_count FROM orders",
            question="Count orders",
        )
        analyzer_result = SimpleNamespace(
            success=True,
            result={"direct_metric_candidates": [{"name": "order_count"}]},
            error=None,
        )
        agent_config = MagicMock()
        agent_config.current_db_config.return_value = SimpleNamespace(type="duckdb")

        with (
            patch(
                "datus.tools.func_tool.semantic_discovery_tools.analyze_metric_candidate_entries",
                return_value=analyzer_result,
            ) as analyze,
            patch(
                "datus.utils.sql_utils.extract_table_names",
                return_value={"orders"},
            ),
        ):
            plan = SqlModelingPlanner(agent_config, "gen_metrics").plan(
                [source],
                existing_metric_catalog=[],
            )

        assert plan.candidate_plan["available"] is True
        assert plan.candidate_plan["direct_metric_candidates"] == [{"name": "order_count"}]
        assert plan.candidate_plan["sql_to_table_lineage"] == [{"source_sql_name": "sql_1", "tables": ["orders"]}]
        assert len(plan.source_fingerprint) == 64
        assert len(plan.metric_catalog_fingerprint) == 64
        entries = analyze.call_args.args[0]
        assert entries[0]["question"] == "Count orders"
        assert "external_knowledge" not in entries[0]

    def test_source_fingerprint_is_stable_for_the_same_input(self):
        source = SourceQueryEvidence(
            source_sql_name="sql_1",
            sql="SELECT COUNT(*) AS order_count FROM orders",
            question="Count orders",
        )
        analyzer_result = SimpleNamespace(success=True, result={}, error=None)
        agent_config = MagicMock()
        agent_config.current_db_config.return_value = SimpleNamespace(type="duckdb")

        with (
            patch(
                "datus.tools.func_tool.semantic_discovery_tools.analyze_metric_candidate_entries",
                return_value=analyzer_result,
            ),
            patch("datus.utils.sql_utils.extract_table_names", return_value={"orders"}),
        ):
            planner = SqlModelingPlanner(agent_config, "gen_metrics")
            first = planner.plan([source], existing_metric_catalog=[])
            second = planner.plan([source], existing_metric_catalog=[])

        assert first.source_fingerprint == second.source_fingerprint

    def test_any_parse_error_makes_the_request_plan_unavailable(self):
        sources = [
            SourceQueryEvidence(
                source_sql_name="valid_revenue",
                sql="SELECT SUM(amount) AS revenue FROM orders",
            ),
            SourceQueryEvidence(
                source_sql_name="broken_query",
                sql="SELECT FROM",
            ),
        ]
        analyzer_result = SimpleNamespace(
            success=True,
            result={
                "metric_requirements": [{"output_id": "valid:statement_1:output_1:revenue"}],
                "parse_errors": [{"source": "broken_query", "error": "cannot parse"}],
            },
            error=None,
        )
        agent_config = MagicMock()
        agent_config.current_db_config.return_value = SimpleNamespace(type="duckdb")

        with (
            patch(
                "datus.tools.func_tool.semantic_discovery_tools.analyze_metric_candidate_entries",
                return_value=analyzer_result,
            ),
            patch("datus.utils.sql_utils.extract_table_names", return_value={"orders"}),
        ):
            plan = SqlModelingPlanner(agent_config, "gen_metrics").plan(
                sources,
                existing_metric_catalog=[],
            )

        assert plan.candidate_plan["available"] is False
        assert "broken_query" in plan.candidate_plan["error"]

    @pytest.mark.parametrize("dialect", ["mysql", "starrocks", "sqlite"])
    def test_reads_dialect_from_db_config_type(self, dialect):
        agent_config = MagicMock()
        agent_config.current_db_config.return_value = DbConfig(type=dialect)

        assert _agent_config_dialect(agent_config) == dialect
