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
)


class TestSqlModelingPlanTools:
    @staticmethod
    def _tool(semantic_source_inspector=None):
        evidence = GenerationEvidence()
        accepted = []
        tool = SqlModelingPlanTools(
            agent_config=MagicMock(),
            sub_agent_name="gen_metrics",
            generation_evidence=evidence,
            plan_consumer=accepted.append,
            semantic_source_inspector=semantic_source_inspector,
        )
        return tool, evidence, accepted

    def test_exposes_the_standard_native_tool_group_contract(self):
        tool, _, _ = self._tool()

        assert tool.all_tools_name() == ["prepare_sql_modeling_plan"]
        assert [item.name for item in tool.available_tools()] == ["prepare_sql_modeling_plan"]

    def test_empty_entries_mark_the_invoked_preflight_unresolved(self):
        tool, evidence, accepted = self._tool()

        result = tool.prepare_sql_modeling_plan([])

        assert result.success == 0
        assert "must contain every SQL statement" in result.error
        assert evidence.sql_modeling_plan_status == "unresolved"
        assert accepted == []

    def test_invalid_entries_mark_the_invoked_preflight_unresolved(self):
        tool, evidence, accepted = self._tool()

        result = tool.prepare_sql_modeling_plan([{"source_index": 1, "name": "order_count"}])

        assert result.success == 0
        assert "Invalid sql_entries" in result.error
        assert evidence.sql_modeling_plan_status == "unresolved"
        assert accepted == []

    def test_whitespace_only_sql_is_rejected(self):
        tool, evidence, accepted = self._tool()

        result = tool.prepare_sql_modeling_plan([{"source_index": 1, "name": "order_count", "sql": "  \n"}])

        assert result.success == 0
        assert "sql must not be empty" in result.error
        assert evidence.sql_modeling_plan_status == "unresolved"
        assert accepted == []

    def test_sql_read_from_a_user_specified_file_can_be_submitted(self):
        sql = "SELECT region, SUM(amount) AS revenue FROM orders GROUP BY region;"
        tool, evidence, accepted = self._tool()
        plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            candidate_plan={"available": True},
        )

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=plan,
        ) as planner:
            result = tool.prepare_sql_modeling_plan([{"source_index": 1, "name": "regional_revenue", "sql": sql}])

        assert result.success == 1
        assert planner.call_args.args[0][0].sql == sql
        assert evidence.sql_modeling_plan_status == "ready"
        assert accepted == [plan]

    def test_source_index_controls_planner_order(self):
        first_sql = "SELECT COUNT(*) AS order_count FROM orders"
        second_sql = "SELECT SUM(amount) AS revenue FROM orders"
        tool, evidence, accepted = self._tool()
        plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            candidate_plan={"available": True},
        )

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=plan,
        ) as planner:
            result = tool.prepare_sql_modeling_plan(
                [
                    {"source_index": 2, "name": "revenue", "sql": second_sql},
                    {"source_index": 1, "name": "order_count", "sql": first_sql},
                ]
            )

        assert result.success == 1
        assert [source.sql for source in planner.call_args.args[0]] == [first_sql, second_sql]
        assert evidence.sql_modeling_plan_status == "ready"
        assert accepted == [plan]

    def test_collects_multiple_batches_and_plans_once_on_finalize(self):
        first_sql = "SELECT COUNT(*) AS order_count FROM orders"
        second_sql = "SELECT SUM(amount) AS revenue FROM orders"
        tool, evidence, accepted = self._tool()
        plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            candidate_plan={"available": True},
        )

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=plan,
        ) as planner:
            collecting = tool.prepare_sql_modeling_plan(
                [{"source_index": 1, "name": "order_count", "sql": first_sql}],
                finalize=False,
            )
            ready = tool.prepare_sql_modeling_plan(
                [{"source_index": 2, "name": "revenue", "sql": second_sql}],
                finalize=True,
            )

        assert collecting.result == {"status": "collecting", "received_count": 1, "source_indexes": [1]}
        assert ready.success == 1
        assert [source.sql for source in planner.call_args.args[0]] == [first_sql, second_sql]
        planner.assert_called_once()
        assert evidence.sql_modeling_plan_status == "ready"
        assert accepted == [plan]

    def test_empty_final_batch_finalizes_collected_entries(self):
        sql = "SELECT COUNT(*) AS order_count FROM orders"
        tool, evidence, _ = self._tool()
        plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            candidate_plan={"available": True},
        )

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=plan,
        ) as planner:
            tool.prepare_sql_modeling_plan(
                [{"source_index": 10, "name": "order_count", "sql": sql}],
                finalize=False,
            )
            result = tool.prepare_sql_modeling_plan([], finalize=True)

        assert result.success == 1
        assert planner.call_args.args[0][0].sql == sql
        assert evidence.sql_modeling_plan_status == "ready"

    def test_conflicting_source_index_does_not_overwrite_an_earlier_batch(self):
        first_sql = "SELECT COUNT(*) AS order_count FROM orders"
        second_sql = "SELECT COUNT(*) AS customer_count FROM customers"
        tool, evidence, accepted = self._tool()
        plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            candidate_plan={"available": True},
        )

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=plan,
        ) as planner:
            tool.prepare_sql_modeling_plan(
                [{"source_index": 1, "name": "order_count", "sql": first_sql}],
                finalize=False,
            )
            conflict = tool.prepare_sql_modeling_plan(
                [{"source_index": 1, "name": "customer_count", "sql": second_sql}],
                finalize=True,
            )
            ready = tool.prepare_sql_modeling_plan(
                [{"source_index": 2, "name": "customer_count", "sql": second_sql}],
                finalize=True,
            )

        assert conflict.success == 0
        assert conflict.result["conflicting_source_indexes"] == [1]
        assert [source.sql for source in planner.call_args.args[0]] == [first_sql, second_sql]
        planner.assert_called_once()
        assert ready.success == 1
        assert evidence.sql_modeling_plan_status == "ready"
        assert accepted == [plan]

    def test_identical_source_index_retry_can_finalize_the_collection(self):
        entry = {
            "source_index": 1,
            "name": "order_count",
            "sql": "SELECT COUNT(*) AS order_count FROM orders",
        }
        tool, evidence, _ = self._tool()
        plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            candidate_plan={"available": True},
        )

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=plan,
        ) as planner:
            tool.prepare_sql_modeling_plan([entry], finalize=False)
            result = tool.prepare_sql_modeling_plan([entry], finalize=True)

        assert result.success == 1
        assert len(planner.call_args.args[0]) == 1
        assert evidence.sql_modeling_plan_status == "ready"

    def test_submitted_sql_preserves_leading_optimizer_hint(self):
        sql = "/*+ SET_VAR(query_timeout = 10) */\nSELECT COUNT(*) AS order_count FROM orders;"
        tool, evidence, accepted = self._tool()
        plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            candidate_plan={"available": True},
        )

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=plan,
        ) as planner:
            result = tool.prepare_sql_modeling_plan([{"source_index": 1, "name": "order_count", "sql": sql}])

        assert result.success == 1
        assert planner.call_args.args[0][0].sql == sql
        assert evidence.sql_modeling_plan_status == "ready"
        assert accepted == [plan]

    def test_identical_sql_in_distinct_request_positions_keeps_both_entries(self):
        sql = "SELECT COUNT(*) AS order_count FROM orders;"
        tool, evidence, accepted = self._tool()
        plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            candidate_plan={"available": True},
        )

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=plan,
        ) as planner:
            result = tool.prepare_sql_modeling_plan(
                [
                    {"source_index": 1, "name": "first_order_count", "sql": sql},
                    {"source_index": 2, "name": "second_order_count", "sql": sql},
                ]
            )

        assert result.success == 1
        assert [source.sql for source in planner.call_args.args[0]] == [sql, sql]
        assert evidence.sql_modeling_plan_status == "ready"
        assert accepted == [plan]

    def test_exact_non_select_sql_does_not_depend_on_extractor_support(self):
        sql = "SHOW TABLES;"
        tool, evidence, accepted = self._tool()
        plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            candidate_plan={"available": True},
        )

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=plan,
        ) as planner:
            result = tool.prepare_sql_modeling_plan([{"source_index": 1, "name": "available_tables", "sql": sql}])

        assert result.success == 1
        assert planner.call_args.args[0][0].sql == sql
        assert evidence.sql_modeling_plan_status == "ready"
        assert accepted == [plan]

    def test_complete_cte_is_copied_and_business_name_is_preserved(self):
        sql = "WITH daily AS (SELECT user_id FROM logins) SELECT COUNT(*) AS users FROM daily;"
        tool, evidence, accepted = self._tool()
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
                [
                    {
                        "source_index": 1,
                        "name": "Active Users",
                        "question": "Count active users",
                        "sql": sql,
                    }
                ]
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
        tool, evidence, accepted = self._tool()
        fixed_plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            candidate_plan={"available": True},
        )

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=fixed_plan,
        ) as planner:
            entries = [{"source_index": 1, "name": "order_count", "question": "Count orders", "sql": sql}]
            first = tool.prepare_sql_modeling_plan(entries)
            repeated = tool.prepare_sql_modeling_plan(entries)

        assert first.success == 1
        assert repeated.success == 1
        planner.assert_called_once()
        assert evidence.sql_modeling_plan_status == "ready"
        assert accepted == [fixed_plan]

    def test_changed_sql_after_a_fixed_plan_is_ignored_without_downgrading_it(self):
        original_sql = "SELECT COUNT(*) AS order_count FROM orders"
        changed_sql = "SELECT COUNT(*) AS customer_count FROM customers"
        tool, evidence, accepted = self._tool()
        fixed_plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            candidate_plan={"available": True},
        )
        original_entries = [
            {
                "source_index": 1,
                "name": "order_count",
                "question": "Count orders",
                "sql": original_sql,
            }
        ]

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=fixed_plan,
        ):
            tool.prepare_sql_modeling_plan(original_entries)
            changed = tool.prepare_sql_modeling_plan(
                [
                    {
                        "source_index": 1,
                        "name": "customer_count",
                        "question": "Count customers",
                        "sql": changed_sql,
                    }
                ]
            )
            repeated = tool.prepare_sql_modeling_plan(original_entries)

        assert changed.success == 0
        assert changed.result["status"] == "ready"
        assert repeated.success == 1
        assert evidence.sql_modeling_plan_status == "ready"
        assert accepted == [fixed_plan]

    def test_sql_plan_includes_automatic_semantic_source_inspection(self):
        sql = "SELECT SUM(amount) AS revenue FROM orders"
        inspected = {
            "status": "ready",
            "tables": [{"table_name": "orders"}],
            "relationships": [],
        }
        inspector = MagicMock(return_value=inspected)
        tool, _, accepted = self._tool(semantic_source_inspector=inspector)
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
            result = tool.prepare_sql_modeling_plan([{"source_index": 1, "name": "revenue", "sql": sql}])

        assert result.success == 1
        assert result.result["semantic_source_evidence"] == inspected
        inspector.assert_called_once_with(plan)
        assert accepted == [plan]

    def test_sparse_source_indexes_are_allowed(self):
        sql = "SELECT COUNT(*) AS orders FROM orders"
        tool, evidence, accepted = self._tool()
        plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            candidate_plan={"available": True},
        )

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=plan,
        ) as planner:
            result = tool.prepare_sql_modeling_plan([{"source_index": 20, "name": "order_count", "sql": sql}])

        assert result.success == 1
        assert planner.call_args.args[0][0].sql == sql
        assert evidence.sql_modeling_plan_status == "ready"
        assert accepted == [plan]

    def test_tool_preserves_literal_whitespace_and_statement_terminator(self):
        raw_sql = "SELECT 'a  b' AS label;"
        tool, evidence, accepted = self._tool()
        plan = SqlModelingPlan(
            source_fingerprint="source",
            metric_catalog_fingerprint="catalog",
            candidate_plan={"available": True},
        )

        with patch(
            "datus.tools.func_tool.sql_modeling_planner.SqlModelingPlanner.plan",
            return_value=plan,
        ) as planner:
            result = tool.prepare_sql_modeling_plan([{"source_index": 1, "name": "label_value", "sql": raw_sql}])

        assert result.success == 1
        assert planner.call_args.args[0][0].sql == raw_sql
        assert evidence.sql_modeling_plan_status == "ready"
        assert accepted == [plan]

    def test_generic_sql_index_is_not_a_business_name(self):
        raw_sql = "SELECT COUNT(*) AS order_count FROM orders"
        tool, evidence, accepted = self._tool()

        result = tool.prepare_sql_modeling_plan([{"source_index": 1, "name": "sql_1", "sql": raw_sql}])

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
