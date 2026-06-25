# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/func_tool/semantic_discovery_tools.py"""

import json
from unittest.mock import MagicMock, patch

from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.semantic_discovery_tools import SemanticDiscoveryTools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db_tool(agent_config=None, sub_agent_name="test_agent"):
    """Build a mock DBFuncTool."""
    db_tool = MagicMock()
    db_tool.agent_config = agent_config or MagicMock()
    db_tool.sub_agent_name = sub_agent_name
    return db_tool


def _make_tools(db_tool=None) -> SemanticDiscoveryTools:
    if db_tool is None:
        db_tool = _make_db_tool()
    return SemanticDiscoveryTools(db_tool=db_tool)


# ---------------------------------------------------------------------------
# get_multiple_tables_ddl
# ---------------------------------------------------------------------------


class TestGetMultipleTablesDDL:
    def test_success_single_table(self):
        db_tool = _make_db_tool()
        db_tool.get_table_ddl.return_value = FuncToolResult(
            success=1, result={"definition": "CREATE TABLE orders (id INT)"}
        )
        tools = _make_tools(db_tool)
        result = tools.get_multiple_tables_ddl(["orders"])
        assert result.success == 1
        assert len(result.result) == 1
        assert result.result[0]["table_name"] == "orders"

    def test_success_multiple_tables(self):
        db_tool = _make_db_tool()
        db_tool.get_table_ddl.return_value = FuncToolResult(success=1, result={"definition": "CREATE TABLE t (id INT)"})
        tools = _make_tools(db_tool)
        result = tools.get_multiple_tables_ddl(["orders", "customers"])
        assert result.success == 1
        assert len(result.result) == 2

    def test_partial_failure(self):
        db_tool = _make_db_tool()

        def side_effect(table, *args, **kwargs):
            if table == "orders":
                return FuncToolResult(success=1, result={"definition": "CREATE TABLE orders (id INT)"})
            return FuncToolResult(success=0, error="Table not found")

        db_tool.get_table_ddl.side_effect = side_effect
        tools = _make_tools(db_tool)
        result = tools.get_multiple_tables_ddl(["orders", "missing"])
        assert result.success == 1
        assert result.result[0]["table_name"] == "orders"
        assert "error" in result.result[1]

    def test_exception_returns_error(self):
        db_tool = _make_db_tool()
        db_tool.get_table_ddl.side_effect = Exception("DB error")
        tools = _make_tools(db_tool)
        result = tools.get_multiple_tables_ddl(["orders"])
        assert result.success == 0
        assert "DB error" in result.error

    def test_empty_tables_list(self):
        tools = _make_tools()
        result = tools.get_multiple_tables_ddl([])
        assert result.success == 1
        assert result.result == []


# ---------------------------------------------------------------------------
# _extract_foreign_keys_from_ddl
# ---------------------------------------------------------------------------


class TestExtractForeignKeys:
    def test_extracts_foreign_key(self):
        ddl = """CREATE TABLE orders (
            id INT,
            customer_id INT,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        )"""
        db_tool = _make_db_tool()
        db_tool.get_table_ddl.return_value = FuncToolResult(success=1, result={"definition": ddl})
        tools = _make_tools(db_tool)
        result = tools._extract_foreign_keys_from_ddl(["orders"], "", "", "")
        assert len(result) == 1
        assert result[0]["source_table"] == "orders"
        assert result[0]["source_column"] == "customer_id"
        assert result[0]["target_table"] == "customers"
        assert result[0]["confidence"] == "high"

    def test_no_foreign_keys(self):
        ddl = "CREATE TABLE orders (id INT, name VARCHAR(100))"
        db_tool = _make_db_tool()
        db_tool.get_table_ddl.return_value = FuncToolResult(success=1, result={"definition": ddl})
        tools = _make_tools(db_tool)
        result = tools._extract_foreign_keys_from_ddl(["orders"], "", "", "")
        assert result == []

    def test_ddl_fetch_failure_skipped(self):
        db_tool = _make_db_tool()
        db_tool.get_table_ddl.return_value = FuncToolResult(success=0, error="Not found")
        tools = _make_tools(db_tool)
        result = tools._extract_foreign_keys_from_ddl(["missing"], "", "", "")
        assert result == []


# ---------------------------------------------------------------------------
# _infer_from_column_names
# ---------------------------------------------------------------------------


class TestInferFromColumnNames:
    def test_infers_relationship_from_column_name(self):
        db_tool = _make_db_tool()

        # "customer_id" strips "_id" -> "customer", so target table must be "customer"
        orders_result = FuncToolResult(
            success=1,
            result={"columns": [{"name": "id"}, {"name": "customer_id"}]},
        )
        customer_result = FuncToolResult(
            success=1,
            result={"columns": [{"name": "id"}, {"name": "name"}]},
        )

        call_count = [0]

        def describe_side_effect(*args, **kwargs):
            # First call -> orders, second call -> customer
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                return orders_result
            return customer_result

        db_tool.describe_table.side_effect = describe_side_effect
        tools = _make_tools(db_tool)
        result = tools._infer_from_column_names(["orders", "customer"], "", "", "")
        assert len(result) == 1
        assert result[0]["source_table"] == "orders"
        assert result[0]["source_column"] == "customer_id"
        assert result[0]["target_table"] == "customer"
        assert result[0]["confidence"] == "low"
        assert result[0]["evidence"] == "column_name"

    def test_no_matching_columns(self):
        db_tool = _make_db_tool()
        db_tool.describe_table.return_value = FuncToolResult(
            success=1, result={"columns": [{"name": "name"}, {"name": "value"}]}
        )
        tools = _make_tools(db_tool)
        result = tools._infer_from_column_names(["t1", "t2"], "", "", "")
        assert result == []

    def test_schema_fetch_failure_skipped(self):
        db_tool = _make_db_tool()
        db_tool.describe_table.return_value = FuncToolResult(success=0, error="Error")
        tools = _make_tools(db_tool)
        result = tools._infer_from_column_names(["t1"], "", "", "")
        assert result == []


# ---------------------------------------------------------------------------
# _deduplicate_relationships
# ---------------------------------------------------------------------------


class TestDeduplicateRelationships:
    def test_removes_duplicates(self):
        rels = [
            {
                "source_table": "a",
                "source_column": "id",
                "target_table": "b",
                "target_column": "a_id",
                "confidence": "high",
                "evidence": "fk",
            },
            {
                "source_table": "a",
                "source_column": "id",
                "target_table": "b",
                "target_column": "a_id",
                "confidence": "medium",
                "evidence": "join",
            },
        ]
        tools = _make_tools()
        result = tools._deduplicate_relationships(rels)
        assert len(result) == 1
        # First by confidence order: high wins
        assert result[0]["confidence"] == "high"

    def test_sorts_by_confidence(self):
        rels = [
            {
                "source_table": "a",
                "source_column": "x",
                "target_table": "b",
                "target_column": "y",
                "confidence": "low",
                "evidence": "col",
            },
            {
                "source_table": "c",
                "source_column": "p",
                "target_table": "d",
                "target_column": "q",
                "confidence": "high",
                "evidence": "fk",
            },
        ]
        tools = _make_tools()
        result = tools._deduplicate_relationships(rels)
        assert result[0]["confidence"] == "high"
        assert result[1]["confidence"] == "low"

    def test_empty_list(self):
        tools = _make_tools()
        result = tools._deduplicate_relationships([])
        assert result == []


# ---------------------------------------------------------------------------
# _analyze_join_patterns_from_history
# ---------------------------------------------------------------------------


class TestAnalyzeJoinPatterns:
    def test_no_agent_config_returns_empty(self):
        db_tool = _make_db_tool(agent_config=None)
        db_tool.agent_config = None
        tools = _make_tools(db_tool)
        result = tools._analyze_join_patterns_from_history(["orders", "customers"], 10)
        assert result == []

    def test_finds_join_pattern(self):
        db_tool = _make_db_tool()
        sql_entry = {"sql": "SELECT * FROM orders o JOIN customers c ON orders.customer_id = customers.id"}
        mock_rag = MagicMock()
        mock_rag.search_reference_sql.return_value = [sql_entry]
        # ReferenceSqlRAG is imported locally inside the method body
        with patch("datus.storage.reference_sql.store.ReferenceSqlRAG", return_value=mock_rag):
            tools = _make_tools(db_tool)
            result = tools._analyze_join_patterns_from_history(["orders", "customers"], 10)
        assert len(result) >= 1
        assert any(r["evidence"] == "join_pattern" for r in result)

    def test_finds_alias_join_pattern(self):
        db_tool = _make_db_tool()
        sql_entry = {"sql": "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id"}
        mock_rag = MagicMock()
        mock_rag.search_reference_sql.return_value = [sql_entry]
        with patch("datus.storage.reference_sql.store.ReferenceSqlRAG", return_value=mock_rag):
            tools = _make_tools(db_tool)
            result = tools._analyze_join_patterns_from_history(["orders", "customers"], 10)
        assert result == [
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "id",
                "confidence": "medium",
                "evidence": "join_pattern",
            }
        ]

    def test_search_exception_handled_gracefully(self):
        db_tool = _make_db_tool()
        mock_rag = MagicMock()
        mock_rag.search_reference_sql.side_effect = Exception("DB unavailable")
        with patch("datus.storage.reference_sql.store.ReferenceSqlRAG", return_value=mock_rag):
            tools = _make_tools(db_tool)
            result = tools._analyze_join_patterns_from_history(["orders"], 10)
        assert result == []


# ---------------------------------------------------------------------------
# analyze_table_relationships (integration of strategies)
# ---------------------------------------------------------------------------


class TestAnalyzeTableRelationships:
    def test_returns_relationships_from_fk(self):
        ddl = "CREATE TABLE a (id INT, b_id INT, FOREIGN KEY (b_id) REFERENCES b(id))"
        db_tool = _make_db_tool()
        db_tool.get_table_ddl.return_value = FuncToolResult(success=1, result={"definition": ddl})
        mock_rag = MagicMock()
        mock_rag.search_reference_sql.return_value = []
        with patch("datus.storage.reference_sql.store.ReferenceSqlRAG", return_value=mock_rag):
            tools = _make_tools(db_tool)
            result = tools.analyze_table_relationships(["a", "b"])
        assert result.success == 1
        assert "relationships" in result.result
        assert result.result["relationships"][0]["confidence"] == "high"

    def test_falls_back_to_column_names_when_no_fk_or_join(self):
        db_tool = _make_db_tool()
        db_tool.get_table_ddl.return_value = FuncToolResult(
            success=1, result={"definition": "CREATE TABLE a (id INT, b_id INT)"}
        )

        def describe_side(table, *args):
            if table == "a":
                return FuncToolResult(success=1, result={"columns": [{"name": "id"}, {"name": "b_id"}]})
            elif table == "b":
                return FuncToolResult(success=1, result={"columns": [{"name": "id"}]})
            return FuncToolResult(success=0, error="not found")

        db_tool.describe_table.side_effect = describe_side
        mock_rag = MagicMock()
        mock_rag.search_reference_sql.return_value = []
        with patch("datus.storage.reference_sql.store.ReferenceSqlRAG", return_value=mock_rag):
            tools = _make_tools(db_tool)
            result = tools.analyze_table_relationships(["a", "b"])
        assert result.success == 1

    def test_exception_returns_error(self):
        db_tool = _make_db_tool()
        db_tool.get_table_ddl.side_effect = Exception("crash")
        tools = _make_tools(db_tool)
        result = tools.analyze_table_relationships(["a"])
        assert result.success == 0


# ---------------------------------------------------------------------------
# analyze_column_usage_patterns
# ---------------------------------------------------------------------------


class TestAnalyzeColumnUsagePatterns:
    def test_no_agent_config_returns_error(self):
        db_tool = _make_db_tool(agent_config=None)
        db_tool.agent_config = None
        tools = _make_tools(db_tool)
        result = tools.analyze_column_usage_patterns("orders")
        assert result.success == 0
        assert "agent_config" in result.error

    def test_describe_table_failure(self):
        db_tool = _make_db_tool()
        db_tool.describe_table.return_value = FuncToolResult(success=0, error="not found")
        tools = _make_tools(db_tool)
        result = tools.analyze_column_usage_patterns("orders")
        assert result.success == 0

    def test_empty_sql_history(self):
        db_tool = _make_db_tool()
        db_tool.describe_table.return_value = FuncToolResult(
            success=1,
            result={"columns": [{"name": "status"}, {"name": "amount"}]},
        )
        mock_rag = MagicMock()
        mock_rag.search_reference_sql.return_value = []
        with patch("datus.storage.reference_sql.store.ReferenceSqlRAG", return_value=mock_rag):
            tools = _make_tools(db_tool)
            result = tools.analyze_column_usage_patterns("orders", sample_sql_queries=5)
        assert result.success == 1
        assert result.result["column_patterns"] == {}

    def test_finds_operator_pattern(self):
        db_tool = _make_db_tool()
        db_tool.describe_table.return_value = FuncToolResult(success=1, result={"columns": [{"name": "status"}]})
        sql_entries = [{"sql": "SELECT * FROM orders WHERE status = 1"}]
        mock_rag = MagicMock()
        mock_rag.search_reference_sql.return_value = sql_entries
        with patch("datus.storage.reference_sql.store.ReferenceSqlRAG", return_value=mock_rag):
            tools = _make_tools(db_tool)
            result = tools.analyze_column_usage_patterns("orders", columns=["status"])
        assert result.success == 1
        assert "status" in result.result["column_patterns"]
        assert "=" in result.result["column_patterns"]["status"]["operators"]

    def test_finds_function_pattern(self):
        db_tool = _make_db_tool()
        db_tool.describe_table.return_value = FuncToolResult(success=1, result={"columns": [{"name": "tags"}]})
        sql_entries = [{"sql": "SELECT * FROM orders WHERE FIND_IN_SET('vip', tags)"}]
        mock_rag = MagicMock()
        mock_rag.search_reference_sql.return_value = sql_entries
        with patch("datus.storage.reference_sql.store.ReferenceSqlRAG", return_value=mock_rag):
            tools = _make_tools(db_tool)
            result = tools.analyze_column_usage_patterns("orders", columns=["tags"])
        assert result.success == 1
        assert "tags" in result.result["column_patterns"]
        assert "FIND_IN_SET" in result.result["column_patterns"]["tags"]["functions"]

    def test_filters_sql_not_containing_table(self):
        db_tool = _make_db_tool()
        db_tool.describe_table.return_value = FuncToolResult(success=1, result={"columns": [{"name": "status"}]})
        sql_entries = [{"sql": "SELECT * FROM other_table WHERE status = 1"}]
        mock_rag = MagicMock()
        mock_rag.search_reference_sql.return_value = sql_entries
        with patch("datus.storage.reference_sql.store.ReferenceSqlRAG", return_value=mock_rag):
            tools = _make_tools(db_tool)
            result = tools.analyze_column_usage_patterns("orders", columns=["status"])
        assert result.success == 1
        # SQL doesn't mention 'orders', so patterns should be empty
        assert result.result["column_patterns"] == {}

    def test_specific_columns_subset(self):
        db_tool = _make_db_tool()
        db_tool.describe_table.return_value = FuncToolResult(
            success=1,
            result={"columns": [{"name": "status"}, {"name": "amount"}, {"name": "date"}]},
        )
        sql_entries = [{"sql": "SELECT * FROM orders WHERE status = 1"}]
        mock_rag = MagicMock()
        mock_rag.search_reference_sql.return_value = sql_entries
        with patch("datus.storage.reference_sql.store.ReferenceSqlRAG", return_value=mock_rag):
            tools = _make_tools(db_tool)
            # Only analyze "status" column
            result = tools.analyze_column_usage_patterns("orders", columns=["status"])
        assert result.success == 1

    def test_exception_returns_error(self):
        db_tool = _make_db_tool()
        db_tool.describe_table.side_effect = Exception("crash")
        tools = _make_tools(db_tool)
        result = tools.analyze_column_usage_patterns("orders")
        assert result.success == 0


# ---------------------------------------------------------------------------
# analyze_metric_candidates_from_history
# ---------------------------------------------------------------------------


class TestAnalyzeMetricCandidatesFromHistory:
    def test_available_tools_includes_metric_candidate_mining(self):
        tools = _make_tools()
        tool_names = {tool.name for tool in tools.available_tools()}
        assert {
            "analyze_table_relationships",
            "get_multiple_tables_ddl",
            "analyze_column_usage_patterns",
            "analyze_metric_candidates_from_history",
        }.issubset(tool_names)

    def test_ratio_candidate_preserves_base_measures(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                """
                SELECT dt, SUM(paid_amount) / COUNT(DISTINCT user_id) AS paid_arppu
                FROM orders
                WHERE status = 'paid'
                GROUP BY dt
                """
            ]
        )

        assert result.success == 1
        candidates = result.result["metric_candidates"]
        assert len(candidates) == 1
        assert candidates[0]["name"] == "paid_arppu"
        assert candidates[0]["metric_type"] == "ratio"
        assert candidates[0]["candidate_classification"] == "exact_metric"
        assert candidates[0]["expression_kind"] == "aggregate_ratio_expr"
        assert candidates[0]["equivalence"] == "exact"
        assert candidates[0]["requires_validation"] is False
        assert candidates[0]["dimensions"] == ["dt"]
        assert candidates[0]["filters"] == ["status = 'paid'"]
        assert {m["agg"] for m in candidates[0]["base_measures"]} == {"SUM", "COUNT_DISTINCT"}

    def test_expr_candidate_for_measure_expression(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                """
                SELECT (SUM(revenue) - SUM(cost)) / SUM(revenue) AS gross_margin_rate
                FROM orders
                """
            ]
        )

        candidate = result.result["metric_candidates"][0]
        assert candidate["name"] == "gross_margin_rate"
        assert candidate["metric_type"] == "expr"
        assert candidate["candidate_classification"] == "exact_metric"
        assert candidate["equivalence"] == "exact"
        assert len(candidate["base_measures"]) == 2

    def test_derived_candidate_for_existing_metric_expression(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=["SELECT revenue / ad_spend AS roas FROM metric_table"],
            existing_metric_catalog_json=(
                '[{"name": "revenue", "type": "measure_proxy", "subject_path": "finance"}, '
                '{"name": "ad_spend", "type": "measure_proxy", "subject_path": "finance"}]'
            ),
        )

        candidate = result.result["metric_candidates"][0]
        assert candidate["name"] == "roas"
        assert candidate["metric_type"] == "derived"
        assert result.result["direct_metric_candidates"] == []
        assert result.result["derived_metric_candidates"] == [candidate]
        assert candidate["referenced_metrics"] == [
            {"name": "ad_spend", "type": "measure_proxy", "subject_path": "finance"},
            {"name": "revenue", "type": "measure_proxy", "subject_path": "finance"},
        ]

    def test_existing_metric_passthrough_is_identity_reference_not_derived_candidate(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=["SELECT revenue AS revenue FROM metric_table"],
            existing_metric_catalog_json='[{"name": "revenue", "type": "measure_proxy"}]',
        )

        assert result.result["metric_candidates"] == []
        assert result.result["direct_metric_candidates"] == []
        assert result.result["derived_metric_candidates"] == []
        assert result.result["identity_metric_references"] == [
            {
                "evidence_kind": "identity_metric_reference",
                "name": "revenue",
                "expression": "revenue",
                "source_alias": "revenue",
                "source_sql_name": "sql_1",
                "referenced_metrics": [{"name": "revenue", "type": "measure_proxy"}],
                "reason": "projection references existing metric without a new business formula",
            }
        ]

    def test_reference_sql_search_keeps_all_unique_entries(self):
        tools = _make_tools()
        with patch("datus.storage.reference_sql.store.ReferenceSqlRAG") as rag_cls:
            rag_cls.return_value.search_reference_sql.return_value = [
                {"sql": "SELECT SUM(amount) AS revenue FROM orders"},
                {"sql": "SELECT SUM(cost) AS cost FROM orders"},
            ]

            result = tools.analyze_metric_candidates_from_history(query_text="orders")

        assert result.success == 1
        assert {candidate["name"] for candidate in result.result["metric_candidates"]} == {"revenue", "cost"}

    def test_cumulative_candidate_for_window_expression(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                """
                SELECT dt, SUM(revenue) OVER (ORDER BY dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7d_revenue
                FROM orders
                """
            ]
        )

        candidate = next(item for item in result.result["metric_candidates"] if item["name"] == "rolling_7d_revenue")
        assert candidate["name"] == "rolling_7d_revenue"
        assert candidate["metric_type"] == "cumulative"
        assert candidate["window"] == "7 days"
        assert candidate["window_aggregation"] == "sum"
        assert candidate["expression"] == "SUM(revenue)"
        assert candidate["base_metric_name"] == "revenue"

    def test_window_candidate_uses_catalog_base_metric_without_synthetic_measure(self):
        tools = _make_tools()
        detail = {
            "name": "running_order_count",
            "base_metric_name": "order_count",
            "base_expression": "COUNT(DISTINCT order_id)",
            "aggregate": "SUM",
            "window_aggregation": "sum",
            "grain_to_date": "month",
            "dimensions": ["metric_time__month"],
        }

        candidate = tools._window_metric_candidate_from_detail(
            detail,
            base_candidate=None,
            existing_metric_catalog={"order_count": {"name": "order_count", "type": "aggregate"}},
        )

        assert candidate["base_measures"] == []
        assert candidate["referenced_metrics"] == [{"name": "order_count", "type": "aggregate"}]

    def test_window_candidate_signature_includes_order_by_and_time_grain(self):
        tools = _make_tools()
        base = {
            "name": "running_order_count",
            "metric_type": "cumulative",
            "expression": "COUNT(DISTINCT order_id)",
            "window_aggregation": "sum",
            "window_order_by": ["metric_time__month"],
            "time_grain": "month",
        }
        different_grain = {**base, "time_grain": "week"}
        different_order = {**base, "window_order_by": ["created_month"]}

        base_signature = tools._metric_candidate_formula_signature(base)

        assert base_signature != tools._metric_candidate_formula_signature(different_grain)
        assert base_signature != tools._metric_candidate_formula_signature(different_order)

    def test_window_aggregate_candidate_resolves_cte_base_metric(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                """
                WITH monthly AS (
                    SELECT
                        DATE_TRUNC('month', order_date) AS metric_time__month,
                        COUNT(DISTINCT order_id) AS order_count
                    FROM fact_orders
                    WHERE order_date >= '2025-04-01'
                    GROUP BY DATE_TRUNC('month', order_date)
                )
                SELECT
                    metric_time__month,
                    order_count,
                    SUM(order_count) OVER (
                        ORDER BY metric_time__month
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS running_order_count,
                    AVG(order_count) OVER (
                        ORDER BY metric_time__month
                        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                    ) AS moving_3_month_order_count_avg,
                    COUNT(*) OVER (
                        ORDER BY metric_time__month
                        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                    ) AS moving_window_month_count
                FROM monthly
                ORDER BY metric_time__month
                """
            ]
        )

        candidates = {candidate["name"]: candidate for candidate in result.result["metric_candidates"]}
        assert {
            "order_count",
            "running_order_count",
            "moving_3_month_order_count_avg",
            "moving_window_month_count",
        } <= set(candidates)
        assert candidates["order_count"]["expression"] == "COUNT(DISTINCT order_id)"

        running = candidates["running_order_count"]
        assert running["expression"] == "COUNT(DISTINCT order_id)"
        assert running["grain_to_date"] == "month"
        assert running["window_aggregation"] == "sum"
        assert running["base_metric_name"] == "order_count"

        moving_avg = candidates["moving_3_month_order_count_avg"]
        assert moving_avg["window"] == "3 months"
        assert moving_avg["window_aggregation"] == "avg"
        assert moving_avg["base_metric_name"] == "order_count"

        row_count = candidates["moving_window_month_count"]
        assert row_count["window"] == "3 months"
        assert row_count["window_aggregation"] == "row_count"

    def test_lag_period_aggregation_becomes_previous_and_delta_metrics(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                """
                WITH monthly_orders AS (
                    SELECT
                        DATE_TRUNC('month', order_date) AS metric_month,
                        COUNT(DISTINCT order_id) AS order_count
                    FROM fact_orders
                    WHERE order_date >= '2025-01-01' AND order_date < '2025-07-01'
                    GROUP BY DATE_TRUNC('month', order_date)
                ),
                period_comparison AS (
                    SELECT
                        metric_month,
                        order_count,
                        LAG(order_count) OVER (ORDER BY metric_month) AS order_count_previous_period
                    FROM monthly_orders
                )
                SELECT
                    metric_month,
                    order_count,
                    order_count_previous_period,
                    order_count - order_count_previous_period AS order_count_period_delta
                FROM period_comparison
                ORDER BY metric_month
                """
            ]
        )

        assert result.success == 1
        assert [candidate["name"] for candidate in result.result["direct_metric_candidates"]] == ["order_count"]

        derived = {candidate["name"]: candidate for candidate in result.result["derived_metric_candidates"]}
        assert sorted(derived) == ["order_count_period_delta", "order_count_previous_period"]

        previous = derived["order_count_previous_period"]
        assert previous["metric_type"] == "derived"
        assert previous["metric_kind"] == "derived"
        assert previous["expression"] == "order_count_previous_period"
        assert previous["inputs"] == [
            {
                "name": "order_count",
                "alias": "order_count_previous_period",
                "offset_window": "1 month",
            }
        ]

        delta = derived["order_count_period_delta"]
        assert delta["metric_type"] == "derived"
        assert delta["metric_kind"] == "derived"
        assert delta["expression"] == "order_count - order_count_previous_period"
        assert delta["inputs"] == [
            {"name": "order_count"},
            {
                "name": "order_count",
                "alias": "order_count_previous_period",
                "offset_window": "1 month",
            },
        ]

    def test_period_shift_source_lookup_accepts_legacy_from_key(self):
        from sqlglot import parse_one

        tools = _make_tools()
        select = parse_one("SELECT order_id FROM fact_orders")
        from_clause = select.args.pop("from_", None)
        if from_clause is None:
            from_clause = select.args.get("from")
        select.args["from"] = from_clause

        assert tools._direct_source_names(select) == ["fact_orders"]

    def test_monthly_order_count_generates_previous_month_and_delta_metrics(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_entries_json=json.dumps(
                [
                    {
                        "name": "monthly_order_count_mom",
                        "question": "Month-over-month order count from April to October",
                        "sql": """
                        WITH monthly AS (
                            SELECT
                                DATE_TRUNC('month', order_date) AS order_month,
                                COUNT(DISTINCT order_id) AS order_count
                            FROM fact_orders
                            WHERE order_date >= '2025-04-01' AND order_date <= '2025-10-31'
                            GROUP BY DATE_TRUNC('month', order_date)
                        ),
                        compared AS (
                            SELECT
                                order_month,
                                order_count,
                                LAG(order_count) OVER (ORDER BY order_month)
                                    AS previous_month_order_count
                            FROM monthly
                        )
                        SELECT
                            order_month,
                            order_count,
                            previous_month_order_count,
                            order_count - previous_month_order_count AS order_count_mom_delta
                        FROM compared
                        ORDER BY order_month
                        """,
                    }
                ]
            ),
            existing_metric_catalog_json=json.dumps([{"name": "order_count", "type": "aggregate"}]),
        )

        assert result.success == 1
        derived = {candidate["name"]: candidate for candidate in result.result["derived_metric_candidates"]}
        assert sorted(derived) == ["order_count_mom_delta", "previous_month_order_count"]
        assert derived["previous_month_order_count"]["inputs"] == [
            {
                "name": "order_count",
                "alias": "previous_month_order_count",
                "offset_window": "1 month",
            }
        ]
        assert derived["order_count_mom_delta"]["inputs"] == [
            {"name": "order_count"},
            {
                "name": "order_count",
                "alias": "previous_month_order_count",
                "offset_window": "1 month",
            },
        ]

    def test_inline_lag_metric_math_uses_source_time_grain_context(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                """
                WITH monthly_orders AS (
                    SELECT
                        DATE_TRUNC('month', order_date) AS metric_month,
                        COUNT(DISTINCT order_id) AS order_count
                    FROM fact_orders
                    GROUP BY DATE_TRUNC('month', order_date)
                )
                SELECT
                    metric_month,
                    order_count,
                    order_count - LAG(order_count) OVER (ORDER BY metric_month) AS order_count_period_delta
                FROM monthly_orders
                ORDER BY metric_month
                """
            ]
        )

        assert result.success == 1
        matches = [
            candidate
            for candidate in result.result["derived_metric_candidates"]
            if candidate["name"] == "order_count_period_delta"
        ]
        assert len(matches) == 1
        delta = matches[0]
        assert delta["expression"] == "order_count - order_count_prev"
        assert delta["inputs"] == [
            {"name": "order_count"},
            {
                "name": "order_count",
                "alias": "order_count_prev",
                "offset_window": "1 month",
            },
        ]

    def test_period_shift_aliases_are_scoped_to_source_select(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                """
                WITH monthly_orders AS (
                    SELECT
                        DATE_TRUNC('month', order_date) AS metric_month,
                        COUNT(DISTINCT order_id) AS order_count
                    FROM fact_orders
                    GROUP BY DATE_TRUNC('month', order_date)
                ),
                weekly_orders AS (
                    SELECT
                        DATE_TRUNC('week', order_date) AS metric_week,
                        COUNT(DISTINCT order_id) AS order_count
                    FROM fact_orders
                    GROUP BY DATE_TRUNC('week', order_date)
                ),
                monthly_comparison AS (
                    SELECT
                        metric_month,
                        order_count,
                        LAG(order_count) OVER (ORDER BY metric_month) AS order_count_previous_period
                    FROM monthly_orders
                ),
                weekly_comparison AS (
                    SELECT
                        metric_week,
                        order_count,
                        LAG(order_count) OVER (ORDER BY metric_week) AS order_count_previous_period
                    FROM weekly_orders
                )
                SELECT
                    metric_month,
                    order_count,
                    order_count_previous_period,
                    order_count - order_count_previous_period AS order_count_period_delta
                FROM monthly_comparison
                ORDER BY metric_month
                """
            ]
        )

        assert result.success == 1
        delta = next(
            candidate
            for candidate in result.result["derived_metric_candidates"]
            if candidate["name"] == "order_count_period_delta"
        )
        assert delta["inputs"] == [
            {"name": "order_count"},
            {
                "name": "order_count",
                "alias": "order_count_previous_period",
                "offset_window": "1 month",
            },
        ]

    def test_conditional_aggregation_keeps_case_measure_evidence(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                """
                SELECT SUM(CASE WHEN status = 'paid' THEN amount ELSE 0 END) AS paid_revenue
                FROM orders
                """
            ]
        )

        candidate = result.result["metric_candidates"][0]
        assert candidate["metric_type"] == "measure_proxy"
        assert candidate["base_measures"][0]["expr"] == "CASE WHEN status = 'paid' THEN amount ELSE 0 END"

    def test_filter_only_sql_becomes_non_metric_evidence(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=["SELECT * FROM users WHERE is_test = 0 AND country = 'US'"]
        )

        assert result.result["metric_candidates"] == []
        evidence = result.result["non_metric_evidence"][0]
        assert evidence["tables"] == ["users"]
        assert evidence["filters"] == ["is_test = 0 AND country = 'US'"]

    def test_raw_ratio_with_rate_context_becomes_llm_review_candidate(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_entries_json=json.dumps(
                [
                    {
                        "question": "Please list the lowest three eligible free rates for students aged 5-17.",
                        "sql": """
                        SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`
                        FROM frpm
                        WHERE `Educational Option Type` = 'Continuation School'
                        ORDER BY `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` ASC
                        LIMIT 3
                        """,
                    }
                ]
            )
        )

        assert result.success == 1
        assert result.result["non_metric_evidence"] == []
        assert result.result["direct_metric_candidates"] == []
        candidate = result.result["llm_review_candidates"][0]
        assert candidate["evidence_kind"] == "llm_review_projection"
        assert candidate["candidate_classification"] == "llm_review_candidate"
        assert candidate["expression_kind"] == "row_ratio_expr"
        assert candidate["equivalence"] == "lifted"
        assert candidate["requires_validation"] is True
        assert candidate["name"] == "free_meal_count_ages_5_17_rate"
        assert candidate["metric_type"] == "ratio"
        assert candidate["requires_name_translation"] is True
        assert candidate["source_context"] == "Please list the lowest three eligible free rates for students aged 5-17."
        measures_by_role = {measure["role"]: measure for measure in candidate["base_measures"]}
        assert measures_by_role["numerator"]["agg"] == "SUM"
        assert measures_by_role["numerator"]["expr"] == '"Free Meal Count (Ages 5-17)"'
        assert measures_by_role["denominator"]["agg"] == "SUM"
        assert measures_by_role["denominator"]["expr"] == '"Enrollment (Ages 5-17)"'

    def test_detail_success_story_keeps_detail_sql_non_metric(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_entries_json=json.dumps(
                [
                    {
                        "question": "Please list the lowest three eligible free rates for students aged 5-17.",
                        "sql": """
                        SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`
                        FROM frpm
                        WHERE `Educational Option Type` = 'Continuation School'
                        ORDER BY `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` ASC
                        LIMIT 3
                        """,
                    },
                    {
                        "question": "Please list the zip code of all charter schools.",
                        "sql": """
                        SELECT T2.Zip
                        FROM frpm AS T1
                        INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode
                        WHERE T1.`District Name` = 'Fresno County Office of Education'
                          AND T1.`Charter School (Y/N)` = 1
                        """,
                    },
                ]
            )
        )

        assert [candidate["metric_type"] for candidate in result.result["llm_review_candidates"]] == ["ratio"]
        assert result.result["direct_metric_candidates"] == []
        assert len(result.result["non_metric_evidence"]) == 1
        assert result.result["non_metric_evidence"][0]["source_sql_name"] == "sql_2"
        assert result.result["source_classifications"] == [
            {"source_sql_name": "sql_1", "classification": "llm_review_candidate", "reason": ""},
            {"source_sql_name": "sql_2", "classification": "cohort_or_dataset_only", "reason": ""},
        ]

    def test_raw_division_without_rate_context_becomes_llm_review_candidate(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=["SELECT price / quantity FROM order_lines WHERE quantity > 0"]
        )

        assert result.result["non_metric_evidence"] == []
        candidate = result.result["llm_review_candidates"][0]
        assert candidate["name"] == "price_per_quantity"
        assert candidate["metric_type"] == "ratio"
        assert candidate["confidence"] == "low"
        assert candidate["equivalence"] == "lifted"
        assert candidate["requires_validation"] is True

    def test_percentage_scaled_raw_ratio_becomes_llm_review_candidate(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=["SELECT paid_users * 100.0 / total_users AS paid_user_pct FROM cohorts"]
        )

        candidate = result.result["llm_review_candidates"][0]
        assert candidate["name"] == "paid_user_pct"
        assert candidate["metric_type"] == "ratio"
        measures_by_role = {measure["role"]: measure for measure in candidate["base_measures"]}
        assert measures_by_role["numerator"]["expr"] == "paid_users"
        assert measures_by_role["denominator"]["expr"] == "total_users"

    def test_wrapped_raw_ratio_becomes_llm_review_candidate(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=["SELECT ROUND(CAST(a / NULLIF(b, 0) AS DOUBLE), 2) AS ratio_value FROM t"]
        )

        candidate = result.result["llm_review_candidates"][0]
        assert candidate["expression_kind"] == "row_ratio_expr"
        measures_by_role = {measure["role"]: measure for measure in candidate["base_measures"]}
        assert measures_by_role["numerator"]["expr"] == "a"
        assert measures_by_role["denominator"]["expr"] == "b"

    def test_count_star_with_distinct_business_count_is_support_measure(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                """
                SELECT COUNT(*) AS order_row_count,
                       COUNT(DISTINCT order_id) AS order_count
                FROM fact_orders
                WHERE buyer_name LIKE '%test%' AND FIND_IN_SET('priority', order_tags)
                """
            ]
        )

        assert result.success == 1
        assert [candidate["name"] for candidate in result.result["direct_metric_candidates"]] == ["order_count"]
        assert [candidate["name"] for candidate in result.result["metric_candidates"]] == ["order_count"]
        assert result.result["support_measure_candidates"][0]["name"] == "order_row_count"
        assert result.result["support_measure_candidates"][0]["evidence_kind"] == "support_measure"
        assert result.result["support_measure_candidates"][0]["base_measures"][0]["agg"] == "COUNT"

    def test_count_star_without_distinct_business_count_stays_direct_metric(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=["SELECT COUNT(*) AS order_count, SUM(amount) AS revenue FROM orders"]
        )

        assert result.success == 1
        assert [candidate["name"] for candidate in result.result["direct_metric_candidates"]] == [
            "order_count",
            "revenue",
        ]
        assert result.result["support_measure_candidates"] == []

    def test_repeated_aliases_are_merged(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                "SELECT SUM(amount) AS revenue FROM orders",
                "SELECT SUM(amount) AS revenue FROM payments",
            ]
        )

        candidates = result.result["metric_candidates"]
        assert len(candidates) == 1
        assert candidates[0]["name"] == "revenue"
        assert candidates[0]["source_count"] == 2

    def test_same_alias_with_different_formulas_are_not_merged(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                "SELECT SUM(amount) AS revenue FROM orders",
                "SELECT COUNT(*) AS revenue FROM orders",
            ]
        )

        candidates = sorted(result.result["metric_candidates"], key=lambda item: item["expression"])
        assert len(candidates) == 2
        assert {candidate["expression"] for candidate in candidates} == {"COUNT(*)", "SUM(amount)"}
        assert all(candidate["name"] == "revenue" for candidate in candidates)
        assert all(candidate["source_count"] == 1 for candidate in candidates)

    def test_repeated_blocked_candidates_do_not_reappear_as_direct_candidates(self):
        tools = _make_tools()
        ranked_sql = """
            WITH f_data AS (
                SELECT
                    dt,
                    store_id,
                    module,
                    SUM(product_count) / SUM(non_prime_tc) AS sell_hitrate
                FROM store_daily
                GROUP BY dt, store_id, module
            ),
            rank_data AS (
                SELECT
                    f.*,
                    RANK() OVER (
                        PARTITION BY f.dt, f.module
                        ORDER BY f.sell_hitrate ASC
                    ) AS rank_no
                FROM f_data f
            )
            SELECT store_id, COUNT(*) AS time_count
            FROM rank_data
            WHERE rank_no <= 10
            GROUP BY store_id
        """
        result = tools.analyze_metric_candidates_from_history(sql_queries=[ranked_sql, ranked_sql])

        assert result.result["query_classification"] == "metric_plus_derived_datasource"
        assert result.result["metric_candidates"][0]["source_sql_name"] == "sql_1, sql_2"
        assert result.result["direct_metric_candidates"] == []
        assert len(result.result["blocked_direct_metric_candidates"]) == 2

    def test_invalid_sql_does_not_block_other_queries(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                "SELECT FROM",
                "SELECT SUM(amount) AS revenue FROM orders",
            ]
        )

        assert len(result.result["parse_errors"]) == 1
        assert result.result["metric_candidates"][0]["name"] == "revenue"

    def test_mysql_dialect_fallback_parses_backtick_aliases(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=["SELECT COUNT(DISTINCT `user_id`) AS `人数` FROM `orders`"]
        )

        assert result.result["parse_errors"] == []
        candidate = result.result["metric_candidates"][0]
        assert candidate["source_alias"] == "人数"
        assert candidate["requires_name_translation"] is True
        assert candidate["name_source"] == "expression_fallback"
        assert candidate["name"] == "count_distinct_user_id"

    def test_ranked_window_blocks_direct_metric_and_recommends_datasource(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                """
                WITH f_data AS (
                    SELECT
                        dt,
                        store_id,
                        module,
                        SUM(product_count) / SUM(non_prime_tc) AS sell_hitrate
                    FROM store_daily
                    GROUP BY dt, store_id, module
                ),
                rank_data AS (
                    SELECT
                        f.*,
                        RANK() OVER (
                            PARTITION BY f.dt, f.module
                            ORDER BY f.sell_hitrate ASC
                        ) AS rank_no
                    FROM f_data f
                    WHERE f.sell_hitrate > 0
                )
                SELECT store_id, COUNT(*) AS time_count
                FROM rank_data
                WHERE rank_no <= 10
                GROUP BY store_id
                HAVING COUNT(*) >= 10
                """
            ]
        )

        assert result.result["query_classification"] == "metric_plus_derived_datasource"
        assert result.result["direct_metric_candidates"] == []
        assert result.result["blocked_direct_metric_candidates"][0]["name"] == "time_count"
        assert result.result["metric_generation_skips"] == [
            {
                "source_sql_name": "sql_1",
                "reason": (
                    "rank/window TopN query returns row-level or post-window results; skip during metric generation"
                ),
                "sql_shape": "ranked_window",
                "window": {
                    "function": "RANK",
                    "partition_by": ["f.dt", "f.module"],
                    "order_by": [{"expr": "f.sell_hitrate", "direction": "ASC"}],
                },
                "rank_alias": "rank_no",
                "rank_filters": ["rank_no <= 10"],
            }
        ]
        recommendation = result.result["derived_datasource_recommendations"][0]
        assert recommendation["source_cte"] == "rank_data"
        assert recommendation["rank_alias"] == "rank_no"
        assert recommendation["window"]["function"] == "RANK"
        assert recommendation["window"]["partition_by"] == ["f.dt", "f.module"]
        assert recommendation["window"]["order_by"] == [{"expr": "f.sell_hitrate", "direction": "ASC"}]
        assert recommendation["ordering_metric_evidence"] == [
            {"name": "sell_hitrate", "expression": "SUM(product_count) / SUM(non_prime_tc)"}
        ]
        assert result.result["post_aggregation_constraints"] == [
            {
                "source_sql_name": "sql_1",
                "constraint": "COUNT(*) >= 10",
                "clause": "HAVING",
                "reason": "post-aggregation constraint must be preserved as a query filter or later derived data source",
            }
        ]

    def test_inline_ranked_subquery_blocks_direct_metric_and_recommends_datasource(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                """
                SELECT store_id, COUNT(*) AS time_count
                FROM (
                    SELECT
                        store_id,
                        amount,
                        ROW_NUMBER() OVER (
                            PARTITION BY store_id
                            ORDER BY amount DESC
                        ) AS rn
                    FROM orders
                ) ranked
                WHERE rn = 1
                GROUP BY store_id
                """
            ]
        )

        assert result.result["query_classification"] == "metric_plus_derived_datasource"
        assert result.result["direct_metric_candidates"] == []
        assert result.result["blocked_direct_metric_candidates"][0]["name"] == "time_count"
        recommendation = result.result["derived_datasource_recommendations"][0]
        assert recommendation["source_cte"] == "ranked"
        assert recommendation["rank_alias"] == "rn"
        assert recommendation["window"]["function"] == "ROW_NUMBER"
        assert recommendation["rank_filters"] == ["rn = 1"]

    def test_row_number_main_entity_distribution_recommends_datasource(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                """
                WITH user_total_playtime_per_mode AS (
                    SELECT vplayerid, modename, SUM(roundtime) AS total_playtime
                    FROM mode_roundrecord
                    GROUP BY vplayerid, modename
                ),
                user_main_mode AS (
                    SELECT
                        vplayerid,
                        modename,
                        ROW_NUMBER() OVER (
                            PARTITION BY vplayerid
                            ORDER BY total_playtime DESC
                        ) AS rn
                    FROM user_total_playtime_per_mode
                )
                SELECT modename AS `主玩玩法`, COUNT(*) AS `人数`
                FROM user_main_mode
                WHERE rn = 1
                GROUP BY modename
                """
            ]
        )

        assert result.result["query_classification"] == "metric_plus_derived_datasource"
        assert result.result["blocked_direct_metric_candidates"][0]["source_alias"] == "人数"
        assert result.result["blocked_direct_metric_candidates"][0]["requires_name_translation"] is True
        recommendation = result.result["derived_datasource_recommendations"][0]
        assert recommendation["source_cte"] == "user_main_mode"
        assert recommendation["rank_alias"] == "rn"
        assert recommendation["window"]["function"] == "ROW_NUMBER"
        assert recommendation["rank_filters"] == ["rn = 1"]

    def test_simple_aggregation_stays_direct_metric(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=["SELECT dt, SUM(amount) AS revenue FROM orders GROUP BY dt"]
        )

        assert result.result["query_classification"] == "direct_metric"
        assert result.result["derived_datasource_recommendations"] == []
        assert result.result["blocked_direct_metric_candidates"] == []
        assert result.result["direct_metric_candidates"][0]["name"] == "revenue"

    def test_literal_values_and_time_grain_are_reported_as_preservation_evidence(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                """
                SELECT
                    store_code,
                    CURDATE() AS part_dt,
                    COUNT(DISTINCT table_source) AS table_count,
                    MAX(CASE WHEN table_source = '7day_app_sale_rate_2_0_1' THEN 1 ELSE 0 END) AS seven_day_flag
                FROM (
                    SELECT
                        co_4 AS store_code,
                        '7day_app_sale_rate_2_0_1' AS table_source,
                        create_time
                    FROM app_event_source
                    WHERE DATE(create_time) = CURDATE()
                ) combined_data
                GROUP BY store_code
                """
            ]
        )

        assert {
            "source_sql_name": "sql_1",
            "alias": "table_source",
            "value": "7day_app_sale_rate_2_0_1",
            "expression": "'7day_app_sale_rate_2_0_1'",
            "projection": "'7day_app_sale_rate_2_0_1' AS table_source",
            "preservation_rule": "preserve literal values verbatim; only MetricFlow object names may be normalized",
        } in result.result["literal_mappings"]

        time_evidence = result.result["time_grain_evidence"]
        assert any(
            item["alias"] == "part_dt"
            and item["expression"] in {"CURRENT_DATE", "CURDATE()"}
            and item["evidence_type"] == "projected_time_dimension"
            for item in time_evidence
        )
        assert any(
            item["expression"] in {"CAST(create_time AS DATE)", "DATE(create_time)"}
            and item["evidence_type"] == "date_filter"
            and ("CURRENT_DATE" in item["predicate"] or "CURDATE()" in item["predicate"])
            for item in time_evidence
        )

    def test_date_trunc_time_grain_uses_projection_unit(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                """
                SELECT
                    DATE_TRUNC('month', created_at) AS month_dt,
                    SUM(amount) AS revenue
                FROM orders
                WHERE DATE_TRUNC('week', created_at) = DATE_TRUNC('week', CURRENT_DATE)
                GROUP BY DATE_TRUNC('month', created_at)
                """
            ]
        )

        time_evidence = result.result["time_grain_evidence"]
        assert any(
            item["alias"] == "month_dt"
            and item["evidence_type"] == "projected_time_dimension"
            and item["grain"] == "MONTH"
            for item in time_evidence
        )
        assert any(
            item["evidence_type"] == "date_filter"
            and item["grain"] == "WEEK"
            and "DATE_TRUNC('WEEK'" in item["expression"]
            for item in time_evidence
        )
