# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/func_tool/semantic_discovery_tools.py"""

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

    def test_reuses_cached_ddl_result(self):
        db_tool = _make_db_tool()
        db_tool.get_table_ddl.return_value = FuncToolResult(success=1, result={"definition": "CREATE TABLE t (id INT)"})
        tools = _make_tools(db_tool)

        first = tools.get_multiple_tables_ddl(["orders"])
        second = tools.get_multiple_tables_ddl(["orders"])

        assert first.result == second.result
        db_tool.get_table_ddl.assert_called_once()

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
        assert len(candidate["base_measures"]) == 2

    def test_derived_candidate_for_existing_metric_expression(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=["SELECT revenue / ad_spend AS roas FROM metric_table"],
            existing_metric_catalog_json=(
                '[{"name": "revenue", "type": "measure_proxy", "subject_path": "finance", '
                '"base_measures": ["total_revenue"], "dimensions": ["dt"], "entities": ["account_id"], '
                '"semantic_model": "orders"}, '
                '{"name": "ad_spend", "type": "measure_proxy", "subject_path": "finance", '
                '"base_measures": ["total_ad_spend"], "dimensions": ["dt"], "entities": ["account_id"], '
                '"semantic_model": "ads"}]'
            ),
        )

        candidate = result.result["metric_candidates"][0]
        assert candidate["name"] == "roas"
        assert candidate["metric_type"] == "derived"
        assert result.result["direct_metric_candidates"] == []
        assert result.result["derived_metric_candidates"] == [candidate]
        assert candidate["referenced_metrics"] == [
            {
                "name": "ad_spend",
                "type": "measure_proxy",
                "subject_path": "finance",
                "base_measures": ["total_ad_spend"],
                "dimensions": ["dt"],
                "entities": ["account_id"],
                "semantic_model": "ads",
            },
            {
                "name": "revenue",
                "type": "measure_proxy",
                "subject_path": "finance",
                "base_measures": ["total_revenue"],
                "dimensions": ["dt"],
                "entities": ["account_id"],
                "semantic_model": "orders",
            },
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

        candidate = result.result["metric_candidates"][0]
        assert candidate["name"] == "rolling_7d_revenue"
        assert candidate["metric_type"] == "cumulative"

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

    def test_same_formula_with_different_aliases_is_merged_with_alias_mapping(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                "SELECT COUNT(DISTINCT ip) AS new_and_ip_activity_count FROM activity WHERE ac_tag = '1,5'",
                "SELECT COUNT(DISTINCT ip) AS new_ip_activity_count FROM activity WHERE ac_tag = '1,5'",
            ]
        )

        candidates = result.result["metric_candidates"]
        assert len(candidates) == 1
        assert candidates[0]["name"] == "new_and_ip_activity_count"
        assert candidates[0]["source_count"] == 2
        assert candidates[0]["candidate_names"] == ["new_and_ip_activity_count", "new_ip_activity_count"]
        assert {
            "source_alias": "new_ip_activity_count",
            "candidate_name": "new_ip_activity_count",
            "canonical_name": "new_and_ip_activity_count",
            "source_sql_name": "sql_2",
        } in result.result["metric_aliases"]

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

    def test_low_information_ascii_alias_requires_business_name_translation(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=["SELECT COUNT(DISTINCT user_id) AS co_0 FROM orders"]
        )

        candidate = result.result["metric_candidates"][0]
        assert candidate["source_alias"] == "co_0"
        assert candidate["requires_name_translation"] is True
        assert candidate["name_translation_reason"] == "source alias looks like a generated short prefix plus ordinal"
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

    def test_having_blocks_direct_metric_and_recommends_post_aggregation_datasource(self):
        tools = _make_tools()
        result = tools.analyze_metric_candidates_from_history(
            sql_queries=[
                """
                SELECT store_id, SUM(amount) AS high_revenue
                FROM orders
                WHERE status = 'paid'
                GROUP BY store_id
                HAVING SUM(amount) >= 100
                """
            ]
        )

        assert result.result["query_classification"] == "metric_plus_derived_datasource"
        assert result.result["direct_metric_candidates"] == []
        assert result.result["blocked_direct_metric_candidates"][0]["name"] == "high_revenue"
        assert (
            result.result["blocked_direct_metric_candidates"][0]["block_reason"]
            == "post-aggregation filtering must be preserved in a derived data source before defining metrics"
        )
        recommendation = result.result["derived_datasource_recommendations"][0]
        assert recommendation["recommendation_type"] == "post_aggregation"
        assert recommendation["post_aggregation_constraints"] == ["SUM(amount) >= 100"]
        assert recommendation["generated_columns"] == ["store_id", "high_revenue"]
        assert result.result["modeling_plan"][0]["recommendation_type"] == "post_aggregation"
        assert {
            "source_sql_name": "sql_1",
            "clause": "WHERE",
            "value": "paid",
            "expression": "'paid'",
            "predicate": "status = 'paid'",
            "literal_type": "string",
            "preservation_rule": "preserve literal values verbatim; only MetricFlow object names may be normalized",
        } in result.result["literal_mappings"]
        assert {
            "source_sql_name": "sql_1",
            "clause": "HAVING",
            "value": "100",
            "expression": "100",
            "predicate": "SUM(amount) >= 100",
            "literal_type": "numeric",
            "preservation_rule": "preserve literal values verbatim; only MetricFlow object names may be normalized",
        } in result.result["literal_mappings"]

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
        literal_values = {item.get("value") for item in result.result["literal_mappings"]}
        assert "MONTH" not in literal_values
        assert "WEEK" not in literal_values
