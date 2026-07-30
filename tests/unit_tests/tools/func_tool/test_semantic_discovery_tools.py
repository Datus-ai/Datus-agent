# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/func_tool/semantic_discovery_tools.py"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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


def _make_tools(
    db_tool: MagicMock | None = None,
    enable_semantic_model_profiler: bool = False,
    source_sql_provider=None,
) -> SemanticDiscoveryTools:
    if db_tool is None:
        db_tool = _make_db_tool()
    return SemanticDiscoveryTools(
        db_tool=db_tool,
        enable_semantic_model_profiler=enable_semantic_model_profiler,
        source_sql_provider=source_sql_provider,
    )


# ---------------------------------------------------------------------------
# Batched semantic source discovery
# ---------------------------------------------------------------------------


class TestInspectSemanticSources:
    def test_combines_schema_relationship_and_request_sql_usage_in_one_call(self):
        db_tool = _make_db_tool()

        def ddl(table, *_args):
            definitions = {
                "orders": (
                    "CREATE TABLE orders (id INT, customer_id INT, amount DECIMAL, "
                    "FOREIGN KEY (customer_id) REFERENCES customers(id))"
                ),
                "customers": "CREATE TABLE customers (id INT, region VARCHAR)",
            }
            return FuncToolResult(success=1, result={"definition": definitions[table]})

        def schema(table, *_args):
            columns = {
                "orders": [
                    {"name": "id", "type": "INT"},
                    {"name": "customer_id", "type": "INT"},
                    {"name": "amount", "type": "DECIMAL"},
                ],
                "customers": [
                    {"name": "id", "type": "INT"},
                    {"name": "region", "type": "VARCHAR"},
                ],
            }
            return FuncToolResult(success=1, result={"columns": columns[table]})

        db_tool.get_table_ddl.side_effect = ddl
        db_tool.describe_table.side_effect = schema
        tools = _make_tools(
            db_tool,
            source_sql_provider=lambda: [
                {
                    "name": "revenue_by_region",
                    "sql": (
                        "SELECT c.region, SUM(o.amount) AS revenue "
                        "FROM orders o JOIN customers c ON o.customer_id = c.id "
                        "GROUP BY c.region"
                    ),
                }
            ],
        )

        result = tools.inspect_semantic_sources(["orders", "customers"])

        assert result.success == 1
        assert [table["table_name"] for table in result.result["tables"]] == ["orders", "customers"]
        assert result.result["source_sql_count"] == 1
        assert len(result.result["relationships"]) == 1
        assert result.result["relationships"][0]["evidence"] == "foreign_key"
        orders = result.result["tables"][0]
        customers = result.result["tables"][1]
        assert orders["sql_usage"]["field_usage_statistics"]["amount"]["aggregate_count"] == 1
        assert customers["sql_usage"]["field_usage_statistics"]["region"]["group_by_count"] == 1
        assert db_tool.get_table_ddl.call_count == 2
        assert db_tool.describe_table.call_count == 2

    def test_reports_partial_table_inspection_without_repeating_calls(self):
        db_tool = _make_db_tool()
        db_tool.get_table_ddl.return_value = FuncToolResult(success=0, error="DDL unavailable")
        db_tool.describe_table.return_value = FuncToolResult(
            success=1,
            result={"columns": [{"name": "id", "type": "INT"}]},
        )

        result = _make_tools(db_tool).inspect_semantic_sources(["orders", "orders"])

        assert result.success == 1
        assert len(result.result["tables"]) == 1
        assert result.result["tables"][0]["ddl_error"] == "DDL unavailable"
        db_tool.get_table_ddl.assert_called_once()
        db_tool.describe_table.assert_called_once()

    def test_rejects_empty_table_scope(self):
        result = _make_tools().inspect_semantic_sources([])

        assert result.success == 0
        assert "at least one" in result.error


class TestValidateSemanticKeyCandidatesBatch:
    def test_verifies_multiple_candidates_in_one_tool_call(self):
        db_tool = _make_db_tool()
        db_tool.read_query.side_effect = [
            FuncToolResult(success=1, result={"compressed_data": "row_count,null_key_rows\n12,0\n"}),
            FuncToolResult(
                success=1,
                result={"compressed_data": "duplicate_group_count,duplicate_row_count\n0,0\n"},
            ),
            FuncToolResult(success=1, result={"compressed_data": "row_count,null_key_rows\n8,1\n"}),
            FuncToolResult(
                success=1,
                result={"compressed_data": "duplicate_group_count,duplicate_row_count\n0,0\n"},
            ),
        ]

        result = _make_tools(db_tool).validate_semantic_key_candidates(
            [
                {"table_name": "customers", "columns": ["customer_id"]},
                {"table_name": "stores", "columns": ["tenant_id", "store_id"]},
            ]
        )

        assert result.success == 1
        assert len(result.result["validations"]) == 2
        assert result.result["validations"][0]["is_valid_logical_key"] is True
        assert result.result["validations"][1]["is_valid_logical_key"] is False
        assert db_tool.read_query.call_count == 4

    def test_deduplicates_identical_candidates(self):
        db_tool = _make_db_tool()
        db_tool.read_query.side_effect = [
            FuncToolResult(success=1, result={"compressed_data": "row_count,null_key_rows\n12,0\n"}),
            FuncToolResult(
                success=1,
                result={"compressed_data": "duplicate_group_count,duplicate_row_count\n0,0\n"},
            ),
        ]

        result = _make_tools(db_tool).validate_semantic_key_candidates(
            [
                {"table_name": "customers", "columns": ["customer_id"]},
                {"table_name": "CUSTOMERS", "columns": ["CUSTOMER_ID"]},
            ]
        )

        assert result.success == 1
        assert len(result.result["validations"]) == 1
        assert db_tool.read_query.call_count == 2

    def test_requires_at_least_one_candidate(self):
        result = _make_tools().validate_semantic_key_candidates([])

        assert result.success == 0
        assert "Provide every" in result.error


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
# validate_semantic_key_candidate
# ---------------------------------------------------------------------------


class TestValidateSemanticKeyCandidate:
    def test_accepts_full_table_non_null_unique_composite_key(self):
        db_tool = _make_db_tool()
        db_tool.read_query.side_effect = [
            FuncToolResult(
                success=1,
                result={"compressed_data": "index,row_count,null_key_rows\n0,12,0\n"},
            ),
            FuncToolResult(
                success=1,
                result={"compressed_data": ("index,duplicate_group_count,duplicate_row_count\n0,0,0\n")},
            ),
        ]
        tools = _make_tools(db_tool)

        result = tools.validate_semantic_key_candidate(
            "customers",
            ["tenant_id", "customer_id"],
            schema_name="analytics",
        )

        assert result.success == 1
        assert result.result["is_valid_logical_key"] is True
        assert result.result["recommended_osi_declaration"] == "unique_keys"
        assert result.result["primary_key_inferred"] is False
        assert result.result["verification_scope"] == "full_table"
        assert "GROUP BY tenant_id, customer_id" in db_tool.read_query.call_args_list[1].args[0]
        assert "FROM analytics.customers" in db_tool.read_query.call_args_list[0].args[0]

    def test_accepts_case_insensitive_profile_column_names(self):
        db_tool = _make_db_tool()
        db_tool.read_query.side_effect = [
            FuncToolResult(
                success=1,
                result={"compressed_data": "index,ROW_COUNT,NULL_KEY_ROWS\n0,12,0\n"},
            ),
            FuncToolResult(
                success=1,
                result={"compressed_data": ("index,DUPLICATE_GROUP_COUNT,DUPLICATE_ROW_COUNT\n0,0,0\n")},
            ),
        ]

        result = _make_tools(db_tool).validate_semantic_key_candidate("customers", ["tenant_id", "customer_id"])

        assert result.success == 1
        assert result.result["is_valid_logical_key"] is True

    def test_rejects_candidate_with_nulls_or_duplicates(self):
        db_tool = _make_db_tool()
        db_tool.read_query.side_effect = [
            FuncToolResult(
                success=1,
                result={"compressed_data": "index,row_count,null_key_rows\n0,20,2\n"},
            ),
            FuncToolResult(
                success=1,
                result={"compressed_data": ("index,duplicate_group_count,duplicate_row_count\n0,3,4\n")},
            ),
        ]
        result = _make_tools(db_tool).validate_semantic_key_candidate("customers", ["tenant_id", "customer_id"])

        assert result.success == 1
        assert result.result["is_non_null"] is False
        assert result.result["is_unique"] is False
        assert result.result["is_valid_logical_key"] is False
        assert result.result["recommended_osi_declaration"] == "none"

    def test_empty_table_is_not_supporting_key_evidence(self):
        db_tool = _make_db_tool()
        db_tool.read_query.side_effect = [
            FuncToolResult(
                success=1,
                result={"compressed_data": "index,row_count,null_key_rows\n0,0,\n"},
            ),
            FuncToolResult(
                success=1,
                result={"compressed_data": ("index,duplicate_group_count,duplicate_row_count\n0,0,0\n")},
            ),
        ]
        result = _make_tools(db_tool).validate_semantic_key_candidate("customers", ["customer_id"])

        assert result.success == 1
        assert result.result["is_valid_logical_key"] is False
        assert "empty" in result.result["reason"]

    def test_query_failure_is_not_reported_as_verification(self):
        db_tool = _make_db_tool()
        db_tool.read_query.return_value = FuncToolResult(success=0, error="permission denied")
        result = _make_tools(db_tool).validate_semantic_key_candidate("customers", ["customer_id"])

        assert result.success == 0
        assert "permission denied" in result.error

    def test_rejects_empty_or_duplicate_column_list(self):
        tools = _make_tools()
        empty = tools.validate_semantic_key_candidate("customers", [])
        duplicate = tools.validate_semantic_key_candidate("customers", ["customer_id", "CUSTOMER_ID"])

        assert empty.success == 0
        assert duplicate.success == 0


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

    def test_extracts_composite_foreign_key_as_one_ordered_relationship(self):
        ddl = """CREATE TABLE order_items (
            tenant_id INT,
            order_id INT,
            FOREIGN KEY (tenant_id, order_id)
              REFERENCES orders(tenant_id, id)
        )"""
        db_tool = _make_db_tool()
        db_tool.get_table_ddl.return_value = FuncToolResult(success=1, result={"definition": ddl})

        result = _make_tools(db_tool)._extract_foreign_keys_from_ddl(["order_items"], "", "", "")

        assert len(result) == 1
        assert result[0]["source_columns"] == ["tenant_id", "order_id"]
        assert result[0]["target_columns"] == ["tenant_id", "id"]
        assert result[0]["key_arity"] == 2
        assert result[0]["target_key_status"] == "declared"


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
                "source_columns": ["customer_id"],
                "target_table": "customers",
                "target_column": "id",
                "target_columns": ["id"],
                "key_arity": 1,
                "confidence": "medium",
                "evidence": "join_pattern",
                "target_key_status": "candidate_unverified",
                "requires_target_key_validation": True,
            }
        ]

    def test_groups_one_join_clause_into_composite_relationship(self):
        tools = _make_tools()
        result = tools._extract_join_relationships_from_sql(
            """
            SELECT *
            FROM orders o
            JOIN customers c
              ON o.tenant_id = c.tenant_id
             AND o.customer_id = c.id
            """,
            {"orders": "orders", "customers": "customers"},
        )

        assert len(result) == 1
        assert result[0]["source_columns"] == ["tenant_id", "customer_id"]
        assert result[0]["target_columns"] == ["tenant_id", "id"]
        assert result[0]["key_arity"] == 2
        assert result[0]["target_key_status"] == "candidate_unverified"

    def test_groups_comma_join_predicates_into_composite_relationship(self):
        result = _make_tools()._extract_join_relationships_from_sql(
            """
            SELECT *
            FROM orders o, customers c
            WHERE o.tenant_id = c.tenant_id
              AND o.customer_id = c.id
            """,
            {"orders": "orders", "customers": "customers"},
        )

        assert len(result) == 1
        assert result[0]["source_columns"] == ["tenant_id", "customer_id"]
        assert result[0]["target_columns"] == ["tenant_id", "id"]

    def test_merges_on_and_where_join_predicates(self):
        result = _make_tools()._extract_join_relationships_from_sql(
            """
            SELECT *
            FROM orders o
            JOIN customers c ON o.tenant_id = c.tenant_id
            WHERE o.customer_id = c.id
              AND o.status = 'paid'
            """,
            {"orders": "orders", "customers": "customers"},
        )

        assert len(result) == 1
        assert result[0]["source_columns"] == ["tenant_id", "customer_id"]
        assert result[0]["target_columns"] == ["tenant_id", "id"]

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
        sql_entries = [{"sql": "SELECT * FROM orders WHERE CUSTOM_MATCH(tags, 'vip')"}]
        mock_rag = MagicMock()
        mock_rag.search_reference_sql.return_value = sql_entries
        with patch("datus.storage.reference_sql.store.ReferenceSqlRAG", return_value=mock_rag):
            tools = _make_tools(db_tool)
            result = tools.analyze_column_usage_patterns("orders", columns=["tags"])
        assert result.success == 1
        assert "tags" in result.result["column_patterns"]
        assert "CUSTOM_MATCH" in result.result["column_patterns"]["tags"]["functions"]
        assert "Function predicates: CUSTOM_MATCH" in result.result["column_patterns"]["tags"]["usage_description"]

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
# profile_semantic_model_evidence
# ---------------------------------------------------------------------------


class TestProfileSemanticModelEvidence:
    def test_sql_only_mines_fields_filters_aggregates_and_joins(self):
        tools = _make_tools()
        result = tools.profile_semantic_model_evidence(
            sql_queries=[
                """
                SELECT c.region, SUM(o.amount) AS revenue
                FROM orders o
                JOIN customers c ON o.customer_id = c.id
                WHERE o.status = 'paid'
                GROUP BY c.region
                """
            ],
            profile_mode="sql_only",
        )

        assert result.success == 1
        assert result.result["data_profiled"] is False
        tables = result.result["tables"]
        assert set(tables) == {"orders", "customers"}
        assert tables["orders"]["field_usage_statistics"]["status"]["operators"] == ["="]
        assert tables["orders"]["common_filter_conditions"][0]["condition"] == "o.status = '<REDACTED>'"
        filter_template = tables["orders"]["common_business_filter_templates"][0]
        assert filter_template["condition_template"] == "o.status = '<REDACTED>'"
        assert filter_template["fields"] == ["status"]
        assert filter_template["literal_values"] == ["paid"]
        assert filter_template["usage_kind"] == "categorical_filter"
        assert tables["orders"]["aggregate_expressions"][0]["expression"] == "SUM(o.amount)"
        assert tables["customers"]["group_by_expressions"][0]["expression"] == "c.region"
        assert tables["orders"]["join_relationships"][0]["evidence"] == "historical_sql_join"
        assert "compact distribution notes" in result.result["yaml_guidance"]

    def test_sql_only_keeps_composite_join_components_together(self):
        result = _make_tools().profile_semantic_model_evidence(
            sql_queries=[
                """
                SELECT SUM(o.amount) AS revenue
                FROM orders o
                JOIN customers c
                  ON o.tenant_id = c.tenant_id
                 AND o.customer_id = c.id
                """
            ],
            profile_mode="sql_only",
        )

        assert result.success == 1
        relationship = result.result["tables"]["orders"]["join_relationships"][0]
        assert relationship["source_columns"] == ["tenant_id", "customer_id"]
        assert relationship["target_columns"] == ["tenant_id", "id"]
        assert relationship["key_arity"] == 2
        assert relationship["target_key_status"] == "candidate_unverified"

    def test_sql_only_groups_comma_join_components(self):
        result = _make_tools().profile_semantic_model_evidence(
            sql_queries=[
                """
                SELECT SUM(o.amount) AS revenue
                FROM orders o, customers c
                WHERE o.tenant_id = c.tenant_id
                  AND o.customer_id = c.id
                """
            ],
            profile_mode="sql_only",
        )

        assert result.success == 1
        relationship = result.result["tables"]["orders"]["join_relationships"][0]
        assert relationship["source_columns"] == ["tenant_id", "customer_id"]
        assert relationship["target_columns"] == ["tenant_id", "id"]

    def test_lightweight_profiles_used_columns(self):
        db_tool = _make_db_tool()
        db_tool.describe_table.return_value = FuncToolResult(
            success=1,
            result={
                "columns": [
                    {"name": "status", "type": "VARCHAR"},
                    {"name": "amount", "type": "DECIMAL"},
                ]
            },
        )
        db_tool.read_query.side_effect = [
            FuncToolResult(success=1, result={"compressed_data": "index,row_count\n0,10\n"}),
            FuncToolResult(
                success=1,
                result={"compressed_data": "index,row_count,non_null_count,distinct_count\n0,10,9,2\n"},
            ),
            FuncToolResult(success=1, result={"compressed_data": "index,value,count\n0,paid,7\n1,refund,2\n"}),
            FuncToolResult(
                success=1,
                result={
                    "compressed_data": "index,row_count,non_null_count,distinct_count,min_value,max_value\n0,10,10,8,1,99\n"
                },
            ),
            FuncToolResult(
                success=1,
                result={"compressed_data": "index,p25,p50,p75,p90,p95\n0,10,50,75,90,95\n"},
            ),
        ]
        tools = _make_tools(db_tool)

        result = tools.profile_semantic_model_evidence(
            sql_queries=["SELECT SUM(amount) AS revenue FROM orders WHERE status = 'paid'"],
            profile_mode="lightweight",
            max_columns_per_table=2,
        )

        assert result.success == 1
        assert result.result["data_profiled"] is True
        profile = result.result["tables"]["orders"]["data_distribution_profile"]
        assert profile["row_count"] == 10
        assert profile["columns"]["status"]["stats"]["null_rate"] == 0.1
        assert profile["columns"]["status"]["top_values"][0] == {"value": "paid", "count": 7}
        assert profile["columns"]["amount"]["stats"]["min_value"] == 1
        assert profile["columns"]["amount"]["stats"]["max_value"] == 99
        assert profile["columns"]["amount"]["percentiles"]["p50"] == 50

    def test_top_values_profile_skips_error_rows(self):
        db_tool = _make_db_tool()
        db_tool.read_query.side_effect = [
            FuncToolResult(
                success=1, result={"compressed_data": "index,row_count,non_null_count,distinct_count\n0,10,9,2\n"}
            ),
            FuncToolResult(success=0, result=[{"error": "top values failed"}]),
        ]
        tools = _make_tools(db_tool)

        profile = tools._profile_single_column(
            table_ref="orders",
            column_name="status",
            column_type="VARCHAR",
            kind="categorical",
            database="",
            top_n=2,
        )

        assert "top_values_sql" in profile
        assert "top_values" not in profile

    def test_deep_profiles_explicit_table_without_sql_evidence(self):
        db_tool = _make_db_tool()
        db_tool.describe_table.return_value = FuncToolResult(
            success=1,
            result={"columns": [{"name": "amount", "type": "DECIMAL"}]},
        )
        db_tool.read_query.side_effect = [
            FuncToolResult(success=1, result={"compressed_data": "index,row_count\n0,10\n"}),
            FuncToolResult(
                success=1,
                result={
                    "compressed_data": "index,row_count,non_null_count,distinct_count,min_value,max_value\n0,10,10,8,1,99\n"
                },
            ),
            FuncToolResult(
                success=1,
                result={"compressed_data": "index,p25,p50,p75,p90,p95\n0,10,50,75,90,95\n"},
            ),
        ]
        tools = _make_tools(db_tool)

        result = tools.profile_semantic_model_evidence(
            tables=["orders"],
            profile_mode="deep",
            max_columns_per_table=1,
        )

        assert result.success == 1
        assert result.result["tables"]["orders"]["query_count"] == 0
        profile = result.result["tables"]["orders"]["data_distribution_profile"]
        assert profile["columns"]["amount"]["stats"]["max_value"] == 99
        assert profile["columns"]["amount"]["percentiles"]["p90"] == 90

    def test_deep_profiles_temporal_span_and_duration_pairs(self):
        db_tool = _make_db_tool()
        db_tool.describe_table.return_value = FuncToolResult(
            success=1,
            result={
                "columns": [
                    {"name": "opened_at", "type": "DATE"},
                    {"name": "closed_at", "type": "DATE"},
                ]
            },
        )
        db_tool.read_query.side_effect = [
            FuncToolResult(success=1, result={"compressed_data": "index,row_count\n0,3\n"}),
            FuncToolResult(
                success=1,
                result={
                    "compressed_data": (
                        "index,row_count,non_null_count,distinct_count,min_value,max_value\n"
                        "0,3,3,3,2025-01-01,2025-01-05\n"
                    )
                },
            ),
            FuncToolResult(
                success=1,
                result={
                    "compressed_data": (
                        "index,row_count,non_null_count,distinct_count,min_value,max_value\n"
                        "0,3,3,3,2025-01-03,2025-01-10\n"
                    )
                },
            ),
            FuncToolResult(
                success=1,
                result={
                    "compressed_data": (
                        "index,left_value,right_value\n"
                        "0,2025-01-01,2025-01-03\n"
                        "1,2025-01-02,2025-01-05\n"
                        "2,2025-01-05,2025-01-10\n"
                    )
                },
            ),
        ]
        tools = _make_tools(db_tool)

        result = tools.profile_semantic_model_evidence(
            tables=["events"],
            profile_mode="deep",
            max_columns_per_table=2,
        )

        assert result.success == 1
        profile = result.result["tables"]["events"]["data_distribution_profile"]
        assert profile["columns"]["opened_at"]["temporal_summary"]["span_days"] == 4
        duration = profile["date_duration_profiles"][0]
        assert duration["candidate_reason"] == "shared_stem_boundary_tokens"
        assert duration["left_column"] == "opened_at"
        assert duration["right_column"] == "closed_at"
        assert duration["delta_days"] == {"min": 2, "p50": 3, "p90": 5, "max": 5}

    def test_join_relationship_profile_reports_coverage_and_fanout_generically(self):
        db_tool = _make_db_tool()
        db_tool.read_query.return_value = FuncToolResult(
            success=1,
            result={
                "compressed_data": (
                    "index,source_rows,non_null_source_rows,distinct_source_keys,"
                    "matched_join_rows,matched_distinct_source_keys\n"
                    "0,10,9,3,8,2\n"
                )
            },
        )
        tools = _make_tools(db_tool)

        profiles = tools._profile_join_relationship_profiles(
            relationships=[
                {
                    "source_table": "events",
                    "source_column": "user_id",
                    "target_table": "users",
                    "target_column": "id",
                }
            ],
            catalog="",
            database="",
            schema_name="",
        )

        assert profiles[0]["referential_coverage"] == 0.666667
        assert profiles[0]["join_fanout_ratio"] == 0.888889
        assert profiles[0]["join_cardinality_hint"] == "many_to_one_or_one_to_one"
        assert "matched_row_ratio" not in profiles[0]


# ---------------------------------------------------------------------------
# Internal metric-candidate analyzer
# ---------------------------------------------------------------------------


class TestMetricCandidateAnalyzer:
    def test_parser_tries_configured_datasource_dialect_first(self, monkeypatch):
        import sqlglot

        calls = []
        original_parse = sqlglot.parse

        def record_parse(sql, read=None):
            calls.append(read)
            return original_parse(sql)

        monkeypatch.setattr(sqlglot, "parse", record_parse)
        tools = SemanticDiscoveryTools(
            agent_config=SimpleNamespace(
                current_datasource="analytics",
                current_db_config=lambda _name: SimpleNamespace(type="starrocks"),
            )
        )

        tools._parse_sql("SELECT 1")

        assert calls == ["starrocks"]

    def test_parser_uses_datasource_type_instead_of_connection_name(self, monkeypatch):
        import sqlglot

        calls = []
        original_parse = sqlglot.parse

        def record_parse(sql, read=None):
            calls.append(read)
            return original_parse(sql)

        monkeypatch.setattr(sqlglot, "parse", record_parse)
        tools = SemanticDiscoveryTools(
            agent_config=SimpleNamespace(
                current_datasource="warehouse_prod",
                current_db_config=lambda _name: SimpleNamespace(type="mysql"),
            )
        )

        tools._parse_sql("SELECT 1")

        assert calls == ["mysql"]

    def test_parser_uses_adapter_registered_parser_dialect(self, monkeypatch):
        import sqlglot

        from datus.tools.db_tools import connector_registry

        calls = []
        original_parse = sqlglot.parse

        def record_parse(sql, read=None):
            calls.append(read)
            return original_parse(sql)

        monkeypatch.setattr(sqlglot, "parse", record_parse)
        monkeypatch.setattr(
            connector_registry,
            "get_parser_dialect",
            lambda dialect: "postgres" if dialect == "hologres" else None,
            raising=False,
        )
        tools = SemanticDiscoveryTools(
            agent_config=SimpleNamespace(
                current_datasource="analytics",
                current_db_config=lambda _name: SimpleNamespace(type="hologres"),
            )
        )

        tools._parse_sql("SELECT 1")

        assert calls == ["postgres"]

    def test_available_tools_without_db_is_empty(self):
        tools = SemanticDiscoveryTools(
            db_tool=None,
            agent_config=MagicMock(),
            sub_agent_name="gen_metrics",
        )

        tool_names = {tool.name for tool in tools.available_tools()}

        assert tool_names == set()

    def test_available_tools_without_db_omits_enabled_profiler(self):
        tools = SemanticDiscoveryTools(
            db_tool=None,
            enable_semantic_model_profiler=True,
        )

        tool_names = {tool.name for tool in tools.available_tools()}

        assert tool_names == set()

    def test_history_analyzer_accepts_direct_sql_without_db(self):
        tools = SemanticDiscoveryTools(db_tool=None)

        result = tools._analyze_metric_candidates(sql_queries=["SELECT SUM(amount) AS revenue FROM orders"])

        assert result.success == 1
        assert [candidate["name"] for candidate in result.result["metric_candidates"]] == ["revenue"]

    def test_available_tools_omits_internal_metric_candidate_analyzer(self):
        tools = _make_tools()
        tool_names = {tool.name for tool in tools.available_tools()}
        assert tool_names == {
            "inspect_semantic_sources",
            "validate_semantic_key_candidates",
        }
        assert "_analyze_metric_candidates" not in tool_names
        assert "profile_semantic_model_evidence" not in tool_names

    def test_available_tools_includes_profiler_when_enabled(self):
        tools = _make_tools(enable_semantic_model_profiler=True)
        tool_names = {tool.name for tool in tools.available_tools()}
        assert "profile_semantic_model_evidence" in tool_names

    def test_ratio_candidate_preserves_base_measures(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
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
        result = tools._analyze_metric_candidates(
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
        result = tools._analyze_metric_candidates(
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
        result = tools._analyze_metric_candidates(
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

            result = tools._analyze_metric_candidates(query_text="orders")

        assert result.success == 1
        assert {candidate["name"] for candidate in result.result["metric_candidates"]} == {"revenue", "cost"}

    def test_cumulative_candidate_for_window_expression(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
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
        result = tools._analyze_metric_candidates(
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

    def test_lag_period_aggregation_becomes_fixed_period_metrics(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
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
        direct = {candidate["name"]: candidate for candidate in result.result["direct_metric_candidates"]}
        assert sorted(direct) == ["order_count", "order_count_period_delta", "order_count_previous_period"]
        assert result.result["derived_metric_candidates"] == []

        previous = direct["order_count_previous_period"]
        assert previous["metric_type"] == "period_over_period"
        assert previous["expression"] == "COUNT(DISTINCT order_id)"
        assert previous["time_dimension"] == "order_date"
        assert previous["period_over_period"] == {
            "time_grain": "month",
            "offset_window": "1 month",
            "calculation": "previous_value",
        }
        assert "inputs" not in previous

        delta = direct["order_count_period_delta"]
        assert delta["metric_type"] == "period_over_period"
        assert delta["expression"] == "COUNT(DISTINCT order_id)"
        assert delta["source_expression"] == "order_count - order_count_previous_period"
        assert delta["time_dimension"] == "order_date"
        assert delta["period_over_period"] == {
            "time_grain": "month",
            "offset_window": "1 month",
            "calculation": "delta",
        }
        assert "inputs" not in delta

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
        result = tools._analyze_metric_candidates(
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
        direct = {candidate["name"]: candidate for candidate in result.result["direct_metric_candidates"]}
        assert sorted(direct) == ["order_count_mom_delta", "previous_month_order_count"]
        assert result.result["derived_metric_candidates"] == []

        previous = direct["previous_month_order_count"]
        assert previous["metric_type"] == "period_over_period"
        assert previous["expression"] == "COUNT(DISTINCT order_id)"
        assert previous["period_over_period"] == {
            "time_grain": "month",
            "offset_window": "1 month",
            "calculation": "previous_value",
        }
        assert "inputs" not in previous

        delta = direct["order_count_mom_delta"]
        assert delta["metric_type"] == "period_over_period"
        assert delta["expression"] == "COUNT(DISTINCT order_id)"
        assert delta["period_over_period"] == {
            "time_grain": "month",
            "offset_window": "1 month",
            "calculation": "delta",
        }
        assert "inputs" not in delta

    def test_inline_lag_metric_math_uses_source_time_grain_context(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
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
            for candidate in result.result["direct_metric_candidates"]
            if candidate["name"] == "order_count_period_delta"
        ]
        assert len(matches) == 1
        delta = matches[0]
        assert delta["metric_type"] == "period_over_period"
        assert delta["expression"] == "COUNT(DISTINCT order_id)"
        assert delta["source_expression"] == "order_count - order_count_prev"
        assert delta["period_over_period"] == {
            "time_grain": "month",
            "offset_window": "1 month",
            "calculation": "delta",
        }
        assert "inputs" not in delta

    def test_inline_lag_aliases_do_not_leak_to_later_plain_columns(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
            sql_queries=[
                """
                WITH monthly_orders AS (
                    SELECT
                        DATE_TRUNC('month', order_date) AS metric_month,
                        COUNT(DISTINCT order_id) AS order_count,
                        SUM(previous_count) AS order_count_prev
                    FROM fact_orders
                    GROUP BY DATE_TRUNC('month', order_date)
                )
                SELECT
                    metric_month,
                    order_count - LAG(order_count) OVER (ORDER BY metric_month) AS order_count_period_delta,
                    order_count_prev
                FROM monthly_orders
                ORDER BY metric_month
                """
            ]
        )

        assert result.success == 1
        direct = result.result["direct_metric_candidates"]
        delta = next(candidate for candidate in direct if candidate["name"] == "order_count_period_delta")
        assert delta["metric_type"] == "period_over_period"
        assert delta["period_over_period"] == {
            "time_grain": "month",
            "offset_window": "1 month",
            "calculation": "delta",
        }
        leaked = [
            candidate
            for candidate in direct
            if candidate["name"] == "order_count_prev" and candidate.get("metric_type") == "period_over_period"
        ]
        assert leaked == []

    def test_lag_percent_change_becomes_fixed_period_metric(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
            sql_queries=[
                """
                WITH monthly_orders AS (
                    SELECT
                        DATE_TRUNC('month', order_date) AS metric_month,
                        COUNT(DISTINCT order_id) AS order_count
                    FROM fact_orders
                    GROUP BY DATE_TRUNC('month', order_date)
                ),
                compared AS (
                    SELECT
                        metric_month,
                        order_count,
                        LAG(order_count) OVER (ORDER BY metric_month) AS previous_month_order_count
                    FROM monthly_orders
                )
                SELECT
                    metric_month,
                    order_count,
                    (order_count - previous_month_order_count) * 1.0
                        / NULLIF(previous_month_order_count, 0) AS order_count_mom_percent_change
                FROM compared
                ORDER BY metric_month
                """
            ]
        )

        assert result.success == 1
        percent_change = next(
            candidate
            for candidate in result.result["direct_metric_candidates"]
            if candidate["name"] == "order_count_mom_percent_change"
        )
        assert percent_change["metric_type"] == "period_over_period"
        assert percent_change["expression"] == "COUNT(DISTINCT order_id)"
        assert percent_change["period_over_period"] == {
            "time_grain": "month",
            "offset_window": "1 month",
            "calculation": "percent_change",
        }
        assert "inputs" not in percent_change

    def test_lag_ratio_becomes_fixed_period_metric(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
            sql_queries=[
                """
                WITH monthly_orders AS (
                    SELECT
                        DATE_TRUNC('month', order_date) AS metric_month,
                        COUNT(DISTINCT order_id) AS order_count
                    FROM fact_orders
                    GROUP BY DATE_TRUNC('month', order_date)
                ),
                compared AS (
                    SELECT
                        metric_month,
                        order_count,
                        LAG(order_count) OVER (ORDER BY metric_month) AS previous_month_order_count
                    FROM monthly_orders
                )
                SELECT
                    metric_month,
                    order_count,
                    order_count * 1.0 / NULLIF(previous_month_order_count, 0) AS order_count_mom_ratio
                FROM compared
                ORDER BY metric_month
                """
            ]
        )

        assert result.success == 1
        ratio = next(
            candidate
            for candidate in result.result["direct_metric_candidates"]
            if candidate["name"] == "order_count_mom_ratio"
        )
        assert ratio["metric_type"] == "period_over_period"
        assert ratio["expression"] == "COUNT(DISTINCT order_id)"
        assert ratio["period_over_period"] == {
            "time_grain": "month",
            "offset_window": "1 month",
            "calculation": "ratio",
        }
        assert "inputs" not in ratio

    def test_lag_with_explicit_offset_count_sets_offset_window(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
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
                    LAG(order_count, 2) OVER (ORDER BY metric_month) AS order_count_two_months_ago
                FROM monthly_orders
                ORDER BY metric_month
                """
            ]
        )

        assert result.success == 1
        previous = next(
            candidate
            for candidate in result.result["direct_metric_candidates"]
            if candidate["name"] == "order_count_two_months_ago"
        )
        assert previous["metric_type"] == "period_over_period"
        assert previous["period_over_period"] == {
            "time_grain": "month",
            "offset_window": "2 months",
            "calculation": "previous_value",
        }
        assert "inputs" not in previous

    def test_period_shift_aliases_are_scoped_to_source_select(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
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
            for candidate in result.result["direct_metric_candidates"]
            if candidate["name"] == "order_count_period_delta"
        )
        assert delta["metric_type"] == "period_over_period"
        assert delta["period_over_period"] == {
            "time_grain": "month",
            "offset_window": "1 month",
            "calculation": "delta",
        }
        assert "inputs" not in delta

    def test_conditional_aggregation_keeps_case_measure_evidence(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
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
        result = tools._analyze_metric_candidates(
            sql_queries=["SELECT * FROM users WHERE is_test = 0 AND country = 'US'"]
        )

        assert result.result["metric_candidates"] == []
        evidence = result.result["non_metric_evidence"][0]
        assert evidence["tables"] == ["users"]
        assert evidence["filters"] == ["is_test = 0 AND country = 'US'"]

    def test_raw_ratio_with_rate_context_becomes_llm_review_candidate(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
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
        result = tools._analyze_metric_candidates(
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
        result = tools._analyze_metric_candidates(
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
        result = tools._analyze_metric_candidates(
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
        result = tools._analyze_metric_candidates(
            sql_queries=["SELECT ROUND(CAST(a / NULLIF(b, 0) AS DOUBLE), 2) AS ratio_value FROM t"]
        )

        candidate = result.result["llm_review_candidates"][0]
        assert candidate["expression_kind"] == "row_ratio_expr"
        measures_by_role = {measure["role"]: measure for measure in candidate["base_measures"]}
        assert measures_by_role["numerator"]["expr"] == "a"
        assert measures_by_role["denominator"]["expr"] == "b"

    def test_count_star_with_distinct_business_count_is_support_measure(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
            sql_queries=[
                """
                SELECT COUNT(*) AS order_row_count,
                       COUNT(DISTINCT order_id) AS order_count
                FROM fact_orders
                WHERE buyer_name LIKE '%test%' AND CUSTOM_MATCH(order_tags, 'priority')
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
        result = tools._analyze_metric_candidates(
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
        result = tools._analyze_metric_candidates(
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
        result = tools._analyze_metric_candidates(
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

    def test_period_over_period_merge_key_includes_time_dimension(self):
        tools = _make_tools()
        candidate = {
            "name": "order_count_yoy",
            "metric_type": "period_over_period",
            "expression": "COUNT(DISTINCT order_id)",
            "time_dimension": "order_date",
            "period_over_period": {
                "time_grain": "month",
                "offset_window": "1 year",
                "calculation": "percent_change",
            },
        }
        alternate = {**candidate, "time_dimension": "ship_date"}

        assert tools._metric_candidate_merge_key(candidate) != tools._metric_candidate_merge_key(alternate)

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
        result = tools._analyze_metric_candidates(sql_queries=[ranked_sql, ranked_sql])

        assert result.result["query_classification"] == "metric_plus_derived_datasource"
        assert result.result["metric_candidates"][0]["source_sql_name"] == "sql_1, sql_2"
        assert result.result["direct_metric_candidates"] == []
        assert len(result.result["blocked_direct_metric_candidates"]) == 2

    def test_invalid_sql_does_not_block_other_queries(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
            sql_queries=[
                "SELECT FROM",
                "SELECT SUM(amount) AS revenue FROM orders",
            ]
        )

        assert len(result.result["parse_errors"]) == 1
        assert result.result["metric_candidates"][0]["name"] == "revenue"

    def test_mysql_dialect_fallback_parses_backtick_aliases(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
            sql_queries=["SELECT COUNT(DISTINCT `user_id`) AS `***` FROM `orders`"]
        )

        assert result.result["parse_errors"] == []
        candidate = result.result["metric_candidates"][0]
        assert candidate["source_alias"] == "***"
        assert candidate["requires_name_translation"] is True
        assert candidate["name_source"] == "expression_fallback"
        assert candidate["name"] == "count_distinct_user_id"

    def test_ranked_window_blocks_direct_metric_and_recommends_datasource(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
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
        result = tools._analyze_metric_candidates(
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
        result = tools._analyze_metric_candidates(
            sql_queries=[
                """
                WITH customer_total_amount_per_category AS (
                    SELECT customer_id, category_name, SUM(amount) AS total_amount
                    FROM transactions
                    GROUP BY customer_id, category_name
                ),
                customer_preferred_category AS (
                    SELECT
                        customer_id,
                        category_name,
                        ROW_NUMBER() OVER (
                            PARTITION BY customer_id
                            ORDER BY total_amount DESC
                        ) AS rn
                    FROM customer_total_amount_per_category
                )
                SELECT category_name AS preferred_category, COUNT(*) AS `***`
                FROM customer_preferred_category
                WHERE rn = 1
                GROUP BY category_name
                """
            ]
        )

        assert result.result["query_classification"] == "metric_plus_derived_datasource"
        assert result.result["blocked_direct_metric_candidates"][0]["source_alias"] == "***"
        assert result.result["blocked_direct_metric_candidates"][0]["requires_name_translation"] is True
        recommendation = result.result["derived_datasource_recommendations"][0]
        assert recommendation["source_cte"] == "customer_preferred_category"
        assert recommendation["rank_alias"] == "rn"
        assert recommendation["window"]["function"] == "ROW_NUMBER"
        assert recommendation["rank_filters"] == ["rn = 1"]

    def test_simple_aggregation_stays_direct_metric(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
            sql_queries=["SELECT dt, SUM(amount) AS revenue FROM orders GROUP BY dt"]
        )

        assert result.result["query_classification"] == "direct_metric"
        assert result.result["derived_datasource_recommendations"] == []
        assert result.result["blocked_direct_metric_candidates"] == []
        assert result.result["direct_metric_candidates"][0]["name"] == "revenue"
        assert result.result["queryability_contracts"] == [
            {
                "source": "sql_1",
                "dimension_hints": ["dt"],
                "metric_hints": ["revenue"],
                "metric_output_ids": ["sql_1:statement_1:output_2:revenue"],
                "contract_source": "final_group_by",
                "time_group_hints": [
                    {
                        "alias": "dt",
                        "base_expr": "dt",
                        "grain": "day",
                    }
                ],
            }
        ]

    def test_queryability_contract_reuses_scope_lineage_for_cte_aliases(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
            sql_queries=[
                """
                WITH prepared AS (
                    SELECT event_date AS activity_date,
                           raw_channel AS login_channel,
                           user_id
                    FROM activity_events
                )
                SELECT activity_date AS metric_time__day,
                       login_channel,
                       COUNT(DISTINCT user_id) AS active_user_count
                FROM prepared
                GROUP BY activity_date, login_channel
                """
            ]
        )

        assert result.result["queryability_contracts"] == [
            {
                "source": "sql_1",
                "dimension_hints": ["metric_time__day", "login_channel"],
                "metric_hints": ["active_user_count"],
                "metric_output_ids": ["sql_1:statement_1:output_3:active_user_count"],
                "contract_source": "final_group_by",
                "dimension_expr_hints": [
                    {
                        "alias": "login_channel",
                        "expr": "raw_channel",
                        "column": "raw_channel",
                    }
                ],
                "time_group_hints": [
                    {
                        "alias": "metric_time__day",
                        "base_expr": "event_date",
                        "grain": "day",
                    }
                ],
            }
        ]

    def test_positional_window_does_not_hide_directly_lowerable_metric_output(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
            sql_queries=[
                """
                SELECT
                    region,
                    SUM(amount) AS revenue,
                    ROW_NUMBER() OVER (ORDER BY SUM(amount) DESC) AS ranking
                FROM orders
                GROUP BY region
                """
            ]
        )

        assert [
            (item["output_name"], item["output_role"], item["lowering_status"])
            for item in result.result["output_contracts"]
        ] == [
            ("region", "dimension", "dimension"),
            ("revenue", "metric", "direct"),
            ("ranking", "non_metric", "non_metric"),
        ]
        assert [(item["preferred_name"], item["target_mode"]) for item in result.result["metric_requirements"]] == [
            ("revenue", "direct_metric")
        ]
        assert result.result["dataset_requirements"] == []

    def test_top_level_union_is_preserved_as_query_backed_output_contract(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
            sql_queries=[
                """
                SELECT dt, SUM(amount) AS revenue
                FROM current_orders
                GROUP BY dt
                UNION ALL
                SELECT dt, SUM(amount) AS revenue
                FROM archived_orders
                GROUP BY dt
                """
            ]
        )

        assert [(item["output_name"], item["output_role"]) for item in result.result["output_contracts"]] == [
            ("dt", "dimension"),
            ("revenue", "metric"),
        ]
        assert result.result["metric_requirements"][0]["target_mode"] == "query_backed_metric"
        assert result.result["dataset_requirements"][0]["output_grain"] == ["dt"]
        assert result.result["queryability_contracts"][0]["contract_source"] == "query_backed_output_grain"

    def test_final_group_key_is_not_promoted_from_upstream_cte_aggregate(self):
        tools = _make_tools()
        sql = """
            WITH daily AS (
                SELECT dt, SUM(amount) AS revenue
                FROM orders
                GROUP BY dt
            )
            SELECT revenue, COUNT(*) AS day_count
            FROM daily
            GROUP BY revenue
        """

        result = tools._analyze_metric_candidates(
            sql_entries_json=json.dumps(
                [
                    {
                        "sql": sql,
                        "question": "Count grouped days for each daily revenue value",
                    }
                ]
            )
        )

        assert [item["output_role"] for item in result.result["output_contracts"]] == [
            "dimension",
            "metric",
        ]
        assert [item["preferred_name"] for item in result.result["metric_requirements"]] == ["day_count"]
        assert result.result["query_classification"] == "query_backed_then_metric"
        dataset_requirement = result.result["dataset_requirements"][0]
        assert dataset_requirement["sql"] == sql
        assert dataset_requirement["question"] == "Count grouped days for each daily revenue value"
        assert dataset_requirement["output_grain"] == ["revenue"]
        assert dataset_requirement["requirement_id"].startswith("query_dataset:")
        assert dataset_requirement["suggested_name"] == "day_count_query_dataset"
        metric_requirement = result.result["metric_requirements"][0]
        assert metric_requirement["dataset_requirement_id"] == dataset_requirement["requirement_id"]
        assert metric_requirement["dataset_name_hint"] == dataset_requirement["suggested_name"]

    @pytest.mark.parametrize("group_by", ["1", "revenue_bucket"])
    def test_final_group_key_supports_ordinal_and_alias_grouping(self, group_by):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
            sql_queries=[
                f"""
                WITH daily AS (
                    SELECT order_date, SUM(amount) AS revenue
                    FROM orders
                    GROUP BY order_date
                )
                SELECT revenue AS revenue_bucket, COUNT(*) AS day_count
                FROM daily
                GROUP BY {group_by}
                """
            ]
        )

        assert [item["output_role"] for item in result.result["output_contracts"]] == [
            "dimension",
            "metric",
        ]
        assert [item["preferred_name"] for item in result.result["metric_requirements"]] == ["day_count"]

    def test_query_backed_dataset_identity_uses_exact_sql_and_keeps_business_name_hint(self):
        tools = _make_tools()
        first_sql = """
            WITH daily AS (
                SELECT dt, SUM(amount) AS revenue FROM orders GROUP BY dt
            )
            SELECT revenue, COUNT(*) AS day_count FROM daily GROUP BY revenue
        """
        second_sql = first_sql.replace("orders", "archived_orders")

        first = tools._analyze_metric_candidates(
            sql_entries_json=json.dumps([{"name": "active_order_revenue_distribution", "sql": first_sql}])
        )
        second = tools._analyze_metric_candidates(
            sql_entries_json=json.dumps([{"name": "archived_order_revenue_distribution", "sql": second_sql}])
        )

        first_requirement = first.result["dataset_requirements"][0]
        second_requirement = second.result["dataset_requirements"][0]
        assert first_requirement["source_sql_name"] == "active_order_revenue_distribution"
        assert second_requirement["source_sql_name"] == "archived_order_revenue_distribution"
        assert first_requirement["requirement_id"] != second_requirement["requirement_id"]
        assert first_requirement["suggested_name"] == "active_order_revenue_distribution_query_dataset"
        assert second_requirement["suggested_name"] == "archived_order_revenue_distribution_query_dataset"
        assert first_requirement["sql_fingerprint"] not in first_requirement["suggested_name"]

        same_sql_different_name = tools._analyze_metric_candidates(
            sql_entries_json=json.dumps([{"name": "renamed_distribution", "sql": first_sql}])
        )
        renamed_requirement = same_sql_different_name.result["dataset_requirements"][0]
        assert renamed_requirement["requirement_id"] == first_requirement["requirement_id"]
        assert renamed_requirement["suggested_name"] == "renamed_distribution_query_dataset"

    def test_inline_row_flags_can_stay_direct_metrics(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
            sql_queries=[
                """
                SELECT
                    SUM(is_paid) AS paid_count,
                    SUM(is_new) AS new_count
                FROM (
                    SELECT
                        CASE WHEN status = 'paid' THEN 1 ELSE 0 END AS is_paid,
                        CASE WHEN created_at >= CURRENT_DATE THEN 1 ELSE 0 END AS is_new
                    FROM orders
                ) flags
                """
            ]
        )

        assert result.result["query_classification"] == "direct_metric"
        assert result.result["dataset_requirements"] == []
        assert [item["target_mode"] for item in result.result["metric_requirements"]] == [
            "direct_metric",
            "direct_metric",
        ]

    def test_passthrough_union_of_aligned_aggregates_preserves_query_boundary(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
            sql_queries=[
                """
                SELECT part_dt, metric_value
                FROM (
                    SELECT dt AS part_dt, SUM(amount) AS metric_value
                    FROM current_orders
                    GROUP BY dt
                    UNION ALL
                    SELECT dt AS part_dt, SUM(amount) AS metric_value
                    FROM archived_orders
                    GROUP BY dt
                ) combined
                """
            ]
        )

        assert result.result["query_classification"] == "query_backed_then_metric"
        assert len(result.result["dataset_requirements"]) == 1
        assert result.result["dataset_requirements"][0]["modeling_mode"] == "query_backed"
        assert result.result["metric_requirements"][0]["preferred_name"] == "metric_value"
        assert result.result["metric_requirements"][0]["target_mode"] == "query_backed_metric"

    def test_cte_star_expands_final_metric_outputs(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
            sql_queries=[
                """
                WITH daily AS (
                    SELECT dt, SUM(amount) AS revenue
                    FROM orders
                    GROUP BY dt
                )
                SELECT * FROM daily
                """
            ]
        )

        assert [item["output_name"] for item in result.result["output_contracts"]] == ["dt", "revenue"]
        assert [item["output_role"] for item in result.result["output_contracts"]] == ["dimension", "metric"]
        assert [item["preferred_name"] for item in result.result["metric_requirements"]] == ["revenue"]
        assert result.result["dataset_requirements"][0]["output_grain"] == ["dt"]

    def test_repeated_aliases_keep_distinct_stable_output_ids(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
            sql_entries_json=json.dumps(
                [
                    {
                        "name": "sql_a",
                        "source_context_id": "sales:sql_1:case",
                        "sql": "SELECT SUM(amount) AS total FROM current_orders",
                    },
                    {
                        "name": "sql_b",
                        "source_context_id": "sales:sql_2:case",
                        "sql": "SELECT SUM(amount) AS total FROM archived_orders",
                    },
                ]
            )
        )

        requirements = result.result["metric_requirements"]
        assert [item["preferred_name"] for item in requirements] == ["total", "total"]
        assert [item["output_id"] for item in requirements] == [
            "sql_a:statement_1:output_1:total",
            "sql_b:statement_1:output_1:total",
        ]

    def test_literal_values_and_time_grain_are_reported_as_preservation_evidence(self):
        tools = _make_tools()
        result = tools._analyze_metric_candidates(
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
        result = tools._analyze_metric_candidates(
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
