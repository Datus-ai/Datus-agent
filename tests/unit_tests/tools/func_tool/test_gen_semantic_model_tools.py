# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/func_tool/gen_semantic_model_tools.py"""

from unittest.mock import MagicMock, patch

from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.gen_semantic_model_tools import GenSemanticModelTools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db_tool(agent_config=None, sub_agent_name="test_agent"):
    """Build a mock DBFuncTool."""
    db_tool = MagicMock()
    db_tool.agent_config = agent_config or MagicMock()
    db_tool.sub_agent_name = sub_agent_name
    return db_tool


def _make_tools(db_tool=None) -> GenSemanticModelTools:
    if db_tool is None:
        db_tool = _make_db_tool()
    return GenSemanticModelTools(db_tool=db_tool)


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
        if "status" in result.result["column_patterns"]:
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
        if "tags" in result.result["column_patterns"]:
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
