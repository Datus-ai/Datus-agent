# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/func_tool/semantic_discovery_tools.py"""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import sqlglot

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
    db_tool.dialect_operations.return_value = None
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
    def test_sql_analysis_without_database_and_host_pinned_file_provenance(self, tmp_path):
        source = tmp_path / "original.sql"
        source.write_text("SELECT COUNT(*) n FROM orders")
        tools = SemanticDiscoveryTools(root_path=str(tmp_path))
        assert {t.name for t in tools.available_tools()} == {"inspect_semantic_sources", "get_semantic_source_evidence"}
        tools.pin_source_files(["original.sql"])
        result = tools.inspect_semantic_sources()
        statement = result.result["calculation_evidence"]["statements"][0]
        assert statement["input_kind"] == "original_file"
        oid = statement["outputs"][0]
        assert tools.get_semantic_source_evidence([oid]).success
        tools.inspect_semantic_sources(sql_queries=["SELECT SUM(price) total FROM prices"])
        assert tools.get_semantic_source_evidence([oid]).success
        source.write_text("SELECT COUNT(DISTINCT store) n FROM orders")
        changed = tools.inspect_semantic_sources()
        assert changed.result["calculation_evidence"]["statements"][0]["input_kind"] == "file"
        assert not tools.get_semantic_source_evidence([oid]).success
        assert not tools.get_semantic_source_evidence(123).success
        tools.reset_request_cache()
        assert not tools._original_file_digests

    def test_parallel_inspections_keep_all_inline_sources_and_coherent_metadata_cache(self):
        db = _make_db_tool()
        db.get_table_ddl.return_value = FuncToolResult(result={"definition": "CREATE TABLE orders(v INT)"})
        db.describe_table.return_value = FuncToolResult(result={"columns": [{"name": "v"}]})
        tools = _make_tools(db)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda sql: tools.inspect_semantic_sources(sql_queries=[sql]),
                    ["SELECT COUNT(*) n FROM orders", "SELECT SUM(v) total FROM orders"],
                )
            )
        assert all(result.success for result in results)
        snapshot = tools.source_snapshot()
        assert len(snapshot["analysis"]["statements"]) == 2
        assert db.describe_table.call_count == 1

    def test_details_expand_count_rowset_and_distinct_without_reparsing(self):
        db = _make_db_tool()
        db.get_table_ddl.return_value = FuncToolResult(result={"definition": "CREATE TABLE orders(store TEXT)"})
        db.describe_table.return_value = FuncToolResult(result={"columns": [{"name": "store"}]})
        tools = _make_tools(db)
        response = tools.inspect_semantic_sources(
            sql_queries=["SELECT COUNT(*) n FROM (SELECT DISTINCT store FROM orders WHERE store IS NOT NULL) d"]
        )
        inventory = response.result["calculation_evidence"]
        assert len(inventory["outputs"]) == 1 and len(inventory["queries"]) == 1
        calls = db.describe_table.call_count
        expanded = tools.get_semantic_source_evidence([inventory["outputs"][0]["id"]])
        assert expanded.success
        assert len(expanded.result["queries"]) == 2
        assert any(o["name"] == "store" for o in expanded.result["outputs"])
        condition = expanded.result["queries"][0]["filters"]["where"].removeprefix("WHERE ")
        assert sqlglot.parse_one(condition) == sqlglot.parse_one("store IS NOT NULL")
        assert db.describe_table.call_count == calls

    def test_details_include_computed_filter_and_join_keys(self):
        tools = _make_tools()
        response = tools.inspect_semantic_sources(
            sql_queries=[
                "SELECT COUNT(*) n FROM "
                "(SELECT CASE WHEN state=1 THEN 1 ELSE 0 END eligible, store+1 join_key FROM orders) c "
                "JOIN shops s ON c.join_key=s.id WHERE c.eligible=1"
            ]
        )
        output_ids = response.result["calculation_evidence"]["statements"][0]["outputs"]
        expanded = tools.get_semantic_source_evidence(output_ids)
        by_name = {o["name"]: o for o in expanded.result["outputs"]}
        assert "CASE WHEN state = 1" in by_name["eligible"]["expression"]
        assert by_name["join_key"]["expression"] == "store + 1"

    def test_sql_file_sources_are_deduplicated_refreshed_and_analyzed_without_execution(self, tmp_path):
        db = _make_db_tool()
        db.get_table_ddl.return_value = FuncToolResult(result={"definition": "CREATE TABLE orders(k INT, v INT)"})
        db.describe_table.return_value = FuncToolResult(result={"columns": [{"name": "k"}, {"name": "v"}]})
        source = tmp_path / "etl.sql"
        sql = "SELECT k,SUM(v) n FROM orders WHERE k IN (1,2) GROUP BY k; SELECT COUNT(*) n FROM orders"
        source.write_text(sql)
        tools = SemanticDiscoveryTools(
            db, root_path=str(tmp_path), source_sql_provider=lambda: [{"name": "provided", "sql": sql}]
        )
        first = tools.inspect_semantic_sources(sql_files=["etl.sql"], sql_queries=[sql])
        assert first.success and first.result["source_sql_count"] == 1
        assert len(first.result["calculation_evidence"]["statements"]) == 2
        aliases = first.result["sources"][0]["aliases"]
        assert aliases[:2] == ["provided", "etl.sql"] and aliases[2].startswith("inline_")
        source.write_text("SELECT COUNT(DISTINCT k) n FROM orders")
        changed = tools.inspect_semantic_sources(sql_files=["etl.sql"])
        assert changed.result["source_revision"] != first.result["source_revision"]
        assert changed.result["source_sql_count"] == 2
        assert len(changed.result["calculation_evidence"]["statements"]) == 3
        assert db.get_table_ddl.call_count == 1 and db.describe_table.call_count == 1
        db.read_query.assert_not_called()

    def test_failed_file_input_is_reported_and_does_not_reuse_old_file(self, tmp_path):
        db = _make_db_tool()
        db.get_table_ddl.return_value = FuncToolResult(result={"definition": "CREATE TABLE t(v INT)"})
        db.describe_table.return_value = FuncToolResult(result={"columns": [{"name": "v"}]})
        source = tmp_path / "etl.sql"
        source.write_text("SELECT SUM(v) n FROM t")
        tools = SemanticDiscoveryTools(db, root_path=str(tmp_path))
        assert tools.inspect_semantic_sources(sql_files=["etl.sql"]).result["source_sql_count"] == 1
        source.unlink()
        result = tools.inspect_semantic_sources(sql_files=["etl.sql"])
        assert not result.result["complete"] and result.result["source_sql_count"] == 0
        assert result.result["issues"][0]["reason"] == "source_file_unavailable"
        blocked = tools.inspect_semantic_sources(sql_files=[str(tmp_path.parent / "outside.sql")])
        assert "outside the readable workspace" in blocked.result["issues"][0]["error"]

    def test_absolute_and_relative_file_paths_replace_the_same_source(self, tmp_path):
        source = tmp_path / "etl.sql"
        source.write_text("SELECT COUNT(*) n FROM orders")
        tools = SemanticDiscoveryTools(_make_db_tool(), root_path=str(tmp_path))
        tools.inspect_semantic_sources(sql_files=["etl.sql"])
        source.write_text("SELECT COUNT(DISTINCT store) n FROM orders")
        refreshed = tools.inspect_semantic_sources(sql_files=[str(source)])
        assert refreshed.result["source_sql_count"] == 1
        assert refreshed.result["calculation_evidence"]["outputs"][0]["expression"] == "COUNT(DISTINCT store)"
        source.unlink()
        assert tools.inspect_semantic_sources(sql_files=["etl.sql"]).result["source_sql_count"] == 0

    @pytest.mark.parametrize("enable_profiler", [False, True])
    def test_exposes_source_inspection_and_optional_profiling(self, enable_profiler):
        tools = _make_tools(enable_semantic_model_profiler=enable_profiler)

        expected = {"inspect_semantic_sources", "get_semantic_source_evidence"}
        if enable_profiler:
            expected.add("profile_semantic_model_evidence")
        assert {tool.name for tool in tools.available_tools()} == expected

    def test_preserves_ddl_keys_schema_and_scoped_calculation_evidence(self):
        db_tool = _make_db_tool()
        definitions = {
            "orders": (
                "CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT, amount DECIMAL, "
                "FOREIGN KEY (customer_id) REFERENCES customers(id))"
            ),
            "customers": (
                "CREATE TABLE customers (id INT PRIMARY KEY, region VARCHAR, UNIQUE (region, id));\n"
                "CREATE UNIQUE INDEX customer_region ON customers (region) WHERE region IS NOT NULL;\n"
                "CREATE UNIQUE INDEX customer_region_lower ON customers (lower(region));"
            ),
        }

        def ddl(table, *_args):
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
        assert orders["schema"]["columns"][2] == {"name": "amount", "type": "DECIMAL"}
        assert customers["schema"]["columns"][1] == {"name": "region", "type": "VARCHAR"}
        assert orders["ddl"] == {"definition": definitions["orders"]}
        assert customers["ddl"] == {"definition": definitions["customers"]}
        evidence = result.result["calculation_evidence"]
        assert evidence["outputs"][-1]["expression"] == "SUM(o.amount)"
        details = tools.get_semantic_source_evidence([evidence["outputs"][-1]["id"]])
        assert details.result["queries"][0]["joins"][0]["condition"] == "o.customer_id = c.id"
        assert db_tool.get_table_ddl.call_count == 2
        assert db_tool.describe_table.call_count == 2

        repeated = tools.inspect_semantic_sources(["orders", "customers"])

        assert repeated.success == 1
        assert repeated.result["tables"] == result.result["tables"]
        assert db_tool.get_table_ddl.call_count == 2
        assert db_tool.describe_table.call_count == 2

        tools.reset_request_cache()
        refreshed = tools.inspect_semantic_sources(["orders", "customers"])

        assert refreshed.success == 1
        assert db_tool.get_table_ddl.call_count == 4
        assert db_tool.describe_table.call_count == 4
        db_tool.read_query.assert_not_called()

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
        assert result.result["tables"][0]["schema"]["columns"] == [{"name": "id", "type": "INT"}]
        db_tool.get_table_ddl.assert_called_once()
        db_tool.describe_table.assert_called_once()
        db_tool.read_query.assert_not_called()

    def test_retains_ddl_when_schema_is_unavailable(self):
        db_tool = _make_db_tool()
        db_tool.get_table_ddl.return_value = FuncToolResult(
            success=1,
            result={"definition": "CREATE TABLE orders (id INT)"},
        )
        db_tool.describe_table.return_value = FuncToolResult(success=0, error="Describe unavailable")

        result = _make_tools(db_tool).inspect_semantic_sources(["orders"])

        assert result.success == 1
        assert result.result["tables"] == [
            {
                "table_name": "orders",
                "schema_error": "Describe unavailable",
                "ddl": {"definition": "CREATE TABLE orders (id INT)"},
            }
        ]

    def test_rejects_empty_table_scope(self):
        result = _make_tools().inspect_semantic_sources([])

        assert result.success == 0
        assert "at least one" in result.error

    def test_retains_exact_calculation_instead_of_lossy_usage_statistics(self):
        db_tool = _make_db_tool()
        db_tool.get_table_ddl.return_value = FuncToolResult(
            success=1,
            result={"definition": "CREATE TABLE orders (amount DECIMAL)"},
        )
        db_tool.describe_table.return_value = FuncToolResult(
            success=1,
            result={"columns": [{"name": "amount", "type": "DECIMAL"}]},
        )
        tools = _make_tools(
            db_tool,
            source_sql_provider=lambda: [{"name": "revenue", "sql": "SELECT SUM(amount) FROM orders"}],
        )

        result = tools.inspect_semantic_sources(["orders"])

        assert result.success == 1
        assert result.result["tables"][0]["ddl"] == {"definition": "CREATE TABLE orders (amount DECIMAL)"}
        assert result.result["calculation_evidence"]["outputs"][0]["expression"] == "SUM(amount)"
        assert result.result["calculation_evidence"]["outputs"][0]["kind"] == "aggregate"


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
# Shared profiling SQL identifiers
# ---------------------------------------------------------------------------


class TestProfileSQLIdentifiers:
    def test_keeps_simple_identifiers_unquoted(self):
        db_tool = _make_db_tool()
        tools = _make_tools(db_tool)

        assert tools._quote_sql_identifier("customer_id") == "customer_id"
        db_tool.quote_sql_identifier.assert_not_called()

    def test_legacy_db_tool_keeps_ansi_fallback_for_non_simple_identifiers(self):
        tools = SemanticDiscoveryTools(db_tool=SimpleNamespace())

        assert tools._quote_sql_identifier("order-detail") == '"order-detail"'


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

    def test_profiles_use_registered_dialect_operations(self):
        db_tool = _make_db_tool()
        operations = SimpleNamespace(
            quote_identifier=lambda name: f'"{name.upper()}"',
            render_limit=lambda sql, limit: f"{sql} FETCH FIRST {limit} ROWS ONLY",
        )
        db_tool.dialect_operations.return_value = operations
        db_tool.read_query.side_effect = [
            FuncToolResult(
                success=1,
                result={"compressed_data": "index,row_count,non_null_count,distinct_count\n0,2,2,2\n"},
            ),
            FuncToolResult(
                success=1,
                result={"compressed_data": "index,value,count\n0,open,1\n1,closed,1\n"},
            ),
        ]
        tools = _make_tools(db_tool)

        profile = tools._profile_single_column(
            table_ref='"ORDERS"',
            column_name="status",
            column_type="VARCHAR2",
            kind="categorical",
            database="oracle_prod",
            top_n=2,
        )

        assert 'COUNT("STATUS")' in profile["stats_sql"]
        assert profile["top_values_sql"].endswith("FETCH FIRST 2 ROWS ONLY")
        assert " LIMIT " not in profile["top_values_sql"]
        db_tool.dialect_operations.assert_called_with(database="oracle_prod")

    def test_duration_profile_uses_registered_limit_syntax(self):
        db_tool = _make_db_tool()
        db_tool.dialect_operations.return_value = SimpleNamespace(
            quote_identifier=lambda name: f'"{name.upper()}"',
            render_limit=lambda sql, limit: f"{sql} FETCH FIRST {limit} ROWS ONLY",
        )
        db_tool.read_query.return_value = FuncToolResult(
            success=1,
            result={"compressed_data": ("index,left_value,right_value\n0,2025-01-01,2025-01-03\n")},
        )
        tools = _make_tools(db_tool)

        profiles = tools._profile_date_duration_pairs(
            table_ref='"EVENTS"',
            columns=[
                {"name": "opened_at", "type": "DATE"},
                {"name": "closed_at", "type": "DATE"},
            ],
            database="oracle_prod",
        )

        assert profiles[0]["delta_days"]["max"] == 2
        sql = db_tool.read_query.call_args.args[0]
        assert '"OPENED_AT"' in sql
        assert sql.endswith("FETCH FIRST 1000 ROWS ONLY")
        assert " LIMIT " not in sql

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
