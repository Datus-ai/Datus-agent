# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/func_tool/semantic_discovery_tools.py"""

from types import SimpleNamespace
from unittest.mock import MagicMock

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
) -> SemanticDiscoveryTools:
    if db_tool is None:
        db_tool = _make_db_tool()
    return SemanticDiscoveryTools(
        db_tool=db_tool,
        enable_semantic_model_profiler=enable_semantic_model_profiler,
    )


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
