# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for metric queryability contract helpers."""

from datus.schemas.semantic_agentic_node_models import SourceQueryEvidence
from datus.tools.func_tool.metric_queryability import (
    extract_metric_queryability_contracts_from_sources,
    extract_sql_snippets,
    link_queryability_contracts_to_metric_outputs,
    query_backed_queryability_contracts,
    summarize_queryability_contracts,
)


class TestExtractSqlSnippets:
    def test_accepts_new_adapter_fence_labels(self):
        prompt = "```doris\nSELECT 1 AS doris_value\n```\n```hologres\nSELECT 2 AS hologres_value\n```"

        assert extract_sql_snippets(prompt) == [
            "SELECT 1 AS doris_value",
            "SELECT 2 AS hologres_value",
        ]

    def test_trims_natural_language_after_unfenced_sql(self):
        prompt = (
            "Generate a metric from the following SQL:\n"
            "SELECT region, SUM(amount) AS revenue FROM orders GROUP BY region\n"
            "Name the metric revenue."
        )

        assert extract_sql_snippets(prompt, preserve_source=True, dialect="mysql") == [
            "SELECT region, SUM(amount) AS revenue FROM orders GROUP BY region"
        ]

    def test_labeled_sql_stops_before_following_markdown_section(self):
        prompt = (
            "Query 6:\n"
            "Question: Revenue?\n"
            "SQL:\n"
            "SELECT region, SUM(amount) AS revenue FROM orders GROUP BY region;\n\n"
            "## Additional Instructions\n"
            "Use concise English names."
        )

        assert extract_sql_snippets(prompt, preserve_source=True, dialect="mysql") == [
            "SELECT region, SUM(amount) AS revenue FROM orders GROUP BY region;"
        ]


class TestSummarizeQueryabilityContracts:
    def test_formats_parts(self):
        contracts = [
            {"source": "sql_1", "dimension_hints": ["order_date"], "metric_hints": ["revenue"]},
            {"source": "sql_2", "dimension_hints": ["region"], "metric_hints": ["orders"]},
        ]
        result = summarize_queryability_contracts(contracts)
        assert "sql_1 group-by [order_date] metrics [revenue]" in result
        assert "sql_2 group-by [region] metrics [orders]" in result

    def test_empty(self):
        assert summarize_queryability_contracts([]) == ""

    def test_includes_time_grain_guidance(self):
        contracts = [
            {
                "source": "sql_1",
                "dimension_hints": ["metric_date"],
                "metric_hints": ["revenue"],
                "time_group_hints": [
                    {"alias": "metric_date", "base_expr": "CAST(ordered_at AS DATETIME)", "grain": "day"}
                ],
            }
        ]
        result = summarize_queryability_contracts(contracts)
        assert "sql_1 group-by [metric_date] metrics [revenue]" in result
        assert "time_granularity='day'" in result

    def test_without_time_hints_has_no_grain_guidance(self):
        contracts = [{"source": "sql_1", "dimension_hints": ["region"], "metric_hints": ["orders"]}]
        assert "time_granularity" not in summarize_queryability_contracts(contracts)


class TestExtractMetricQueryabilityContractsFromSources:
    def test_preserves_structured_source_name(self):
        contracts = extract_metric_queryability_contracts_from_sources(
            [
                SourceQueryEvidence(
                    source_sql_name="sql_9",
                    sql="SELECT region, SUM(amount) AS revenue FROM orders GROUP BY region",
                )
            ]
        )

        assert len(contracts) == 1
        assert contracts[0]["source"] == "sql_9"
        assert contracts[0]["dimension_hints"] == ["region"]
        assert contracts[0]["metric_hints"] == ["revenue"]

    def test_traces_grouped_alias_through_subquery(self):
        contracts = extract_metric_queryability_contracts_from_sources(
            [
                SourceQueryEvidence(
                    source_sql_name="channel_activity",
                    sql=(
                        "SELECT activity_date AS metric_time__day, login_channel, "
                        "COUNT(DISTINCT user_id) AS active_user_count "
                        "FROM (SELECT event_date AS activity_date, raw_channel AS login_channel, user_id "
                        "FROM activity_events) AS metric_source "
                        "GROUP BY activity_date, login_channel"
                    ),
                )
            ]
        )

        assert contracts[0]["dimension_expr_hints"] == [
            {
                "alias": "login_channel",
                "expr": "raw_channel",
                "column": "raw_channel",
            },
        ]
        assert contracts[0]["time_group_hints"] == [
            {
                "alias": "metric_time__day",
                "base_expr": "event_date",
                "grain": "day",
            }
        ]

    def test_traces_grouped_alias_through_cte(self):
        contracts = extract_metric_queryability_contracts_from_sources(
            [
                SourceQueryEvidence(
                    source_sql_name="regional_sales",
                    sql=(
                        "WITH prepared AS ("
                        "SELECT raw_region AS reporting_region, amount FROM orders"
                        ") SELECT reporting_region, SUM(amount) AS revenue "
                        "FROM prepared GROUP BY reporting_region"
                    ),
                )
            ]
        )

        assert contracts[0]["dimension_expr_hints"] == [
            {
                "alias": "reporting_region",
                "expr": "raw_region",
                "column": "raw_region",
            }
        ]

    def test_links_source_alias_to_stable_metric_output_id(self):
        contracts = [
            {
                "source": "sql_9",
                "dimension_hints": ["week_start"],
                "metric_hints": ["iusernum"],
            }
        ]
        requirements = [
            {
                "source_sql_name": "sql_9",
                "preferred_name": "iusernum",
                "output_id": "sql_9:statement_1:output_2:iusernum",
            }
        ]

        linked = link_queryability_contracts_to_metric_outputs(contracts, requirements)

        assert linked[0]["metric_output_ids"] == ["sql_9:statement_1:output_2:iusernum"]
        assert linked[0]["metric_hints"] == ["iusernum"]

    def test_query_backed_contract_uses_authoritative_final_output_grain(self):
        output_id = "retention:statement_1:output_4:retained_players"
        contracts = query_backed_queryability_contracts(
            {
                "metric_requirements": [
                    {
                        "output_id": output_id,
                        "preferred_name": "retained_players",
                    }
                ],
                "dataset_requirements": [
                    {
                        "source_sql_name": "retention_metrics",
                        "output_grain": ["gameplay_name", "cohort_date", "retention_day"],
                        "metric_output_ids": [output_id],
                    }
                ],
            }
        )

        assert contracts == [
            {
                "source": "retention_metrics",
                "dimension_hints": ["gameplay_name", "cohort_date", "retention_day"],
                "metric_hints": ["retained_players"],
                "metric_output_ids": [output_id],
                "contract_source": "query_backed_output_grain",
            }
        ]
