# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for metric queryability contract helpers."""

from datus.schemas.semantic_agentic_node_models import SourceQueryEvidence
from datus.tools.func_tool.metric_queryability import (
    extract_metric_queryability_contracts_from_sources,
    summarize_queryability_contracts,
)


<<<<<<< HEAD
=======
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


>>>>>>> 1681803 ([Enhancement] Integrate Doris and Hologres adapters (#1218))
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
