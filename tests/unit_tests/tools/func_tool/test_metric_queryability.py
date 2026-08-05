# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for compact metric queryability contract formatting."""

from datus.tools.func_tool.metric_queryability import summarize_queryability_contracts


def test_summarizes_complete_grouping_contract():
    result = summarize_queryability_contracts(
        [
            {
                "contract_id": "orders:group_1",
                "source_id": "orders",
                "metric_output_ids": ["orders:output:revenue"],
                "dimensions": ["orders.order_date", "orders.region"],
                "time_grain": "month",
            }
        ]
    )

    assert "orders:group_1 group-by [orders.order_date, orders.region]" in result
    assert "metrics [orders:output:revenue]" in result
    assert "time_granularity='month'" in result


def test_prefers_bound_metric_names_and_handles_empty_contracts():
    result = summarize_queryability_contracts(
        [
            {
                "contract_id": "orders:group_1",
                "metric_output_ids": ["orders:output:revenue"],
                "metric_hints": ["revenue_total"],
                "dimensions": ["orders.region"],
            }
        ]
    )

    assert "metrics [revenue_total]" in result
    assert summarize_queryability_contracts([]) == ""
