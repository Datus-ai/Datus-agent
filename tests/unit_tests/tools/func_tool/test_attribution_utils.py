"""Tests for attribution result validation and runtime guardrails."""

import math

import pytest

from datus.tools.func_tool.attribution_utils import (
    AttributionAnalysisResult,
    AttributionValidationException,
    DimensionAttributionUtil,
)
from datus.tools.semantic_tools.models import QueryResult


class ScriptedAdapter:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def query_metrics(self, **kwargs):
        self.calls.append(kwargs)
        scripted_result = self.results.pop(0)
        if isinstance(scripted_result, Exception):
            raise scripted_result
        return scripted_result


def result(columns, *rows):
    return QueryResult(columns=columns, data=list(rows))


async def analyze(adapter, **kwargs):
    return await DimensionAttributionUtil(adapter).attribution_analyze(
        metric_name="revenue",
        candidate_dimensions=kwargs.pop("candidate_dimensions", ["orders.region"]),
        baseline_start=kwargs.pop("baseline_start", "2026-01-01"),
        baseline_end=kwargs.pop("baseline_end", "2026-01-08"),
        current_start=kwargs.pop("current_start", "2026-01-08"),
        current_end=kwargs.pop("current_end", "2026-01-15"),
        **kwargs,
    )


@pytest.mark.ci
class TestAttributionAnalysis:
    @pytest.mark.asyncio
    async def test_passes_scope_and_builds_per_dimension_filters(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 30}),
                result(["revenue"], {"revenue": 50}),
                result(
                    ["region", "revenue"],
                    {"region": 7, "revenue": 10},
                    {"region": None, "revenue": 20},
                ),
                result(
                    ["region", "revenue"],
                    {"region": 7, "revenue": 30},
                    {"region": None, "revenue": 20},
                ),
            ]
        )

        output = await analyze(
            adapter,
            where="game = 'demo'",
            path=["games", "revenue"],
            max_dimension_values=25,
        )

        assert all(call["where"] == "game = 'demo'" for call in adapter.calls)
        assert all(call["path"] == ["games", "revenue"] for call in adapter.calls)
        assert [call.get("limit") for call in adapter.calls] == [None, None, 26, 26]
        assert [(call["time_start"], call["time_end"]) for call in adapter.calls] == [
            ("2026-01-01", "2026-01-08"),
            ("2026-01-08", "2026-01-15"),
            ("2026-01-01", "2026-01-08"),
            ("2026-01-08", "2026-01-15"),
        ]
        contributions = output.per_dimension["orders.region"].contributions
        assert contributions[0].filter_hint.model_dump() == {
            "dimension": "orders.region",
            "operator": "eq",
            "value": 7,
        }
        assert contributions[1].dimension_values == {"orders.region": "(null)"}
        assert contributions[1].filter_hint.operator == "is_null"
        assert output.per_dimension["orders.region"].additivity_check.status == "passed"
        assert output.dimension_analysis_status == "complete"
        assert output.comparison_metadata["time_range_semantics"] == "[start, end)"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("baseline", "expected_code"),
        [
            (result([]), "NO_DATA_BASELINE"),
            (result(["revenue"], {"revenue": None}), "NULL_TOTAL_BASELINE"),
            (result(["revenue"], {"revenue": 0}), "ZERO_OR_NO_DATA_BASELINE"),
        ],
    )
    async def test_totals_only_uses_the_same_coverage_warnings(self, baseline, expected_code):
        adapter = ScriptedAdapter([baseline, result(["revenue"], {"revenue": 10})])

        output = await analyze(adapter, candidate_dimensions=[])

        assert output.per_dimension == {}
        assert output.dimension_ranking == []
        assert output.selected_dimensions == []
        assert output.comparison_metadata["baseline"]["total"] == 0
        assert output.warnings[0].code == expected_code

    @pytest.mark.asyncio
    async def test_empty_grouped_period_preserves_zero_total_coverage_warning(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 10}),
                result(["revenue"], {"revenue": 0}),
                result(["region", "revenue"], {"region": "US", "revenue": 10}),
                result([]),
            ]
        )

        output = await analyze(adapter)

        assert "ZERO_OR_NO_DATA_CURRENT" in [warning.code for warning in output.warnings]
        contribution = output.per_dimension["orders.region"].contributions[0]
        assert contribution.current == 0
        assert contribution.delta == -10

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("bad_total", "expected_code"),
        [
            (
                result(["revenue"], {"revenue": 1}, {"revenue": 2}),
                "MULTI_ROW_TOTAL",
            ),
            (result(["other"], {"other": 1}), "MISSING_METRIC_COLUMN"),
            (
                result(["revenue"], {"revenue": math.nan}),
                "NON_NUMERIC_METRIC_VALUE",
            ),
        ],
    )
    async def test_rejects_invalid_total_shapes(self, bad_total, expected_code):
        adapter = ScriptedAdapter([bad_total, result(["revenue"], {"revenue": 10})])

        with pytest.raises(AttributionValidationException) as exc_info:
            await analyze(adapter, candidate_dimensions=[])

        assert exc_info.value.payload.code == expected_code
        assert exc_info.value.payload.period == "baseline"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("grouped", "expected_code"),
        [
            (
                result(["revenue"], {"revenue": 10}),
                "MISSING_DIMENSION_COLUMN",
            ),
            (
                result(["region", "revenue"], {"region": "US", "revenue": None}),
                "NON_NUMERIC_METRIC_VALUE",
            ),
            (
                result(
                    ["region", "revenue"],
                    {"region": "US", "revenue": 5},
                    {"region": "US", "revenue": 5},
                ),
                "DUPLICATE_DIMENSION_KEY",
            ),
        ],
    )
    async def test_isolates_invalid_grouped_shapes(self, grouped, expected_code):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 10}),
                result(["revenue"], {"revenue": 10}),
                grouped,
                result(["region", "revenue"], {"region": "US", "revenue": 10}),
            ]
        )

        output = await analyze(adapter)

        error = output.per_dimension["orders.region"].error
        assert error.error_type == "validation_error"
        assert error.code == expected_code
        assert error.period == "baseline"
        assert output.dimension_analysis_status == "unavailable"
        assert output.dimension_ranking == []
        assert output.selected_dimensions == []
        assert output.warnings[-1].code == "DIMENSION_ANALYSIS_FAILED"

    @pytest.mark.asyncio
    async def test_query_failure_isolated_and_next_dimension_still_analyzed(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 100}),
                result(["revenue"], {"revenue": 150}),
                RuntimeError("ambiguous organization column"),
                result(["channel", "revenue"], {"channel": "direct", "revenue": 100}),
                result(["channel", "revenue"], {"channel": "direct", "revenue": 150}),
            ]
        )

        output = await analyze(
            adapter,
            candidate_dimensions=["orders.organization", "orders.channel"],
        )

        failed = output.per_dimension["orders.organization"]
        assert failed.error.error_type == "query_error"
        assert failed.error.code == "DIMENSION_QUERY_FAILED"
        assert failed.error.period == "baseline"
        assert output.per_dimension["orders.channel"].error is None
        assert [ranking.dimension for ranking in output.dimension_ranking] == ["orders.channel"]
        assert output.selected_dimensions == ["orders.channel"]
        assert output.dimension_analysis_status == "partial"
        assert len(adapter.calls) == 5

    @pytest.mark.asyncio
    async def test_deduplicates_candidate_dimensions_before_querying_and_reporting(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 100}),
                result(["revenue"], {"revenue": 150}),
                result(["region", "revenue"], {"region": "US", "revenue": 100}),
                result(["region", "revenue"], {"region": "US", "revenue": 150}),
            ]
        )

        output = await analyze(
            adapter,
            candidate_dimensions=["orders.region", "orders.region"],
        )

        assert output.candidate_dimensions == ["orders.region"]
        assert [ranking.dimension for ranking in output.dimension_ranking] == ["orders.region"]
        assert output.selected_dimensions == ["orders.region"]
        assert list(output.per_dimension) == ["orders.region"]
        assert len(output.top_dimension_values) == 1
        assert output.dimension_analysis_status == "complete"
        assert len(adapter.calls) == 4

    @pytest.mark.asyncio
    async def test_current_query_failure_discards_partial_dimension_result(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 100}),
                result(["revenue"], {"revenue": 150}),
                result(["region", "revenue"], {"region": "US", "revenue": 100}),
                RuntimeError("current query failed"),
            ]
        )

        output = await analyze(adapter)

        dimension = output.per_dimension["orders.region"]
        assert dimension.error.period == "current"
        assert dimension.contributions == []
        assert output.dimension_analysis_status == "unavailable"

    @pytest.mark.asyncio
    async def test_all_dimension_failures_keep_valid_total_comparison(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 100}),
                result(["revenue"], {"revenue": 150}),
                RuntimeError("region query failed"),
                RuntimeError("channel query failed"),
            ]
        )

        output = await analyze(
            adapter,
            candidate_dimensions=["orders.region", "orders.channel"],
        )

        assert output.comparison_metadata["total_delta"] == 50
        assert output.dimension_analysis_status == "unavailable"
        assert output.dimension_ranking == []
        assert all(item.error is not None for item in output.per_dimension.values())
        assert [warning.code for warning in output.warnings] == [
            "DIMENSION_ANALYSIS_FAILED",
            "DIMENSION_ANALYSIS_FAILED",
        ]

    @pytest.mark.asyncio
    async def test_checks_each_period_additivity_even_when_delta_residual_cancels(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 100}),
                result(["revenue"], {"revenue": 100}),
                result(
                    ["region", "revenue"],
                    {"region": "US", "revenue": 60},
                    {"region": "EU", "revenue": 50},
                ),
                result(
                    ["region", "revenue"],
                    {"region": "US", "revenue": 70},
                    {"region": "EU", "revenue": 40},
                ),
            ]
        )

        output = await analyze(adapter)

        check = output.per_dimension["orders.region"].additivity_check
        assert check.status == "failed"
        assert check.baseline_residual == 10
        assert check.current_residual == 10
        assert check.delta_residual_pct is None
        assert "NON_ADDITIVE_DIMENSION" in [warning.code for warning in output.warnings]

    @pytest.mark.asyncio
    async def test_zero_total_still_checks_additivity_and_reports_offsetting_changes(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 0}),
                result(["revenue"], {"revenue": 0}),
                result(
                    ["region", "revenue"],
                    {"region": "US", "revenue": 10},
                    {"region": "EU", "revenue": -10},
                ),
                result(
                    ["region", "revenue"],
                    {"region": "US", "revenue": 20},
                    {"region": "EU", "revenue": -20},
                ),
            ]
        )

        output = await analyze(adapter)

        dimension = output.per_dimension["orders.region"]
        assert dimension.additivity_check.status == "passed"
        assert dimension.score == 0
        assert all(item.contribution_pct_of_total_delta == 0 for item in dimension.contributions)
        assert "ZERO_TOTAL_DELTA_WITH_COMPONENT_CHANGES" in [warning.code for warning in output.warnings]

    @pytest.mark.asyncio
    async def test_effectively_zero_delta_omits_delta_residual_percentage(self):
        baseline_total = 1_000_000_000
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": baseline_total}),
                result(["revenue"], {"revenue": baseline_total + 0.001}),
                result(["region", "revenue"], {"region": "US", "revenue": baseline_total}),
                result(["region", "revenue"], {"region": "US", "revenue": baseline_total + 1}),
            ]
        )

        output = await analyze(adapter)

        dimension = output.per_dimension["orders.region"]
        assert dimension.additivity_check.status == "passed"
        assert dimension.additivity_check.delta_residual_pct is None
        assert dimension.score == 0
        assert dimension.contributions[0].contribution_pct_of_total_delta == 0

    @pytest.mark.asyncio
    async def test_union_cardinality_truncates_dimension_and_excludes_legacy_ranking(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 2}),
                result(["revenue"], {"revenue": 2}),
                result(
                    ["region", "revenue"],
                    {"region": "A", "revenue": 1},
                    {"region": "B", "revenue": 1},
                ),
                result(
                    ["region", "revenue"],
                    {"region": "B", "revenue": 1},
                    {"region": "C", "revenue": 1},
                ),
            ]
        )

        output = await analyze(adapter, max_dimension_values=2)

        dimension = output.per_dimension["orders.region"]
        assert adapter.calls[2]["limit"] == 3
        assert dimension.truncated is True
        assert dimension.score is None
        assert dimension.additivity_check.status == "skipped"
        assert output.dimension_ranking == []
        assert output.selected_dimensions == []
        assert output.dimension_analysis_status == "unavailable"
        assert output.warnings[-1].code == "HIGH_CARDINALITY_DIMENSION"

    @pytest.mark.asyncio
    async def test_hard_caps_dimension_limit_and_records_requested_value(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 1}),
                result(["revenue"], {"revenue": 1}),
                result(["region", "revenue"], {"region": "A", "revenue": 1}),
                result(["region", "revenue"], {"region": "A", "revenue": 1}),
            ]
        )

        output = await analyze(adapter, max_dimension_values=5000)

        assert adapter.calls[2]["limit"] == 1001
        assert output.comparison_metadata["requested_max_dimension_values"] == 5000
        assert output.comparison_metadata["effective_max_dimension_values"] == 1000

    @pytest.mark.asyncio
    async def test_warns_for_unequal_concrete_windows(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 1}),
                result(["revenue"], {"revenue": 1}),
            ]
        )

        output = await analyze(
            adapter,
            candidate_dimensions=[],
            baseline_end="2026-01-04",
        )

        assert output.comparison_metadata["baseline"]["days"] == 3
        assert output.comparison_metadata["current"]["days"] == 7
        assert "UNEQUAL_WINDOWS" in [warning.code for warning in output.warnings]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("start", "end"),
        [
            ("2026-01-01", "2026-01-01"),
            ("2026-01-02", "2026-01-01"),
        ],
    )
    async def test_rejects_empty_or_reversed_half_open_window_before_query(self, start, end):
        adapter = ScriptedAdapter([])

        with pytest.raises(AttributionValidationException) as exc_info:
            await analyze(
                adapter,
                candidate_dimensions=[],
                baseline_start=start,
                baseline_end=end,
            )

        assert exc_info.value.payload.code == "INVALID_TIME_WINDOW"
        assert exc_info.value.payload.period == "baseline"
        assert adapter.calls == []

    @pytest.mark.asyncio
    async def test_single_day_snapshot_uses_next_day_as_exclusive_end(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 100}),
                result(["revenue"], {"revenue": 150}),
            ]
        )

        output = await analyze(
            adapter,
            candidate_dimensions=[],
            baseline_start="2026-06-30",
            baseline_end="2026-07-01",
            current_start="2026-07-31",
            current_end="2026-08-01",
        )

        assert output.comparison_metadata["baseline"]["days"] == 1
        assert output.comparison_metadata["current"]["days"] == 1
        assert output.dimension_analysis_status == "not_requested"

    def test_old_serialized_result_gets_defaults_for_new_fields(self):
        output = AttributionAnalysisResult.model_validate(
            {
                "metric_name": "revenue",
                "candidate_dimensions": ["region"],
                "dimension_ranking": [{"dimension": "region", "score": 1.0}],
                "selected_dimensions": ["region"],
                "top_dimension_values": [
                    {
                        "dimension_values": {"region": "US"},
                        "baseline": 1,
                        "current": 2,
                        "delta": 1,
                        "contribution_pct_of_total_delta": 100,
                    }
                ],
                "comparison_metadata": {
                    "baseline": {"start": "2026-01-01", "end": "2026-01-01"},
                    "current": {"start": "2026-01-02", "end": "2026-01-02"},
                },
            }
        )

        assert output.per_dimension == {}
        assert output.warnings == []
        assert output.dimension_analysis_status == "complete"
        assert output.top_dimension_values[0].filter_hint is None
