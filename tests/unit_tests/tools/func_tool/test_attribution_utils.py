"""Tests for the generic attribution fallback and its public result contract."""

import pytest

from datus.tools.func_tool.attribution_utils import (
    AttributionValidationException,
    GenericAttributeAnalyzer,
)
from datus.tools.semantic_tools.models import (
    AttributionRequest,
    AttributionWindow,
    QueryResult,
)


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


async def analyze(adapter, **overrides):
    request = AttributionRequest(
        metric="revenue",
        dimensions=overrides.pop("dimensions", ["orders.region"]),
        baseline=AttributionWindow(
            start=overrides.pop("baseline_start", "2026-01-01"),
            end=overrides.pop("baseline_end", "2026-01-08"),
        ),
        current=AttributionWindow(
            start=overrides.pop("current_start", "2026-01-08"),
            end=overrides.pop("current_end", "2026-01-15"),
        ),
        **overrides,
    )
    return await GenericAttributeAnalyzer(adapter).attribute(request)


@pytest.mark.ci
class TestGenericAttributeAnalyzer:
    @pytest.mark.asyncio
    async def test_returns_unified_contract_and_drill_down_sql(self):
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
            where_sql="game = 'demo'",
            path=["games", "revenue"],
            max_values_per_dimension=25,
            params={"currency": "USD"},
        )

        assert output.implementation == "generic"
        assert output.strategy == "term_wise"
        assert output.metric == "revenue"
        assert output.total_change.delta == 20
        assert output.comparison_metadata.queries_executed == 4
        assert all(call["where"] == "game = 'demo'" for call in adapter.calls)
        assert all(call["path"] == ["games", "revenue"] for call in adapter.calls)
        assert all(call["params"] == {"currency": "USD"} for call in adapter.calls)
        assert [call.get("limit") for call in adapter.calls] == [None, None, 26, 26]
        values = output.per_dimension["orders.region"].values
        assert values[0].drill_down.where_sql == "orders.region = 7"
        assert values[1].value == "(null)"
        assert values[1].drill_down.where_sql == "orders.region IS NULL"
        payload = output.model_dump(exclude_none=True)
        assert "metric_name" not in payload
        assert "total_delta" not in payload["comparison_metadata"]
        assert "filter_hint" not in payload["per_dimension"]["orders.region"]["values"][0]

    @pytest.mark.asyncio
    async def test_marks_entered_and_exited_segments_and_escapes_strings(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 30}),
                result(["revenue"], {"revenue": 45}),
                result(
                    ["region", "revenue"],
                    {"region": "old", "revenue": 30},
                ),
                result(
                    ["region", "revenue"],
                    {"region": "O'Reilly", "revenue": 45},
                ),
            ]
        )

        output = await analyze(adapter)

        values = {item.value: item for item in output.per_dimension["orders.region"].values}
        assert values["old"].segment_kind == "exited"
        assert values["O'Reilly"].segment_kind == "entered"
        assert values["O'Reilly"].drill_down.where_sql == "orders.region = 'O''Reilly'"

    @pytest.mark.asyncio
    async def test_total_no_data_is_zero_with_lowercase_warning(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"]),
                result(["revenue"], {"revenue": 10}),
            ]
        )

        output = await analyze(adapter, dimensions=[])

        assert output.strategy == "unsupported"
        assert output.unsupported_reason.code == "dimensions_required"
        assert output.total_change.baseline_value == 0
        assert output.total_change.pct_change is None
        assert [warning.code for warning in output.warnings] == ["no_data_baseline"]

    @pytest.mark.asyncio
    async def test_rejects_multi_row_total(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 10}, {"revenue": 20}),
                result(["revenue"], {"revenue": 30}),
            ]
        )

        with pytest.raises(AttributionValidationException) as exc_info:
            await analyze(adapter)

        assert exc_info.value.payload.code == "MULTI_ROW_TOTAL"
        assert exc_info.value.payload.period == "baseline"

    @pytest.mark.asyncio
    async def test_dimension_failure_is_isolated(self):
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
            dimensions=["orders.organization", "orders.channel"],
        )

        assert output.strategy == "term_wise"
        assert "orders.organization" not in output.per_dimension
        assert output.selected_dimensions == ["orders.channel"]
        assert output.comparison_metadata.queries_executed == 5
        assert output.warnings[-1].code == "dimension_analysis_failed"

    @pytest.mark.asyncio
    async def test_non_additive_dimension_is_not_ranked(self):
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

        detail = output.per_dimension["orders.region"]
        assert detail.non_additive is True
        assert detail.reconciliation.passed is False
        assert detail.reconciliation.baseline_residual == 10
        assert output.strategy == "unsupported"
        assert output.dimension_ranking == []
        assert output.selected_dimensions == []
        assert "non_additive_dimension" in [warning.code for warning in output.warnings]

    @pytest.mark.asyncio
    async def test_zero_total_delta_uses_null_percentages(self):
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

        detail = output.per_dimension["orders.region"]
        assert detail.score is None
        assert all(item.contribution_pct is None for item in detail.values)
        assert "zero_total_delta_with_component_changes" in [warning.code for warning in output.warnings]

    @pytest.mark.asyncio
    async def test_high_cardinality_dimension_is_truncated_and_unsupported(self):
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

        output = await analyze(adapter, max_values_per_dimension=2)

        assert adapter.calls[2]["limit"] == 3
        assert output.per_dimension["orders.region"].truncated is True
        assert output.strategy == "unsupported"
        assert output.warnings[-1].code == "high_cardinality_dimension"

    @pytest.mark.asyncio
    async def test_dimension_limit_is_hard_capped(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 1}),
                result(["revenue"], {"revenue": 2}),
                result(["region", "revenue"], {"region": "A", "revenue": 1}),
                result(["region", "revenue"], {"region": "A", "revenue": 2}),
            ]
        )

        await analyze(adapter, max_values_per_dimension=5000)

        assert adapter.calls[2]["limit"] == 1001

    @pytest.mark.asyncio
    async def test_unequal_windows_are_recorded_in_metadata(self):
        adapter = ScriptedAdapter(
            [
                result(["revenue"], {"revenue": 1}),
                result(["revenue"], {"revenue": 1}),
            ]
        )

        output = await analyze(
            adapter,
            dimensions=[],
            baseline_end="2026-01-04",
        )

        assert output.comparison_metadata.baseline_days == 3
        assert output.comparison_metadata.current_days == 7
        assert output.comparison_metadata.equal_length_windows is False
        assert "unequal_windows" in [warning.code for warning in output.warnings]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("start", "end"),
        [
            ("2026-01-01", "2026-01-01"),
            ("2026-01-02", "2026-01-01"),
            ("not-a-date", "2026-01-01"),
        ],
    )
    async def test_rejects_invalid_half_open_window_before_query(self, start, end):
        adapter = ScriptedAdapter([])

        with pytest.raises(AttributionValidationException) as exc_info:
            await analyze(adapter, baseline_start=start, baseline_end=end)

        assert exc_info.value.payload.code == "INVALID_TIME_WINDOW"
        assert adapter.calls == []
