# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Real Agent tool -> Dosi adapter -> engine -> DuckDB attribution coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from datus.configuration.agent_config import AgentConfig, NodeConfig
from datus.tools.func_tool.semantic_tools import SemanticTools
from tests.nightly_requirements import import_required

# Dosi is an optional runtime adapter and a required CI dependency. Local
# source-only environments may omit it; Coverage Check installs it from the
# semantic-adapter checkout before collecting this acceptance suite.
import_required(
    "datus_semantic_dosi",
    reason="Dosi attribution integration requires the datus-semantic-dosi CI dependency",
)

pytestmark = [pytest.mark.acceptance, pytest.mark.nightly]

FIXTURE_DIR = Path(__file__).parents[2] / "data" / "semantic_models" / "attribution"
MODEL_PATH = FIXTURE_DIR / "model.yaml"
SEED_PATH = FIXTURE_DIR / "seed.sql"


@pytest.fixture
def attribution_tool(tmp_path) -> SemanticTools:
    database = tmp_path / "attribution.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute(SEED_PATH.read_text(encoding="utf-8"))

    config = AgentConfig(
        nodes={"semantic": NodeConfig(model="mock", input=None)},
        home=str(tmp_path / "home"),
        project_name="dosi_attribution_acceptance",
        project_root=str(tmp_path / "workspace"),
        target="mock",
        models={
            "mock": {
                "type": "openai",
                "api_key": "unused",
                "model": "unused",
                "base_url": "http://127.0.0.1:1",
            }
        },
        services={
            "datasources": {
                "attribution": {
                    "type": "duckdb",
                    "uri": str(database),
                    "default": True,
                }
            },
            "semantic_layer": {
                "dosi": {
                    "type": "dosi",
                    "default": True,
                    "semantic_model_path": str(MODEL_PATH),
                }
            },
        },
    )
    config.current_datasource = "attribution"

    # Attribution does not use MetricRAG. Replacing that unrelated storage
    # dependency keeps this integration test focused on the real semantic
    # adapter, engine binding, SQL execution, and Agent tool contract.
    with patch("datus.tools.func_tool.semantic_tools.MetricRAG"):
        tool = SemanticTools(agent_config=config, adapter_type="dosi")

    assert type(tool.adapter).__module__ == "datus_semantic_dosi.adapter"
    return tool


def _attribute(
    tool: SemanticTools,
    metric: str,
    *,
    baseline: tuple[str, str] = ("2024-01-01", "2024-02-01"),
    current: tuple[str, str] = ("2024-02-01", "2024-03-01"),
    dimensions: list[str] | None = None,
    **kwargs,
):
    return tool.attribution_analyze(
        metric_name=metric,
        candidate_dimensions=["ledger.cat"] if dimensions is None else dimensions,
        baseline_start=baseline[0],
        baseline_end=baseline[1],
        current_start=current[0],
        current_end=current[1],
        **kwargs,
    )


def _values_by_name(payload: dict) -> dict[str, dict]:
    values = payload["per_dimension"]["ledger.cat"]["values"]
    return {value["value"]: value for value in values}


def test_native_term_wise_reconciles_segment_deltas(attribution_tool):
    result = _attribute(attribution_tool, "net")

    assert result.success == 1, result.error
    assert result.result["implementation"] == "dosi"
    assert result.result["strategy"] == "term_wise"
    assert result.result["total_change"] == {
        "baseline_value": 120.0,
        "current_value": 140.0,
        "delta": 20.0,
        "pct_change": pytest.approx(100 / 6),
    }
    values = _values_by_name(result.result)
    assert values["a"]["delta"] == 40.0
    assert values["b"]["delta"] == -20.0
    assert sum(value["delta"] for value in values.values()) == 20.0


def test_native_mix_shift_reconciles_factor_totals(attribution_tool):
    result = _attribute(attribution_tool, "avg_x")

    assert result.success == 1, result.error
    assert result.result["strategy"] == "mix_shift"
    assert result.result["total_change"]["baseline_value"] == 75.0
    assert result.result["total_change"]["current_value"] == pytest.approx(230 / 3)
    assert result.result["factor_totals"]["residual"] == pytest.approx(0)
    values = _values_by_name(result.result)
    assert values["a"]["mix_effect"] + values["a"]["rate_effect"] == pytest.approx(values["a"]["delta"])


def test_native_factor_shapley_matches_hand_computed_oracle(attribution_tool):
    result = _attribute(attribution_tool, "xy")

    assert result.success == 1, result.error
    assert result.result["strategy"] == "factor_shapley"
    assert result.result["total_change"]["delta"] == 16200.0
    factors = {factor["factor"]: factor for factor in result.result["factors"]}
    assert factors["ledger_x_sum"]["effect"] == 4800.0
    assert factors["ledger_y_sum"]["effect"] == 11400.0
    assert sum(factor["effect"] for factor in factors.values()) == 16200.0
    values = _values_by_name(result.result)
    assert values["a"]["delta"] == 17400.0
    assert values["b"]["delta"] == -1200.0


def test_native_unsupported_result_is_not_a_tool_failure(attribution_tool):
    result = _attribute(attribution_tool, "unique_categories")

    assert result.success == 1, result.error
    assert result.result["strategy"] == "unsupported"
    assert result.result["unsupported_reason"]["code"] == "non_sum_tier_measure"
    assert result.result["unsupported_reason"]["distinct"] is True


def test_null_and_literal_null_members_keep_distinct_drilldowns(attribution_tool):
    result = _attribute(
        attribution_tool,
        "net",
        baseline=("2024-03-01", "2024-04-01"),
        current=("2024-04-01", "2024-05-01"),
    )

    assert result.success == 1, result.error
    values = result.result["per_dimension"]["ledger.cat"]["values"]
    assert len(values) == 2
    assert {value["value"] for value in values} == {"(null)"}
    assert {value["drill_down"]["where_sql"] for value in values} == {
        "ledger.cat IS NULL",
        "ledger.cat = '(null)'",
    }
    assert sorted(value["delta"] for value in values) == [20.0, 30.0]


def test_value_cap_preserves_the_large_negative_mover(attribution_tool):
    result = _attribute(
        attribution_tool,
        "net",
        baseline=("2024-05-01", "2024-06-01"),
        current=("2024-06-01", "2024-07-01"),
        max_dimension_values=1,
    )

    assert result.success == 1, result.error
    detail = result.result["per_dimension"]["ledger.cat"]
    assert detail["truncated"] is True
    assert any(warning["code"] == "truncated" for warning in result.result["warnings"])
    shrink = next(value for value in detail["values"] if value["value"] == "shrink")
    assert shrink["baseline_value"] == 3000.0
    assert shrink["current_value"] == 100.0
    assert shrink["delta"] == -2900.0


def test_zero_delta_omits_undefined_percentages(attribution_tool):
    result = _attribute(
        attribution_tool,
        "net",
        baseline=("2024-02-01", "2024-03-01"),
        current=("2024-02-01", "2024-03-01"),
    )

    assert result.success == 1, result.error
    assert result.result["total_change"]["delta"] == 0.0
    assert any(warning["code"] == "zero_total_delta" for warning in result.result["warnings"])
    values = result.result["per_dimension"]["ledger.cat"]["values"]
    assert all("contribution_pct" not in value for value in values)


def test_parameter_binding_reaches_native_attribution(attribution_tool):
    result = _attribute(
        attribution_tool,
        "x_above",
        baseline=("2024-11-01", "2024-12-01"),
        current=("2024-12-01", "2025-01-01"),
        params={"t": 100},
    )

    assert result.success == 1, result.error
    assert result.result["strategy"] == "term_wise"
    assert result.result["total_change"]["baseline_value"] == 200.0
    assert result.result["total_change"]["current_value"] == 420.0
    assert result.result["comparison_metadata"]["params"] == {"t": 100}
    values = _values_by_name(result.result)
    assert values["a"]["delta"] == 100.0
    assert values["b"]["delta"] == 120.0


@pytest.mark.parametrize(
    ("metric", "dimensions", "baseline", "params", "expected_code"),
    [
        ("avg_x", ["ledger.cat"], ("2023-01-01", "2023-02-01"), None, "denominator_nonpositive"),
        ("net", [], ("2024-01-01", "2024-02-01"), None, "dimensions_required"),
        ("nett", ["ledger.cat"], ("2024-01-01", "2024-02-01"), None, "unknown_metric"),
        (
            "x_above",
            ["ledger.cat"],
            ("2024-11-01", "2024-12-01"),
            {"t": [50, 100]},
            "param_out_of_domain",
        ),
    ],
)
def test_native_rejections_keep_structured_error_codes(
    attribution_tool,
    metric,
    dimensions,
    baseline,
    params,
    expected_code,
):
    current = ("2024-12-01", "2025-01-01") if metric == "x_above" else ("2024-02-01", "2024-03-01")
    result = _attribute(
        attribution_tool,
        metric,
        dimensions=dimensions,
        baseline=baseline,
        current=current,
        params=params,
    )

    assert result.success == 0
    assert result.result["error_type"] == "semantic_validation_error"
    assert result.result["code"] == expected_code
