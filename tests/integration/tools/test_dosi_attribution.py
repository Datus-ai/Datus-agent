# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Real Agent tool -> bundled Dosi engine -> DuckDB attribution coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from datus.configuration.agent_config import AgentConfig, NodeConfig
from datus.tools.func_tool.semantic_tools import SemanticTools

pytestmark = [pytest.mark.acceptance, pytest.mark.nightly]

FIXTURE_DIR = Path(__file__).parents[2] / "data" / "semantic_models" / "attribution"
MODEL_PATH = FIXTURE_DIR / "model.yaml"
SEED_PATH = FIXTURE_DIR / "seed.sql"


@pytest.fixture
def attribution_tool(tmp_path, monkeypatch) -> SemanticTools:
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
        },
    )
    config.current_datasource = "attribution"
    monkeypatch.setattr(config.path_manager, "semantic_model_path", lambda datasource: FIXTURE_DIR)

    # Attribution does not use MetricRAG. Replacing that unrelated storage
    # dependency keeps this integration test focused on the real semantic
    # engine binding, SQL execution, and Agent tool contract.
    with patch("datus.tools.func_tool.semantic_tools.MetricRAG"):
        tool = SemanticTools(agent_config=config)

    assert type(tool.runtime).__module__ == "datus.tools.semantic_tools.dosi.runtime"
    return tool


JAN = ("2024-01-01", "2024-02-01")
FEB = ("2024-02-01", "2024-03-01")


def _periods(baseline, current):
    return {
        "baseline_start": baseline[0],
        "baseline_end": baseline[1],
        "current_start": current[0],
        "current_end": current[1],
    }


def _by_formula(tool: SemanticTools, metric: str, *, baseline=JAN, current=FEB, **kwargs):
    return tool.attribute_metrics(metric=metric, **_periods(baseline, current), **kwargs)


def _by_groups(tool: SemanticTools, metric: str, *, dimensions=("ledger.cat",), baseline=JAN, current=FEB, **kwargs):
    return tool.attribute_dimensions(
        metric=metric, dimensions=list(dimensions), **_periods(baseline, current), **kwargs
    )


def _tree(result) -> list[str]:
    """The lines after the status, issue and note lines."""
    assert result.success == 1, result.error
    lines = result.result.splitlines()
    return lines[lines.index("") + 1 :]


# January has 31 days and February 29: every call over them says so.
UNEQUAL = "note the periods differ in length (31 days and 29 days); values are not scaled to one length"


def test_two_term_metric_by_formula_and_by_group(attribution_tool):
    by_formula = _by_formula(attribution_tool, "net")
    assert by_formula.result.splitlines()[:2] == ["status complete", UNEQUAL]
    assert _tree(by_formula) == [
        "net = SUM(ledger.x) - SUM(ledger.y) | 120 → 140 | +20 (100.0%)",
        "  x_total [input] | 150 → 230 | +80 (400.0%)",
        "  SUM(y) [input] | 30 → 90 | -60 (-300.0%)",
    ]
    assert _tree(_by_groups(attribution_tool, "net")) == [
        "net by ledger.cat | 120 → 140 | change +20",
        "- a | 80 → 120 | +40 (200.0%)",
        "- b | 40 → 20 | -20 (-100.0%)",
    ]


def test_a_ratio_splits_its_change_between_groups_whose_own_value_did_not_move(attribution_tool):
    assert _tree(_by_formula(attribution_tool, "avg_x")) == [
        "avg_x = AVG(x) [input] | 75 → 76.6667 | +1.66667 (100.0%)",
    ]
    # a's average stayed at 100 but its weight grew: its effect is not its own change.
    assert _tree(_by_groups(attribution_tool, "avg_x")) == [
        "avg_x by ledger.cat | 75 → 76.6667 | change +1.66667",
        "- a | 100 → 100 | +10 (600.0%)",
        "- b | 50 → 30 | -8.33333 (-500.0%)",
    ]


def test_a_product_shares_its_change_between_its_factors(attribution_tool):
    # Shapley over the two inputs: (230 - 150) * (30 + 90) / 2 and (90 - 30) * (150 + 230) / 2.
    assert _tree(_by_formula(attribution_tool, "xy")) == [
        "xy = SUM(ledger.x) * SUM(ledger.y) | 4500 → 20700 | +16200 (100.0%)",
        "  SUM(y) [input] | 30 → 90 | +11400 (70.4%)",
        "  x_total [input] | 150 → 230 | +4800 (29.6%)",
    ]


def test_a_distinct_count_by_group_is_an_answer_not_a_tool_failure(attribution_tool):
    result = _by_groups(attribution_tool, "unique_categories")

    assert result.success == 1, result.error
    assert result.result.splitlines()[:2] == [
        "status unsupported",
        "issue error insufficient_state: `COUNT(DISTINCT cat)` counts distinct values: its groups do not add up to"
        " the whole, and the state that would is not fetched",
    ]
    # get_metric says so before any call.
    detail = attribution_tool.get_metric(name="unique_categories")
    assert detail.result.splitlines()[-1] == (
        "attribution: attribute_metrics yes; attribute_dimensions no (insufficient_state)"
    )
    net = attribution_tool.get_metric(name="net").result
    assert net.splitlines()[-1] == "attribution: attribute_metrics yes; attribute_dimensions yes"


def test_null_and_literal_null_groups_stay_apart(attribution_tool):
    march, april = ("2024-03-01", "2024-04-01"), ("2024-04-01", "2024-05-01")

    assert _tree(_by_groups(attribution_tool, "net", baseline=march, current=april)) == [
        "net by ledger.cat | 140 → 190 | change +50",
        "- NULL | 100 → 130 | +30 (60.0%)",
        "- (null) | 40 → 60 | +20 (40.0%)",
    ]


def test_top_groups_keeps_the_largest_mover_and_sums_the_rest(attribution_tool):
    may, june = ("2024-05-01", "2024-06-01"), ("2024-06-01", "2024-07-01")

    assert _tree(_by_groups(attribution_tool, "net", baseline=may, current=june, top_groups=1)) == [
        "net by ledger.cat | 4000 → 1200 | change -2800",
        "- shrink | 3000 → 100 | -2900 (103.6%)",
        "- 1 other | +100 (-3.6%)",
    ]


def test_no_change_has_no_shares(attribution_tool):
    assert _tree(_by_groups(attribution_tool, "net", baseline=FEB, current=FEB)) == [
        "net by ledger.cat | 140 → 140 | change 0",
        "- a | 120 → 120 | 0 (n/a)",
        "- b | 20 → 20 | 0 (n/a)",
    ]


def test_parameter_binding_reaches_native_attribution(attribution_tool):
    november, december = ("2024-11-01", "2024-12-01"), ("2024-12-01", "2025-01-01")

    result = _by_groups(attribution_tool, "x_above", baseline=november, current=december, params={"t": 100})

    assert _tree(result) == [
        "x_above by ledger.cat | 200 → 420 | change +220",
        "- b | null_value → 120 | +120 (54.5%)",
        "- a | 200 → 300 | +100 (45.5%)",
    ]


def test_an_endpoint_without_a_value_is_not_computable(attribution_tool):
    result = _by_formula(attribution_tool, "avg_x", baseline=("2023-01-01", "2023-02-01"))

    assert result.success == 1, result.error
    lines = result.result.splitlines()
    assert lines[0] == "status not_computable"
    assert lines[1].startswith("issue error undefined_endpoint: the target is NULL in the baseline period")
    assert lines[-1] == "avg_x = AVG(x) | null_value → 76.6667 | effect n/a"


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (
            lambda tool: _by_groups(tool, "net", dimensions=()),
            "dimensions_required: name at least one dimension to explain the metric's change by; `candidates` lists"
            " what `net` can be split by, best first | candidates: ledger.cat",
        ),
        (
            lambda tool: _by_groups(tool, "net", top_groups=0),
            "top_groups_out_of_range: `top_groups` is how many groups the tree lists, from 1 to 200; 0 is not",
        ),
        (
            lambda tool: _by_formula(tool, "net", baseline=("2024-02-01", "2024-01-01")),
            "invalid_comparison: the baseline period [2024-02-01, 2024-01-01) is empty: `start` must be before `end`",
        ),
        (lambda tool: _by_formula(tool, "nett"), None),
    ],
)
def test_request_errors_keep_their_code(attribution_tool, call, expected):
    result = call(attribution_tool)

    assert result.success == 0
    assert result.result["error_type"] == "semantic_validation_error"
    if expected is None:
        assert result.result["code"] == "unknown_metric"
    else:
        assert result.error == expected


def test_several_values_of_one_parameter_are_refused_in_the_result(attribution_tool):
    november, december = ("2024-11-01", "2024-12-01"), ("2024-12-01", "2025-01-01")

    result = _by_formula(attribution_tool, "x_above", baseline=november, current=december, params={"t": [50, 100]})

    assert result.success == 1, result.error
    assert result.result.splitlines()[:2] == [
        "status invalid_request",
        "issue error param_out_of_domain: parameter `t` is bound to several values; an attribution explains one"
        " binding",
    ]
