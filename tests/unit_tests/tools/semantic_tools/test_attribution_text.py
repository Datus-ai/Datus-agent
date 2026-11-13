# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""The text a model reads for an attribution result.

The fixtures under ``tests/data/attribution_results`` are Dosi's own results
on TPC-DS sf=1 (``dosi attribute-metrics`` / ``attribute-dimensions --format
json``), unedited, so the renderer is checked against every field the engine
actually writes.
"""

import copy
import json
from pathlib import Path

import pytest

from datus.tools.semantic_tools.dosi.attribution_text import (
    number,
    render_attribute_dimensions,
    render_attribute_metrics,
    render_attribution,
    summarize_attribution_text,
)

RESULTS = Path(__file__).resolve().parents[3] / "data" / "attribution_results"


def _load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


@pytest.mark.parametrize(
    "value, text",
    [
        (1019954026.3199998, "1019954026"),
        (144.23904684175895, "144.239"),
        (0.0000123456789, "0.0000123457"),
        (-0.331556, "-0.331556"),
        (13.0, "13"),
        (0, "0"),
        (None, "null"),
    ],
)
def test_numbers_keep_six_significant_digits_and_every_integer_digit(value, text):
    assert number(value) == text


def test_metric_tree_one_node_per_line():
    text = render_attribute_metrics(_load("metrics_ratio"))

    assert text.splitlines() == [
        "status complete",
        "",
        "store_sales_vs_channel_average = store_sales_amount / channel_average_sales_amount * 100"
        " | 144.239 → 145.32 | +1.08095 (100.0%)",
        "  store_sales_amount [input] | 1019954026 → 1024805507 | +0.68702 (63.6%)",
        "  channel_average_sales_amount = all_channel_sales_amount / 3 | 707127542 → 705206136 | +0.393926 (36.4%)",
        "    all_channel_sales_amount = CASE WHEN store_sales_amount IS NULL AND catalog_sales_amount IS NULL AND web_s…"
        " | 2121382626 → 2115618407 | +0.393926 (36.4%)",
        "      web_sales_amount [input] | 373090522 → 365010650 | +0.552182 (51.1%)",
        "      store_sales_amount [input] | 1019954026 → 1024805507 | -0.331556 (-30.7%)",
        "      catalog_sales_amount [input] | 728338078 → 725802250 | +0.1733 (16.0%)",
    ]
    # What a model does not act on stays out.
    for absent in ("schema_version", "exact_subset", "queries_executed", "store_sales.ss", "date_dim"):
        assert absent not in text


def test_metric_tree_of_a_grouped_target_names_its_row_once():
    text = render_attribute_metrics(_load("metrics_offset_row"))

    assert text.splitlines()[2:] == [
        "inventory_on_hand_variation_next_month = inventory_on_hand_variation 1 month later"
        " | 0.743596 → 1.01516 | +0.271563 (100.0%)",
        "row d_date__month=2000-01-01 / 2001-01-01, inv_item_sk=265, inv_warehouse_sk=1",
        "  (inventory_on_hand_stddev * 1.000 / NULLIF(inventory_avg_on_hand, 0)) @ d_date__month=2000-02-01 / 2001-02-01"
        " | 0.743596 → 1.01516 | +0.271563 (100.0%)",
        "    inventory_on_hand_stddev [input] @ d_date__month=2000-02-01 / 2001-02-01"
        " | 214.341 → 333.987 | +0.38937 (143.4%)",
        "    inventory_avg_on_hand [input] @ d_date__month=2000-02-01 / 2001-02-01 | 288.25 → 329 | -0.117808 (-43.4%)",
    ]


def test_dimension_tree_lists_groups_below_their_parent_with_others():
    text = render_attribute_dimensions(_load("dims_two_levels"))

    assert text.splitlines() == [
        "status complete, accuracy mixed",
        "",
        "store_sales_vs_channel_average by item.i_category > item.i_class | 144.238 → 145.314 | change +1.07549",
        "- Home | 143.428 → 145.888 | +0.248937 (23.1%) | 16 below, none listed",
        "- Music | 144.108 → 145.801 | +0.17489 (16.3%) | 4 below, none listed",
        "- Jewelry | 144.6 → 146.217 | +0.152667 (14.2%) | 16 below, none listed",
        "- Electronics | 144.057 → 145.22 | +0.118653 (11.0%) | 16 below, none listed",
        "- Sports | 144.657 → 145.739 | +0.107405 (10.0%) | 16 below, none listed",
        "- Children | 143.824 → 144.9 | +0.105782 (9.8%) | 4 below",
        "  - infants | 142.752 → 147.015 | +0.101707 (9.5%)",
        "  - 3 others | +0.00407507 (0.4%)",
        "- Books | 145.037 → 145.959 | +0.0888935 (8.3%) | 16 below, none listed",
        "- 3 others | +0.0782667 (7.3%)",
    ]


def test_groups_without_a_value_of_their_own_say_so():
    text = render_attribute_dimensions(_load("dims_rank_by_store"))

    assert text.splitlines()[2:5] == [
        "store_gross_margin_rank_in_category by store.s_store_name | 13 → 1 | change -12",
        "row i_category=Books, i_class=travel",
        "- ation | no own value | -5 (41.7%)",
    ]


def test_a_refusal_is_its_status_and_issue_without_a_tree():
    text = render_attribution(_load("dims_distinct"))

    assert text == (
        "status unsupported\n"
        "issue error insufficient_state: `COUNT(DISTINCT ss_list_price)` counts distinct values: its groups do not"
        " add up to the whole, and the state that would is not fetched"
    )


def test_a_refusal_with_the_target_keeps_its_line_and_the_suggested_action():
    lines = render_attribution(_load("dims_root_only")).splitlines()

    assert lines[0] == "status unsupported"
    assert lines[1].startswith("issue error dimension_not_aligned: in the baseline period the group")
    assert lines[1].endswith("→ exclude the NULL values with `context_filter` (`field IS NOT NULL`)")
    assert lines[2:] == [
        "",
        "store_sales_vs_channel_average by item.i_category > item.i_class | 144.239 → 145.32 | change +1.08095",
    ]


def test_notes_follow_the_issues():
    text = render_attribution(_load("dims_additive"), notes=["the periods differ in length"])

    assert text.splitlines()[:3] == ["status complete", "note the periods differ in length", ""]


def test_missing_values_say_why_and_filters_are_shown_on_the_input():
    result = copy.deepcopy(_load("metrics_ratio"))
    leaf = result["tree"]["children"][0]
    leaf["baseline"] = None
    leaf["baseline_status"] = "missing_row"
    leaf["binding"]["filters"] = ["store_sales.ss_quantity > 0"]

    line = render_attribute_metrics(result).splitlines()[3]

    assert line == (
        "  store_sales_amount [input, filter store_sales.ss_quantity > 0] | missing_row → 1024805507 | +0.68702 (63.6%)"
    )


def test_sampling_error_is_shown_only_where_it_changes_the_digits():
    result = copy.deepcopy(_load("dims_additive"))
    children = result["tree"]["children"]
    children[0]["uncertainty"] = {"sample_stddev": 1.0, "standard_error": 0.001}
    children[1]["uncertainty"] = {"sample_stddev": 9000.0, "standard_error": 1234.5}

    lines = render_attribute_dimensions(result).splitlines()

    assert lines[3] == "- Children | 99319529 → 102535817 | +3216288 (66.3%)"
    assert lines[4] == "- Home | 102083599 → 104823721 | +2740122 (56.5%) ±1200"


def test_a_group_whose_rank_is_within_noise_is_marked():
    result = copy.deepcopy(_load("dims_additive"))
    result["tree"]["children"][0]["rank_unstable"] = True

    assert render_attribute_dimensions(result).splitlines()[3].startswith("- Children ~ |")


@pytest.mark.parametrize(
    "name, summary",
    [
        ("metrics_ratio", "complete · Δ +1.08095 · web_sales_amount +0.552 (51.1%)"),
        ("metrics_offset_row", "complete · Δ +0.271563 · inventory_on_hand_stddev +0.389 (143.4%)"),
        ("dims_two_levels", "complete · Δ +1.07549 · Home +0.249 (23.1%)"),
        ("dims_rank_by_store", "complete · Δ -12 · ation -5 (41.7%)"),
        ("dims_distinct", "unsupported · insufficient_state"),
        ("dims_root_only", "unsupported · dimension_not_aligned"),
    ],
)
def test_summary_line(name, summary):
    assert summarize_attribution_text(render_attribution(_load(name))) == summary


def test_summary_of_anything_else_is_empty():
    assert summarize_attribution_text({"strategy": "term_wise"}) == ""
    assert summarize_attribution_text("rows: 3") == ""
