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

RESULTS = Path(__file__).resolve().parents[4] / "data" / "attribution_results"


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
        "store_sales_vs_channel_average | 144.239 → 145.32 | +1.08095 (100.0%)",
        "  store_sales_amount | 1019954026 → 1024805507 | +0.68702 (63.6%)",
        "  channel_average_sales_amount | 707127542 → 705206136 | +0.393926 (36.4%)",
        "    all_channel_sales_amount | 2121382626 → 2115618407 | +0.393926 (36.4%)",
        "      web_sales_amount | 373090522 → 365010650 | +0.552182 (51.1%)",
        "      store_sales_amount | 1019954026 → 1024805507 | -0.331556 (-30.7%)",
        "      catalog_sales_amount | 728338078 → 725802250 | +0.1733 (16.0%)",
        # An input read in two places gets its whole effect on one line, so
        # the model quotes the sum rather than one of the parts.
        "store_sales_amount in all, 2 places | +0.355464 (32.9%)",
    ]
    # What a model does not act on stays out, a metric's formula included:
    # get_metric has it.
    for absent in (
        "schema_version",
        "exact_subset",
        "queries_executed",
        "store_sales.ss",
        "date_dim",
        "channel_average_sales_amount * 100",
        "CASE WHEN",
    ):
        assert absent not in text


def test_an_input_is_named_by_its_metric_even_when_it_cannot_be_queried_as_it():
    # A member with a fill of its own reads the bare aggregate inside its
    # parent: Dosi gives no `metric`, and the input is still named by the
    # metric it is written as, not by its expression.
    leaf = {
        "label": "late_orders",
        "metric": None,
        "expression": "SUM(late)",
        "atom": "SUM(late)",
        "binding": {"datasets": ["orders"], "filters": [], "time_dimension": None},
        "baseline": 58.0,
        "current": 179.0,
        "delta": 121.0,
        "effect": -0.0965352,
        "share": 0.869,
    }
    result = {
        "schema_version": 3,
        "status": "complete",
        "metric": "on_time_rate",
        "issues": [],
        "tree": {
            "label": "on_time_rate",
            "metric": "on_time_rate",
            "expression": "1 - late_orders / orders",
            "baseline": 0.946375,
            "current": 0.835294,
            "delta": -0.111081,
            "effect": -0.111081,
            "share": 1.0,
            "children": [leaf],
        },
    }
    lines = render_attribute_metrics(result).splitlines()
    assert lines[3] == "  late_orders | 58 → 179 | -0.0965352 (86.9%)"


def test_metric_tree_of_a_grouped_target_names_its_row_once():
    text = render_attribute_metrics(_load("metrics_offset_row"))

    assert text.splitlines()[2:] == [
        "inventory_on_hand_variation_next_month | 0.743596 → 1.01516 | +0.271563 (100.0%)",
        "row d_date__month=2000-01-01 / 2001-01-01, inv_item_sk=265, inv_warehouse_sk=1",
        "  (inventory_on_hand_stddev * 1.000 / NULLIF(inventory_avg_on_hand, 0)) @ d_date__month=2000-02-01 / 2001-02-01"
        " | 0.743596 → 1.01516 | +0.271563 (100.0%)",
        "    inventory_on_hand_stddev @ d_date__month=2000-02-01 / 2001-02-01 | 214.341 → 333.987 | +0.38937 (143.4%)",
        "    inventory_avg_on_hand @ d_date__month=2000-02-01 / 2001-02-01 | 288.25 → 329 | -0.117808 (-43.4%)",
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


def test_missing_values_say_why_and_a_named_input_carries_its_condition_in_its_name():
    result = copy.deepcopy(_load("metrics_ratio"))
    leaf = result["tree"]["children"][0]
    leaf["baseline"] = None
    leaf["baseline_status"] = "missing_row"
    leaf["binding"]["filters"] = ["store_sales.ss_quantity > 0"]

    line = render_attribute_metrics(result).splitlines()[3]

    assert line == "  store_sales_amount | missing_row → 1024805507 | +0.68702 (63.6%)"


def test_an_input_with_no_name_is_its_aggregate_under_its_condition():
    result = copy.deepcopy(_load("metrics_ratio"))
    leaf = result["tree"]["children"][0]
    leaf["metric"] = None
    leaf["label"] = leaf["atom"] = leaf["expression"] = "SUM(ss_sales_price)"
    leaf["binding"]["filters"] = ["store_sales.ss_quantity > 0"]

    line = render_attribute_metrics(result).splitlines()[3]

    assert line == (
        "  (SUM(ss_sales_price) FILTER (WHERE store_sales.ss_quantity > 0)) | 1019954026 → 1024805507 | +0.68702 (63.6%)"
    )


def test_sampling_error_is_shown_only_where_it_changes_the_digits():
    result = copy.deepcopy(_load("dims_additive"))
    children = result["tree"]["children"]
    children[0]["uncertainty"] = {"sample_stddev": 1.0, "standard_error": 0.001}
    children[1]["uncertainty"] = {"sample_stddev": 9000.0, "standard_error": 1234.5}

    lines = render_attribute_dimensions(result).splitlines()

    assert lines[3] == "- Children | 99319529 → 102535817 | +3216288 (66.3%)"
    assert lines[4] == "- Home | 102083599 → 104823721 | +2740122 (56.5%) ±1200"


def test_sampling_error_counts_against_the_last_integer_digit_of_a_large_effect():
    # +3216288 is written to the unit, so an error of 5 changes a written digit.
    result = copy.deepcopy(_load("dims_additive"))
    result["tree"]["children"][0]["uncertainty"] = {"sample_stddev": 40.0, "standard_error": 5.0}

    assert render_attribute_dimensions(result).splitlines()[3] == (
        "- Children | 99319529 → 102535817 | +3216288 (66.3%) ±5"
    )


def test_a_text_group_that_would_read_as_null_or_break_the_line_is_quoted():
    result = copy.deepcopy(_load("dims_additive"))
    children = result["tree"]["children"]
    children[0]["label"], children[0]["key"] = "NULL", {"item.i_category": "NULL"}
    children[1]["label"], children[1]["key"] = "it's\nnew", {"item.i_category": "it's\nnew"}
    # A text that is already quoted is quoted again, so it does not read as the text NULL.
    children[2]["label"], children[2]["key"] = "'NULL'", {"item.i_category": "'NULL'"}

    lines = render_attribute_dimensions(result).splitlines()

    assert lines[3].startswith("- 'NULL' | ")
    assert lines[4].startswith("- 'it''s\\nnew' | ")
    assert lines[5].startswith("- '''NULL''' | ")


def test_a_null_group_under_a_text_null_group_is_still_null():
    # The key below holds the level above; only the level's own field decides.
    result = copy.deepcopy(_load("dims_two_levels"))
    parent = result["tree"]["children"][5]
    parent["label"], parent["key"] = "(null)", {"i_category": "(null)"}
    child = parent["children"][0]
    child["label"], child["key"] = "(null)", {"i_category": "(null)", "i_class": None}

    lines = render_attribute_dimensions(result).splitlines()

    assert lines[8].startswith("- (null) | ")
    assert lines[9].startswith("  - NULL | ")


def test_every_line_boundary_in_a_label_is_escaped():
    result = copy.deepcopy(_load("dims_additive"))
    plain = render_attribute_dimensions(result).splitlines()
    group = result["tree"]["children"][0]
    group["label"], group["key"] = "North\u2028East", {"i_category": "North\u2028East"}

    text = render_attribute_dimensions(result)
    lines = text.splitlines()

    assert len(lines) == len(plain)
    assert lines[3].startswith("- 'North\\u2028East' | ")
    assert summarize_attribution_text(text) == "complete · Δ +4851480 · 'North\\u2028East' +3216288 (66.3%)"


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


def test_summary_of_a_change_of_zero_names_the_largest_input():
    text = "\n".join(
        [
            "status complete",
            "",
            "net | 140 → 140 | 0 (n/a)",
            "  x_total | 150 → 160 | +10 (n/a)",
            "  (SUM(y)) | 10 → 20 | -10 (n/a)",
        ]
    )
    assert summarize_attribution_text(text) == "complete · Δ 0 · x_total +10 (n/a)"


def test_summary_keeps_a_group_label_that_holds_the_separator():
    text = "\n".join(
        [
            "status complete",
            "",
            "sales by item.i_category | 100 → 130 | change +30",
            "- Home | Garden | 40 → 65 | +25 (83.3%) | 3 below",
            "- Books | 60 → 65 | +5 (16.7%)",
        ]
    )
    assert summarize_attribution_text(text) == "complete · Δ +30 · Home | Garden +25 (83.3%)"


def test_summary_keeps_an_unnamed_input_whole_and_adds_up_its_rows():
    text = "\n".join(
        [
            "status complete",
            "",
            "m | 100 → 105 | +5 (100.0%)",
            "row d=1 / 2",
            "  (SUM(x) FILTER (WHERE r = 'A @ West')) | 10 → 20 | +10 (200.0%)",
            "  (SUM(x) FILTER (WHERE r = 'A @ East')) | 20 → 11 | -9 (-180.0%)",
            "  y_total | 1 → 3 | +2 (40.0%)",
            "  (SUM(z)) @ d=0 / 1 | 3 → 4 | +1 (20.0%)",
            "  (SUM(z)) @ d=2 / 3 | 4 → 5 | +1 (20.0%)",
        ]
    )
    # The two `A @ …` inputs stay apart; `SUM(z)` at two rows is one input.
    assert summarize_attribution_text(text) == "complete · Δ +5 · (SUM(x) FILTER (WHERE r = 'A @ West')) +10 (200.0%)"


def test_summary_of_anything_else_is_empty():
    assert summarize_attribution_text({"strategy": "term_wise"}) == ""
    assert summarize_attribution_text("rows: 3") == ""
