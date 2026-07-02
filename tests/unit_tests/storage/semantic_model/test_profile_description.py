import yaml

from datus.storage.semantic_model.profile_description import (
    _clip_text,
    _duration_profile_phrases,
    _profile_scalar,
    build_column_observed_profile,
    build_table_observed_profile,
    merge_observed_profile,
    refresh_metricflow_yaml_descriptions,
    refresh_osi_yaml_descriptions,
)


def test_merge_observed_profile_replaces_generated_suffix():
    description = "Order status. Observed profile: old values."

    updated = merge_observed_profile(description, "4 distinct non-null values; common values include paid, refund")

    assert updated == (
        "Order status. Observed profile: 4 distinct non-null values; common values include paid, refund."
    )


def test_merge_observed_profile_clamps_long_base_description():
    description = "x" * 80

    updated = merge_observed_profile(description, "y" * 80, max_chars=60)

    assert len(updated) <= 60


def test_merge_observed_profile_handles_empty_observed_and_no_base_clipping():
    assert merge_observed_profile("Order status. Observed profile: old values.", "") == "Order status"

    updated = merge_observed_profile("", "x" * 80, max_chars=24)

    assert updated == "Observed profile: xxx..."


def test_build_column_observed_profile_for_categorical_usage():
    observed = build_column_observed_profile(
        {
            "kind": "categorical",
            "stats": {"distinct_count": 4, "null_rate": 0.02},
            "top_values": [{"value": "paid"}, {"value": "refund"}, {"value": "cancelled"}],
        },
        field_usage={"filter_count": 3, "operators": ["="]},
    )

    assert "4 distinct non-null values" in observed
    assert "common values include paid, refund, cancelled" in observed
    assert "null rate 2.0%" in observed
    assert "categorical filter" in observed


def test_build_profile_descriptions_cover_distribution_fields():
    table_observed = build_table_observed_profile(
        {
            "query_count": 4,
            "common_business_filter_templates": [
                {"fields": ["status"], "operator": "="},
                {"fields": ["created_at"], "operator": "BETWEEN"},
            ],
            "data_distribution_profile": {
                "row_count": 100,
                "date_duration_profiles": [
                    {
                        "left_column": "opened_at",
                        "right_column": "closed_at",
                        "delta_days": {"p50": 3, "p90": 5},
                    }
                ],
            },
        }
    )
    numeric_observed = build_column_observed_profile(
        {
            "kind": "numeric",
            "stats": {"min_value": 10, "max_value": 99, "null_rate": 0.1},
            "percentiles": {"p50": 50, "p90": 90},
        },
        field_usage={"aggregate_count": 2},
    )
    temporal_observed = build_column_observed_profile(
        {
            "kind": "temporal",
            "stats": {"min_value": "2025-01-01", "max_value": "2025-01-10"},
            "temporal_summary": {"freshness_days_from_profile_date": 2},
        }
    )
    join_observed = build_column_observed_profile(
        {"kind": "categorical", "stats": {"distinct_count": 3}},
        join_profile={"referential_coverage": 0.75, "join_cardinality_hint": "many_to_one_or_one_to_one"},
    )

    assert "observed row count 100" in table_observed
    assert "opened_at to closed_at p50 3 days, p90 5 days" in table_observed
    assert "common filters use status, created_at" in table_observed
    assert "observed range 10-99" in numeric_observed
    assert "p50 50, p90 90" in numeric_observed
    assert "null rate 10.0%" in numeric_observed
    assert "latest value 2 days before profiling" in temporal_observed
    assert "referential coverage 75.0%" in join_observed
    assert "many to one or one to one" in join_observed


def test_build_column_observed_profile_formats_future_freshness():
    observed = build_column_observed_profile(
        {
            "kind": "temporal",
            "stats": {"min_value": "2026-01-01", "max_value": "2026-10-31"},
            "temporal_summary": {"freshness_days_from_profile_date": -121},
        }
    )

    assert "latest value 121 days after profiling" in observed


def test_profile_description_edge_cases_for_usage_and_helpers():
    temporal_observed = build_column_observed_profile(
        {
            "kind": "temporal",
            "stats": {},
            "temporal_summary": {"freshness_days_from_profile_date": 0},
        }
    )
    range_observed = build_column_observed_profile(
        {"kind": "categorical", "stats": {"distinct_count": "bad", "null_rate": "bad"}},
        field_usage={"filter_count": 1, "group_by_count": 1, "operators": [">"]},
    )
    generic_observed = build_column_observed_profile(
        {"kind": "unknown", "stats": {}},
        field_usage={"filter_count": 1, "operators": ["LIKE"]},
    )

    assert temporal_observed == "latest value on profiling date"
    assert "frequently used as a range filter" in range_observed
    assert "commonly grouped" in range_observed
    assert generic_observed == "frequently filtered"
    assert _duration_profile_phrases(["invalid", {"left_column": "a", "right_column": "b", "delta_days": {}}]) == []
    assert _profile_scalar(1.23456789) == "1.23457"
    assert _clip_text("abcdef", 0) == ""
    assert _clip_text("abcdef", 2) == ".."


def test_refresh_metricflow_yaml_descriptions_patches_table_and_column():
    docs = [
        yaml.safe_load(
            """
data_source:
  name: orders
  description: Orders table.
  sql_table: marts.orders
  dimensions:
    - name: status
      expr: status
      type: CATEGORICAL
      description: Order status.
"""
        )
    ]
    evidence = {
        "tables": {
            "orders": {
                "query_count": 2,
                "field_usage_statistics": {
                    "status": {"filter_count": 2, "operators": ["="]},
                },
                "data_distribution_profile": {
                    "row_count": 10,
                    "columns": {
                        "status": {
                            "kind": "categorical",
                            "stats": {"distinct_count": 3},
                            "top_values": [{"value": "paid"}, {"value": "refund"}],
                        }
                    },
                },
            }
        }
    }

    changed = refresh_metricflow_yaml_descriptions(docs, evidence)

    assert changed == 2
    data_source = docs[0]["data_source"]
    assert "observed row count 10" in data_source["description"]
    assert "common values include paid, refund" in data_source["dimensions"][0]["description"]


def test_refresh_metricflow_yaml_descriptions_skips_invalid_docs_and_missing_profiles():
    docs = [
        None,
        {"data_source": "not-a-dict"},
        {"data_source": {"name": "orders", "description": "Orders table."}},
    ]

    changed = refresh_metricflow_yaml_descriptions(docs, {"tables": {"customers": {}}})

    assert changed == 0
    assert docs[-1]["data_source"]["description"] == "Orders table."


def test_refresh_metricflow_yaml_descriptions_matches_join_profiles_by_column():
    docs = [
        yaml.safe_load(
            """
data_source:
  name: orders
  dimensions:
    - name: customer_id
      expr: customer_id
      description: Customer identifier.
"""
        )
    ]
    evidence = {
        "tables": {
            "orders": {
                "data_distribution_profile": {
                    "columns": {
                        "customer_id": {
                            "kind": "categorical",
                            "stats": {"distinct_count": 2},
                        }
                    },
                    "join_relationship_profiles": [
                        "invalid",
                        {
                            "source_column": "orders.customer_id",
                            "target_column": "customers.id",
                            "referential_coverage": 1.0,
                        },
                    ],
                }
            }
        }
    }

    changed = refresh_metricflow_yaml_descriptions(docs, evidence)

    assert changed == 1
    assert "referential coverage 100.0%" in docs[0]["data_source"]["dimensions"][0]["description"]


def test_refresh_osi_yaml_descriptions_patches_dataset_and_dimension():
    docs = [
        yaml.safe_load(
            """
semantic_model:
  - name: commerce
    datasets:
      - name: orders
        description: Orders dataset.
        source:
          table: marts.orders
        dimensions:
          - name: status
            description: Order status.
"""
        )
    ]
    evidence = {
        "tables": {
            "orders": {
                "query_count": 1,
                "field_usage_statistics": {"status": {"filter_count": 1, "operators": ["="]}},
                "data_distribution_profile": {
                    "columns": {
                        "status": {
                            "kind": "categorical",
                            "stats": {"distinct_count": 2},
                            "top_values": [{"value": "paid"}],
                        }
                    }
                },
            }
        }
    }

    changed = refresh_osi_yaml_descriptions(docs, evidence)

    dataset = docs[0]["semantic_model"][0]["datasets"][0]
    assert changed == 2
    assert "referenced by 1 historical query" in dataset["description"]
    assert "2 distinct non-null values" in dataset["dimensions"][0]["description"]


def test_refresh_osi_yaml_descriptions_patches_dataset_fields():
    docs = [
        yaml.safe_load(
            """
semantic_model:
  - name: commerce
    datasets:
      - name: orders
        description: Orders dataset.
        source: marts.orders
        fields:
          - name: amount
            expression:
              dialects:
                - dialect: ANSI_SQL
                  expression: amount
            description: Order amount.
"""
        )
    ]
    evidence = {
        "tables": {
            "orders": {
                "query_count": 1,
                "data_distribution_profile": {
                    "row_count": 10,
                    "columns": {
                        "amount": {
                            "kind": "numeric",
                            "stats": {"min_value": 1, "max_value": 99},
                            "percentiles": {"p50": 50, "p90": 90},
                        }
                    },
                },
            }
        }
    }

    changed = refresh_osi_yaml_descriptions(docs, evidence)

    dataset = docs[0]["semantic_model"][0]["datasets"][0]
    assert changed == 2
    assert "observed row count 10" in dataset["description"]
    assert "observed range 1-99" in dataset["fields"][0]["description"]


def test_refresh_osi_yaml_descriptions_skips_nonmatching_dataset():
    docs = [
        None,
        {"semantic_model": "not-a-list"},
        {"datasets": [{"name": "orders", "description": "Orders dataset."}]},
    ]

    changed = refresh_osi_yaml_descriptions(docs, {"tables": {"customers": {}}})

    assert changed == 0
    assert docs[-1]["datasets"][0]["description"] == "Orders dataset."
