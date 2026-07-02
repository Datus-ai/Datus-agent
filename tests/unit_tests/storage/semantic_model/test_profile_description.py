import yaml

from datus.storage.semantic_model.profile_description import (
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
