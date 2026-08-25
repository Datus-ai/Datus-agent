# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import pytest

from datus.tools.semantic_tools.osi_document import load_osi_document
from datus.utils.exceptions import DatusException


def test_load_osi_document_reads_core_fields_and_datus_hints(tmp_path):
    model = tmp_path / "orders.yml"
    model.write_text(
        """
version: 0.2.0.dev0
semantic_model:
  - name: orders_model
    datasets:
      - name: orders
        source: analytics.orders
        primary_key: [order_id]
        fields:
          - name: order_id
            expression: {dialects: [{dialect: ANSI_SQL, expression: order_id}]}
          - name: order_date
            expression: {dialects: [{dialect: ANSI_SQL, expression: order_date}]}
            dimension: {is_time: true}
            custom_extensions:
              - vendor_name: DATUS
                data: '{"v":"1.2","time_granularity":"day"}'
          - name: status
            expression: {dialects: [{dialect: ANSI_SQL, expression: status}]}
            dimension: {}
    metrics:
      - name: running_revenue
        expression:
          dialects: [{dialect: ANSI_SQL, expression: 'SUM(orders.amount)'}]
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v":"1.2","window":{"type":"cumulative","function":"sum"},"subject_path":["sales","revenue"]}'
""".lstrip(),
        encoding="utf-8",
    )

    document = load_osi_document(str(model), "orders_model")

    assert document.name == "orders_model"
    assert document.datasets[0].source.table == "analytics.orders"
    assert document.datasets[0].time_dimension.name == "order_date"
    assert document.datasets[0].time_dimension.granularity == "day"
    assert [item.name for item in document.datasets[0].fields] == ["order_id", "order_date", "status"]
    assert [item.name for item in document.datasets[0].dimensions] == ["status"]
    assert document.datasets[0].fields[0].is_dimension is False
    assert document.datasets[0].fields[1].is_dimension is True
    assert document.datasets[0].fields[2].is_dimension is True
    assert document.metrics[0].dataset == "orders"
    assert document.metrics[0].subject_path == ["sales", "revenue"]
    assert document.metrics[0].window == {"type": "cumulative", "function": "sum"}


def test_load_osi_document_does_not_scan_sibling_artifacts(tmp_path):
    model = tmp_path / "model.yml"
    model.write_text(
        """
version: 0.2.0.dev0
semantic_model:
  - name: finance
    datasets:
      - name: budgets
        source: budgets
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "invalid-sibling.yml").write_text("semantic_model: [", encoding="utf-8")

    document = load_osi_document(str(model), "finance")

    assert [item.name for item in document.datasets] == ["budgets"]
    assert document.metrics == []


def test_load_osi_document_preserves_authored_source_types(tmp_path):
    model = tmp_path / "sources.yml"
    model.write_text(
        """
version: 0.2.0.dev0
semantic_model:
  - name: source_model
    datasets:
      - name: mapped_table
        source: {table: analytics.orders}
      - name: whitespace_query
        source: "SELECT\\tCOUNT(*) FROM analytics.orders"
      - name: extension_query
        source: "VALUES (1)"
        custom_extensions:
          - vendor_name: DATUS
            data: '{"source_type":"query"}'
      - name: extension_table
        source: "SELECT_table"
        custom_extensions:
          - vendor_name: DATUS
            data: '{"source_type":"table"}'
""".lstrip(),
        encoding="utf-8",
    )

    document = load_osi_document(str(model), "source_model")

    assert document.datasets[0].source.table == "analytics.orders"
    assert document.datasets[1].source.query == "SELECT\tCOUNT(*) FROM analytics.orders"
    assert document.datasets[2].source.query == "VALUES (1)"
    assert document.datasets[3].source.table == "SELECT_table"


def test_load_osi_document_leaves_multiple_time_fields_unresolved(tmp_path):
    model = tmp_path / "events.yml"
    model.write_text(
        """
version: 0.2.0.dev0
semantic_model:
  - name: events_model
    datasets:
      - name: events
        source: events
        fields:
          - name: created_at
            dimension: {is_time: true}
          - name: completed_at
            dimension: {is_time: true}
          - name: invalid_dimension_marker
            dimension: true
""".lstrip(),
        encoding="utf-8",
    )

    document = load_osi_document(str(model), "events_model")

    assert document.datasets[0].time_dimension is None
    assert [item.name for item in document.datasets[0].dimensions] == [
        "created_at",
        "completed_at",
    ]
    assert document.datasets[0].fields[-1].is_dimension is False


def test_load_osi_document_rejects_duplicate_model_declarations(tmp_path):
    model = tmp_path / "model.yml"
    model.write_text(
        """
version: 0.2.0.dev0
semantic_model:
  - name: finance
    datasets: []
---
version: 0.2.0.dev0
semantic_model:
  - name: finance
    metrics: []
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(DatusException, match="declared 2 times"):
        load_osi_document(str(model), "finance")


def test_load_osi_document_reads_composite_unique_keys(tmp_path):
    """A dataset can declare uniqueness without a DDL primary key to
    transcribe, and such a key routinely spans several columns."""
    model = tmp_path / "activity.yml"
    model.write_text(
        """
version: 0.2.0.dev0
semantic_model:
  - name: ops
    datasets:
      - name: activity
        source: mart.activity
        unique_keys:
          - [ac_code, subject_seq, product_code]
          - [surrogate_id]
        fields:
          - name: ac_code
            expression: {dialects: [{dialect: ANSI_SQL, expression: ac_code}]}
"""
    )

    dataset = load_osi_document(str(model), semantic_model_name="ops").datasets[0]

    assert dataset.unique_keys == [["ac_code", "subject_seq", "product_code"], ["surrogate_id"]]
    assert dataset.primary_key == []


def test_load_osi_document_defaults_unique_keys_to_empty(tmp_path):
    model = tmp_path / "plain.yml"
    model.write_text(
        """
version: 0.2.0.dev0
semantic_model:
  - name: ops
    datasets:
      - name: activity
        source: mart.activity
        fields:
          - name: ac_code
            expression: {dialects: [{dialect: ANSI_SQL, expression: ac_code}]}
"""
    )

    assert load_osi_document(str(model), semantic_model_name="ops").datasets[0].unique_keys == []
