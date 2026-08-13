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
