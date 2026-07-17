# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for AgenticNode._render_context_hint_part — the look-up hints
block for @-references whose detail couldn't be pre-loaded."""

from types import SimpleNamespace

from datus.agent.node.agentic_node import AgenticNode
from datus.schemas.node_models import Metric, ReferenceSql


def _render(**fields) -> str:
    """Render at-context parts for a bare input carrying only *fields*."""
    node = AgenticNode.__new__(AgenticNode)  # bypass __init__; method only reads getattr
    attrs = {"external_knowledge": "", "schemas": None, "metrics": None, "reference_sql": None, "context_hints": None}
    attrs.update(fields)
    return "\n\n".join(node._render_at_context_parts(SimpleNamespace(**attrs)))


def test_metric_block_includes_subject_path_and_definition():
    m = Metric(
        name="aov",
        description="avg order value",
        subject_path=["Commerce", "Orders"],
        metric_type="ratio",
        measure_expr="SUM(amount)/COUNT(*)",
        dimensions=["platform", "country"],
    )
    out = _render(metrics=[m])
    assert "## Referenced metrics" in out
    assert "subject_path: Commerce/Orders" in out
    assert "avg order value" in out
    assert "type: ratio" in out
    assert "measure: SUM(amount)/COUNT(*)" in out
    assert "dimensions: platform, country" in out


def test_reference_sql_block_includes_subject_path_and_sql():
    r = ReferenceSql(name="raw_customers", sql="select * from raw_customers", subject_path=["main"])
    out = _render(reference_sql=[r])
    assert "## Referenced SQL" in out
    assert "subject_path: main" in out
    assert "```sql\nselect * from raw_customers\n```" in out


def test_empty_hints_render_nothing():
    assert AgenticNode._render_context_hint_part(None) == ""
    assert AgenticNode._render_context_hint_part([]) == ""


def test_metric_hint_points_at_get_metrics():
    out = AgenticNode._render_context_hint_part(
        [{"kind": "metric", "name": "aov", "subject_path": ["Commerce", "Orders"]}]
    )
    assert "## Referenced items to look up" in out
    assert "get_metrics(subject_path=['Commerce', 'Orders'], name=\"aov\")" in out


def test_reference_sql_hint_points_at_get_reference_sql():
    out = AgenticNode._render_context_hint_part(
        [{"kind": "reference_sql", "name": "raw_customers", "subject_path": ["main"]}]
    )
    assert "get_reference_sql(subject_path=['main'], name=\"raw_customers\")" in out


def test_knowledge_hint_has_no_tool_call():
    out = AgenticNode._render_context_hint_part(
        [{"kind": "knowledge", "name": "gmv", "subject_path": ["Domain", "Glossary"]}]
    )
    # No get_* tool exists for knowledge — point at the subject tree instead.
    assert "get_metrics" not in out and "get_reference_sql" not in out
    assert "list_subject_tree" in out
    assert "Domain/Glossary/gmv" in out
