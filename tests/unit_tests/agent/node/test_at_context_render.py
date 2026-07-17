# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for AgenticNode._render_context_hint_part — the look-up hints
block for @-references whose detail couldn't be pre-loaded."""

from datus.agent.node.agentic_node import AgenticNode


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
