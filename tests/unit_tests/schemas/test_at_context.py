# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/schemas/at_context.py.

Covers the shared @-context carrier:
- AtContextInput declares the four reference fields with safe defaults.
- apply_at_context sets / skips / preserves fields correctly.
- The anti-leak contract: every subagent node input inherits AtContextInput,
  so a new reference kind added to the mixin reaches all of them at once.
"""

import pytest

from datus.schemas.ask_metrics_agentic_node_models import AskMetricsNodeInput
from datus.schemas.at_context import AtContextInput, apply_at_context
from datus.schemas.base import BaseInput
from datus.schemas.chat_agentic_node_models import ChatNodeInput
from datus.schemas.explore_agentic_node_models import ExploreNodeInput
from datus.schemas.feedback_agentic_node_models import FeedbackNodeInput
from datus.schemas.gen_dashboard_agentic_node_models import GenDashboardNodeInput
from datus.schemas.gen_report_agentic_node_models import GenReportNodeInput
from datus.schemas.gen_skill_agentic_node_models import SkillCreatorNodeInput
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput
from datus.schemas.gen_visual_dashboard_models import GenVisualDashboardNodeInput
from datus.schemas.gen_visual_report_models import GenVisualReportNodeInput
from datus.schemas.node_models import Metric, ReferenceSql, TableSchema
from datus.schemas.scheduler_agentic_node_models import SchedulerNodeInput
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput

# Every input a subagent funnel (create_node_input / _build_node_input) may
# construct. Each MUST inherit AtContextInput so @-context reaches its node.
SUBAGENT_INPUTS = [
    ChatNodeInput,
    GenSQLNodeInput,
    SemanticNodeInput,
    ExploreNodeInput,
    AskMetricsNodeInput,
    GenReportNodeInput,
    GenVisualReportNodeInput,
    GenVisualDashboardNodeInput,
    GenDashboardNodeInput,
    SqlSummaryNodeInput,
    SchedulerNodeInput,
    SkillCreatorNodeInput,
    FeedbackNodeInput,
]

AT_CONTEXT_FIELDS = ("schemas", "metrics", "reference_sql", "external_knowledge")


def _table() -> TableSchema:
    return TableSchema(
        identifier="c.d.t",
        catalog_name="c",
        database_name="d",
        schema_name="",
        table_name="t",
        definition="CREATE TABLE t(x int)",
    )


@pytest.mark.parametrize("input_cls", SUBAGENT_INPUTS)
def test_subagent_input_inherits_at_context(input_cls):
    """Anti-leak guard: no subagent input may drop the @-context contract."""
    assert issubclass(input_cls, AtContextInput), f"{input_cls.__name__} must inherit AtContextInput"
    for field in AT_CONTEXT_FIELDS:
        assert field in input_cls.model_fields, f"{input_cls.__name__} missing @-context field '{field}'"


def test_at_context_defaults_are_empty():
    si = SemanticNodeInput(user_message="hi")
    assert si.schemas is None
    assert si.metrics is None
    assert si.reference_sql is None
    assert si.external_knowledge == ""


def test_apply_at_context_sets_fields():
    si = SemanticNodeInput(user_message="hi")
    ts = _table()
    m = Metric(name="gmv")
    r = ReferenceSql(name="q", sql="select 1")
    apply_at_context(si, schemas=[ts], metrics=[m], reference_sql=[r], external_knowledge="biz")
    assert si.schemas == [ts]
    assert si.metrics == [m]
    assert si.reference_sql == [r]
    assert si.external_knowledge == "biz"


def test_apply_at_context_none_preserves_existing():
    si = SemanticNodeInput(user_message="hi", external_knowledge="keep")
    apply_at_context(si, schemas=None, metrics=None, reference_sql=None, external_knowledge=None)
    assert si.schemas is None
    assert si.external_knowledge == "keep"


def test_apply_at_context_skips_unknown_fields():
    """A plain BaseInput lacks the fields; helper must not raise (hasattr guard)."""

    class Bare(BaseInput):
        user_message: str

    bare = Bare(user_message="x")
    result = apply_at_context(bare, schemas=[_table()], external_knowledge="biz")
    assert result is bare
    assert not hasattr(bare, "schemas")


def test_reference_sql_accepts_historical_alias():
    """reference_sql keeps the legacy ``historical_sql`` alias on the mixin."""
    node = ChatNodeInput.model_validate({"user_message": "hi", "historical_sql": [{"name": "q", "sql": "select 1"}]})
    assert node.reference_sql and node.reference_sql[0].name == "q"
