# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus/schemas/analysis_artifacts.py``.

Pins the validation contract for every schema written under the
``analysis/`` directory and the per-query ``reasoning.json`` sidecar.
Each schema is exercised for:

* Round-trip serialization (model_validate → model_dump).
* Required-field constraints (``min_length`` / ``pattern``) the LLM
  finalize call is expected to honor.
* Numeric / cardinality bounds (``ge`` / ``le`` / ``min_length`` /
  ``max_length``) — these prevent garbage entries from poisoning the
  follow-up subagent.
* Slug pattern matches the same character set every other artifact id
  in the codebase uses.
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from datus.schemas.analysis_artifacts import (
    ANALYSIS_SLUG_PATTERN,
    FinalizeAnalysisOutput,
    Insight,
    Interpretation,
    ReasoningStep,
    SubjectAssetRef,
    SubjectRefIds,
    SubjectRefs,
    SuggestedQuestion,
)

# --------------------------------------------------------------------------- #
# SubjectRefIds                                                               #
# --------------------------------------------------------------------------- #


class TestSubjectRefIds:
    def test_defaults_are_empty_buckets(self):
        refs = SubjectRefIds()
        assert refs.metrics == []
        assert refs.reference_sql == []
        assert refs.ext_knowledge == []

    def test_round_trip(self):
        refs = SubjectRefIds(metrics=["m_orders"], reference_sql=["rs_top_q"], ext_knowledge=["kb_rules"])
        dumped = refs.model_dump()
        restored = SubjectRefIds.model_validate(dumped)
        assert restored == refs

    def test_extra_field_rejected(self):
        # ``extra=forbid`` keeps future LLM hallucinations from silently
        # landing in the file.
        with pytest.raises(ValidationError):
            SubjectRefIds.model_validate({"metrics": [], "unknown_bucket": ["x"]})


# --------------------------------------------------------------------------- #
# ReasoningStep                                                               #
# --------------------------------------------------------------------------- #


def _full_reasoning_payload(**overrides):
    base = {
        "name": "sales_by_store",
        "goal": "distribution of high-risk signups by month",
        "hypothesis": "high-risk signups cluster around promotional campaigns",
        "uses": {"metrics": ["m_signups"], "reference_sql": [], "ext_knowledge": []},
        "caveats": "Excludes test accounts (signup_email LIKE '%@example.com').",
        "datasource": "primary_pg",
        "created_at": "2026-05-14T10:00:00Z",
    }
    base.update(overrides)
    return base


class TestReasoningStep:
    def test_round_trip_full(self):
        step = ReasoningStep.model_validate(_full_reasoning_payload())
        dumped = step.model_dump()
        restored = ReasoningStep.model_validate(dumped)
        assert restored == step
        assert restored.uses.metrics == ["m_signups"]

    def test_round_trip_minimal(self):
        # ``uses`` / ``caveats`` carry safe defaults.
        minimal = {
            "name": "sales_by_store",
            "goal": "list stores",
            "hypothesis": "stores form a flat list",
            "datasource": "default",
            "created_at": "2026-05-14T10:00:00Z",
        }
        step = ReasoningStep.model_validate(minimal)
        assert step.uses == SubjectRefIds()
        assert step.caveats == ""

    def test_empty_goal_rejected(self):
        with pytest.raises(ValidationError) as exc:
            ReasoningStep.model_validate(_full_reasoning_payload(goal=""))
        assert "goal" in str(exc.value)

    def test_empty_hypothesis_rejected(self):
        with pytest.raises(ValidationError) as exc:
            ReasoningStep.model_validate(_full_reasoning_payload(hypothesis=""))
        assert "hypothesis" in str(exc.value)

    @pytest.mark.parametrize("bad_name", ["Bad-Slug", "UPPER", "with space", "中文", "a" * 65, ""])
    def test_invalid_slug_pattern_rejected(self, bad_name: str):
        with pytest.raises(ValidationError):
            ReasoningStep.model_validate(_full_reasoning_payload(name=bad_name))

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            ReasoningStep.model_validate(_full_reasoning_payload(unknown="x"))


# --------------------------------------------------------------------------- #
# Interpretation                                                              #
# --------------------------------------------------------------------------- #


def _full_interpretation_payload(**overrides):
    base = {
        "audience": ["Ops managers", "Finance leads"],
        "goal": "Understand quarterly revenue movement by region.",
        "focus_questions": [
            "Which regions grew fastest?",
            "Where did revenue contract?",
        ],
        "last_updated": "2026-05-14T10:00:00Z",
    }
    base.update(overrides)
    return base


class TestInterpretation:
    def test_round_trip(self):
        interp = Interpretation.model_validate(_full_interpretation_payload())
        restored = Interpretation.model_validate(interp.model_dump())
        assert restored == interp

    def test_empty_audience_rejected(self):
        with pytest.raises(ValidationError):
            Interpretation.model_validate(_full_interpretation_payload(audience=[]))

    def test_empty_focus_questions_rejected(self):
        with pytest.raises(ValidationError):
            Interpretation.model_validate(_full_interpretation_payload(focus_questions=[]))

    def test_empty_goal_rejected(self):
        with pytest.raises(ValidationError):
            Interpretation.model_validate(_full_interpretation_payload(goal=""))


# --------------------------------------------------------------------------- #
# Insight                                                                     #
# --------------------------------------------------------------------------- #


def _full_insight_payload(**overrides):
    base = {
        "id": "revenue_dipped_in_eu",
        "title": "EU revenue dipped 8% MoM in March",
        "summary": "March EU revenue dipped 8% MoM, driven by APAC over-shipment in February.",
        "confidence": 0.7,
        "evidence_queries": ["rev_by_region_monthly"],
        "informed_by_knowledge": ["kb_shipment_cycles"],
    }
    base.update(overrides)
    return base


class TestInsight:
    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_confidence_within_bounds_accepted(self, confidence: float):
        insight = Insight.model_validate(_full_insight_payload(confidence=confidence))
        assert insight.confidence == confidence

    def test_round_trip(self):
        insight = Insight.model_validate(_full_insight_payload())
        restored = Insight.model_validate(insight.model_dump())
        assert restored == insight

    def test_informed_by_knowledge_defaults_to_empty(self):
        payload = _full_insight_payload()
        payload.pop("informed_by_knowledge")
        insight = Insight.model_validate(payload)
        assert insight.informed_by_knowledge == []

    @pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0, -5.0])
    def test_out_of_range_confidence_rejected(self, confidence: float):
        with pytest.raises(ValidationError):
            Insight.model_validate(_full_insight_payload(confidence=confidence))

    def test_missing_evidence_queries_rejected(self):
        with pytest.raises(ValidationError):
            Insight.model_validate(_full_insight_payload(evidence_queries=[]))

    @pytest.mark.parametrize("bad_id", ["Bad-Slug", "中文", ""])
    def test_invalid_id_pattern_rejected(self, bad_id: str):
        with pytest.raises(ValidationError):
            Insight.model_validate(_full_insight_payload(id=bad_id))


# --------------------------------------------------------------------------- #
# SuggestedQuestion                                                           #
# --------------------------------------------------------------------------- #


def _full_suggested_question_payload(**overrides):
    base = {
        "question": "Which regions drove the March dip?",
        "related_queries": ["rev_by_region_monthly"],
        "related_insight": "revenue_dipped_in_eu",
        "priority": 0.6,
    }
    base.update(overrides)
    return base


class TestSuggestedQuestion:
    def test_round_trip(self):
        sq = SuggestedQuestion.model_validate(_full_suggested_question_payload())
        restored = SuggestedQuestion.model_validate(sq.model_dump())
        assert restored == sq

    def test_related_insight_none_accepted(self):
        sq = SuggestedQuestion.model_validate(_full_suggested_question_payload(related_insight=None))
        assert sq.related_insight is None

    def test_related_queries_default_empty(self):
        payload = _full_suggested_question_payload()
        payload.pop("related_queries")
        sq = SuggestedQuestion.model_validate(payload)
        assert sq.related_queries == []

    @pytest.mark.parametrize("priority", [-0.1, 1.5, 10.0])
    def test_out_of_range_priority_rejected(self, priority: float):
        with pytest.raises(ValidationError):
            SuggestedQuestion.model_validate(_full_suggested_question_payload(priority=priority))

    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            SuggestedQuestion.model_validate(_full_suggested_question_payload(question=""))


# --------------------------------------------------------------------------- #
# SubjectAssetRef / SubjectRefs                                               #
# --------------------------------------------------------------------------- #


class TestSubjectAssetRef:
    def test_defaults_for_unresolved_fields(self):
        ref = SubjectAssetRef(id="m_orders", name="Orders metric")
        assert ref.definition_or_summary == ""
        assert ref.source == ""

    def test_round_trip(self):
        ref = SubjectAssetRef(
            id="m_orders", name="Orders", definition_or_summary="SUM(orders)", source="metrics/m_orders"
        )
        restored = SubjectAssetRef.model_validate(ref.model_dump())
        assert restored == ref


class TestSubjectRefs:
    def test_defaults_are_empty_lists(self):
        refs = SubjectRefs()
        assert refs.metrics == []
        assert refs.reference_sql == []
        assert refs.ext_knowledge == []

    def test_round_trip(self):
        refs = SubjectRefs(
            metrics=[SubjectAssetRef(id="m1", name="m1")],
            reference_sql=[SubjectAssetRef(id="rs1", name="rs1")],
            ext_knowledge=[SubjectAssetRef(id="kb1", name="kb1")],
        )
        restored = SubjectRefs.model_validate(refs.model_dump())
        assert restored == refs


# --------------------------------------------------------------------------- #
# FinalizeAnalysisOutput                                                      #
# --------------------------------------------------------------------------- #


def _finalize_payload(*, n_suggested: int = 5, insights: list | None = None):
    return {
        "interpretation": _full_interpretation_payload(),
        "insights": insights if insights is not None else [_full_insight_payload()],
        "suggested_questions": [_full_suggested_question_payload() for _ in range(n_suggested)],
    }


class TestFinalizeAnalysisOutput:
    def test_round_trip(self):
        output = FinalizeAnalysisOutput.model_validate(_finalize_payload())
        restored = FinalizeAnalysisOutput.model_validate(output.model_dump())
        assert restored == output

    def test_insights_default_empty(self):
        payload = _finalize_payload()
        payload.pop("insights")
        output = FinalizeAnalysisOutput.model_validate(payload)
        assert output.insights == []

    @pytest.mark.parametrize("n_suggested", [1, 5, 8])
    def test_suggested_questions_count_within_bounds(self, n_suggested: int):
        output = FinalizeAnalysisOutput.model_validate(_finalize_payload(n_suggested=n_suggested))
        assert len(output.suggested_questions) == n_suggested

    @pytest.mark.parametrize("n_suggested", [0, 9, 15])
    def test_suggested_questions_count_out_of_bounds_rejected(self, n_suggested: int):
        with pytest.raises(ValidationError):
            FinalizeAnalysisOutput.model_validate(_finalize_payload(n_suggested=n_suggested))

    def test_extra_top_level_field_rejected(self):
        payload = _finalize_payload()
        payload["unexpected"] = "x"
        with pytest.raises(ValidationError):
            FinalizeAnalysisOutput.model_validate(payload)


# --------------------------------------------------------------------------- #
# ANALYSIS_SLUG_PATTERN                                                       #
# --------------------------------------------------------------------------- #


class TestAnalysisSlugPattern:
    """Pin the slug regex so future renames of the regex constant don't
    silently widen what counts as a valid id (insight slugs, query
    slugs, etc.)."""

    _RE = re.compile(ANALYSIS_SLUG_PATTERN)

    @pytest.mark.parametrize("good", ["a", "abc", "abc_123", "x_" * 32, "a" * 64])
    def test_accepts_good_slugs(self, good: str):
        assert self._RE.fullmatch(good) is not None

    @pytest.mark.parametrize(
        "bad",
        [
            "",  # empty
            "A",  # uppercase
            "Has-Hyphen",  # uppercase + dash
            "with space",
            "中文",
            "a" * 65,  # length cap exceeded
            "abc-123",  # dash disallowed
            "abc.123",  # punctuation disallowed
        ],
    )
    def test_rejects_bad_slugs(self, bad: str):
        assert self._RE.fullmatch(bad) is None
