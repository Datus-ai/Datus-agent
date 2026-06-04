# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus/schemas/analysis_artifacts.py``.

Pins the validation contract for every schema written under the
``analysis/`` directory and the per-query ``brief.json`` sidecar.
Each schema is exercised for:

* Round-trip serialization (model_validate → model_dump).
* Required-field constraints (``min_length`` / ``pattern``) the LLM
  finalize call is expected to honor.
* Numeric / cardinality bounds (``ge`` / ``le`` / ``min_length`` /
  ``max_length``) — these prevent garbage entries from poisoning the
  follow-up subagent.
* Slug pattern matches the same character set every other artifact id
  in the codebase uses.

History: an earlier ``Interpretation`` schema also lived here. It was
removed when the corresponding ``analysis/interpretation.json`` file
was deleted (duplicated ``manifest.description`` and was fully covered
by ``insights[].evidence_queries``). The ``ReasoningStep`` schema was
also renamed and trimmed to :class:`QueryBrief` (dropped ``goal`` /
``datasource`` / ``created_at`` — the SQL header comment + result
JSON + file mtime already carry those).
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from datus.schemas.analysis_artifacts import (
    ANALYSIS_SLUG_PATTERN,
    FinalizeAnalysisOutput,
    Insight,
    QueryBrief,
    SubjectAssetRef,
    SubjectRefs,
    SuggestedQuestion,
)

# --------------------------------------------------------------------------- #
# SubjectAssetRef                                                             #
# --------------------------------------------------------------------------- #


def _ref(path: list[str], name: str) -> SubjectAssetRef:
    return SubjectAssetRef(path=path, name=name)


class TestSubjectAssetRef:
    """The ``(path, name)`` pair is the natural key for any subject-library
    asset. Both fields are required and must be non-empty; ``path`` is
    a list so callers can copy the same shape every discovery tool
    already returns without joining/parsing strings."""

    def test_round_trip(self):
        ref = _ref(["Commerce", "Orders"], "average_gross_order_value")
        restored = SubjectAssetRef.model_validate(ref.model_dump())
        assert restored == ref

    def test_empty_path_rejected(self):
        with pytest.raises(ValidationError):
            SubjectAssetRef.model_validate({"path": [], "name": "x"})

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            SubjectAssetRef.model_validate({"path": ["a"], "name": ""})

    def test_missing_path_rejected(self):
        with pytest.raises(ValidationError):
            SubjectAssetRef.model_validate({"name": "x"})

    def test_missing_name_rejected(self):
        with pytest.raises(ValidationError):
            SubjectAssetRef.model_validate({"path": ["a"]})

    def test_extra_field_rejected(self):
        # Old wire shapes used to carry ``id`` / ``definition_or_summary`` /
        # ``source``. Strict validation slams the door on them so a stale
        # producer surfaces immediately instead of silently writing fields
        # the consumer no longer reads.
        with pytest.raises(ValidationError):
            SubjectAssetRef.model_validate({"path": ["a"], "name": "x", "id": "metric:a.x"})


# --------------------------------------------------------------------------- #
# SubjectRefs                                                                 #
# --------------------------------------------------------------------------- #


class TestSubjectRefs:
    def test_defaults_are_empty_buckets(self):
        refs = SubjectRefs()
        assert refs.metrics == []
        assert refs.reference_sql == []

    def test_round_trip(self):
        refs = SubjectRefs(
            metrics=[_ref(["Commerce", "Orders"], "aov")],
            reference_sql=[_ref(["Templates"], "top_q")],
        )
        restored = SubjectRefs.model_validate(refs.model_dump())
        assert restored == refs

    def test_legacy_string_id_form_rejected(self):
        """The old ``["metric:Sales/Revenue.gross"]`` shape must hard-fail
        — that's the LLM-drift form previous PRs tolerated and the new
        wire contract explicitly disallows."""
        with pytest.raises(ValidationError):
            SubjectRefs.model_validate({"metrics": ["metric:Sales/Revenue.gross"]})

    def test_extra_field_rejected(self):
        # ``extra=forbid`` keeps future LLM hallucinations from silently
        # landing in the file.
        with pytest.raises(ValidationError):
            SubjectRefs.model_validate({"metrics": [], "unknown_bucket": []})


# --------------------------------------------------------------------------- #
# QueryBrief                                                                  #
# --------------------------------------------------------------------------- #


def _full_brief_payload(**overrides):
    base = {
        "name": "sales_by_store",
        "hypothesis": "high-risk signups cluster around promotional campaigns",
        "uses": {
            "metrics": [{"path": ["Signups", "Risk"], "name": "high_risk_signups"}],
            "reference_sql": [],
        },
        "caveats": "Excludes test accounts (signup_email LIKE '%@example.com').",
    }
    base.update(overrides)
    return base


class TestQueryBrief:
    def test_round_trip_full(self):
        brief = QueryBrief.model_validate(_full_brief_payload())
        dumped = brief.model_dump()
        restored = QueryBrief.model_validate(dumped)
        assert restored == brief
        assert restored.uses.metrics[0].name == "high_risk_signups"
        assert restored.uses.metrics[0].path == ["Signups", "Risk"]

    def test_round_trip_minimal(self):
        # ``uses`` / ``caveats`` carry safe defaults — the only mandatory
        # fields are ``name`` and ``hypothesis``.
        minimal = {
            "name": "sales_by_store",
            "hypothesis": "stores form a flat list",
        }
        brief = QueryBrief.model_validate(minimal)
        assert brief.uses == SubjectRefs()
        assert brief.caveats == ""

    def test_empty_hypothesis_rejected(self):
        with pytest.raises(ValidationError) as exc:
            QueryBrief.model_validate(_full_brief_payload(hypothesis=""))
        assert "hypothesis" in str(exc.value)

    @pytest.mark.parametrize("bad_name", ["Bad-Slug", "UPPER", "with space", "中文", "a" * 65, ""])
    def test_invalid_slug_pattern_rejected(self, bad_name: str):
        with pytest.raises(ValidationError):
            QueryBrief.model_validate(_full_brief_payload(name=bad_name))

    def test_extra_field_rejected(self):
        # Catches both old-schema fields (``goal`` / ``datasource`` /
        # ``created_at``) and any forward-looking LLM hallucination
        # (``unknown``). ``extra=forbid`` is the schema gate.
        with pytest.raises(ValidationError):
            QueryBrief.model_validate(_full_brief_payload(unknown="x"))

    @pytest.mark.parametrize("dropped_field", ["goal", "datasource", "created_at"])
    def test_legacy_fields_no_longer_accepted(self, dropped_field: str):
        """The brief schema is the trimmed successor of ReasoningStep —
        old fields must hard-fail rather than be silently accepted, so
        a stale producer surfaces during migration instead of writing
        a file the consumer no longer reads."""
        payload = _full_brief_payload()
        payload[dropped_field] = "stale"
        with pytest.raises(ValidationError):
            QueryBrief.model_validate(payload)


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
        "kind": "quick",
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
        # ``quick`` still satisfied by non-empty related_queries alone.
        sq = SuggestedQuestion.model_validate(_full_suggested_question_payload(related_insight=None))
        assert sq.related_insight is None

    def test_related_queries_default_empty(self):
        # Default empty list is fine for deep_dive (no grounding required).
        payload = _full_suggested_question_payload(kind="deep_dive", related_insight=None)
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


class TestSuggestedQuestionKind:
    """Pins the ``kind`` field contract: required, restricted to
    ``quick`` / ``deep_dive``, and the quick-grounding model validator.

    The split exists so the ask-agent's inlined artifact context (insights,
    briefs, preview rows, key-table schemas) actually pays off — quick
    chips MUST be answerable from that context, otherwise clicking one
    forces a full new analysis loop and defeats the inlining contract
    set up by the prior round of optimizations.
    """

    def test_kind_required(self):
        # No default — finalize LLM must explicitly classify every chip.
        payload = _full_suggested_question_payload()
        payload.pop("kind")
        with pytest.raises(ValidationError):
            SuggestedQuestion.model_validate(payload)

    @pytest.mark.parametrize("bad_kind", ["medium", "Quick", "", "follow_up", None])
    def test_invalid_kind_rejected(self, bad_kind):
        with pytest.raises(ValidationError):
            SuggestedQuestion.model_validate(_full_suggested_question_payload(kind=bad_kind))

    def test_quick_with_only_related_queries_accepted(self):
        sq = SuggestedQuestion.model_validate(
            _full_suggested_question_payload(
                kind="quick", related_queries=["rev_by_region_monthly"], related_insight=None
            )
        )
        assert sq.kind == "quick"

    def test_quick_with_only_related_insight_accepted(self):
        sq = SuggestedQuestion.model_validate(
            _full_suggested_question_payload(kind="quick", related_queries=[], related_insight="revenue_dipped_in_eu")
        )
        assert sq.kind == "quick"

    def test_quick_without_any_grounding_rejected(self):
        # The whole point of "quick": it MUST cite where the answer lives.
        # Bare quick with no grounding becomes a hallucination invitation.
        with pytest.raises(ValidationError, match="kind='quick'"):
            SuggestedQuestion.model_validate(
                _full_suggested_question_payload(kind="quick", related_queries=[], related_insight=None)
            )

    def test_deep_dive_without_grounding_accepted(self):
        # deep_dive is exempt — by definition the data isn't in the artifact
        # yet, so pointing at a "closest existing query" is encouraged but
        # not required.
        sq = SuggestedQuestion.model_validate(
            _full_suggested_question_payload(kind="deep_dive", related_queries=[], related_insight=None)
        )
        assert sq.kind == "deep_dive"
        assert sq.related_queries == []
        assert sq.related_insight is None

    def test_deep_dive_with_partial_grounding_accepted(self):
        # Pointing at the closest existing query as a starting hint is the
        # encouraged shape for deep_dive.
        sq = SuggestedQuestion.model_validate(
            _full_suggested_question_payload(
                kind="deep_dive",
                related_queries=["aov_by_store"],
                related_insight="philly_dominates",
            )
        )
        assert sq.kind == "deep_dive"


# --------------------------------------------------------------------------- #
# FinalizeAnalysisOutput                                                      #
# --------------------------------------------------------------------------- #


def _finalize_payload(*, n_suggested: int = 5, insights: list | None = None):
    return {
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

    def test_legacy_interpretation_field_rejected(self):
        """``interpretation`` was removed from the schema. A stale producer
        echoing it must be caught at the schema layer (the finalize
        runtime additionally drops it before model_validate as a
        defensive belt-and-braces, but the schema itself stays strict)."""
        payload = _finalize_payload()
        payload["interpretation"] = {"audience": ["x"], "goal": "y", "focus_questions": ["q"]}
        with pytest.raises(ValidationError):
            FinalizeAnalysisOutput.model_validate(payload)

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
