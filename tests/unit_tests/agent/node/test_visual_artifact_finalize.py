# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus/agent/node/_visual_artifact_finalize.py``.

Covers the four module-level "helper" functions
(``collect_query_briefs``, ``collect_query_previews``,
``aggregate_subject_refs``, ``parse_finalize_output``,
``consistency_check``) plus the end-to-end ``run_finalize_analysis``
orchestrator with a mocked ``model`` instance.

Filesystem state is built per-test via ``tmp_path``. We never touch a
real LLM — ``run_finalize_analysis`` calls
``model.generate_with_json_output``, which we stub with
``unittest.mock.Mock``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from datus.agent.node._visual_artifact_finalize import (
    aggregate_subject_refs,
    collect_query_briefs,
    collect_query_previews,
    consistency_check,
    parse_finalize_output,
    run_finalize_analysis,
)
from datus.schemas.analysis_artifacts import (
    FinalizeAnalysisOutput,
    Insight,
    SubjectRefs,
    SuggestedQuestion,
)

# --------------------------------------------------------------------------- #
# Fixtures and helpers                                                        #
# --------------------------------------------------------------------------- #


def _write_brief(queries_dir: Path, name: str, *, uses: Dict[str, List[str]] | None = None) -> None:
    """Persist a minimal valid brief sidecar at ``queries/<name>.brief.json``."""
    queries_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "hypothesis": f"hypothesis for {name}",
        "uses": uses if uses is not None else {"metrics": [], "reference_sql": [], "ext_knowledge": []},
        "caveats": "",
    }
    (queries_dir / f"{name}.brief.json").write_text(json.dumps(payload), encoding="utf-8")


def _full_finalize_response(*, insights: list | None = None, suggested_questions: list | None = None) -> Dict[str, Any]:
    """Build a ``FinalizeAnalysisOutput``-compatible dict the LLM mock will return."""
    return {
        "insights": insights
        if insights is not None
        else [
            {
                "id": "revenue_dip",
                "title": "EU revenue dipped",
                "summary": "EU revenue dipped 8% in March.",
                "confidence": 0.7,
                "evidence_queries": ["rev_by_region"],
                "informed_by_knowledge": [],
            }
        ],
        "suggested_questions": suggested_questions
        if suggested_questions is not None
        else [
            {
                "question": "Which regions drove the dip?",
                "related_queries": ["rev_by_region"],
                "related_insight": "revenue_dip",
                "priority": 0.6,
            }
        ],
    }


# --------------------------------------------------------------------------- #
# collect_query_briefs                                                        #
# --------------------------------------------------------------------------- #


class TestCollectQueryBriefs:
    def test_missing_dir_returns_empty(self, tmp_path: Path):
        assert collect_query_briefs(tmp_path / "does_not_exist") == []

    def test_empty_dir_returns_empty(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        assert collect_query_briefs(queries_dir) == []

    def test_reads_multiple_files_sorted(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        _write_brief(queries_dir, "alpha")
        _write_brief(queries_dir, "bravo")
        _write_brief(queries_dir, "charlie")
        briefs = collect_query_briefs(queries_dir)
        assert [b["name"] for b in briefs] == ["alpha", "bravo", "charlie"]

    def test_skips_unparseable_file(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        _write_brief(queries_dir, "good")
        (queries_dir / "bad.brief.json").write_text("{not-json", encoding="utf-8")
        briefs = collect_query_briefs(queries_dir)
        assert [b["name"] for b in briefs] == ["good"]


# --------------------------------------------------------------------------- #
# collect_query_previews                                                      #
# --------------------------------------------------------------------------- #


class TestCollectQueryPreviews:
    def test_missing_dir_returns_empty(self, tmp_path: Path):
        assert collect_query_previews(tmp_path / "missing") == []

    def test_report_result_shape(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "alpha.sql").write_text("SELECT 1", encoding="utf-8")
        (queries_dir / "alpha.json").write_text(
            json.dumps(
                {
                    "executed_at": "2026-05-14T10:00:00Z",
                    "datasource": "pg",
                    "row_count": 12,
                    "columns": [{"name": "a", "type": "integer"}],
                    "rows": [{"a": i} for i in range(10)],
                }
            ),
            encoding="utf-8",
        )
        previews = collect_query_previews(queries_dir, max_rows=3)
        assert len(previews) == 1
        assert previews[0]["name"] == "alpha"
        assert previews[0]["kind"] == "report_result"
        assert previews[0]["row_count"] == 12
        # max_rows caps the preview window even when the file has more rows.
        assert len(previews[0]["preview_rows"]) == 3

    def test_dashboard_template_shape(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "rev.sql.j2").write_text("-- @datus-params x:string\nSELECT :x", encoding="utf-8")
        (queries_dir / "rev.params.json").write_text(
            json.dumps(
                {
                    "slug": "rev",
                    "description": "desc",
                    "datasource": "pg",
                    "params": [{"name": "x", "type": "string", "required": True}],
                    "columns": [{"name": "a", "type": "integer"}],
                    "sample_params": {"x": "v"},
                    "sample_row_count": 1,
                    "saved_at": "2026-05-14T10:00:00Z",
                }
            ),
            encoding="utf-8",
        )
        previews = collect_query_previews(queries_dir)
        assert len(previews) == 1
        assert previews[0]["name"] == "rev"
        assert previews[0]["kind"] == "dashboard_template"
        assert previews[0]["sample_params"] == {"x": "v"}
        assert previews[0]["sample_row_count"] == 1

    def test_unknown_kind_when_neither_readable(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "orphan.sql").write_text("SELECT 1", encoding="utf-8")
        previews = collect_query_previews(queries_dir)
        assert len(previews) == 1
        assert previews[0]["name"] == "orphan"
        assert previews[0]["kind"] == "unknown"
        assert "note" in previews[0]


# --------------------------------------------------------------------------- #
# aggregate_subject_refs                                                      #
# --------------------------------------------------------------------------- #


class TestAggregateSubjectRefs:
    def test_empty_dir_returns_empty_buckets(self, tmp_path: Path):
        refs = aggregate_subject_refs(tmp_path / "queries")
        assert refs == SubjectRefs()

    def test_dedupes_ids_across_files(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        _write_brief(queries_dir, "alpha", uses={"metrics": ["m1", "m2"], "reference_sql": ["rs1"]})
        _write_brief(queries_dir, "bravo", uses={"metrics": ["m1", "m3"], "ext_knowledge": ["kb1"]})
        refs = aggregate_subject_refs(queries_dir)
        # m1 appears in two files but only once in the aggregate.
        assert [r.id for r in refs.metrics] == ["m1", "m2", "m3"]
        assert [r.id for r in refs.reference_sql] == ["rs1"]
        assert [r.id for r in refs.ext_knowledge] == ["kb1"]

    def test_preserves_first_seen_order(self, tmp_path: Path):
        """First-seen order within each bucket matters for subagent rendering."""
        queries_dir = tmp_path / "queries"
        # alpha sorts before zulu; insertion order within alpha is preserved.
        _write_brief(queries_dir, "alpha", uses={"metrics": ["m_first", "m_second"]})
        _write_brief(queries_dir, "zulu", uses={"metrics": ["m_third"]})
        refs = aggregate_subject_refs(queries_dir)
        assert [r.id for r in refs.metrics] == ["m_first", "m_second", "m_third"]

    def test_ignores_invalid_uses_field(self, tmp_path: Path):
        """A brief file with a non-dict ``uses`` shouldn't crash the aggregator."""
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        # Hand-build a file where ``uses`` is not the schema shape but the
        # rest is parseable JSON — the aggregator skips it gracefully.
        broken = {
            "name": "broken",
            "hypothesis": "h",
            "uses": "not-a-dict",
            "caveats": "",
        }
        (queries_dir / "broken.brief.json").write_text(json.dumps(broken), encoding="utf-8")
        refs = aggregate_subject_refs(queries_dir)
        assert refs == SubjectRefs()


# --------------------------------------------------------------------------- #
# parse_finalize_output                                                       #
# --------------------------------------------------------------------------- #


class TestParseFinalizeOutput:
    def test_validates_good_dict(self):
        output = parse_finalize_output(_full_finalize_response(), artifact_kind="report")
        assert isinstance(output, FinalizeAnalysisOutput)
        assert len(output.insights) == 1
        assert output.insights[0].id == "revenue_dip"

    def test_rejects_non_dict(self):
        with pytest.raises(ValueError):
            parse_finalize_output(["not", "a", "dict"], artifact_kind="report")

    def test_dashboard_forces_empty_insights(self):
        # LLM mistakenly returned insights for a dashboard run — parser
        # should silently drop them rather than persist conclusions that
        # don't belong on a runtime-parameterized dashboard.
        raw = _full_finalize_response()
        assert raw["insights"]
        output = parse_finalize_output(raw, artifact_kind="dashboard")
        assert output.insights == []

    def test_legacy_interpretation_key_silently_dropped(self):
        """Stale producers may still echo a top-level ``interpretation``
        field; the parser must strip it before schema validation so the
        finalize pipeline keeps working through the migration window
        (the schema itself stays strict — see schema tests)."""
        raw = _full_finalize_response()
        raw["interpretation"] = {"audience": ["x"], "goal": "y", "focus_questions": ["q"]}
        output = parse_finalize_output(raw, artifact_kind="report")
        assert len(output.insights) == 1


# --------------------------------------------------------------------------- #
# consistency_check                                                           #
# --------------------------------------------------------------------------- #


def _make_output(*, insights=None, suggested_questions=None) -> FinalizeAnalysisOutput:
    return FinalizeAnalysisOutput(
        insights=insights or [],
        suggested_questions=suggested_questions
        or [SuggestedQuestion(question="q?", related_queries=[], related_insight=None, priority=0.5)],
    )


class TestConsistencyCheck:
    def test_clean_output_no_warnings(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "alpha.sql").write_text("SELECT 1", encoding="utf-8")
        output = _make_output(
            insights=[
                Insight(
                    id="i1",
                    title="t",
                    summary="s",
                    confidence=0.5,
                    evidence_queries=["alpha"],
                )
            ],
            suggested_questions=[
                SuggestedQuestion(question="q?", related_queries=["alpha"], related_insight="i1", priority=0.5)
            ],
        )
        warnings = consistency_check(queries_dir=queries_dir, output=output)
        assert warnings == []

    def test_warns_when_insight_evidence_missing(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        # No SQL file backs the evidence.
        output = _make_output(
            insights=[
                Insight(
                    id="i1",
                    title="t",
                    summary="s",
                    confidence=0.5,
                    evidence_queries=["ghost_query"],
                )
            ],
        )
        warnings = consistency_check(queries_dir=queries_dir, output=output)
        assert any("ghost_query" in w for w in warnings)

    def test_warns_when_related_insight_missing(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        output = _make_output(
            insights=[],
            suggested_questions=[
                SuggestedQuestion(question="q?", related_queries=[], related_insight="unknown", priority=0.5)
            ],
        )
        warnings = consistency_check(queries_dir=queries_dir, output=output)
        assert any("unknown" in w for w in warnings)


# --------------------------------------------------------------------------- #
# run_finalize_analysis                                                       #
# --------------------------------------------------------------------------- #


def _make_artifact_layout(
    tmp_path: Path, *, with_brief: bool = True, brief_uses: Dict[str, List[str]] | None = None
) -> tuple[Path, Path, Path]:
    """Build the three on-disk paths run_finalize_analysis expects.

    By default seeds one brief with a ``metrics`` ref so the
    ``subject_refs.json`` writer triggers; pass ``brief_uses={}`` to
    exercise the empty-uses path that skips the file.
    """
    artifact_dir = tmp_path / "artifact"
    queries_dir = artifact_dir / "queries"
    analysis_dir = artifact_dir / "analysis"
    queries_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    if with_brief:
        if brief_uses is None:
            brief_uses = {"metrics": ["m_revenue"]}
        _write_brief(queries_dir, "rev_by_region", uses=brief_uses)
        (queries_dir / "rev_by_region.sql").write_text("SELECT 1", encoding="utf-8")
        (queries_dir / "rev_by_region.json").write_text(
            json.dumps(
                {
                    "executed_at": "2026-05-14T10:00:00Z",
                    "datasource": "pg",
                    "row_count": 1,
                    "columns": [{"name": "a", "type": "integer"}],
                    "rows": [{"a": 1}],
                }
            ),
            encoding="utf-8",
        )
    return artifact_dir, queries_dir, analysis_dir


class TestRunFinalizeAnalysis:
    def test_end_to_end_writes_expected_files(self, tmp_path: Path):
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(tmp_path)

        model = Mock(spec=["generate_with_json_output"])
        model.generate_with_json_output.return_value = _full_finalize_response()

        result = run_finalize_analysis(
            model=model,
            artifact_kind="report",
            artifact_dir=artifact_dir,
            queries_dir=queries_dir,
            analysis_dir=analysis_dir,
            actions=[],
        )

        assert result["ok"] is True
        # interpretation.json was removed in the brief.json refactor —
        # it must not be written even if a stale LLM produced one.
        assert not (analysis_dir / "interpretation.json").exists()
        assert (analysis_dir / "insights.json").is_file()
        assert (analysis_dir / "suggested_questions.json").is_file()
        # subject_refs.json is present because the brief declared a metric.
        assert (analysis_dir / "subject_refs.json").is_file()
        refs = json.loads((analysis_dir / "subject_refs.json").read_text(encoding="utf-8"))
        assert any(m["id"] == "m_revenue" for m in refs["metrics"])
        assert result["subject_refs_count"]["metrics"] == 1

    def test_subject_refs_skipped_when_no_uses_declared(self, tmp_path: Path):
        """Present-iff-non-empty: a brief without any subject-library
        ids must NOT produce a ``subject_refs.json`` file — an absent
        file is the honest "no attribution" signal."""
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(
            tmp_path, brief_uses={"metrics": [], "reference_sql": [], "ext_knowledge": []}
        )

        model = Mock(spec=["generate_with_json_output"])
        model.generate_with_json_output.return_value = _full_finalize_response()

        result = run_finalize_analysis(
            model=model,
            artifact_kind="report",
            artifact_dir=artifact_dir,
            queries_dir=queries_dir,
            analysis_dir=analysis_dir,
            actions=[],
        )

        assert result["ok"] is True
        assert not (analysis_dir / "subject_refs.json").exists()
        assert result["subject_refs_count"] == {"metrics": 0, "reference_sql": 0, "ext_knowledge": 0}

    def test_subject_refs_stale_file_removed_when_now_empty(self, tmp_path: Path):
        """Edit-mode rerun where all ``uses`` were dropped: a stale
        ``subject_refs.json`` from a prior run must be deleted so the
        absent-file signal stays accurate."""
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(
            tmp_path, brief_uses={"metrics": [], "reference_sql": [], "ext_knowledge": []}
        )
        stale_path = analysis_dir / "subject_refs.json"
        stale_path.write_text(
            json.dumps({"metrics": [{"id": "old"}], "reference_sql": [], "ext_knowledge": []}),
            encoding="utf-8",
        )

        model = Mock(spec=["generate_with_json_output"])
        model.generate_with_json_output.return_value = _full_finalize_response()

        run_finalize_analysis(
            model=model,
            artifact_kind="report",
            artifact_dir=artifact_dir,
            queries_dir=queries_dir,
            analysis_dir=analysis_dir,
            actions=[],
        )
        assert not stale_path.exists()

    def test_dashboard_does_not_write_insights(self, tmp_path: Path):
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(tmp_path)

        model = Mock(spec=["generate_with_json_output"])
        model.generate_with_json_output.return_value = _full_finalize_response()

        result = run_finalize_analysis(
            model=model,
            artifact_kind="dashboard",
            artifact_dir=artifact_dir,
            queries_dir=queries_dir,
            analysis_dir=analysis_dir,
            actions=[],
        )

        assert result["ok"] is True
        # Dashboard mode never persists insights, even though the LLM
        # returned some.
        assert not (analysis_dir / "insights.json").exists()

    def test_consistency_warnings_surface(self, tmp_path: Path):
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(tmp_path)
        # LLM response references a query that doesn't exist on disk —
        # consistency_check should catch and surface that.
        response = _full_finalize_response(
            insights=[
                {
                    "id": "i1",
                    "title": "t",
                    "summary": "s",
                    "confidence": 0.5,
                    "evidence_queries": ["ghost"],
                    "informed_by_knowledge": [],
                }
            ],
        )

        model = Mock(spec=["generate_with_json_output"])
        model.generate_with_json_output.return_value = response

        result = run_finalize_analysis(
            model=model,
            artifact_kind="report",
            artifact_dir=artifact_dir,
            queries_dir=queries_dir,
            analysis_dir=analysis_dir,
            actions=[],
        )
        assert result["ok"] is True
        assert any("ghost" in w for w in result["warnings"])

    def test_llm_exception_surfaces_as_error(self, tmp_path: Path):
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(tmp_path)

        model = Mock(spec=["generate_with_json_output"])
        model.generate_with_json_output.side_effect = RuntimeError("LLM is down")

        result = run_finalize_analysis(
            model=model,
            artifact_kind="report",
            artifact_dir=artifact_dir,
            queries_dir=queries_dir,
            analysis_dir=analysis_dir,
            actions=[],
        )

        assert result["ok"] is False
        assert "LLM is down" in result["error"]
        # No analysis files were written when the LLM call failed up-front.
        assert not (analysis_dir / "insights.json").exists()

    def test_schema_validation_failure_surfaces_as_error(self, tmp_path: Path):
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(tmp_path)

        model = Mock(spec=["generate_with_json_output"])
        # Missing suggested_questions — schema fails.
        model.generate_with_json_output.return_value = {"insights": []}

        result = run_finalize_analysis(
            model=model,
            artifact_kind="report",
            artifact_dir=artifact_dir,
            queries_dir=queries_dir,
            analysis_dir=analysis_dir,
            actions=[],
        )

        assert result["ok"] is False
        assert "finalize output invalid" in result["error"]
        assert not (analysis_dir / "insights.json").exists()
