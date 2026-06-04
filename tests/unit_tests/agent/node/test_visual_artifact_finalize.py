# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus/agent/node/visual_artifact/_visual_artifact_finalize.py``.

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

from datus.agent.node.visual_artifact._visual_artifact_finalize import (
    _sanitize_curated_intent_md,
    aggregate_referenced_tables,
    aggregate_subject_refs,
    bake_key_tables_schema,
    build_finalize_prompt,
    collect_query_briefs,
    collect_query_previews,
    consistency_check,
    parse_finalize_output,
    run_finalize_analysis,
    run_intent_curation,
    update_manifest_key_tables,
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
        "uses": uses if uses is not None else {"metrics": [], "reference_sql": []},
        "caveats": "",
    }
    (queries_dir / f"{name}.brief.json").write_text(json.dumps(payload), encoding="utf-8")


def _full_finalize_response(*, insights: list | None = None, suggested_questions: list | None = None) -> Dict[str, Any]:
    """Build a ``FinalizeAnalysisOutput``-compatible dict the LLM mock will return.

    Suggested questions default to a single ``kind='quick'`` entry that
    satisfies the schema's grounding rule (cites ``related_queries`` and
    ``related_insight``). Tests exercising the 3-quick + 2-deep_dive
    distribution / consistency_check assemble their own lists.
    """
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
                "kind": "quick",
                "related_queries": ["rev_by_region"],
                "related_insight": "revenue_dip",
                "priority": 0.6,
            }
        ],
    }


def _sq(
    *,
    question: str = "q?",
    kind: str = "deep_dive",
    related_queries: list[str] | None = None,
    related_insight: str | None = None,
    priority: float = 0.5,
) -> SuggestedQuestion:
    """Build a SuggestedQuestion with sensible defaults for tests.

    Defaults to ``kind='deep_dive'`` so tests that don't care about the
    quick-grounding contract get a schema-valid object without having to
    set ``related_queries`` / ``related_insight``. Override ``kind`` when
    exercising the quick path.
    """
    return SuggestedQuestion(
        question=question,
        kind=kind,
        related_queries=related_queries or [],
        related_insight=related_insight,
        priority=priority,
    )


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


def _m(path: list[str], name: str) -> dict:
    """Helper to build a ``uses`` entry in the new ``{path, name}`` shape."""
    return {"path": path, "name": name}


class TestAggregateSubjectRefs:
    def test_empty_dir_returns_empty_buckets(self, tmp_path: Path):
        refs = aggregate_subject_refs(tmp_path / "queries")
        assert refs == SubjectRefs()

    def test_dedupes_by_path_and_name_across_files(self, tmp_path: Path):
        """``(path, name)`` is the natural key — same pair across two
        files dedupes to one entry; different paths with the same leaf
        name survive as separate entries."""
        queries_dir = tmp_path / "queries"
        _write_brief(
            queries_dir,
            "alpha",
            uses={
                "metrics": [_m(["Commerce", "Orders"], "aov"), _m(["Commerce", "Orders"], "order_count")],
                "reference_sql": [_m(["Templates"], "top_q")],
            },
        )
        _write_brief(
            queries_dir,
            "bravo",
            uses={
                # Repeats ``(["Commerce", "Orders"], "aov")`` — should dedupe.
                # Adds a new entry that shares only the leaf name ``aov`` but
                # under a different path — must NOT dedupe.
                "metrics": [_m(["Commerce", "Orders"], "aov"), _m(["Marketing", "Spend"], "aov")],
            },
        )
        refs = aggregate_subject_refs(queries_dir)
        metric_keys = [(m.path, m.name) for m in refs.metrics]
        assert metric_keys == [
            (["Commerce", "Orders"], "aov"),
            (["Commerce", "Orders"], "order_count"),
            (["Marketing", "Spend"], "aov"),
        ]
        assert [(r.path, r.name) for r in refs.reference_sql] == [(["Templates"], "top_q")]

    def test_preserves_first_seen_order(self, tmp_path: Path):
        """First-seen order within each bucket matters for subagent rendering."""
        queries_dir = tmp_path / "queries"
        # alpha sorts before zulu; insertion order within alpha is preserved.
        _write_brief(
            queries_dir,
            "alpha",
            uses={"metrics": [_m(["X"], "m_first"), _m(["X"], "m_second")]},
        )
        _write_brief(queries_dir, "zulu", uses={"metrics": [_m(["X"], "m_third")]})
        refs = aggregate_subject_refs(queries_dir)
        assert [m.name for m in refs.metrics] == ["m_first", "m_second", "m_third"]

    def test_skips_brief_with_invalid_uses(self, tmp_path: Path):
        """A brief whose ``uses`` block fails schema validation (legacy
        string-id form, missing fields, etc.) is skipped with a warning
        — one malformed brief must not strand the whole aggregate."""
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        # Legacy string-id form — the LLM-drift shape we used to tolerate
        # before the path/name redesign. Now rejected at the schema layer.
        broken = {
            "name": "broken",
            "hypothesis": "h",
            "uses": {"metrics": ["metric:Sales/Revenue.gross_revenue"]},
            "caveats": "",
        }
        (queries_dir / "broken.brief.json").write_text(json.dumps(broken), encoding="utf-8")
        # A second brief with a well-formed entry — must still land.
        _write_brief(queries_dir, "good", uses={"metrics": [_m(["A"], "x")]})
        refs = aggregate_subject_refs(queries_dir)
        assert [(m.path, m.name) for m in refs.metrics] == [(["A"], "x")]


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
        suggested_questions=suggested_questions or [_sq()],
    )


def _balanced_suggestions(*, quick_query_slug: str = "alpha", quick_insight_id: str = "i1") -> list[SuggestedQuestion]:
    """A schema-valid 3 quick + 2 deep_dive list for distribution tests.

    Quick chips ground on ``quick_query_slug`` (and ``quick_insight_id`` on
    the third) so consistency_check's resolved-grounding check passes when
    the caller materializes the matching SQL file + insight on disk.
    """
    return [
        _sq(question="quick A?", kind="quick", related_queries=[quick_query_slug]),
        _sq(question="quick B?", kind="quick", related_queries=[quick_query_slug]),
        _sq(question="quick C?", kind="quick", related_insight=quick_insight_id),
        _sq(question="deep A?", kind="deep_dive"),
        _sq(question="deep B?", kind="deep_dive", related_queries=[quick_query_slug]),
    ]


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
            suggested_questions=_balanced_suggestions(),
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
            suggested_questions=[_sq(kind="deep_dive", related_insight="unknown")],
        )
        warnings = consistency_check(queries_dir=queries_dir, output=output)
        assert any("unknown" in w for w in warnings)


class TestConsistencyCheckDistribution:
    """The 3 quick + 2 deep_dive split is the contract every artifact ships
    under. The schema keeps the total range 1..8 forgiving so a marginal
    LLM run isn't outright rejected; consistency_check surfaces drift as a
    warning so ops can spot finalize regressions in the trace log without
    blocking the artifact.
    """

    def test_no_distribution_warning_on_3_quick_2_deep(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "alpha.sql").write_text("SELECT 1", encoding="utf-8")
        output = _make_output(
            insights=[
                Insight(id="i1", title="t", summary="s", confidence=0.5, evidence_queries=["alpha"]),
            ],
            suggested_questions=_balanced_suggestions(),
        )
        warnings = consistency_check(queries_dir=queries_dir, output=output)
        assert not any("distribution off-target" in w for w in warnings)

    def test_distribution_warning_when_all_quick(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "alpha.sql").write_text("SELECT 1", encoding="utf-8")
        five_quick = [_sq(question=f"q{i}", kind="quick", related_queries=["alpha"]) for i in range(5)]
        output = _make_output(suggested_questions=five_quick)
        warnings = consistency_check(queries_dir=queries_dir, output=output)
        assert any("distribution off-target" in w and "quick=5" in w for w in warnings)

    def test_distribution_warning_when_wrong_total(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "alpha.sql").write_text("SELECT 1", encoding="utf-8")
        # 4 entries instead of 5 — schema allows 1..8 but contract is 5.
        four = [
            _sq(question="q1", kind="quick", related_queries=["alpha"]),
            _sq(question="q2", kind="quick", related_queries=["alpha"]),
            _sq(question="q3", kind="quick", related_queries=["alpha"]),
            _sq(question="q4", kind="deep_dive"),
        ]
        output = _make_output(suggested_questions=four)
        warnings = consistency_check(queries_dir=queries_dir, output=output)
        assert any("distribution off-target" in w and "total=4" in w for w in warnings)


class TestConsistencyCheckQuickGrounding:
    """The schema validator only sees that quick has *some* grounding pointer
    (non-empty related_queries OR non-null related_insight). It can't tell
    whether those pointers actually resolve to artifacts on disk. The
    consistency check closes that gap so a quick chip whose only grounding
    is a typo'd slug surfaces as a warning instead of silently failing the
    ask-agent at click time.
    """

    def test_warns_when_quick_grounding_does_not_resolve(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        # Real SQL on disk; the quick chip cites a different (missing) slug.
        (queries_dir / "alpha.sql").write_text("SELECT 1", encoding="utf-8")
        output = _make_output(
            insights=[],
            suggested_questions=[_sq(kind="quick", related_queries=["ghost_query"])],
        )
        warnings = consistency_check(queries_dir=queries_dir, output=output)
        assert any("no resolvable grounding" in w for w in warnings)

    def test_no_quick_grounding_warning_when_insight_resolves(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        output = _make_output(
            insights=[
                Insight(id="i1", title="t", summary="s", confidence=0.5, evidence_queries=["alpha"]),
            ],
            suggested_questions=[_sq(kind="quick", related_insight="i1")],
        )
        # SQL "alpha" exists too so the insight's evidence resolves.
        (queries_dir / "alpha.sql").write_text("SELECT 1", encoding="utf-8")
        warnings = consistency_check(queries_dir=queries_dir, output=output)
        assert not any("no resolvable grounding" in w for w in warnings)

    def test_deep_dive_without_grounding_does_not_warn(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        output = _make_output(
            suggested_questions=[_sq(kind="deep_dive", related_queries=[], related_insight=None)],
        )
        warnings = consistency_check(queries_dir=queries_dir, output=output)
        # deep_dive is exempt — data isn't in the artifact yet, by design.
        assert not any("no resolvable grounding" in w for w in warnings)


class TestBuildFinalizePrompt:
    """Pin the SUGGESTED QUESTIONS CONTRACT section of the finalize prompt.

    The prompt is the only place we communicate the 3-quick + 2-deep_dive
    split to the LLM, so a regression that drops or muddies the wording
    will silently let the finalize step revert to the pre-split behavior
    (all-aspirational chips that trigger a full new analysis loop on
    every click).
    """

    @pytest.fixture
    def report_prompt(self) -> str:
        return build_finalize_prompt(
            artifact_kind="report",
            intent_md="user wanted an AOV report",
            query_briefs=[{"name": "alpha", "hypothesis": "h"}],
            query_previews=[{"name": "alpha", "kind": "report_result", "preview_rows": []}],
            action_history_hints=[],
            existing_insights=None,
            existing_suggested_questions=None,
        )

    def test_prompt_announces_kind_field_in_schema(self, report_prompt: str):
        # Schema section must mention the kind field by name so the LLM
        # actually emits it.
        assert "`kind`" in report_prompt
        assert "'quick'" in report_prompt
        assert "'deep_dive'" in report_prompt

    def test_prompt_pins_3_quick_2_deep_split(self, report_prompt: str):
        # OUTPUT SCHEMA section spells the split in plain prose; the
        # CONTRACT section spells it in code-block form. Both must be
        # present so the LLM has the rule reinforced.
        assert "3 quick + 2 deep_dive" in report_prompt
        assert '3 × `kind="quick"`' in report_prompt
        assert '2 × `kind="deep_dive"`' in report_prompt

    def test_prompt_states_quick_grounding_rule(self, report_prompt: str):
        # The "quick MUST cite grounding" invariant has to be visible in
        # the prompt — the schema validator catches violations, but the
        # prompt is what gets the LLM to produce valid output in the
        # first place.
        assert "MUST cite" in report_prompt
        assert "related_queries" in report_prompt
        assert "related_insight" in report_prompt

    def test_prompt_calls_out_avoid_aspirational_quick(self, report_prompt: str):
        # The single biggest failure mode the contract guards against:
        # "why" / "what factors" questions tagged as quick because the
        # LLM sees them as obvious follow-ups. The prompt must call this
        # out explicitly or the LLM regresses to the old behavior.
        assert '"why"' in report_prompt
        assert "deep_dive" in report_prompt

    def test_prompt_calls_out_self_check(self, report_prompt: str):
        # The "draft a one-sentence answer from preview rows alone"
        # heuristic is the operational version of the quick contract —
        # pinning the exact wording so a future prose-tweak that drops
        # the imperative ("could try drafting…") still fails this test.
        assert "Self-check: before tagging a question quick" in report_prompt

    def test_dashboard_prompt_keeps_kind_split(self):
        prompt = build_finalize_prompt(
            artifact_kind="dashboard",
            intent_md="dashboard intent",
            query_briefs=[],
            query_previews=[],
            action_history_hints=[],
            existing_insights=None,
            existing_suggested_questions=None,
        )
        # Dashboards skip insights but the quick/deep_dive split still
        # applies — quick chips describe the template metadata, deep_dive
        # chips trigger a real query run.
        assert "DASHBOARD MODE" in prompt
        assert "quick" in prompt
        assert "deep_dive" in prompt

    def test_constraints_recap_pins_distribution(self, report_prompt: str):
        # The "CONSTRAINTS RECAP" block is the prompt's TL;DR. The exact
        # "5 entries split as 3 ... + 2 ..." wording must surface here so
        # an LLM that skims past the long-form CONTRACT section still
        # gets the distribution rule.
        assert "5 entries split as" in report_prompt
        assert "kind='quick'" in report_prompt
        assert "kind='deep_dive'" in report_prompt


# --------------------------------------------------------------------------- #
# run_finalize_analysis                                                       #
# --------------------------------------------------------------------------- #


def _seed_manifest(artifact_dir: Path, *, slug: str = "demo_report") -> Path:
    """Write a minimal valid manifest so finalize's key_tables update
    has something to patch in place."""
    payload = {
        "slug": slug,
        "name": "Demo report",
        "description": "Smoke-test artifact used by the finalize unit tests.",
        "kind": "report",
        "created_at": "2026-05-14T10:00:00Z",
    }
    path = artifact_dir / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_artifact_layout(
    tmp_path: Path,
    *,
    with_brief: bool = True,
    brief_uses: Dict[str, Any] | None = None,
    sql_body: str = "SELECT 1",
) -> tuple[Path, Path, Path]:
    """Build the on-disk paths run_finalize_analysis expects.

    Always seeds ``manifest.json`` (so the key_tables updater has a
    target). Optionally seeds one brief + matching SQL/result file; the
    SQL body is parameterised so individual tests can exercise the
    table-extraction path with realistic FROM/JOIN clauses while the
    default keeps prior tests' contract (``SELECT 1`` → no tables).
    """
    artifact_dir = tmp_path / "artifact"
    queries_dir = artifact_dir / "queries"
    analysis_dir = artifact_dir / "analysis"
    queries_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    _seed_manifest(artifact_dir)
    if with_brief:
        if brief_uses is None:
            brief_uses = {"metrics": [{"path": ["Revenue"], "name": "revenue_by_region"}]}
        _write_brief(queries_dir, "rev_by_region", uses=brief_uses)
        (queries_dir / "rev_by_region.sql").write_text(sql_body, encoding="utf-8")
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


# --------------------------------------------------------------------------- #
# bake_key_tables_schema                                                      #
# --------------------------------------------------------------------------- #


def _mock_describe_table_tool(per_table_payload: Dict[str, Any]) -> Mock:
    """Build a mock ``db_func_tool`` whose ``describe_table`` returns
    a pre-canned payload per table name.

    Each entry in ``per_table_payload`` is either:
    * ``{"result": {...}}`` for a successful describe
    * ``{"success": 0, "error": "..."}`` for a connector-side failure
    * an ``Exception`` instance to raise

    Builds a FuncToolResult-shaped Mock object (``success``, ``result``,
    ``error`` attrs) so the bake function's duck-typed access path
    works the same way it does against the real tool.
    """
    tool = Mock()

    def describe(*, table_name: str, **_kwargs: Any) -> Any:
        spec = per_table_payload.get(table_name)
        if isinstance(spec, Exception):
            raise spec
        if spec is None:
            return None
        result_mock = Mock()
        result_mock.success = spec.get("success", 1)
        result_mock.result = spec.get("result")
        result_mock.error = spec.get("error")
        return result_mock

    tool.describe_table = Mock(side_effect=describe)
    return tool


class TestBakeKeyTablesSchema:
    """Snapshot ``describe_table`` output per key_table into the sidecar.

    The bake is best-effort: per-table failures get captured inline so
    a single broken connector / dropped table doesn't strand the whole
    schema sidecar.
    """

    def test_no_db_func_tool_skips_silently(self, tmp_path: Path):
        """A node without a DB tool (rare, but supported in tests / dry
        runs) must skip the bake without writing a misleading empty
        sidecar."""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        warning = bake_key_tables_schema(
            db_func_tool=None,
            key_tables=["jeff_shop.raw_orders"],
            analysis_dir=analysis_dir,
        )
        assert warning is None
        assert not (analysis_dir / "key_tables_schema.json").exists()

    def test_early_return_removes_stale_schema_file(self, tmp_path: Path):
        """Edit-mode rerun where the new SQL set produces no
        ``key_tables`` (or finalize runs without a db tool this time)
        MUST proactively delete any prior ``key_tables_schema.json`` —
        otherwise ask_* would serve the previous artifact's schema
        snapshot indefinitely. Mirrors the present-iff-non-empty
        semantics already used by ``write_subject_refs`` for
        ``subject_refs.json``."""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        stale_path = analysis_dir / "key_tables_schema.json"
        stale_path.write_text(
            json.dumps({"tables": [{"name": "old_tbl", "columns": []}]}),
            encoding="utf-8",
        )
        assert stale_path.is_file()

        warning = bake_key_tables_schema(
            db_func_tool=None,
            key_tables=["some_tbl_that_would_have_been_baked"],
            analysis_dir=analysis_dir,
        )
        assert warning is None
        # Stale file gone — the absent signal is now truthful.
        assert not stale_path.exists()

    def test_stale_cleanup_unlink_failure_surfaces_as_warning(self, tmp_path: Path, monkeypatch):
        """When ``stale.unlink()`` fails (read-only filesystem, immutable
        flag, racing process), the bake must return a warning string so
        ``run_finalize_analysis`` collects it into its ``warnings``
        list. Silently logging would leave the next ask_* turn serving
        a snapshot the consumer treats as fresh — the exact lying-
        snapshot scenario the stale-cleanup was added to prevent."""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        stale_path = analysis_dir / "key_tables_schema.json"
        stale_path.write_text(json.dumps({"tables": []}), encoding="utf-8")

        # Patch ``Path.unlink`` so only the stale-cleanup call raises —
        # other Path operations stay live and the test doesn't trip on
        # incidental mkdir / is_file etc.
        original_unlink = Path.unlink

        def boom(self, *args, **kwargs):  # noqa: ARG001 — match Path.unlink signature
            if self == stale_path:
                raise OSError("read-only filesystem")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", boom)

        warning = bake_key_tables_schema(
            db_func_tool=None,
            key_tables=["bake-would-have-happened"],
            analysis_dir=analysis_dir,
        )
        assert isinstance(warning, str), f"expected a warning string, got {warning!r}"
        # The warning must include both the filename (consumer-facing
        # identifier) and the underlying OSError message (actionable
        # detail) — pin both so a refactor that swallows one trips.
        assert "key_tables_schema.json" in warning
        assert "read-only filesystem" in warning
        # File still on disk (unlink failed) — proves the warning is
        # actually correlated with the lying-snapshot risk, not just
        # a phantom error string.
        assert stale_path.is_file()

    def test_early_return_with_empty_key_tables_also_clears_stale(self, tmp_path: Path):
        """Same cleanup applies when ``key_tables`` is empty — finalize
        re-aggregated the SQL set and there are no tables anymore, so
        the prior snapshot is wrong by definition."""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        stale_path = analysis_dir / "key_tables_schema.json"
        stale_path.write_text(
            json.dumps({"tables": [{"name": "obsolete", "columns": []}]}),
            encoding="utf-8",
        )

        tool = _mock_describe_table_tool({})
        warning = bake_key_tables_schema(
            db_func_tool=tool,
            key_tables=[],
            analysis_dir=analysis_dir,
        )
        assert warning is None
        assert not stale_path.exists()
        # And describe_table was NOT called — empty key_tables is a
        # short-circuit, not "ask the connector about nothing".
        tool.describe_table.assert_not_called()

    def test_empty_key_tables_skips_silently(self, tmp_path: Path):
        """Manifest has no key_tables (e.g. an artifact with only
        literal/constant SQL) ⇒ no schema to bake, no sidecar file."""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        tool = _mock_describe_table_tool({})
        warning = bake_key_tables_schema(
            db_func_tool=tool,
            key_tables=[],
            analysis_dir=analysis_dir,
        )
        assert warning is None
        assert not (analysis_dir / "key_tables_schema.json").exists()
        # describe_table must not be called when there's nothing to describe.
        tool.describe_table.assert_not_called()

    def test_writes_schema_when_describe_succeeds(self, tmp_path: Path):
        """Happy path: one table, describe_table returns columns + an
        optional semantic-model description; the sidecar carries
        name/description/columns shape verbatim."""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        tool = _mock_describe_table_tool(
            {
                "jeff_shop.raw_orders": {
                    "result": {
                        "columns": [
                            {"name": "order_id", "type": "bigint", "comment": "primary key"},
                            {"name": "order_total", "type": "int", "comment": "stored in cents"},
                        ],
                        "table": {
                            "name": "raw_orders",
                            "description": "canonical orders fact table",
                        },
                    }
                }
            }
        )
        warning = bake_key_tables_schema(
            db_func_tool=tool,
            key_tables=["jeff_shop.raw_orders"],
            analysis_dir=analysis_dir,
        )
        assert warning is None
        out = json.loads((analysis_dir / "key_tables_schema.json").read_text())
        # Exact shape pinned so consumers can rely on this contract.
        assert out == {
            "tables": [
                {
                    "name": "jeff_shop.raw_orders",
                    "description": "canonical orders fact table",
                    "columns": [
                        {
                            "name": "order_id",
                            "type": "bigint",
                            "comment": "primary key",
                            "is_dimension": None,
                        },
                        {
                            "name": "order_total",
                            "type": "int",
                            "comment": "stored in cents",
                            "is_dimension": None,
                        },
                    ],
                    "error": None,
                }
            ]
        }

    def test_is_dimension_propagates_when_semantic_model_present(self, tmp_path: Path):
        """When describe_table found a semantic model, the per-column
        ``is_dimension`` flag is preserved through to the sidecar so
        the LLM can tell measures from dimensions without a semantic
        lookup.
        """
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        tool = _mock_describe_table_tool(
            {
                "tbl_a": {
                    "result": {
                        "columns": [
                            {"name": "id", "type": "int", "comment": "", "is_dimension": True},
                            {"name": "amount", "type": "decimal", "comment": "", "is_dimension": False},
                        ],
                        "table": {"name": "tbl_a", "description": ""},
                    }
                }
            }
        )
        bake_key_tables_schema(
            db_func_tool=tool,
            key_tables=["tbl_a"],
            analysis_dir=analysis_dir,
        )
        out = json.loads((analysis_dir / "key_tables_schema.json").read_text())
        cols_by_name = {c["name"]: c for c in out["tables"][0]["columns"]}
        assert cols_by_name["id"]["is_dimension"] is True
        assert cols_by_name["amount"]["is_dimension"] is False

    def test_per_table_describe_failure_captured_inline(self, tmp_path: Path):
        """Mixed success: one table works, one returns ``success=0``,
        one raises. All three appear in the sidecar — the failing two
        with their error strings so the prompt can render a per-table
        "schema unavailable" hint instead of dropping them."""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        tool = _mock_describe_table_tool(
            {
                "good_tbl": {
                    "result": {
                        "columns": [{"name": "c", "type": "int", "comment": ""}],
                    }
                },
                "permission_denied_tbl": {"success": 0, "error": "access denied"},
                "raising_tbl": RuntimeError("connection reset"),
            }
        )
        warning = bake_key_tables_schema(
            db_func_tool=tool,
            key_tables=["good_tbl", "permission_denied_tbl", "raising_tbl"],
            analysis_dir=analysis_dir,
        )
        assert warning is None
        out = json.loads((analysis_dir / "key_tables_schema.json").read_text())
        by_name = {t["name"]: t for t in out["tables"]}
        assert by_name["good_tbl"]["error"] is None
        assert len(by_name["good_tbl"]["columns"]) == 1
        # Connector-side failure: error string from the tool surfaces verbatim.
        assert by_name["permission_denied_tbl"]["columns"] == []
        assert "access denied" in by_name["permission_denied_tbl"]["error"]
        # Exception: bake catches and records the message.
        assert by_name["raising_tbl"]["columns"] == []
        assert "connection reset" in by_name["raising_tbl"]["error"]

    def test_non_dict_payload_captured_as_error(self, tmp_path: Path):
        """A connector that returns the wrong shape (string, list,
        None) gets caught at validation time so the LLM never sees a
        half-populated entry that looks valid."""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        tool = _mock_describe_table_tool({"weird_tbl": {"result": "not a dict"}})
        bake_key_tables_schema(
            db_func_tool=tool,
            key_tables=["weird_tbl"],
            analysis_dir=analysis_dir,
        )
        out = json.loads((analysis_dir / "key_tables_schema.json").read_text())
        assert out["tables"][0]["columns"] == []
        # Pin the substring rather than the exact wording — error
        # message is constructed at the bake site and may be tuned.
        assert "unexpected payload type" in out["tables"][0]["error"]

    def test_columns_without_name_silently_skipped(self, tmp_path: Path):
        """Malformed column entries (non-dict, missing/blank name) are
        skipped at the column level so one bad row doesn't strand the
        rest of the table's schema."""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        tool = _mock_describe_table_tool(
            {
                "tbl": {
                    "result": {
                        "columns": [
                            {"name": "good", "type": "int", "comment": ""},
                            "not a dict",
                            {"name": "", "type": "int"},
                            {"type": "bigint"},  # no name
                            {"name": "also_good", "type": "varchar"},
                        ],
                    }
                }
            }
        )
        bake_key_tables_schema(
            db_func_tool=tool,
            key_tables=["tbl"],
            analysis_dir=analysis_dir,
        )
        out = json.loads((analysis_dir / "key_tables_schema.json").read_text())
        names = [c["name"] for c in out["tables"][0]["columns"]]
        assert names == ["good", "also_good"]


# --------------------------------------------------------------------------- #
# aggregate_referenced_tables                                                 #
# --------------------------------------------------------------------------- #


class TestAggregateReferencedTables:
    def test_missing_dir_returns_empty(self, tmp_path: Path):
        assert aggregate_referenced_tables(tmp_path / "nope") == []

    def test_empty_dir_returns_empty(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        assert aggregate_referenced_tables(queries_dir) == []

    def test_simple_select_picks_up_one_table(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "alpha.sql").write_text("SELECT * FROM Account", encoding="utf-8")
        assert aggregate_referenced_tables(queries_dir) == ["Account"]

    def test_join_picks_up_all_sides(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "join.sql").write_text(
            "SELECT * FROM Account a LEFT JOIN PersonOwnAccount poa ON a.id = poa.id",
            encoding="utf-8",
        )
        assert aggregate_referenced_tables(queries_dir) == ["Account", "PersonOwnAccount"]

    def test_two_part_qualified_preserved(self, tmp_path: Path):
        """``schema.table`` form is kept verbatim — the ask agent can
        copy it straight into a new SQL without inventing a prefix."""
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "schema_q.sql").write_text("SELECT * FROM main.Account", encoding="utf-8")
        assert aggregate_referenced_tables(queries_dir) == ["main.Account"]

    def test_three_part_qualified_preserved(self, tmp_path: Path):
        """Strict-schema dialects (DuckDB / Trino) need ``catalog.schema.table``;
        dropping any segment would force the ask agent to guess on the
        next query it writes."""
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "full_q.sql").write_text(
            "SELECT * FROM finbench.main.Account a JOIN finbench.main.PersonOwnAccount poa ON a.id = poa.id",
            encoding="utf-8",
        )
        assert aggregate_referenced_tables(queries_dir) == [
            "finbench.main.Account",
            "finbench.main.PersonOwnAccount",
        ]

    def test_qualified_beats_bare_in_dedupe(self, tmp_path: Path):
        """Mixed-style project: one file qualifies, another doesn't —
        the qualified form wins so the saved name is always copy-pastable."""
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "bare.sql").write_text("SELECT * FROM Account", encoding="utf-8")
        (queries_dir / "qualified.sql").write_text("SELECT * FROM finbench.main.Account", encoding="utf-8")
        # Only the qualified form survives.
        assert aggregate_referenced_tables(queries_dir) == ["finbench.main.Account"]

    def test_different_catalogs_for_same_bare_name_both_kept(self, tmp_path: Path):
        """Same bare name, different qualifications → really two different
        tables (e.g. prod ``main.Account`` vs audit ``audit.Account``)
        — both must survive dedupe."""
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "prod.sql").write_text("SELECT * FROM finbench.main.Account", encoding="utf-8")
        (queries_dir / "audit.sql").write_text("SELECT * FROM finbench.audit.Account", encoding="utf-8")
        assert aggregate_referenced_tables(queries_dir) == [
            "finbench.audit.Account",
            "finbench.main.Account",
        ]

    def test_cte_aliases_are_filtered(self, tmp_path: Path):
        """A WITH-clause alias must not leak into key_tables — the LLM /
        UI would otherwise see ``monthly`` as if it were a real table."""
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "cte.sql").write_text(
            "WITH monthly AS (SELECT * FROM Account) SELECT * FROM monthly",
            encoding="utf-8",
        )
        # Only the real underlying table survives.
        assert aggregate_referenced_tables(queries_dir) == ["Account"]

    def test_dedup_across_multiple_files(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "a.sql").write_text("SELECT * FROM Account", encoding="utf-8")
        (queries_dir / "b.sql").write_text("SELECT * FROM Account WHERE x=1", encoding="utf-8")
        (queries_dir / "c.sql").write_text("SELECT * FROM Person", encoding="utf-8")
        # Sorted alphabetically; Account appears once despite two refs.
        assert aggregate_referenced_tables(queries_dir) == ["Account", "Person"]

    def test_dashboard_template_jinja_blocks_stripped_before_parse(self, tmp_path: Path):
        """Dashboard ``.sql.j2`` files mix Jinja2 control flow into the SQL.
        The extractor must strip ``{% %}`` / ``{{ }}`` tokens before
        handing the body to sqlglot or the parse will fail and we'd
        silently lose table refs."""
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "filtered.sql.j2").write_text(
            (
                "-- @datus-params region:string\n"
                "SELECT * FROM Account a\n"
                "{% if region %}WHERE a.region = {{ region }}{% endif %}\n"
                "JOIN Person p ON a.person_id = p.id"
            ),
            encoding="utf-8",
        )
        assert aggregate_referenced_tables(queries_dir) == ["Account", "Person"]

    def test_broken_file_does_not_crash_aggregate(self, tmp_path: Path):
        """sqlglot runs in ``error_level=IGNORE`` mode — broken SQL won't
        raise, and any identifiers sqlglot can still salvage are kept
        (better partial recovery than dropping the file on the floor).
        The pin here is purely "doesn't crash; siblings still contribute"."""
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "good.sql").write_text("SELECT * FROM Account", encoding="utf-8")
        (queries_dir / "bad.sql").write_text("SELECT * FROM ((( unclosed", encoding="utf-8")
        result = aggregate_referenced_tables(queries_dir)
        # The good file always contributes.
        assert "Account" in result
        # And it's a sorted list of strings — no exception, no None entries.
        assert result == sorted(result)
        assert all(isinstance(t, str) and t for t in result)


# --------------------------------------------------------------------------- #
# update_manifest_key_tables                                                  #
# --------------------------------------------------------------------------- #


class TestUpdateManifestKeyTables:
    def test_writes_key_tables_to_existing_manifest(self, tmp_path: Path):
        manifest_path = _seed_manifest(tmp_path)
        err = update_manifest_key_tables(manifest_path, ["Account", "Person"])
        assert err is None
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["key_tables"] == ["Account", "Person"]
        # Other fields stay intact.
        assert data["slug"] == "demo_report"
        assert data["name"] == "Demo report"

    def test_overwrites_instead_of_unioning(self, tmp_path: Path):
        """Edit-mode rerun where a query (and its table) was removed:
        ``key_tables`` is code-generated and authoritative each run,
        so the stale entry must NOT survive."""
        manifest_path = _seed_manifest(tmp_path)
        update_manifest_key_tables(manifest_path, ["Account", "Person", "OldTable"])
        update_manifest_key_tables(manifest_path, ["Account", "Person"])
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["key_tables"] == ["Account", "Person"]

    def test_no_op_when_identical(self, tmp_path: Path, monkeypatch):
        """If the value didn't change, skip the disk write (don't bump mtime
        needlessly). We probe by patching the underlying writer."""
        manifest_path = _seed_manifest(tmp_path)
        update_manifest_key_tables(manifest_path, ["Account"])  # establish baseline

        from datus.agent.node.visual_artifact import _visual_artifact_finalize as finalize_mod

        write_calls: list[Path] = []
        original = finalize_mod._atomic_write_text

        def _spy(path, content):
            write_calls.append(path)
            original(path, content)

        monkeypatch.setattr(finalize_mod, "_atomic_write_text", _spy)
        err = update_manifest_key_tables(manifest_path, ["Account"])
        assert err is None
        assert write_calls == []  # no second write because content was identical

    def test_missing_manifest_returns_error_string(self, tmp_path: Path):
        err = update_manifest_key_tables(tmp_path / "nope.json", ["Account"])
        assert err is not None
        assert "manifest missing" in err

    def test_corrupt_manifest_returns_error_string(self, tmp_path: Path):
        path = tmp_path / "manifest.json"
        path.write_text("{not-json", encoding="utf-8")
        err = update_manifest_key_tables(path, ["Account"])
        assert err is not None
        assert "unreadable" in err


# --------------------------------------------------------------------------- #
# _sanitize_curated_intent_md                                                 #
# --------------------------------------------------------------------------- #


_CURATED_BODY = (
    "### [2026-05-18T03:10:06Z] mode: new\n"
    "```\n"
    "Generate a banking user account growth analysis report\n"
    "```\n"
    "\n"
    "### [2026-05-18T03:32:30Z] mode: edit\n"
    "```\n"
    "Focus on risk control analysis\n"
    "```\n"
)


class TestSanitizeCuratedIntentMd:
    def test_plain_body_unchanged(self):
        """Already-clean output passes through verbatim (modulo strip)."""
        out = _sanitize_curated_intent_md(_CURATED_BODY)
        assert out.startswith("### ")
        assert "Focus on risk control" in out

    def test_strips_outer_triple_backtick_fence(self):
        """Outer wrap stripped; per-section inner fences (which are now
        part of the body) survive verbatim."""
        wrapped = f"```\n{_CURATED_BODY}\n```"
        out = _sanitize_curated_intent_md(wrapped)
        assert out.startswith("### ")
        # End-anchored regex picks the LAST closing fence as the wrapper close,
        # so the body's inner fences are preserved.
        assert "Focus on risk control" in out
        assert "```" in out  # inner section fences survived
        # The wrapper added exactly one extra ``` pair (open + close); the body
        # originally has one fence-pair per section.
        section_count = _CURATED_BODY.count("###")
        assert out.count("```") == 2 * section_count

    def test_strips_language_tagged_fence(self):
        """``` ```markdown``` ` and ``` ```md``` ` are common (DeepSeek / GPT)."""
        for tag in ("markdown", "md"):
            wrapped = f"```{tag}\n{_CURATED_BODY}\n```"
            out = _sanitize_curated_intent_md(wrapped)
            assert out.startswith("### "), f"failed for tag={tag!r}"
            # Inner per-section fences preserved; only the outer language-
            # tagged wrapper is gone (the tag itself must not survive).
            assert tag not in out
            assert "Focus on risk control" in out

    def test_strips_leading_preface(self):
        """GPT-style 'Here is the cleaned version:' preface gone."""
        with_preface = f"Here is the cleaned version:\n\n{_CURATED_BODY}"
        out = _sanitize_curated_intent_md(with_preface)
        assert out.startswith("### ")
        assert "Here is the cleaned version" not in out

    def test_strips_preface_with_trailing_chatter_left_alone(self):
        """Preface stripped; trailing chatter past the body is preserved.

        Previously a third pass tried to trim a stray closing ``` line
        after the last ``### `` heading, but per-section fenced code
        blocks introduced by the verbatim-prompt format make that
        heuristic ambiguous (every section legitimately ends with
        ``\n``` ``). Sanitize now stops at "strip preface"; the
        downstream length / no-heading safety checks pick up any LLM
        slop that survives."""
        composite = f"Sure! Here is the curated intent.md:\n\n{_CURATED_BODY}\nLet me know if you'd like further edits."
        out = _sanitize_curated_intent_md(composite)
        assert out.startswith("### ")
        assert "Sure!" not in out
        # Trailing chatter survives — that's the explicit trade-off.
        assert "Let me know if you'd like further edits." in out

    def test_body_with_no_heading_returned_as_is(self):
        """If sanitize can't find a ``### `` heading the input is
        returned (stripped). The caller's safety check then fails the
        write — we don't try to 'fix' un-fixable input here."""
        ill_formed = "I cannot perform this task."
        out = _sanitize_curated_intent_md(ill_formed)
        assert out == "I cannot perform this task."


# --------------------------------------------------------------------------- #
# run_intent_curation                                                         #
# --------------------------------------------------------------------------- #


_ORIGINAL_INTENT = (
    "### [2026-05-18T03:10:06Z] mode: new\n"
    "```\n"
    "Generate a banking user account growth analysis report\n"
    "```\n"
    "\n"
    "### [2026-05-18T03:31:42Z] mode: edit\n"
    "```\n"
    "continue\n"
    "```\n"
    "\n"
    "### [2026-05-18T03:32:30Z] mode: edit\n"
    "```\n"
    "Focus on risk control analysis\n"
    "```\n"
)


def _curation_model(*, returns):
    """Build a Mock that implements ``LLMBaseModel.generate(prompt) -> str``."""
    m = Mock(spec=["generate"])
    if isinstance(returns, Exception):
        m.generate.side_effect = returns
    else:
        m.generate.return_value = returns
    return m


class TestRunIntentCuration:
    def test_missing_file_is_silent_noop(self, tmp_path: Path):
        """Programmatic test setups may skip creating intent.md
        entirely; the curator must noop without warning."""
        result = run_intent_curation(_curation_model(returns="unused"), tmp_path / "absent.md")
        assert result is None

    def test_empty_file_is_silent_noop(self, tmp_path: Path):
        path = tmp_path / "intent.md"
        path.write_text("   \n\n", encoding="utf-8")
        result = run_intent_curation(_curation_model(returns="unused"), path)
        assert result is None
        # File untouched.
        assert path.read_text(encoding="utf-8") == "   \n\n"

    def test_happy_path_rewrites_file(self, tmp_path: Path):
        """LLM returns a clean curated body — sanitize passes through,
        safety checks pass, file rewritten."""
        path = tmp_path / "intent.md"
        path.write_text(_ORIGINAL_INTENT, encoding="utf-8")
        result = run_intent_curation(_curation_model(returns=_CURATED_BODY), path)
        assert result is None
        rewritten = path.read_text(encoding="utf-8")
        # The dropped "continue" section's content line is no longer present.
        assert "\ncontinue\n" not in rewritten
        assert "Generate a banking" in rewritten  # kept
        assert "Focus on risk control" in rewritten  # kept
        # Trailing newline normalised.
        assert rewritten.endswith("\n")

    def test_fence_wrapped_output_sanitized_then_written(self, tmp_path: Path):
        """LLM wraps output in ```markdown — sanitize strips the outer
        wrapper before safety checks; per-section inner fences (which
        are now part of the curated body) survive verbatim."""
        path = tmp_path / "intent.md"
        path.write_text(_ORIGINAL_INTENT, encoding="utf-8")
        wrapped = f"```markdown\n{_CURATED_BODY}\n```"
        result = run_intent_curation(_curation_model(returns=wrapped), path)
        assert result is None
        rewritten = path.read_text(encoding="utf-8")
        assert rewritten.startswith("### ")
        # Outer wrapper's language tag must not have leaked into the file.
        assert "markdown" not in rewritten
        # Inner per-section fences (3-backtick) survived.
        assert "```" in rewritten
        assert "Generate a banking" in rewritten
        assert "Focus on risk control" in rewritten
        assert rewritten.startswith("### ")

    def test_identical_output_skips_write(self, tmp_path: Path, monkeypatch):
        """When the LLM returns the original content unchanged we skip
        the atomic_write so mtime stays stable across no-op finalize
        reruns. Probe via the atomic writer."""
        path = tmp_path / "intent.md"
        path.write_text(_ORIGINAL_INTENT, encoding="utf-8")

        from datus.agent.node.visual_artifact import _visual_artifact_finalize as finalize_mod

        writes: list = []
        original = finalize_mod._atomic_write_text
        monkeypatch.setattr(
            finalize_mod,
            "_atomic_write_text",
            lambda p, c: writes.append(p) or original(p, c),
        )
        result = run_intent_curation(_curation_model(returns=_ORIGINAL_INTENT), path)
        assert result is None
        assert writes == []

    def test_llm_exception_yields_warning_keeps_original(self, tmp_path: Path):
        """LLM call raises (network blip / provider 500) — curator
        records a warning and leaves intent.md untouched."""
        path = tmp_path / "intent.md"
        path.write_text(_ORIGINAL_INTENT, encoding="utf-8")
        result = run_intent_curation(
            _curation_model(returns=RuntimeError("provider 500")),
            path,
        )
        assert result is not None
        assert "LLM call failed" in result
        assert path.read_text(encoding="utf-8") == _ORIGINAL_INTENT

    def test_empty_llm_output_yields_warning_keeps_original(self, tmp_path: Path):
        path = tmp_path / "intent.md"
        path.write_text(_ORIGINAL_INTENT, encoding="utf-8")
        result = run_intent_curation(_curation_model(returns="   \n  "), path)
        assert result is not None
        assert "empty body" in result
        assert path.read_text(encoding="utf-8") == _ORIGINAL_INTENT

    def test_output_without_heading_yields_warning_keeps_original(self, tmp_path: Path):
        """LLM refused / returned prose — sanitize couldn't recover a
        usable body; safety check trips."""
        path = tmp_path / "intent.md"
        path.write_text(_ORIGINAL_INTENT, encoding="utf-8")
        result = run_intent_curation(
            _curation_model(returns="I'm sorry, I cannot perform this task."),
            path,
        )
        assert result is not None
        assert "no '### ' heading" in result
        assert path.read_text(encoding="utf-8") == _ORIGINAL_INTENT

    def test_too_short_output_yields_warning_keeps_original(self, tmp_path: Path):
        """LLM misinterpreted as 'summarise' and returned 1 tiny
        section. Below the 30% length floor → reject."""
        path = tmp_path / "intent.md"
        path.write_text(_ORIGINAL_INTENT, encoding="utf-8")
        tiny = "### [x] mode: y\n> z\n"
        result = run_intent_curation(_curation_model(returns=tiny), path)
        assert result is not None
        assert "too short" in result
        assert path.read_text(encoding="utf-8") == _ORIGINAL_INTENT


# --------------------------------------------------------------------------- #
# run_finalize_analysis                                                       #
# --------------------------------------------------------------------------- #


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
        assert any(m["path"] == ["Revenue"] and m["name"] == "revenue_by_region" for m in refs["metrics"])
        assert result["subject_refs_count"]["metrics"] == 1

    def test_end_to_end_curates_intent_md_when_present(self, tmp_path: Path):
        """When intent.md exists, finalize triggers run_intent_curation
        which calls ``model.generate`` and rewrites the file with the
        cleaned body. The main ``model.generate_with_json_output``
        call still produces insights / suggested_questions in the
        same orchestration."""
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(tmp_path)
        (analysis_dir / "intent.md").write_text(_ORIGINAL_INTENT, encoding="utf-8")

        # This model mock implements BOTH the structured-output call
        # (insights / suggested_questions) AND the plain text call
        # used by intent curation.
        model = Mock(spec=["generate_with_json_output", "generate"])
        model.generate_with_json_output.return_value = _full_finalize_response()
        model.generate.return_value = _CURATED_BODY

        result = run_finalize_analysis(
            model=model,
            artifact_kind="report",
            artifact_dir=artifact_dir,
            queries_dir=queries_dir,
            analysis_dir=analysis_dir,
            actions=[],
        )

        assert result["ok"] is True
        # Both LLM calls fired exactly once.
        assert model.generate_with_json_output.call_count == 1
        assert model.generate.call_count == 1
        # intent.md was curated: placeholder dropped, real intents kept.
        curated = (analysis_dir / "intent.md").read_text(encoding="utf-8")
        assert "\ncontinue\n" not in curated
        assert "Generate a banking" in curated
        assert "Focus on risk control" in curated

    def test_end_to_end_curation_failure_does_not_block_finalize(self, tmp_path: Path):
        """If the curation LLM call fails (or returns garbage), the
        main finalize products still land on disk and the warning
        surfaces in the result. Intent.md is preserved unchanged."""
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(tmp_path)
        (analysis_dir / "intent.md").write_text(_ORIGINAL_INTENT, encoding="utf-8")

        model = Mock(spec=["generate_with_json_output", "generate"])
        model.generate_with_json_output.return_value = _full_finalize_response()
        model.generate.side_effect = RuntimeError("curation provider 500")

        result = run_finalize_analysis(
            model=model,
            artifact_kind="report",
            artifact_dir=artifact_dir,
            queries_dir=queries_dir,
            analysis_dir=analysis_dir,
            actions=[],
        )

        assert result["ok"] is True
        # Main products survived the curation failure.
        assert (analysis_dir / "insights.json").is_file()
        assert (analysis_dir / "suggested_questions.json").is_file()
        # Intent.md untouched.
        assert (analysis_dir / "intent.md").read_text(encoding="utf-8") == _ORIGINAL_INTENT
        # Warning surfaced for monitoring.
        assert any("LLM call failed" in w for w in result["warnings"])

    def test_end_to_end_populates_manifest_key_tables(self, tmp_path: Path):
        """Finalize writes the code-aggregated table list back to
        ``manifest.key_tables`` (the ask agent's preamble surfaces this
        to skip schema-discovery round-trips)."""
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(
            tmp_path,
            sql_body="SELECT a.id FROM Account a JOIN PersonOwnAccount poa ON a.id = poa.id",
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
        assert result["key_tables"] == ["Account", "PersonOwnAccount"]
        manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["key_tables"] == ["Account", "PersonOwnAccount"]

    def test_end_to_end_bakes_key_tables_schema_when_db_tool_passed(self, tmp_path: Path):
        """When the orchestrator is given a ``db_func_tool`` (the
        production wiring), it bakes ``analysis/key_tables_schema.json``
        from the same key_tables it wrote to the manifest. ask_* reads
        this sidecar so SQL planning on the listed tables skips
        ``describe_table`` round-trips."""
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(
            tmp_path,
            sql_body="SELECT * FROM Account",
        )
        model = Mock(spec=["generate_with_json_output"])
        model.generate_with_json_output.return_value = _full_finalize_response()

        db_tool = _mock_describe_table_tool(
            {
                "Account": {
                    "result": {
                        "columns": [
                            {"name": "id", "type": "int", "comment": "primary key"},
                            {"name": "name", "type": "varchar", "comment": ""},
                        ],
                        "table": {"name": "Account", "description": ""},
                    }
                }
            }
        )

        result = run_finalize_analysis(
            model=model,
            artifact_kind="report",
            artifact_dir=artifact_dir,
            queries_dir=queries_dir,
            analysis_dir=analysis_dir,
            actions=[],
            db_func_tool=db_tool,
        )

        assert result["ok"] is True
        schema_path = analysis_dir / "key_tables_schema.json"
        assert schema_path.is_file()
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        # The bake walks ``key_tables`` (already populated by the
        # update_manifest_key_tables step in the same orchestrator pass)
        # so the sidecar tables list mirrors the manifest entries.
        assert [t["name"] for t in schema["tables"]] == ["Account"]
        assert {c["name"] for c in schema["tables"][0]["columns"]} == {"id", "name"}

    def test_end_to_end_skips_schema_bake_without_db_tool(self, tmp_path: Path):
        """Backwards-compatible default: no ``db_func_tool`` ⇒ no schema
        sidecar. Existing finalize call sites (and the older
        BaseVisualArtifactAgenticNode signature before this PR) keep
        working unchanged."""
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(
            tmp_path,
            sql_body="SELECT * FROM Account",
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
            # No db_func_tool — explicit None to make the default-
            # behaviour assertion clear.
            db_func_tool=None,
        )

        assert result["ok"] is True
        assert result["key_tables"] == ["Account"]
        # Sidecar absent — only the deterministic manifest update fired.
        assert not (analysis_dir / "key_tables_schema.json").exists()

    def test_end_to_end_preserves_qualified_table_references(self, tmp_path: Path):
        """Real-world SQL is usually fully qualified (``finbench.main.Account``);
        the saved key_tables must keep that form so the ask agent can paste
        it into a new SQL on a strict-schema dialect (DuckDB) without
        having to guess the catalog/schema prefix."""
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(
            tmp_path,
            sql_body=(
                "SELECT * FROM finbench.main.Account a "
                "LEFT JOIN finbench.main.PersonOwnAccount poa ON a.accountId = poa.accountId"
            ),
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
        manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["key_tables"] == [
            "finbench.main.Account",
            "finbench.main.PersonOwnAccount",
        ]

    def test_subject_refs_skipped_when_no_uses_declared(self, tmp_path: Path):
        """Present-iff-non-empty: a brief without any subject-library
        ids must NOT produce a ``subject_refs.json`` file — an absent
        file is the honest "no attribution" signal."""
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(
            tmp_path, brief_uses={"metrics": [], "reference_sql": []}
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
        assert result["subject_refs_count"] == {"metrics": 0, "reference_sql": 0}

    def test_subject_refs_stale_file_removed_when_now_empty(self, tmp_path: Path):
        """Edit-mode rerun where all ``uses`` were dropped: a stale
        ``subject_refs.json`` from a prior run must be deleted so the
        absent-file signal stays accurate."""
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(
            tmp_path, brief_uses={"metrics": [], "reference_sql": []}
        )
        stale_path = analysis_dir / "subject_refs.json"
        stale_path.write_text(
            json.dumps(
                {
                    "metrics": [{"path": ["Stale"], "name": "old"}],
                    "reference_sql": [],
                }
            ),
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
        # Narrative outputs (insights / suggested_questions) are LLM-gated and
        # therefore absent.
        assert not (analysis_dir / "insights.json").exists()
        assert not (analysis_dir / "suggested_questions.json").exists()

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
        assert not (analysis_dir / "suggested_questions.json").exists()

    def test_subject_refs_and_key_tables_land_when_llm_exception(self, tmp_path: Path):
        """Deterministic outputs are decoupled from the LLM call.

        ``subject_refs.json`` is aggregated by walking ``queries/*.brief.json``
        and ``manifest.key_tables`` is aggregated by parsing ``queries/*.sql``
        with sqlglot — neither needs the model. The follow-up ``ask_*``
        consultant depends on both to function, so a failed finalize LLM
        must not strand them.
        """
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(
            tmp_path,
            brief_uses={"metrics": [{"path": ["Sales", "Revenue"], "name": "gross_revenue"}]},
            sql_body="SELECT * FROM finbench.main.Account",
        )

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

        # subject_refs.json was aggregated from the brief and persisted.
        refs_path = analysis_dir / "subject_refs.json"
        assert refs_path.is_file(), "subject_refs.json must be written even when the LLM fails"
        refs = json.loads(refs_path.read_text(encoding="utf-8"))
        assert [(r["path"], r["name"]) for r in refs["metrics"]] == [(["Sales", "Revenue"], "gross_revenue")]

        # manifest.key_tables was aggregated by sqlglot and persisted.
        manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["key_tables"] == ["finbench.main.Account"]

        # The result dict surfaces the deterministic counts so callers can
        # see what landed alongside the error.
        assert result["key_tables"] == ["finbench.main.Account"]
        assert result["subject_refs_count"] == {"metrics": 1, "reference_sql": 0}

    def test_subject_refs_and_key_tables_land_when_schema_validation_fails(self, tmp_path: Path):
        """Same decoupling guarantee, but exercised on the schema-validation
        failure path (LLM returned JSON the model rejected). Both fall back
        through the same orchestrator branch."""
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(
            tmp_path,
            brief_uses={"metrics": [{"path": ["Sales", "Revenue"], "name": "gross_revenue"}]},
            sql_body="SELECT * FROM finbench.main.Account",
        )

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

        refs_path = analysis_dir / "subject_refs.json"
        assert refs_path.is_file()
        refs = json.loads(refs_path.read_text(encoding="utf-8"))
        assert [(r["path"], r["name"]) for r in refs["metrics"]] == [(["Sales", "Revenue"], "gross_revenue")]

        manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["key_tables"] == ["finbench.main.Account"]

    def test_stale_narrative_files_removed_on_llm_failure(self, tmp_path: Path):
        """An edit-mode rerun whose finalize LLM call fails must leave the
        ``analysis/`` directory in a state consistent with the failure
        return contract (insights / suggested_questions absent).

        Mirrors the present-iff-non-empty cleanup ``write_subject_refs``
        already enforces for ``subject_refs.json`` — without this, a
        consumer reading ``analysis/insights.json`` after a failed
        rerun would see stale narrative from the previous successful
        run that doesn't match the current queries on disk.
        """
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(tmp_path)
        # Pretend a previous run produced narrative files on disk.
        stale_insights = analysis_dir / "insights.json"
        stale_sq = analysis_dir / "suggested_questions.json"
        stale_insights.write_text(json.dumps({"insights": []}), encoding="utf-8")
        stale_sq.write_text(json.dumps({"suggested_questions": []}), encoding="utf-8")

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
        assert not stale_insights.exists(), "stale insights.json must be removed when finalize LLM fails"
        assert not stale_sq.exists(), "stale suggested_questions.json must be removed when finalize LLM fails"

    def test_intent_curation_skipped_when_main_llm_fails(self, tmp_path: Path):
        """Intent curation is itself an LLM call; skipping it when the
        primary finalize call has already failed avoids burning a second
        request on a model that just returned us nothing usable."""
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(tmp_path)
        (analysis_dir / "intent.md").write_text(
            "### [2026-05-14T10:00:00Z] mode: new\n> real intent\n",
            encoding="utf-8",
        )

        model = Mock(spec=["generate_with_json_output", "generate"])
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
        # ``model.generate`` is the intent curation entry point — it must
        # not have been called when the primary finalize LLM blew up.
        assert model.generate.call_count == 0

    def test_on_progress_fires_for_each_narrative_stage(self, tmp_path: Path):
        """The streaming hook drives stage bubbles off ``on_progress``;
        the narrative path must signal stage 1 (insights LLM), 2 (intent
        curation) and 3 (schema bake) in order."""
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(tmp_path)
        (analysis_dir / "intent.md").write_text(_ORIGINAL_INTENT, encoding="utf-8")

        model = Mock(spec=["generate_with_json_output", "generate"])
        model.generate_with_json_output.return_value = _full_finalize_response()
        model.generate.return_value = _CURATED_BODY

        stages: list[int] = []
        result = run_finalize_analysis(
            model=model,
            artifact_kind="report",
            artifact_dir=artifact_dir,
            queries_dir=queries_dir,
            analysis_dir=analysis_dir,
            actions=[],
            on_progress=stages.append,
        )

        assert result["ok"] is True
        assert stages == [1, 2, 3]

    def test_on_progress_callback_error_does_not_break_finalize(self, tmp_path: Path):
        """A throwing ``on_progress`` must not abort finalize — the stage
        callbacks are best-effort UI nudges."""
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(tmp_path)

        model = Mock(spec=["generate_with_json_output", "generate"])
        model.generate_with_json_output.return_value = _full_finalize_response()

        def _boom(_stage: int) -> None:
            raise RuntimeError("queue closed")

        result = run_finalize_analysis(
            model=model,
            artifact_kind="report",
            artifact_dir=artifact_dir,
            queries_dir=queries_dir,
            analysis_dir=analysis_dir,
            actions=[],
            on_progress=_boom,
        )

        assert result["ok"] is True
        assert (analysis_dir / "insights.json").is_file()


class TestRunFinalizeSkipNarrative:
    """``skip_narrative=True`` (render-only edit): no LLM call, existing
    narrative files untouched, deterministic aggregations still run."""

    def _seed_existing_narrative(self, analysis_dir: Path) -> tuple[str, str]:
        insights = json.dumps([{"id": "i1", "title": "t", "summary": "s"}]) + "\n"
        sq = json.dumps([{"question": "q?", "kind": "quick"}]) + "\n"
        (analysis_dir / "insights.json").write_text(insights, encoding="utf-8")
        (analysis_dir / "suggested_questions.json").write_text(sq, encoding="utf-8")
        return insights, sq

    def test_skips_llm_and_preserves_existing_files(self, tmp_path: Path):
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(tmp_path)
        insights, sq = self._seed_existing_narrative(analysis_dir)
        (analysis_dir / "intent.md").write_text(_ORIGINAL_INTENT, encoding="utf-8")

        model = Mock(spec=["generate_with_json_output", "generate"])

        result = run_finalize_analysis(
            model=model,
            artifact_kind="report",
            artifact_dir=artifact_dir,
            queries_dir=queries_dir,
            analysis_dir=analysis_dir,
            actions=[],
            skip_narrative=True,
        )

        assert result["ok"] is True
        assert result["skipped_narrative"] is True
        # Neither LLM entry point fired.
        assert model.generate_with_json_output.call_count == 0
        assert model.generate.call_count == 0
        # Prior narrative files + intent.md left exactly as they were.
        assert (analysis_dir / "insights.json").read_text(encoding="utf-8") == insights
        assert (analysis_dir / "suggested_questions.json").read_text(encoding="utf-8") == sq
        assert (analysis_dir / "intent.md").read_text(encoding="utf-8") == _ORIGINAL_INTENT

    def test_still_refreshes_subject_refs_and_key_tables(self, tmp_path: Path):
        # A real FROM clause so key_tables aggregation has something to record.
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(
            tmp_path, sql_body="SELECT region FROM finbench.main.Account"
        )

        result = run_finalize_analysis(
            model=Mock(spec=["generate_with_json_output", "generate"]),
            artifact_kind="report",
            artifact_dir=artifact_dir,
            queries_dir=queries_dir,
            analysis_dir=analysis_dir,
            actions=[],
            skip_narrative=True,
        )

        assert result["ok"] is True
        # Deterministic outputs still produced from on-disk briefs/SQL.
        assert (analysis_dir / "subject_refs.json").is_file()
        assert result["subject_refs_count"]["metrics"] == 1
        assert "finbench.main.Account" in result["key_tables"]
        manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        assert "finbench.main.Account" in manifest["key_tables"]

    def test_does_not_create_narrative_files_when_absent(self, tmp_path: Path):
        # No prior insights/suggested_questions on disk: skip must not mint them.
        artifact_dir, queries_dir, analysis_dir = _make_artifact_layout(tmp_path)

        result = run_finalize_analysis(
            model=Mock(spec=["generate_with_json_output", "generate"]),
            artifact_kind="report",
            artifact_dir=artifact_dir,
            queries_dir=queries_dir,
            analysis_dir=analysis_dir,
            actions=[],
            skip_narrative=True,
        )

        assert result["ok"] is True
        assert not (analysis_dir / "insights.json").exists()
        assert not (analysis_dir / "suggested_questions.json").exists()
