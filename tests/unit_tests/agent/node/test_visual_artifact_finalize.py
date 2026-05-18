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
    _sanitize_curated_intent_md,
    aggregate_referenced_tables,
    aggregate_subject_refs,
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
    brief_uses: Dict[str, List[str]] | None = None,
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
            brief_uses = {"metrics": ["m_revenue"]}
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

        from datus.agent.node import _visual_artifact_finalize as finalize_mod

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
    "> Generate a banking user account growth analysis report\n"
    "\n"
    "### [2026-05-18T03:32:30Z] mode: edit\n"
    "> Focus on risk control analysis\n"
)


class TestSanitizeCuratedIntentMd:
    def test_plain_body_unchanged(self):
        """Already-clean output passes through verbatim (modulo strip)."""
        out = _sanitize_curated_intent_md(_CURATED_BODY)
        assert out.startswith("### ")
        assert "Focus on risk control" in out

    def test_strips_outer_triple_backtick_fence(self):
        wrapped = f"```\n{_CURATED_BODY}\n```"
        out = _sanitize_curated_intent_md(wrapped)
        assert out.startswith("### ")
        assert "```" not in out
        assert "Focus on risk control" in out

    def test_strips_language_tagged_fence(self):
        """``` ```markdown``` ` and ``` ```md``` ` are common (DeepSeek / GPT)."""
        for tag in ("markdown", "md"):
            wrapped = f"```{tag}\n{_CURATED_BODY}\n```"
            out = _sanitize_curated_intent_md(wrapped)
            assert out.startswith("### "), f"failed for tag={tag!r}"
            assert "```" not in out

    def test_strips_leading_preface(self):
        """GPT-style 'Here is the cleaned version:' preface gone."""
        with_preface = f"Here is the cleaned version:\n\n{_CURATED_BODY}"
        out = _sanitize_curated_intent_md(with_preface)
        assert out.startswith("### ")
        assert "Here is the cleaned version" not in out

    def test_strips_preface_and_fence_combined(self):
        """Worst case: preface + fence + trailing chatter all at once."""
        composite = (
            "Sure! Here is the curated intent.md:\n"
            "\n"
            f"```markdown\n{_CURATED_BODY}\n```\n"
            "\n"
            "Let me know if you'd like further edits."
        )
        out = _sanitize_curated_intent_md(composite)
        assert out.startswith("### ")
        assert "Sure!" not in out
        assert "```" not in out
        # Trailing chatter is intentionally NOT stripped — telling where
        # a legitimate blockquote ends from where LLM chatter begins is
        # a tough call. Downstream safety checks pick up the slack.

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
    "> Generate a banking user account growth analysis report\n"
    "\n"
    "### [2026-05-18T03:31:42Z] mode: edit\n"
    "> continue\n"
    "\n"
    "### [2026-05-18T03:32:30Z] mode: edit\n"
    "> Focus on risk control analysis\n"
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
        assert "> continue" not in rewritten  # dropped
        assert "Generate a banking" in rewritten  # kept
        assert "Focus on risk control" in rewritten  # kept
        # Trailing newline normalised.
        assert rewritten.endswith("\n")

    def test_fence_wrapped_output_sanitized_then_written(self, tmp_path: Path):
        """LLM wraps output in ```markdown — sanitize strips before
        the safety checks see it; write proceeds."""
        path = tmp_path / "intent.md"
        path.write_text(_ORIGINAL_INTENT, encoding="utf-8")
        wrapped = f"```markdown\n{_CURATED_BODY}\n```"
        result = run_intent_curation(_curation_model(returns=wrapped), path)
        assert result is None
        rewritten = path.read_text(encoding="utf-8")
        assert "```" not in rewritten
        assert rewritten.startswith("### ")

    def test_identical_output_skips_write(self, tmp_path: Path, monkeypatch):
        """When the LLM returns the original content unchanged we skip
        the atomic_write so mtime stays stable across no-op finalize
        reruns. Probe via the atomic writer."""
        path = tmp_path / "intent.md"
        path.write_text(_ORIGINAL_INTENT, encoding="utf-8")

        from datus.agent.node import _visual_artifact_finalize as finalize_mod

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
        assert any(m["id"] == "m_revenue" for m in refs["metrics"])
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
        assert "> continue" not in curated
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
