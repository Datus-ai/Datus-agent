# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Finalize stage shared by ``GenVisualReportAgenticNode`` and
``GenVisualDashboardAgenticNode``.

Runs after ``validate_render`` succeeds (but before ``_post_validate_hook``)
and produces the two LLM-authored analysis files plus a code-aggregated
``analysis/subject_refs.json``:

* ``analysis/insights.json``       — confirmed findings (REPORT ONLY;
                                     dashboards write ``[]``)
* ``analysis/suggested_questions.json`` — 5 follow-up suggestions
* ``analysis/subject_refs.json``   — index of every subject-library id
                                     mentioned across queries/*.brief.json.
                                     **Present-iff-non-empty**: skipped
                                     entirely when no query declared any
                                     subject-library asset.

Implementation choices worth remembering:

* **Single LLM call** producing both LLM-authored files in one shot
  (schema ``FinalizeAnalysisOutput``). Independent call rather than
  reusing the main loop's last turn — see
  ``docs/analysis_artifacts.md`` §7 for the rationale.
* **No ``interpretation.json``**: an earlier iteration produced a
  separate "audience / goal / focus_questions" file, but it duplicated
  ``manifest.description`` and was redundant with
  ``insights[].evidence_queries``. The follow-up consultant reads the
  manifest and the insights directly.
* **subject_refs aggregation is id-only in this PR.** The schema reserves
  ``name`` / ``definition_or_summary`` / ``source`` for future
  population by a subject-library lookup pass; for now they're empty
  strings and the subagent reads ids only. The metadata snapshot is
  scheduled for the subagent-introduction PR which will inject the
  semantic-model / reference-sql / ext-knowledge stores.
* **Best-effort**: finalize failures are logged and surfaced on the
  node result but never break the main artifact (which is already on
  disk by the time finalize runs).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import sqlglot
from sqlglot.expressions import CTE, Table

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.schemas.analysis_artifacts import (
    FinalizeAnalysisOutput,
    SubjectAssetRef,
    SubjectRefs,
)
from datus.schemas.artifact_manifest import ArtifactManifest
from datus.tools.func_tool._visual_artifact_helpers import _atomic_write_text
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Jinja2 control structures + comments we strip before handing dashboard
# templates to sqlglot. Variable interpolations ``{{ ... }}`` inside SQL
# bodies (``WHERE x = {{ region }}``) would otherwise break the parser
# even though we only care about identifiers in FROM / JOIN clauses.
_JINJA_BLOCK_RE = re.compile(r"\{%-?.*?-?%\}", re.DOTALL)
_JINJA_COMMENT_RE = re.compile(r"\{#.*?#\}", re.DOTALL)
_JINJA_INTERP_RE = re.compile(r"\{\{.*?\}\}", re.DOTALL)


# Subject-library-aware tool names whose action history we surface as
# "reminder cards" in the finalize prompt — limits the LLM's chance of
# forgetting a subject asset it actually consulted earlier in the loop.
SUBJECT_TOOL_NAMES = {
    "get_metrics",
    "query_metrics",
    "read_reference_sql",
    "read_ext_knowledge",
    "list_subject_tree",
}


# --------------------------------------------------------------------------- #
# Prompt construction                                                         #
# --------------------------------------------------------------------------- #


def build_finalize_prompt(
    *,
    artifact_kind: str,
    intent_md: str,
    query_briefs: List[Dict[str, Any]],
    query_previews: List[Dict[str, Any]],
    action_history_hints: List[Dict[str, Any]],
    existing_insights: Optional[List[Dict[str, Any]]],
    existing_suggested_questions: Optional[List[Dict[str, Any]]],
) -> str:
    """Compose the single-shot finalize prompt.

    The prompt is intentionally long-form / declarative rather than
    chatty: the LLM has to emit one strict JSON object matching
    :class:`FinalizeAnalysisOutput`, so we want every constraint visible
    in one place.
    """
    is_dashboard = artifact_kind == "dashboard"
    sections: List[str] = []

    sections.append(
        "You are finalizing the analysis-artifact bundle for a visual "
        f"{artifact_kind} that has just been generated. Your job is to "
        "produce exactly ONE JSON object containing the confirmed findings "
        "and recommended follow-up questions."
    )

    sections.append("## OUTPUT SCHEMA (strict)")
    sections.append(
        "Return a single JSON object with the following top-level keys:\n"
        "  - `insights`: array of objects with `id` (slug, "
        "[a-z0-9_]{1,64}), `title`, `summary`, `confidence` (0..1), "
        "`evidence_queries` (string[]), `informed_by_knowledge` "
        "(string[]). 3–8 entries typical.\n"
        "  - `suggested_questions`: array of objects with `question`, "
        "`related_queries` (string[]), `related_insight` (string or "
        "null), `priority` (0..1). Aim for exactly 5 entries."
    )
    if is_dashboard:
        sections.append(
            "**DASHBOARD MODE**: dashboard queries are runtime-parameterized "
            "templates with no statically-known results. You MUST return "
            "`insights: []` (empty array). Suggested questions should focus "
            "on `how to use the dashboard` and `which filters/dimensions to "
            "explore`, NOT on data conclusions."
        )

    sections.append("## RAW USER PROMPTS (intent.md)")
    sections.append(intent_md.strip() or "(empty)")

    if existing_insights or existing_suggested_questions:
        sections.append("## PREVIOUS FINALIZE OUTPUT (edit mode)")
        sections.append(
            "An earlier finalize already produced the following. Treat it "
            "as a revisable draft: reuse what still holds, revise what's "
            "outdated, drop what's been refuted by newer queries."
        )
        if existing_insights:
            sections.append("### Previous insights")
            sections.append(json.dumps(existing_insights, ensure_ascii=False, indent=2))
        if existing_suggested_questions:
            sections.append("### Previous suggested_questions")
            sections.append(json.dumps(existing_suggested_questions, ensure_ascii=False, indent=2))

    sections.append("## QUERIES (briefs)")
    if query_briefs:
        sections.append(json.dumps(query_briefs, ensure_ascii=False, indent=2))
    else:
        sections.append("(no query briefs recorded — this is unexpected)")

    sections.append("## QUERY RESULT PREVIEWS")
    sections.append(
        "First few rows of each query result. Use these for grounding "
        "insights; do not invent statistics that don't appear here."
    )
    sections.append(json.dumps(query_previews, ensure_ascii=False, indent=2, default=str))

    if action_history_hints:
        sections.append("## SUBJECT-LIBRARY TOOL CALLS (reminder)")
        sections.append(
            "These are subject-library tools you invoked during this run. "
            "Any metric / reference-sql / ext-knowledge id that ACTUALLY "
            "informed a query should already be declared in that query's "
            "`uses` block above. This list is a sanity-check that nothing "
            "was forgotten — not a fresh source to invent ids from."
        )
        sections.append(json.dumps(action_history_hints, ensure_ascii=False, indent=2, default=str))

    sections.append(
        "## CONSTRAINTS RECAP\n"
        f"  - artifact_kind = {artifact_kind!r}\n"
        f"  - `insights` MUST be {'an empty array' if is_dashboard else '3–8 entries'}.\n"
        "  - Every `evidence_queries` / `related_queries` entry MUST be a "
        "query name that appears in the reasoning steps above.\n"
        "  - Every `related_insight` MUST reference an `id` you declare in "
        "this same response (or be null).\n"
        "  - `last_updated` MUST be a current ISO-8601 UTC timestamp."
    )

    return "\n\n".join(sections)


# --------------------------------------------------------------------------- #
# Helpers used by the base node to assemble the prompt inputs                 #
# --------------------------------------------------------------------------- #


def collect_query_briefs(queries_dir: Path) -> List[Dict[str, Any]]:
    """Load every ``<name>.brief.json`` in queries/ as a dict.

    Files that fail to parse are skipped with a warning — the artifact
    is still useful; missing brief entries just mean less context
    for the finalize call.
    """
    briefs: List[Dict[str, Any]] = []
    if not queries_dir.is_dir():
        return briefs
    for path in sorted(queries_dir.glob("*.brief.json")):
        try:
            briefs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)
    return briefs


def collect_query_previews(queries_dir: Path, *, max_rows: int = 5) -> List[Dict[str, Any]]:
    """Per-query columns + a few preview rows.

    Handles both report (``<name>.json`` carries a full result file with
    ``rows``) and dashboard (``<name>.params.json`` carries
    ``columns`` + ``sample_params`` only; no rows). We do not require
    either to be present — dashboards in particular may have a missing
    preview, in which case we emit just the slug + a note.
    """
    previews: List[Dict[str, Any]] = []
    if not queries_dir.is_dir():
        return previews
    for sql_path in sorted(queries_dir.glob("*.sql")) + sorted(queries_dir.glob("*.sql.j2")):
        # ``foo.sql.j2`` → slug ``foo``; ``foo.sql`` → slug ``foo``.
        slug = sql_path.name.split(".", 1)[0]
        # Report result file shape.
        result_path = queries_dir / f"{slug}.json"
        params_path = queries_dir / f"{slug}.params.json"
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                rows = payload.get("rows") or []
                previews.append(
                    {
                        "name": slug,
                        "kind": "report_result",
                        "columns": payload.get("columns", []),
                        "row_count": payload.get("row_count", len(rows)),
                        "preview_rows": rows[:max_rows],
                    }
                )
                continue
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to read %s: %s", result_path, exc)
        if params_path.is_file():
            try:
                payload = json.loads(params_path.read_text(encoding="utf-8"))
                previews.append(
                    {
                        "name": slug,
                        "kind": "dashboard_template",
                        "columns": payload.get("columns", []),
                        "sample_params": payload.get("sample_params", {}),
                        "sample_row_count": payload.get("sample_row_count", 0),
                    }
                )
                continue
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to read %s: %s", params_path, exc)
        # Couldn't read either form — still record the slug so the LLM
        # knows the query exists.
        previews.append({"name": slug, "kind": "unknown", "note": "no result file readable"})
    return previews


def collect_action_history_hints(actions: Iterable[ActionHistory]) -> List[Dict[str, Any]]:
    """Pull subject-library tool calls out of the action history.

    Returns ``[{tool, input, ids}, ...]`` — small enough to inline into
    the prompt without blowing the context budget. We only include
    SUCCESS actions; failed tool calls don't establish "the LLM saw this
    asset's content".
    """
    hints: List[Dict[str, Any]] = []
    for a in actions:
        if a.role != ActionRole.TOOL or a.status != ActionStatus.SUCCESS:
            continue
        if a.action_type not in SUBJECT_TOOL_NAMES:
            continue
        hint: Dict[str, Any] = {"tool": a.action_type}
        if isinstance(a.input, dict):
            hint["input"] = {k: a.input[k] for k in list(a.input)[:6]}  # cap keys
        # Best-effort: surface common id-bearing fields from the output.
        if isinstance(a.output, dict):
            interesting_fields = ("name", "subject_path", "metric", "id", "title")
            digest: Dict[str, Any] = {}
            for field in interesting_fields:
                if field in a.output:
                    digest[field] = a.output[field]
            if digest:
                hint["output_digest"] = digest
        hints.append(hint)
    return hints


# --------------------------------------------------------------------------- #
# Write phase                                                                 #
# --------------------------------------------------------------------------- #


def load_intent_md(analysis_dir: Path) -> str:
    """Return ``analysis/intent.md`` contents (or empty string if missing)."""
    path = analysis_dir / "intent.md"
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return ""


def load_existing_finalize_output(
    analysis_dir: Path,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    """Load the previous finalize pair if present (edit mode)."""

    def _load(name: str) -> Optional[Any]:
        path = analysis_dir / name
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return None

    return _load("insights.json"), _load("suggested_questions.json")


def parse_finalize_output(raw: Any, *, artifact_kind: str) -> FinalizeAnalysisOutput:
    """Validate the LLM's response against :class:`FinalizeAnalysisOutput`.

    Dashboard ``insights`` field is forced empty here (rather than only
    via the prompt) — the LLM might still emit insights even when told
    not to, and we'd rather quietly drop them than persist conclusions
    that should never have been minted from runtime-parameterized
    queries.

    Stray legacy ``interpretation`` keys produced by old prompts are
    silently dropped — the field was removed in the brief.json
    refactor; failing the whole finalize because a model still echoes
    it would be a needless regression.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Finalize LLM response must be a dict; got {type(raw).__name__}")
    if artifact_kind == "dashboard" and raw.get("insights"):
        logger.info("Dashboard finalize returned %d insights; discarding per artifact kind.", len(raw["insights"]))
        raw = dict(raw)
        raw["insights"] = []
    if "interpretation" in raw:
        raw = {k: v for k, v in raw.items() if k != "interpretation"}
    return FinalizeAnalysisOutput.model_validate(raw)


def write_finalize_output(analysis_dir: Path, *, output: FinalizeAnalysisOutput, artifact_kind: str) -> List[str]:
    """Persist insights + suggested_questions.

    Returns a list of warning strings for fields that wrote partially or
    not at all. Insight file is skipped on dashboards.
    """
    warnings: List[str] = []
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if artifact_kind == "report":
        try:
            _atomic_write_text(
                analysis_dir / "insights.json",
                json.dumps([i.model_dump() for i in output.insights], ensure_ascii=False, indent=2) + "\n",
            )
        except OSError as exc:
            warnings.append(f"failed to write insights.json: {exc}")
    try:
        _atomic_write_text(
            analysis_dir / "suggested_questions.json",
            json.dumps([q.model_dump() for q in output.suggested_questions], ensure_ascii=False, indent=2) + "\n",
        )
    except OSError as exc:
        warnings.append(f"failed to write suggested_questions.json: {exc}")
    return warnings


def aggregate_subject_refs(queries_dir: Path) -> SubjectRefs:
    """Build ``analysis/subject_refs.json`` by walking brief sidecars.

    First-PR scope: id collection + dedup only. Metadata snapshot
    (``name`` / ``definition_or_summary`` / ``source``) is left as
    empty strings — populating them needs the subject-library RAG
    stores wired in, which lands with the subagent-introduction PR.
    Empty-string fallbacks make consumers (subagent UI) degrade
    gracefully when the id-only mode is in effect.
    """
    metrics: Dict[str, SubjectAssetRef] = {}
    reference_sql: Dict[str, SubjectAssetRef] = {}
    ext_knowledge: Dict[str, SubjectAssetRef] = {}

    for brief in collect_query_briefs(queries_dir):
        uses = brief.get("uses") or {}
        if not isinstance(uses, dict):
            continue
        for asset_id in uses.get("metrics") or []:
            if isinstance(asset_id, str) and asset_id and asset_id not in metrics:
                metrics[asset_id] = SubjectAssetRef(id=asset_id, name="", definition_or_summary="", source="")
        for asset_id in uses.get("reference_sql") or []:
            if isinstance(asset_id, str) and asset_id and asset_id not in reference_sql:
                reference_sql[asset_id] = SubjectAssetRef(id=asset_id, name="", definition_or_summary="", source="")
        for asset_id in uses.get("ext_knowledge") or []:
            if isinstance(asset_id, str) and asset_id and asset_id not in ext_knowledge:
                ext_knowledge[asset_id] = SubjectAssetRef(id=asset_id, name="", definition_or_summary="", source="")

    return SubjectRefs(
        metrics=list(metrics.values()),
        reference_sql=list(reference_sql.values()),
        ext_knowledge=list(ext_knowledge.values()),
    )


def _extract_tables_from_one_sql(sql_text: str) -> set[str]:
    """Return the set of bare (unqualified) table names referenced by a
    single SQL statement, with CTE aliases filtered out.

    Returns an empty set when sqlglot can't parse the input — finalize
    treats table aggregation as best-effort and never raises.
    """
    if not sql_text or not sql_text.strip():
        return set()
    try:
        # ``dialect=None`` lets sqlglot pick its generic parser. The
        # finalize-stage manifest doesn't carry a stable dialect label
        # (``datasources`` holds logical labels like ``finbench``, not
        # ``postgres`` / ``duckdb``), and the generic parser handles
        # SELECT / WITH / JOIN syntax across every dialect this codebase
        # supports.
        parsed = sqlglot.parse_one(sql_text, dialect=None, error_level=sqlglot.ErrorLevel.IGNORE)
    except Exception as exc:  # sqlglot raises a bag of subclasses; broad catch is intentional
        logger.debug("sqlglot parse failed during key_tables extraction: %s", exc)
        return set()
    if parsed is None:
        return set()

    cte_names = set()
    for cte in parsed.find_all(CTE):
        alias = getattr(cte, "alias", None)
        if alias:
            cte_names.add(alias.lower())

    tables: set[str] = set()
    for tbl in parsed.find_all(Table):
        name = tbl.name
        if not name:
            continue
        if name.lower() in cte_names:
            continue
        tables.add(name)
    return tables


def _strip_jinja(template_text: str) -> str:
    """Remove Jinja2 control / comment / interpolation tokens so sqlglot
    can parse the underlying SQL skeleton. We only need the FROM / JOIN
    identifiers for key_tables, so over-stripping the parameter slots is
    fine — it can't accidentally invent table references."""
    without_blocks = _JINJA_BLOCK_RE.sub(" ", template_text)
    without_comments = _JINJA_COMMENT_RE.sub(" ", without_blocks)
    return _JINJA_INTERP_RE.sub(" ", without_comments)


def aggregate_referenced_tables(queries_dir: Path) -> List[str]:
    """Walk ``queries/*.sql`` + ``queries/*.sql.j2`` and return every
    distinct bare table name referenced across them, sorted alphabetically.

    Used to populate ``manifest.key_tables`` at finalize time so the
    follow-up ask agent doesn't need to grep every SQL file (or call
    ``list_tables`` / ``describe_table``) to discover which tables the
    artifact actually touches. CTEs are filtered out; aliases preserve
    case as the LLM wrote them (most likely PascalCase matching the
    underlying database).

    Best-effort: a file that fails to parse contributes nothing to the
    aggregate. An empty list is a legitimate result (no queries on
    disk, or every parse failed) and the manifest writer treats it as
    "no key tables to record" rather than an error.
    """
    tables: set[str] = set()
    if not queries_dir.is_dir():
        return []
    for sql_path in sorted(queries_dir.glob("*.sql")):
        try:
            text = sql_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read %s for key_tables aggregation: %s", sql_path, exc)
            continue
        tables.update(_extract_tables_from_one_sql(text))
    for tpl_path in sorted(queries_dir.glob("*.sql.j2")):
        try:
            text = _strip_jinja(tpl_path.read_text(encoding="utf-8"))
        except OSError as exc:
            logger.warning("Failed to read %s for key_tables aggregation: %s", tpl_path, exc)
            continue
        tables.update(_extract_tables_from_one_sql(text))
    return sorted(tables)


def update_manifest_key_tables(manifest_path: Path, key_tables: List[str]) -> Optional[str]:
    """Refresh ``manifest.key_tables`` in place.

    Read-validate-replace cycle (similar to ``upsert_manifest_after_save``)
    so any other manifest mutation that happened mid-run survives. We
    always overwrite ``key_tables`` rather than union-add: the field is
    code-generated and each finalize run reflects the current SQL set
    authoritatively, including queries that were deleted in an
    edit-mode rerun.

    Returns an error string on failure, ``None`` on success. Missing /
    corrupt manifest is non-fatal — finalize is the *last* step, the
    artifact's primary contract is already on disk by the time this
    runs.
    """
    if not manifest_path.is_file():
        return "manifest missing — cannot update key_tables"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"manifest unreadable while updating key_tables: {exc}"
    try:
        manifest = ArtifactManifest.model_validate(raw)
    except Exception as exc:
        return f"manifest schema validation failed during key_tables update: {exc}"
    if manifest.key_tables == key_tables:
        # No-op: skip the disk write so we don't bump mtime needlessly.
        return None
    manifest.key_tables = key_tables
    try:
        _atomic_write_text(
            manifest_path,
            json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2) + "\n",
        )
        return None
    except OSError as exc:
        logger.warning("Failed to write key_tables back to %s: %s", manifest_path, exc)
        return f"failed to write key_tables: {exc}"


def write_subject_refs(analysis_dir: Path, refs: SubjectRefs) -> Optional[str]:
    """Write ``subject_refs.json`` iff any bucket is non-empty.

    Present-iff-non-empty semantics: an absent file means "no
    subject-library attribution recorded"; a present file with empty
    arrays would lie to the follow-up consultant ("we looked, found
    nothing"). Skipping the write is the honest default. We also
    proactively delete any stale file from a prior run so the absent
    signal stays accurate after an edit-mode rerun drops all uses.
    """
    if not (refs.metrics or refs.reference_sql or refs.ext_knowledge):
        stale = analysis_dir / "subject_refs.json"
        if stale.is_file():
            try:
                stale.unlink()
            except OSError as exc:
                logger.warning("Failed to remove stale subject_refs.json: %s", exc)
        return None
    try:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            analysis_dir / "subject_refs.json",
            json.dumps(refs.model_dump(), ensure_ascii=False, indent=2) + "\n",
        )
        return None
    except OSError as exc:
        logger.warning("Failed to write subject_refs.json: %s", exc)
        return f"failed to write subject_refs.json: {exc}"


# --------------------------------------------------------------------------- #
# Self-check                                                                  #
# --------------------------------------------------------------------------- #


def consistency_check(
    *,
    queries_dir: Path,
    output: FinalizeAnalysisOutput,
) -> List[str]:
    """Best-effort referential check; returns a list of warning strings.

    Failures here never block the write — see docs §10. The warnings are
    logged and exposed on the node result so we can monitor LLM
    reference quality over time.
    """
    warnings: List[str] = []
    existing_query_slugs = {p.name.split(".", 1)[0] for p in queries_dir.iterdir()} if queries_dir.is_dir() else set()
    insight_ids = {i.id for i in output.insights}

    for insight in output.insights:
        for q in insight.evidence_queries:
            if q not in existing_query_slugs:
                warnings.append(f"insight {insight.id!r}.evidence_queries references missing query {q!r}")
    for sq in output.suggested_questions:
        for q in sq.related_queries:
            if q not in existing_query_slugs:
                warnings.append(f"suggested_question references missing query {q!r}")
        if sq.related_insight is not None and sq.related_insight not in insight_ids:
            warnings.append(f"suggested_question.related_insight {sq.related_insight!r} not in insights")

    for w in warnings:
        logger.warning("finalize consistency: %s", w)
    return warnings


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #


def run_finalize_analysis(
    *,
    model: Any,
    artifact_kind: str,
    artifact_dir: Path,
    queries_dir: Path,
    analysis_dir: Path,
    actions: Iterable[ActionHistory],
) -> Dict[str, Any]:
    """Top-level orchestrator. Returns a result dict::

        {
            "ok": True,
            "warnings": [...],
            "subject_refs_count": {"metrics": n, "reference_sql": n, "ext_knowledge": n},
            "key_tables": [...],
        }

    Or, on hard failure (LLM call exception or schema validation
    failure)::

        {
            "ok": False,
            "warnings": [...],
            "error": "...",
        }
    """
    warnings: List[str] = []

    intent_md = load_intent_md(analysis_dir)
    query_briefs = collect_query_briefs(queries_dir)
    query_previews = collect_query_previews(queries_dir)
    action_hints = collect_action_history_hints(actions)
    existing_insights, existing_sq = load_existing_finalize_output(analysis_dir)

    prompt = build_finalize_prompt(
        artifact_kind=artifact_kind,
        intent_md=intent_md,
        query_briefs=query_briefs,
        query_previews=query_previews,
        action_history_hints=action_hints,
        existing_insights=existing_insights,
        existing_suggested_questions=existing_sq,
    )

    try:
        raw = model.generate_with_json_output(prompt)
    except Exception as exc:
        logger.warning("Finalize LLM call failed: %s", exc)
        return {"ok": False, "warnings": warnings, "error": f"finalize llm call failed: {exc}"}

    try:
        output = parse_finalize_output(raw, artifact_kind=artifact_kind)
    except Exception as exc:
        logger.warning("Finalize output validation failed: %s", exc)
        return {"ok": False, "warnings": warnings, "error": f"finalize output invalid: {exc}"}

    warnings.extend(write_finalize_output(analysis_dir, output=output, artifact_kind=artifact_kind))

    refs = aggregate_subject_refs(queries_dir)
    write_err = write_subject_refs(analysis_dir, refs)
    if write_err:
        warnings.append(write_err)

    # Refresh manifest.key_tables — code-aggregated, LLM-free hint the
    # follow-up ask agent reads to skip schema discovery round-trips.
    # Soft-fail: warnings surface in the result, never block the artifact.
    key_tables = aggregate_referenced_tables(queries_dir)
    kt_err = update_manifest_key_tables(artifact_dir / "manifest.json", key_tables)
    if kt_err:
        warnings.append(kt_err)

    warnings.extend(consistency_check(queries_dir=queries_dir, output=output))

    return {
        "ok": True,
        "warnings": warnings,
        "key_tables": key_tables,
        "subject_refs_count": {
            "metrics": len(refs.metrics),
            "reference_sql": len(refs.reference_sql),
            "ext_knowledge": len(refs.ext_knowledge),
        },
    }
