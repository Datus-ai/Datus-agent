# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Finalize stage shared by ``GenVisualReportAgenticNode`` and
``GenVisualDashboardAgenticNode``.

Runs after ``validate_render`` succeeds (but before ``_post_validate_hook``)
and produces the three LLM-authored analysis files plus a code-aggregated
``analysis/subject_refs.json``:

* ``analysis/interpretation.json`` — LLM's structured read of intent.md
* ``analysis/insights.json``       — confirmed findings (REPORT ONLY;
                                     dashboards write ``[]``)
* ``analysis/suggested_questions.json`` — 5 follow-up suggestions
* ``analysis/subject_refs.json``   — index of every subject-library id
                                     mentioned across queries/*.reasoning.json

Implementation choices worth remembering:

* **Single LLM call** producing all three LLM-authored files in one shot
  (schema ``FinalizeAnalysisOutput``). Independent call rather than
  reusing the main loop's last turn — see
  ``docs/analysis_artifacts.md`` §7 for the rationale.
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.schemas.analysis_artifacts import (
    FinalizeAnalysisOutput,
    SubjectAssetRef,
    SubjectRefs,
)
from datus.tools.func_tool._visual_artifact_helpers import _atomic_write_text, utc_now_iso
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


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
    reasoning_steps: List[Dict[str, Any]],
    query_previews: List[Dict[str, Any]],
    action_history_hints: List[Dict[str, Any]],
    existing_interpretation: Optional[Dict[str, Any]],
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
        "produce exactly ONE JSON object describing the user's intent, "
        "the confirmed findings, and recommended follow-up questions."
    )

    sections.append("## OUTPUT SCHEMA (strict)")
    sections.append(
        "Return a single JSON object with the following top-level keys:\n"
        "  - `interpretation`: object with `audience` (string[]), `goal` "
        "(string), `focus_questions` (string[]), `last_updated` "
        "(ISO-8601 UTC string).\n"
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

    if existing_interpretation or existing_insights or existing_suggested_questions:
        sections.append("## PREVIOUS FINALIZE OUTPUT (edit mode)")
        sections.append(
            "An earlier finalize already produced the following. Treat it "
            "as a revisable draft: reuse what still holds, revise what's "
            "outdated, drop what's been refuted by newer queries."
        )
        if existing_interpretation:
            sections.append("### Previous interpretation")
            sections.append(json.dumps(existing_interpretation, ensure_ascii=False, indent=2))
        if existing_insights:
            sections.append("### Previous insights")
            sections.append(json.dumps(existing_insights, ensure_ascii=False, indent=2))
        if existing_suggested_questions:
            sections.append("### Previous suggested_questions")
            sections.append(json.dumps(existing_suggested_questions, ensure_ascii=False, indent=2))

    sections.append("## QUERIES (reasoning steps)")
    if reasoning_steps:
        sections.append(json.dumps(reasoning_steps, ensure_ascii=False, indent=2))
    else:
        sections.append("(no reasoning steps recorded — this is unexpected)")

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


def collect_reasoning_steps(queries_dir: Path) -> List[Dict[str, Any]]:
    """Load every ``<name>.reasoning.json`` in queries/ as a dict.

    Files that fail to parse are skipped with a warning — the artifact
    is still useful; missing reasoning entries just mean less context
    for the finalize call.
    """
    steps: List[Dict[str, Any]] = []
    if not queries_dir.is_dir():
        return steps
    for path in sorted(queries_dir.glob("*.reasoning.json")):
        try:
            steps.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)
    return steps


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
) -> tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    """Load the previous finalize trio if present (edit mode)."""

    def _load(name: str) -> Optional[Any]:
        path = analysis_dir / name
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return None

    return _load("interpretation.json"), _load("insights.json"), _load("suggested_questions.json")


def parse_finalize_output(raw: Any, *, artifact_kind: str) -> FinalizeAnalysisOutput:
    """Validate the LLM's response against :class:`FinalizeAnalysisOutput`.

    Dashboard ``insights`` field is forced empty here (rather than only
    via the prompt) — the LLM might still emit insights even when told
    not to, and we'd rather quietly drop them than persist conclusions
    that should never have been minted from runtime-parameterized
    queries.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Finalize LLM response must be a dict; got {type(raw).__name__}")
    if artifact_kind == "dashboard" and raw.get("insights"):
        logger.info("Dashboard finalize returned %d insights; discarding per artifact kind.", len(raw["insights"]))
        raw = dict(raw)
        raw["insights"] = []
    interp = raw.get("interpretation")
    if isinstance(interp, dict) and "last_updated" not in interp:
        interp["last_updated"] = utc_now_iso()
    return FinalizeAnalysisOutput.model_validate(raw)


def write_finalize_output(analysis_dir: Path, *, output: FinalizeAnalysisOutput, artifact_kind: str) -> List[str]:
    """Persist interpretation / insights / suggested_questions.

    Returns a list of warning strings for fields that wrote partially or
    not at all. Insight file is skipped on dashboards.
    """
    warnings: List[str] = []
    analysis_dir.mkdir(parents=True, exist_ok=True)
    try:
        _atomic_write_text(
            analysis_dir / "interpretation.json",
            json.dumps(output.interpretation.model_dump(), ensure_ascii=False, indent=2) + "\n",
        )
    except OSError as exc:
        warnings.append(f"failed to write interpretation.json: {exc}")
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
    """Build ``analysis/subject_refs.json`` by walking reasoning sidecars.

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

    for step in collect_reasoning_steps(queries_dir):
        uses = step.get("uses") or {}
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


def write_subject_refs(analysis_dir: Path, refs: SubjectRefs) -> Optional[str]:
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
    reasoning_steps = collect_reasoning_steps(queries_dir)
    query_previews = collect_query_previews(queries_dir)
    action_hints = collect_action_history_hints(actions)
    existing_interp, existing_insights, existing_sq = load_existing_finalize_output(analysis_dir)

    prompt = build_finalize_prompt(
        artifact_kind=artifact_kind,
        intent_md=intent_md,
        reasoning_steps=reasoning_steps,
        query_previews=query_previews,
        action_history_hints=action_hints,
        existing_interpretation=existing_interp,
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

    # Stamp last_updated server-side so the field is authoritative even
    # if the LLM produced a stale or omitted value.
    output.interpretation.last_updated = utc_now_iso()

    warnings.extend(write_finalize_output(analysis_dir, output=output, artifact_kind=artifact_kind))

    refs = aggregate_subject_refs(queries_dir)
    write_err = write_subject_refs(analysis_dir, refs)
    if write_err:
        warnings.append(write_err)

    warnings.extend(consistency_check(queries_dir=queries_dir, output=output))

    return {
        "ok": True,
        "warnings": warnings,
        "subject_refs_count": {
            "metrics": len(refs.metrics),
            "reference_sql": len(refs.reference_sql),
            "ext_knowledge": len(refs.ext_knowledge),
        },
    }
