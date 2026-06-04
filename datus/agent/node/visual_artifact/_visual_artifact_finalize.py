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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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


# User-visible status text for each finalize stage. The node streams these
# via a single chat bubble (CREATE then UPDATE_MESSAGE on the same id) so
# the user sees the bubble's content swap stage-to-stage while finalize's
# 10-15 s of LLM + describe_table work runs.
FINALIZE_STAGE_TEXT = {
    1: "Generating insights and follow-up questions...",
    2: "Refining analysis intent...",
    3: "Caching referenced table schemas, almost done...",
}


# Subject-library-aware tool names whose action history we surface as
# "reminder cards" in the finalize prompt — limits the LLM's chance of
# forgetting a subject asset it actually consulted earlier in the loop.
SUBJECT_TOOL_NAMES = {
    "get_metrics",
    "query_metrics",
    "read_reference_sql",
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
        "`kind` (`'quick'` or `'deep_dive'` — see SUGGESTED QUESTIONS "
        "CONTRACT below), `related_queries` (string[]), `related_insight` "
        "(string or null), `priority` (0..1). EXACTLY 5 entries, split as "
        "3 quick + 2 deep_dive."
    )
    if is_dashboard:
        sections.append(
            "**DASHBOARD MODE**: dashboard queries are runtime-parameterized "
            "templates with no statically-known results. You MUST return "
            "`insights: []` (empty array). Suggested questions should focus "
            "on `how to use the dashboard` and `which filters/dimensions to "
            "explore`, NOT on data conclusions. The quick / deep_dive split "
            "still applies: `quick` = answerable from the inlined template "
            "metadata + sample params + columns (e.g. 'which filters does "
            "this dashboard expose'); `deep_dive` = needs a real query run "
            "(e.g. 'what trend does dimension X show under filter Y')."
        )

    sections.append("## SUGGESTED QUESTIONS CONTRACT (3 quick + 2 deep_dive)")
    sections.append(
        "Generate EXACTLY 5 follow-up questions. They are surfaced to the "
        "user as chips in the artifact detail view; clicking a chip launches "
        "the ask-agent against this artifact's pre-loaded context. The split "
        "matters because the artifact already inlines insights, query briefs, "
        "preview rows, and key-table schemas into the ask prompt — chips that "
        "match that inlined context answer in ~0 tool calls; chips that go "
        "beyond it trigger a full new analysis loop.\n\n"
        '### 3 × `kind="quick"` — answerable from the inlined artifact\n'
        "The answer must be fully derivable from the QUERIES (briefs) and "
        "QUERY RESULT PREVIEWS shown above (plus the report's insights for "
        "report mode). A user clicking the chip must get an answer the "
        "ask-agent can write WITHOUT calling any tools.\n"
        '  - Frame as descriptive / lookup questions: "which months had X", '
        '"how does Y compare between A and B", "what share of Z falls '
        'above N", "which insight has the highest confidence".\n'
        "  - MUST cite at least one `related_queries` slug whose preview "
        "rows contain the answer, OR one `related_insight` whose `summary` "
        "already states the answer.\n"
        '  - AVOID "why" / "what caused" / "can we replicate" / "what '
        'factors contribute" — those require decomposition the artifact '
        "didn't pre-compute and belong in deep_dive.\n"
        "  - Self-check: before tagging a question quick, draft a one-sentence "
        "answer from the preview rows alone. If you cannot, it is deep_dive.\n\n"
        '### 2 × `kind="deep_dive"` — requires new analysis\n'
        "The question legitimately needs new SQL, new tables, customer / "
        "product / segment joins, or counterfactual analysis the existing "
        "queries didn't cover. The UI will warn the user that this takes "
        "longer (~5–10 tool calls expected).\n"
        '  - These are the "why" / "what factors" / "can we forecast" / '
        '"what segments drive" questions a sharp analyst would ask after '
        "reading the report.\n"
        "  - `related_queries` and `related_insight` are OPTIONAL but "
        "encouraged — point at the closest existing query as a starting "
        "hint so the consultant knows where to begin extending.\n"
        "  - Use this tag honestly: a question that needs columns NOT in the "
        "existing query previews is deep_dive even if the topic feels "
        "related to an existing insight.\n\n"
        "Distribution: aim for EXACTLY 3 quick + 2 deep_dive in the 5-entry "
        "output. The schema rejects quick questions with no grounding, and a "
        "post-validation check warns when the distribution drifts off-target."
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
        "  - `suggested_questions` MUST contain exactly 5 entries split as "
        "3 `kind='quick'` + 2 `kind='deep_dive'`.\n"
        "  - Every `kind='quick'` entry MUST cite a non-empty "
        "`related_queries` slug OR a non-null `related_insight` so the "
        "consultant has explicit grounding in the inlined artifact context.\n"
        "  - Every `evidence_queries` / `related_queries` entry MUST be a "
        "query name that appears in the reasoning steps above.\n"
        "  - Every `related_insight` MUST reference an `id` you declare in "
        "this same response (or be null)."
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

    For each ``queries/<name>.brief.json`` we parse the ``uses`` block
    through :class:`SubjectRefs`'s pydantic validator and dedup entries
    by ``(tuple(path), name)`` — the natural key for any subject-library
    asset. Briefs whose ``uses`` block fails validation (legacy
    string-id form, missing ``path`` / ``name``, etc.) are skipped with
    a warning so a single malformed brief doesn't strand the whole
    aggregate; ``coerce_uses_arg`` in ``save_query`` catches the same
    shape errors earlier on the write path, so this is a belt-and-
    suspenders boundary.

    First-occurrence wins per dedup key, preserving the order the LLM
    declared assets in across queries. The ``ask_*`` consultant
    reading the resulting file uses each ``(path, name)`` to call
    ``get_metrics`` / ``get_reference_sql`` for the canonical
    definition; no other metadata is snapshotted here because the
    subject library is the source of truth.
    """
    metrics: Dict[Tuple[Tuple[str, ...], str], SubjectAssetRef] = {}
    reference_sql: Dict[Tuple[Tuple[str, ...], str], SubjectAssetRef] = {}

    for brief in collect_query_briefs(queries_dir):
        raw_uses = brief.get("uses") or {}
        try:
            uses = SubjectRefs.model_validate(raw_uses)
        except Exception as exc:
            logger.warning(
                "Skipping brief %r — uses block failed schema validation: %s",
                brief.get("name", "<unknown>"),
                exc,
            )
            continue
        for bucket, ref_list in (
            (metrics, uses.metrics),
            (reference_sql, uses.reference_sql),
        ):
            for ref in ref_list:
                key = (tuple(ref.path), ref.name)
                bucket.setdefault(key, ref)

    return SubjectRefs(
        metrics=list(metrics.values()),
        reference_sql=list(reference_sql.values()),
    )


def _extract_tables_from_one_sql(sql_text: str) -> set[str]:
    """Return every table reference in a single SQL statement, preserving
    whatever qualification the SQL itself used.

    - ``FROM Account``                  → ``"Account"``
    - ``FROM main.Account``             → ``"main.Account"``
    - ``FROM finbench.main.Account``    → ``"finbench.main.Account"``

    The follow-up ask agent can paste the saved name straight into a new
    SQL it writes; stripping the prefix would force it to guess the
    catalog/schema and may produce queries that don't run on a strict-
    schema dialect (DuckDB / Trino). CTE aliases are filtered out so a
    ``WITH monthly AS (...)`` doesn't leak ``monthly`` into key_tables.

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
        # ``Table`` exposes ``catalog`` / ``db`` / ``name`` separately.
        # Reassemble in the order the SQL actually wrote them — empty
        # parts mean "not qualified at that level".
        parts = [p for p in (tbl.catalog, tbl.db, tbl.name) if p]
        tables.add(".".join(parts))
    return tables


def _dedupe_table_references(raw: set[str]) -> List[str]:
    """Collapse same-table-different-qualification entries.

    When a project writes one query as ``FROM Account`` and another as
    ``FROM finbench.main.Account``, sqlglot reports both. We want the
    qualified form (it's strictly more informative — the ask agent can
    paste it without guessing) and drop the bare alias. Two genuinely
    different tables that happen to share a bare name (e.g.
    ``finbench.main.Account`` and ``warehouse.audit.Account``) both
    survive — they have distinct fully-qualified strings.
    """
    by_bare: Dict[str, set[str]] = {}
    for name in raw:
        if not name:
            continue
        bare = name.rsplit(".", 1)[-1]
        by_bare.setdefault(bare, set()).add(name)

    result: set[str] = set()
    for variants in by_bare.values():
        qualified = {v for v in variants if "." in v}
        # If at least one qualified variant exists, keep them and drop the
        # bare alias (it's a strict subset of any qualified entry).
        # Otherwise the bare form is all we have.
        if qualified:
            result.update(qualified)
        else:
            result.update(variants)
    return sorted(result)


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
    distinct table reference, preserving the qualification each SQL
    used (``finbench.main.Account`` stays as is; bare ``Account``
    stays bare). Sorted alphabetically for diff stability.

    Used to populate ``manifest.key_tables`` at finalize time so the
    follow-up ask agent doesn't need to grep every SQL file (or call
    ``list_tables`` / ``describe_table``) to discover which tables the
    artifact actually touches, AND can paste the saved name straight
    into a new SQL without guessing the catalog/schema prefix.

    When the same table is referenced both qualified and unqualified
    across different files (one query writes ``FROM Account``, another
    writes ``FROM finbench.main.Account``), the qualified form wins
    and the bare alias is dropped — see ``_dedupe_table_references``.

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
    return _dedupe_table_references(tables)


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


def bake_key_tables_schema(
    *,
    db_func_tool: Optional[Any],
    key_tables: List[str],
    analysis_dir: Path,
) -> Optional[str]:
    """Snapshot ``describe_table`` output for every ``manifest.key_tables``
    entry into ``analysis/key_tables_schema.json``.

    The follow-up ``ask_*`` consultant inlines this file into the system
    prompt so SQL planning on the listed tables skips
    ``describe_table`` round-trips. Companion file to
    ``manifest.key_tables`` (names only) — that field stays a small
    list of strings for list-page consumers; the schema detail lives
    in this sidecar instead.

    Best-effort:

    * ``db_func_tool is None`` → silently skip (CLI tests, dry runs).
    * Empty ``key_tables`` → skip (no schema to bake).
    * ``describe_table`` per-table failure → record the error string
      on the table entry rather than aborting the whole bake — the
      LLM gets a partial schema plus a "schema unavailable" hint per
      missing table, and can re-fetch via ``describe_table`` for the
      blanks if the user asks.
    * Write OSError → returned as a warning string for the caller to
      surface; the artifact's primary contract (queries/, render/,
      manifest.json) is already on disk by the time this runs.

    Returns ``None`` on success / skip; a warning string on write
    failure. Per-table errors are NOT propagated as warnings — they
    travel with the sidecar so the prompt can be specific about which
    tables were unavailable.
    """
    if db_func_tool is None or not key_tables:
        # Present-iff-bakeable semantics: if a prior finalize wrote a
        # schema sidecar but the current run has nothing to bake
        # (key_tables emptied after an edit-mode rerun, or this run has
        # no db tool), the stale file would lie to the follow-up
        # consultant. Proactively unlink so the "absent" signal stays
        # accurate. Mirrors what ``write_subject_refs`` does for
        # subject_refs.json.
        stale = analysis_dir / "key_tables_schema.json"
        if stale.is_file():
            try:
                stale.unlink()
            except OSError as exc:
                # Surface to the caller's ``warnings`` list — silently
                # dropping the failure would let the next ask_* session
                # serve a snapshot the renderer believes is fresh.
                # ``run_finalize_analysis`` already appends the
                # write-side warnings from this function to the same
                # list; the unlink-side warning uses the same string
                # shape so consumers parsing warnings don't need a new
                # branch.
                logger.warning("Failed to remove stale %s: %s", stale, exc)
                return f"failed to remove stale key_tables_schema.json at {stale}: {exc}"
        return None

    # Lazy import keeps the schema module out of the dependency graph
    # for callers that only run the LLM-authored side of finalize.
    from datus.schemas.key_tables_schema import KeyTableColumn, KeyTableSchema, KeyTablesSchemaFile

    tables_out: List[KeyTableSchema] = []
    for table_name in key_tables:
        try:
            result = db_func_tool.describe_table(table_name=table_name)
        except Exception as exc:
            # Broad except: connector implementations raise their own
            # exception types and we don't want to enumerate them. The
            # bake is best-effort; a single misbehaving connector
            # shouldn't strand the whole sidecar.
            logger.warning("describe_table raised for %s during key_tables bake: %s", table_name, exc)
            tables_out.append(KeyTableSchema(name=table_name, error=f"describe_table raised: {exc}"))
            continue

        if result is None or getattr(result, "success", 1) == 0:
            err = (
                (getattr(result, "error", None) or "describe_table returned no usable result")
                if result
                else "describe_table returned None"
            )
            tables_out.append(KeyTableSchema(name=table_name, error=str(err)))
            continue

        payload = getattr(result, "result", None)
        if not isinstance(payload, dict):
            tables_out.append(
                KeyTableSchema(
                    name=table_name,
                    error=f"describe_table returned unexpected payload type: {type(payload).__name__}",
                )
            )
            continue

        raw_columns = payload.get("columns") or []
        if not isinstance(raw_columns, list):
            tables_out.append(
                KeyTableSchema(
                    name=table_name,
                    error=f"describe_table returned non-list columns: {type(raw_columns).__name__}",
                )
            )
            continue

        cols: List[KeyTableColumn] = []
        for raw_col in raw_columns:
            if not isinstance(raw_col, dict):
                continue
            cname = raw_col.get("name")
            if not isinstance(cname, str) or not cname:
                continue
            # ``is_dimension`` may legitimately be missing (no semantic
            # model). Distinguish "absent" from "False" via None.
            is_dimension = raw_col.get("is_dimension")
            if not isinstance(is_dimension, bool):
                is_dimension = None
            cols.append(
                KeyTableColumn(
                    name=cname,
                    type=str(raw_col.get("type") or ""),
                    comment=str(raw_col.get("comment") or ""),
                    is_dimension=is_dimension,
                )
            )

        table_meta = payload.get("table") if isinstance(payload.get("table"), dict) else {}
        description = str(table_meta.get("description") or "") if table_meta else ""

        tables_out.append(KeyTableSchema(name=table_name, description=description, columns=cols))

    schema_file = KeyTablesSchemaFile(tables=tables_out)
    out_path = analysis_dir / "key_tables_schema.json"
    try:
        _atomic_write_text(
            out_path,
            json.dumps(schema_file.model_dump(), ensure_ascii=False, indent=2) + "\n",
        )
    except OSError as exc:
        logger.warning("Failed to write %s: %s", out_path, exc)
        return f"failed to write key_tables_schema.json: {exc}"
    return None


def write_subject_refs(analysis_dir: Path, refs: SubjectRefs) -> Optional[str]:
    """Write ``subject_refs.json`` iff any bucket is non-empty.

    Present-iff-non-empty semantics: an absent file means "no
    subject-library attribution recorded"; a present file with empty
    arrays would lie to the follow-up consultant ("we looked, found
    nothing"). Skipping the write is the honest default. We also
    proactively delete any stale file from a prior run so the absent
    signal stays accurate after an edit-mode rerun drops all uses.
    """
    if not (refs.metrics or refs.reference_sql):
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
# Intent.md curation (independent LLM call)                                   #
# --------------------------------------------------------------------------- #
#
# A second, dedicated LLM call cleans ``analysis/intent.md`` after the
# main finalize call writes insights / suggested_questions. We use a
# plain ``model.generate(prompt) -> str`` pass-through (per the project
# rule "no hardcoded LLM calls in nodes — go through LLMBaseModel") and
# accept a full markdown body back; the index-based alternative was
# considered and rejected — see commit history for the design trail.
#
# Three guard rails keep this safe:
#
# 1. **Prompt contract** — the LLM is told the response must START with
#    ``#`` and contain ONLY the curated markdown (no fence, no preface,
#    no commentary). It's also told to make a *binary* keep/delete
#    decision per section: never rewrite a user's words.
# 2. **Sanitize** — Python strips outer ```...``` fences (with or
#    without language tag) and any preface text before the first
#    ``### `` heading. Trailing chatter is left alone — the per-section
#    fenced code blocks introduced for verbatim prompt capture mean a
#    stray ``\n```\s*`` line after the last heading is now
#    indistinguishable from a section's own closing fence, so heuristic
#    trimming there would shred legitimate content.
# 3. **Safety checks** — empty body, no ``### `` heading, or shrinking
#    below 30% of original length all abort the rewrite. The original
#    file survives unchanged; a warning surfaces in the finalize
#    result for monitoring.

# Match an entire-body code fence with optional language tag, e.g.
# ``` or ```markdown or ```md. ``DOTALL`` lets ``.*?`` cross newlines.
_CURATED_OUTER_FENCE_RE = re.compile(
    r"^```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)\n```\s*$",
    re.DOTALL,
)

# Minimum size of curated output as a fraction of the original. Below
# this threshold we assume the LLM misinterpreted the task as
# "summarise" (or refused) and refuse to overwrite.
_CURATED_MIN_LENGTH_RATIO = 0.30

# Absolute floor — even tiny intent.md files shouldn't be allowed to
# shrink past this many characters. Prevents the ratio check from
# greenlighting "30% of 50 chars = 15 chars" obvious nonsense.
_CURATED_MIN_LENGTH_ABSOLUTE = 50


def _build_intent_curation_prompt(intent_md: str) -> str:
    """Compose the standalone curation prompt.

    Long and declarative: the LLM has one shot to emit a valid markdown
    body, and we want every constraint visible in one place. Examples
    bias the model toward the "binary keep/delete" decision and away
    from the tempting "let me rephrase this" behavior.
    """
    return (
        "You are curating an analysis-artifact `intent.md` file for a visual "
        "report or dashboard. The file records raw user prompts as timestamped "
        "sections — your job is to REMOVE sections that carry no real direction "
        "and KEEP sections that express genuine intent, in their original wording.\n"
        "\n"
        "## CROSS-LANGUAGE NOTE\n"
        "\n"
        "User prompts may be in any natural language (English, Chinese,\n"
        "Japanese, Korean, etc.). The rules below apply by semantic meaning,\n"
        "NOT by exact string match — apply them across languages. The examples\n"
        "below are illustrative in English; the same categories apply to\n"
        "non-English equivalents.\n"
        "\n"
        "## DELETE these section types\n"
        "\n"
        "**Operational nudges** — agreement / continuation prompts the user\n"
        "sent just to keep the agent loop going. Examples:\n"
        "  - `continue` / `keep going` / `go ahead` / `next` / `proceed`\n"
        "  - `ok` / `sure` / `done` / `that works` / `looks good`\n"
        "  - Same-meaning prompts in any other language follow the same rule.\n"
        "\n"
        "**Pure render / styling adjustments** — visual tweaks the follow-up\n"
        "ask agent (a read-only data consultant) cannot act on and gets no\n"
        "value from. Examples:\n"
        "  - Color / background / font size / spacing / margin / padding\n"
        "  - Layout reflow (`change to 2 columns` / `center-align` /\n"
        "    `switch to grid layout`)\n"
        "  - Component cosmetics (`rounded corners` / `add shadow` /\n"
        "    `bolder font` / `add animation`)\n"
        "  - Examples to DELETE: `change Overview background to red` /\n"
        "    `switch to 2-column layout` / `make the font larger` /\n"
        "    `add some border radius`\n"
        "\n"
        "## KEEP everything else\n"
        "\n"
        "Anything that carries a data, analysis, scope, metric, or analytical-\n"
        "view signal — even if the section also mentions a render tweak\n"
        "alongside it:\n"
        "  - New data dimensions / metrics / analysis angles\n"
        "    (`add a user growth activity bar chart` / `add an LTV analysis` /\n"
        "    `break down by industry`)\n"
        "  - Scope shifts (time window, geography, segment, filter)\n"
        "  - Tone / audience direction (`make it more executive-facing` /\n"
        "    `add a conclusion paragraph`)\n"
        "  - **Mixed prompts** containing ANY data/analysis signal alongside\n"
        "    render adjustments — keep the WHOLE section. Example:\n"
        "    `change Overview background to red, and add a user growth\n"
        "    activity bar chart` → KEEP (the 'new chart' part is real data\n"
        "    intent).\n"
        "  - Chart-type changes that imply a different analytical view\n"
        "    (`switch bar chart to pie chart`, `change monthly to quarterly`)\n"
        "    — these encode comparison semantics, not pure styling.\n"
        "\n"
        "**When in doubt: KEEP.** Losing a section the user genuinely cared\n"
        "about is far worse than keeping one noisy section.\n"
        "\n"
        "## ABSOLUTE RULES\n"
        "\n"
        "1. **Preserve verbatim.** Do NOT rewrite, translate, summarise, or\n"
        "   reformat user prose. A section either survives UNCHANGED, character\n"
        "   for character, or is deleted entirely. Never edit inside a section.\n"
        "   This applies regardless of the section's language — do NOT\n"
        "   translate Chinese / Japanese / Korean prompts into English (or\n"
        "   any other language).\n"
        "2. **Preserve section structure.** Each surviving section keeps its\n"
        "   `### [timestamp] mode: ...` heading line and its fenced code block\n"
        "   body (opening fence, content, closing fence) exactly as written.\n"
        "   The fence may be 3 or more backticks — preserve whatever length\n"
        "   the original section used. Sections are separated by one blank\n"
        "   line.\n"
        "3. **Order preserved.** Surviving sections appear in the original order.\n"
        "4. **No additions.** Do not insert notes, commentary, or new sections.\n"
        "5. **Binary decision per section.** Keep the whole section or delete\n"
        "   the whole section. Never `keep half` / `merge two` / `extract part`.\n"
        "\n"
        "## CRITICAL OUTPUT FORMAT\n"
        "\n"
        "- Output ONLY the curated markdown body.\n"
        "- Do NOT wrap the WHOLE output in an extra outer code fence (no\n"
        "  ```, no ```markdown around the entire response). The per-section\n"
        "  fenced code blocks ARE the section bodies — keep them; just\n"
        "  don't add a wrapping fence on top.\n"
        "- Do NOT add any preface text before the first `### ` heading\n"
        "  (no `Here is the cleaned version:` or similar lead-ins, in any\n"
        "  language).\n"
        "- Do NOT add any closing remark after the last section\n"
        "  (no `Hope this helps` or similar sign-offs, in any language).\n"
        "- The FIRST character of your response must be `#` (the start of\n"
        "  the first `### ` heading).\n"
        "- If after applying the rules every section should be deleted,\n"
        "  return the original file unchanged — never produce empty output.\n"
        "\n"
        "## CURRENT intent.md (curate this)\n"
        "\n"
        f"{intent_md.rstrip()}\n"
    )


def _sanitize_curated_intent_md(text: str) -> str:
    """Strip common LLM-output wrappers around the cleaned body.

    Two passes, narrow on purpose:

      1. **Whole-body fence**: ``` ... ``` or ```markdown ... ``` —
         the fence wraps the entire response.
      2. **Leading preface**: ``Here is the cleaned version:\\n\\n###...``
         — everything before the first ``### `` heading is dropped.

    Trailing commentary is intentionally NOT stripped: now that each
    legitimate section ends with its own fenced code block, a stray
    ``\\n```\\s*`` line after the last ``### `` heading is
    indistinguishable from the section's own closing fence. The
    downstream safety checks ("must contain ``###``", minimum length)
    plus the human-readable structure of intent.md make residual
    chatter tolerable.
    """
    s = text.strip()

    fence_match = _CURATED_OUTER_FENCE_RE.match(s)
    if fence_match:
        s = fence_match.group(1).strip()

    if not s.startswith("### "):
        first_heading = s.find("\n### ")
        if first_heading != -1:
            s = s[first_heading + 1 :].lstrip()

    return s.strip()


def run_intent_curation(model: Any, intent_md_path: Path) -> Optional[str]:
    """Read ``intent.md``, ask the LLM to curate it, sanitize, and write back.

    This is an INDEPENDENT LLM call — separate from the main finalize
    call that produces insights / suggested_questions. We keep it
    independent for two reasons:

    1. Schema separation: ``FinalizeAnalysisOutput`` is the contract
       for *persisted* outputs; intent curation is a transient action
       that doesn't deserve a slot there.
    2. Failure isolation: a curation hiccup must never block the
       finalize products from landing on disk.

    Returns a warning string when the curation degraded gracefully
    (LLM error / empty output / sanity-check fail) and the original
    file was preserved. Returns ``None`` on a clean rewrite or a clean
    no-op (LLM returned identical content / file missing).
    """
    if not intent_md_path.is_file():
        # Programmatic test runs may create no intent.md; nothing to curate.
        return None
    try:
        original = intent_md_path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"intent curation: failed to read intent.md ({exc})"

    if not original.strip():
        # Empty file — leave it alone, nothing to curate.
        return None

    prompt = _build_intent_curation_prompt(original)
    try:
        raw = model.generate(prompt)
    except Exception as exc:
        # LLMBaseModel.generate raises a bag of subclasses depending on
        # the provider; broad catch is intentional so finalize main
        # path is never blocked by curation failures.
        return f"intent curation: LLM call failed ({exc}); intent.md left unchanged"

    if not isinstance(raw, str) or not raw.strip():
        return "intent curation: LLM returned empty body; intent.md left unchanged"

    curated = _sanitize_curated_intent_md(raw)

    # Safety check 1: must contain at least one section heading. If the
    # LLM returned only a preface / fence / refusal, sanitize won't
    # have rescued a usable body.
    if "### " not in curated:
        return "intent curation: LLM output contained no '### ' heading after sanitize; intent.md left unchanged"

    # Safety check 2: minimum length floor. Below this we assume the LLM
    # misread the task as "summarise" or refused. Both ratio and
    # absolute floors apply.
    min_len = max(_CURATED_MIN_LENGTH_ABSOLUTE, int(len(original.strip()) * _CURATED_MIN_LENGTH_RATIO))
    if len(curated) < min_len:
        return f"intent curation: output too short ({len(curated)} chars vs floor {min_len}); intent.md left unchanged"

    # Normalise trailing newline so the file looks like every other one
    # the agent writes.
    new_body = curated.rstrip("\n") + "\n"

    if new_body == original:
        # No-op: LLM judged everything as real intent. Skip the write so
        # mtime stays stable across no-op finalize reruns.
        return None

    try:
        _atomic_write_text(intent_md_path, new_body)
    except OSError as exc:
        return f"intent curation: failed to rewrite intent.md ({exc})"
    return None


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
        # ``kind='quick'`` already had schema-level validation that at least
        # one of related_queries / related_insight is set. We re-check here
        # against the *resolved* world: a quick chip whose grounding points
        # at slugs that don't exist on disk (or an insight id this same
        # finalize call didn't declare) can't be answered from the inlined
        # context either — the schema validator only sees field presence,
        # not whether the referenced names resolve.
        if sq.kind == "quick":
            valid_query_refs = [q for q in sq.related_queries if q in existing_query_slugs]
            valid_insight_ref = sq.related_insight is not None and sq.related_insight in insight_ids
            if not valid_query_refs and not valid_insight_ref:
                warnings.append(
                    f"suggested_question {sq.question[:60]!r} kind='quick' has no "
                    "resolvable grounding (related_queries point to missing slugs "
                    "and related_insight is null/unknown); ask-agent will be unable "
                    "to answer from the inlined artifact"
                )

    # Distribution check — the contract asks for exactly 3 quick + 2 deep_dive
    # out of 5. The schema keeps the count range 1..8 forgiving (so a single
    # off-by-one doesn't strand the whole finalize), so we surface drift as
    # a warning here for ops visibility instead of as a hard failure.
    quick_count = sum(1 for sq in output.suggested_questions if sq.kind == "quick")
    deep_count = sum(1 for sq in output.suggested_questions if sq.kind == "deep_dive")
    total = len(output.suggested_questions)
    if total != 5 or quick_count != 3 or deep_count != 2:
        warnings.append(
            f"suggested_questions distribution off-target: total={total} "
            f"quick={quick_count} deep_dive={deep_count} (expected 5 / 3 / 2)"
        )

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
    db_func_tool: Optional[Any] = None,
    on_progress: Optional[Callable[[int], None]] = None,
    skip_narrative: bool = False,
) -> Dict[str, Any]:
    """Top-level orchestrator. Returns a result dict::

        {
            "ok": True,
            "warnings": [...],
            "subject_refs_count": {"metrics": n, "reference_sql": n},
            "key_tables": [...],
        }

    Or, on LLM failure (exception or schema validation failure)::

        {
            "ok": False,
            "warnings": [...],
            "error": "...",
            "key_tables": [...],
            "subject_refs_count": {"metrics": n, "reference_sql": n},
        }

    The deterministic, code-only outputs (``subject_refs.json`` and
    ``manifest.key_tables``) are produced whether or not the finalize
    LLM call succeeded — both are aggregated from on-disk artifacts
    (``queries/*.brief.json`` / ``queries/*.sql``) and have no LLM
    dependency. The follow-up ``ask_*`` consultant depends on
    ``subject_refs.json`` for subject-library attribution and on
    ``key_tables`` to skip schema-discovery round-trips, so we keep
    those usable even when the analytical narrative
    (``insights.json`` / ``suggested_questions.json``) is unavailable.

    Side effects on disk (all best-effort, surfaced via ``warnings``):

    * ``analysis/insights.json`` (report only, LLM-gated)
    * ``analysis/suggested_questions.json`` (LLM-gated)
    * ``analysis/intent.md`` — curated in place by a separate, dedicated
      LLM call (:func:`run_intent_curation`) that returns a fresh
      markdown body via ``model.generate``; safety checks reject the
      rewrite on empty / fence-only / dramatically-shrunk output.
      Gated on the main finalize LLM succeeding so we don't burn a
      second LLM call when the first one is already broken.
    * ``analysis/subject_refs.json`` — present iff non-empty. LLM-free.
    * ``manifest.json`` — ``key_tables`` field refreshed. LLM-free.

    ``skip_narrative`` short-circuits the two LLM stages (insights /
    suggested_questions + intent curation) for render-only edits where the
    underlying data is unchanged. The existing ``insights.json`` /
    ``suggested_questions.json`` are left in place (NOT deleted — they're
    still valid), and the deterministic aggregations below still run so the
    ask_* consultant's indices stay fresh. The result carries
    ``skipped_narrative: True`` and ``ok: True``.
    """
    warnings: List[str] = []
    output: Optional[FinalizeAnalysisOutput] = None
    llm_error: Optional[str] = None

    if not skip_narrative:
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

        if on_progress is not None:
            try:
                on_progress(1)
            except Exception as exc:
                logger.debug("Finalize on_progress(1) raised: %s", exc)

        try:
            raw = model.generate_with_json_output(prompt)
        except Exception as exc:
            logger.warning("Finalize LLM call failed: %s", exc)
            llm_error = f"finalize llm call failed: {exc}"

        if llm_error is None:
            try:
                output = parse_finalize_output(raw, artifact_kind=artifact_kind)
            except Exception as exc:
                logger.warning("Finalize output validation failed: %s", exc)
                llm_error = f"finalize output invalid: {exc}"

        if output is not None:
            warnings.extend(write_finalize_output(analysis_dir, output=output, artifact_kind=artifact_kind))

            if on_progress is not None:
                try:
                    on_progress(2)
                except Exception as exc:
                    logger.debug("Finalize on_progress(2) raised: %s", exc)

            # Independent LLM call: curate intent.md by stripping operational
            # nudges + pure render adjustments. Failures degrade to "leave the
            # original file unchanged + record a warning" — never blocks the
            # main artifact (which is already on disk by now). Gated on the
            # main finalize LLM succeeding so we don't keep hammering the
            # model when it's already returned us nothing usable.
            curation_warning = run_intent_curation(model, analysis_dir / "intent.md")
            if curation_warning:
                warnings.append(curation_warning)

    # Deterministic aggregations below — these only read files on disk
    # and never call the LLM, so they run regardless of LLM success.
    # That way an ask_* consultant always has its subject-library
    # attribution index and key_tables hint to work with, even when the
    # narrative-side outputs are missing.
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

    if on_progress is not None:
        try:
            on_progress(3)
        except Exception as exc:
            logger.debug("Finalize on_progress(3) raised: %s", exc)

    # Snapshot column metadata for every key_table into
    # ``analysis/key_tables_schema.json``. ask_* inlines this so SQL
    # planning skips ``describe_table`` round-trips on tables the
    # artifact already touched. Best-effort: per-table errors travel
    # with the sidecar (so the prompt can be specific about gaps);
    # only a write OSError surfaces as a warning here.
    schema_err = bake_key_tables_schema(
        db_func_tool=db_func_tool,
        key_tables=key_tables,
        analysis_dir=analysis_dir,
    )
    if schema_err:
        warnings.append(schema_err)

    subject_refs_count = {
        "metrics": len(refs.metrics),
        "reference_sql": len(refs.reference_sql),
    }

    if skip_narrative:
        # Render-only edit: narrative outputs were intentionally not
        # regenerated and the prior insights / suggested_questions stay on
        # disk untouched. No consistency_check (no fresh ``output`` to check).
        return {
            "ok": True,
            "skipped_narrative": True,
            "warnings": warnings,
            "key_tables": key_tables,
            "subject_refs_count": subject_refs_count,
        }

    if llm_error is not None:
        # LLM failed but the deterministic outputs are already on disk.
        # Surface both signals: ``ok=False`` (narrative outputs missing,
        # consumers should treat insights / suggested_questions as
        # absent) AND the counts (so callers can see we DID land
        # subject_refs / key_tables).
        #
        # Proactively delete any stale narrative files from a prior
        # successful run so the "absent" signal stays accurate after an
        # edit-mode rerun whose finalize LLM call fails. Mirrors the
        # same stale-cleanup contract :func:`write_subject_refs` already
        # enforces for ``subject_refs.json`` (present-iff-non-empty).
        # Best-effort: a failed unlink degrades to a warning so finalize
        # never raises on cleanup.
        for stale_name in ("insights.json", "suggested_questions.json"):
            stale_path = analysis_dir / stale_name
            if stale_path.is_file():
                try:
                    stale_path.unlink()
                except OSError as exc:
                    logger.warning("Failed to remove stale %s: %s", stale_name, exc)
                    warnings.append(f"failed to remove stale {stale_name}: {exc}")
        return {
            "ok": False,
            "warnings": warnings,
            "error": llm_error,
            "key_tables": key_tables,
            "subject_refs_count": subject_refs_count,
        }

    warnings.extend(consistency_check(queries_dir=queries_dir, output=output))

    return {
        "ok": True,
        "warnings": warnings,
        "key_tables": key_tables,
        "subject_refs_count": subject_refs_count,
    }
