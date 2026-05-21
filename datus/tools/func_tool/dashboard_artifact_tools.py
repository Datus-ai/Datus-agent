# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tools for producing dashboard artifacts (render/*.jsx + queries/*.sql.j2).

Three complementary tools live here:

* ``DashboardArtifactTools.save_query_template`` — renders a Jinja2 SQL
  template with the LLM-supplied ``sample_params``, runs the resulting
  SQL through the existing ``DBFuncTool`` connector, infers column
  semantic types, and atomically persists ``<slug>.sql.j2`` and
  ``<slug>.params.json`` under the dashboard's ``queries/`` directory.
* ``DashboardArtifactTools.validate_render`` — the terminal action of
  this subagent. Walks ``dashboards/<id>/render/`` and verifies the
  entry point, import graph, every ``useQuerySql(sqlId, params)``
  literal's slug existence, and the params-key contract against the
  template's declaration.
* ``DashboardFilesystemFuncTool`` wraps the standard
  ``FilesystemFuncTool`` to reject writes/edits targeting ``queries/*``
  (use ``save_query_template``) and keep ``render/*`` writes limited to
  JSX/JS/CSS files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agents import Tool
from jinja2 import StrictUndefined
from jinja2.exceptions import TemplateError
from jinja2.sandbox import SandboxedEnvironment

from datus.schemas.artifact_manifest import ArtifactManifest
from datus.schemas.gen_visual_dashboard_models import (
    DASHBOARD_SLUG_RE,
    QUERY_SLUG_RE,
    QueryTemplateMetaFile,
    TemplateParamDecl,
    extract_query_slug,
    parse_datus_params_header,
)
from datus.tools.func_tool._artifact_filesystem_base import ArtifactFilesystemFuncTool
from datus.tools.func_tool._visual_artifact_helpers import (
    append_intent_section,
    coerce_uses_arg,
    upsert_manifest_after_save,
    utc_now_iso,
    write_query_brief,
)
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.func_tool.report_artifact_tools import (
    _DEFAULT_EXPORT_RE,
    _IMPORT_PATH_RE,
    ALLOWED_BARE_MODULES,
    _atomic_write_text,
    _infer_column_type,
    _looks_like_select,
    _module_key,
    _normalize_value,
    _resolve_relative_import,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


_MAX_TEMPLATE_BYTES = 256 * 1024  # 256 KB hard cap per template body
_MAX_PARAMS_META_BYTES = 64 * 1024  # 64 KB hard cap per params.json

# Capture every `useQuerySql(...)` call with **both** arguments. The second
# argument is matched as a balanced ``{...}`` block (with a permissive depth
# of 4 so simple nested object literals work) plus the trailing close paren.
# Calls without a second argument are surfaced as a separate issue — a
# dashboard always parameterizes its queries.
_USE_QUERY_SQL_LITERAL_RE = re.compile(
    r"""
    useQuerySql\s*\(\s*
    ['"]([^'"\\\n]+)['"]                 # 1: sqlId literal
    \s*,\s*
    (\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\})  # 2: params object literal (depth ≤ 3)
    \s*\)
    """,
    re.VERBOSE,
)

# A useQuerySql call with only one argument — the dashboard subagent rejects
# this; pass an explicit params object (use `{}` when the template takes none).
_USE_QUERY_SQL_NO_PARAMS_RE = re.compile(r"useQuerySql\s*\(\s*['\"][^'\"\\\n]+['\"]\s*\)")

# Property keys in the params object literal — only static string / bare
# identifier keys are statically checked. Computed keys or spread syntax
# defer to the runtime, but get flagged as a warning.
_PARAMS_KEY_RE = re.compile(
    r"""
    (?:^|[\s,{])                     # boundary
    (?:['"]([a-z_][a-z0-9_]*)['"]    # 1: 'name' or "name"
    |   ([a-z_][a-z0-9_]*))          # 2: bare name
    \s*:                             # mandatory colon
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ES6 shorthand property names: { foo, bar } ≡ { foo: foo, bar: bar }. The
# boundary intentionally excludes whitespace and the lookahead excludes `:`,
# so identifiers sitting in value position (e.g. `bar` in `{ foo: bar }`) are
# not captured.
_PARAMS_SHORTHAND_KEY_RE = re.compile(
    r"""
    (?:^|[,{])                       # boundary: open brace or comma only
    \s*
    ([a-z_][a-z0-9_]*)               # 1: bare identifier
    \s*
    (?=[,}])                         # lookahead: , or } (never :)
    """,
    re.VERBOSE | re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Jinja2 sandbox + bind-value resolution                                      #
# --------------------------------------------------------------------------- #


_JINJA_ENV = SandboxedEnvironment(
    autoescape=False,
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


def sql_quote_scalar(value: Any) -> str:
    """Quote a scalar literal for inline SQL substitution.

    Used both by the **trial render** inside ``save_query_template`` and by
    the view-time render in ``Datus-backend.services.dashboard_service``.
    Strings get their embedded single-quotes doubled (SQL-89 escape);
    numbers / booleans go in unquoted; ``None`` becomes ``NULL``.

    Note: this is **not** a substitute for driver-level bind values when
    the parameter source is untrusted user input. The dashboard service
    layer is expected to validate / coerce values against the declared
    types in ``-- @datus-params`` before handing them off here.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    # Strings / dates: escape embedded single-quotes by doubling.
    text = str(value).replace("'", "''")
    return f"'{text}'"


# Underscore alias kept for backwards compatibility with internal callers.
_sql_quote_scalar = sql_quote_scalar


# ``:name`` placeholder lexer — avoid matching things like ``::cast`` and
# quoted strings. We intentionally do not parse SQL here; the double-colon
# dodge keeps PostgreSQL casts working, and a single colon inside a quoted
# literal is rare and out-of-scope for the trial render.
_BIND_PLACEHOLDER_RE = re.compile(r"(?<![:'\w]):([a-z_][a-z0-9_]*)\b")


def resolve_bind_placeholders(rendered_sql: str, params_decl: List[TemplateParamDecl], values: Dict[str, Any]) -> str:
    """Replace ``:name`` bind placeholders with quoted literal values.

    Arrays expand to ``(a, b, c)``. Missing required params raise; missing
    optional params leave their ``:name`` placeholder in place to be caught
    by the SQL parser at execute time (the LLM almost certainly forgot to
    wrap the reference in a ``{% if %}``).
    """
    decl_by_name = {p.name: p for p in params_decl}

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        decl = decl_by_name.get(name)
        if decl is None:
            # Unknown :name — leave it for the SQL parser to flag.
            return match.group(0)
        if name not in values:
            if decl.required:
                raise ValueError(f"Required parameter {name!r} missing from sample_params.")
            return match.group(0)
        raw = values[name]
        if decl.is_array:
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"Parameter {name!r} is declared as an array but sample value is not a list.")
            if not raw:
                # Empty IN list is illegal SQL. Fall back to a guaranteed-false predicate.
                return "(NULL)"
            return "(" + ", ".join(_sql_quote_scalar(item) for item in raw) + ")"
        return _sql_quote_scalar(raw)

    return _BIND_PLACEHOLDER_RE.sub(replace, rendered_sql)


def extract_bind_names(sql_template: str) -> Set[str]:
    """Return the set of ``:name`` placeholders referenced by the SQL body.

    Comments (``-- ...`` and ``/* ... */``) are stripped first so the
    ``-- @datus-params`` header — whose ``name:type`` tokens look like binds
    to the lexer — is excluded from the scan. The same lookbehind regex as
    :func:`resolve_bind_placeholders` is used, so ``::cast`` does not
    false-match and string-literal boundaries behave identically.

    Operates on the **pre-Jinja2** template body. Names referenced only
    inside a ``{% if %}`` / ``{% for %}`` block via ``:name`` still count
    because the block's literal text contains the placeholder regardless of
    whether Jinja2 ultimately includes it at render time.
    """
    body = strip_sql_comments(sql_template)
    return {m.group(1) for m in _BIND_PLACEHOLDER_RE.finditer(body)}


_resolve_bind_placeholders = resolve_bind_placeholders


def strip_sql_comments(sql: str) -> str:
    """Strip SQL ``-- to EOL`` and ``/* ... */`` comments, preserving literals.

    Called between Jinja2 rendering and bind-placeholder substitution because
    the saved template's first line is the ``-- @datus-params <name>:<type>[:optional]``
    header — without this strip, the ``:optional`` tail would survive
    ``resolve_bind_placeholders`` (lookbehind sees ``]`` so the regex matches,
    and unknown names fall through to ``match.group(0)``) and the connector's
    SQL parser (SQLAlchemy / DuckDB) would treat the ``:optional`` inside the
    comment as an unbound placeholder. The same applies to any inline ``--``
    or ``/* */`` comment the LLM adds whose body happens to contain a ``:foo``
    token.

    String-literal-aware: ``'foo -- bar'`` and ``"col_with -- in name"``
    survive untouched, and SQL's ``''`` escape sequence is honored. Block
    comments are not nested (standard SQL).
    """
    out: List[str] = []
    i = 0
    n = len(sql)
    in_squote = False
    in_dquote = False
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if not in_squote and not in_dquote:
            if ch == "-" and nxt == "-":
                # Line comment — drop everything up to (but excluding) the newline.
                j = sql.find("\n", i)
                if j == -1:
                    break
                i = j
                continue
            if ch == "/" and nxt == "*":
                # Block comment — drop through ``*/``.
                j = sql.find("*/", i + 2)
                if j == -1:
                    # Unterminated block comment — drop to EOF rather than leave dangling text.
                    break
                i = j + 2
                continue
        if not in_dquote and ch == "'":
            if in_squote and sql[i + 1 : i + 2] == "'":
                out.append("''")
                i += 2
                continue
            in_squote = not in_squote
        elif not in_squote and ch == '"':
            in_dquote = not in_dquote
        out.append(ch)
        i += 1
    return "".join(out)


def render_dashboard_template(sql_template: str, params_decl: List[TemplateParamDecl], values: Dict[str, Any]) -> str:
    """Render a dashboard's Jinja2 SQL template and substitute bind placeholders.

    Used by both the save-time trial path (``DashboardArtifactTools.save_query_template``)
    and the view-time live path (``Datus-backend.services.dashboard_service.run_query``).
    Caller is responsible for **validating / coercing** ``values`` against the
    declared types before calling — this function trusts what it receives.

    Returns a ready-to-execute SQL string.
    """
    try:
        template = _JINJA_ENV.from_string(sql_template)
    except TemplateError as exc:
        raise ValueError(f"Jinja2 parse error: {exc}") from exc

    # Provide a Jinja2 context that includes every declared param. Missing
    # optional params land as ``None`` so ``{% if param %}`` skips the
    # clause; missing required params trip StrictUndefined the moment the
    # template body references them.
    context: Dict[str, Any] = {decl.name: None for decl in params_decl}
    context.update(values)

    try:
        rendered = template.render(**context)
    except TemplateError as exc:
        raise ValueError(f"Jinja2 render error: {exc}") from exc

    # Strip SQL comments before bind substitution — the ``-- @datus-params``
    # header would otherwise leak its ``:optional`` tail through to the
    # connector and trigger "A value is required for bind parameter 'optional'".
    rendered = strip_sql_comments(rendered)

    return resolve_bind_placeholders(rendered, params_decl, values)


# Underscore alias kept for backwards compatibility.
_render_template_for_trial = render_dashboard_template


# --------------------------------------------------------------------------- #
# Filesystem wrapper                                                          #
# --------------------------------------------------------------------------- #


class DashboardFilesystemFuncTool(ArtifactFilesystemFuncTool):
    """Filesystem tool that protects the dashboard artifact tree.

    * ``dashboards/<id>/queries/*`` — read-only via the filesystem layer.
      Writes must go through ``save_query_template`` so the Jinja2 template
      is parsed, rendered against ``sample_params``, executed, and the
      column types are inferred.
    * ``dashboards/<id>/render/*`` — writable, but only ``.jsx`` / ``.js`` /
      ``.css`` files.
    * Anything else under the project root inherits the parent's policy.
    """

    ARTIFACT_ROOT_DIR_NAME = "dashboards"
    SAVE_QUERY_TOOL_NAME = "save_query_template"
    ARTIFACT_KIND = "dashboard"


# --------------------------------------------------------------------------- #
# Artifact tools                                                              #
# --------------------------------------------------------------------------- #


class DashboardArtifactTools:
    """LLM-facing tools that produce the dashboard artifact tree.

    Lifecycle mirrors :class:`ReportArtifactTools`:

    1. The owning node constructs one instance per execution with no
       active dashboard id.
    2. The LLM declares intent and binds the active dashboard by calling
       **exactly one** of ``start_new_dashboard`` or
       ``bind_existing_dashboard``.
    3. ``save_query_template`` persists Jinja2 SQL templates after a
       trial render against ``sample_params``. ``write_file`` /
       ``edit_file`` / ``delete_file`` (from the filesystem tool) put
       JSX/JS/CSS under ``dashboards/<id>/render/``.
    4. ``validate_render`` walks the render tree, checks the entry
       point, verifies every ``useQuerySql`` slug exists, and confirms
       every ``params`` literal's keys match the template's declaration.
    """

    def __init__(
        self,
        *,
        agent_config,
        db_func_tool,
        user_message: str = "",
    ) -> None:
        self.agent_config = agent_config
        self._db_func_tool = db_func_tool
        # See ReportArtifactTools.__init__ for the rationale — mirror logic.
        self._user_message = user_message or ""

        project_root = Path(getattr(agent_config, "project_root", "")).resolve()
        if not project_root or str(project_root) == ".":
            raise ValueError("agent_config.project_root must be a non-empty directory")
        self._project_root = project_root

        # Lazy state — populated by start_new_dashboard / bind_existing_dashboard.
        self.dashboard_slug: Optional[str] = None
        self.dashboard_dir: Optional[Path] = None
        self.queries_dir: Optional[Path] = None
        self.render_dir: Optional[Path] = None
        self.analysis_dir: Optional[Path] = None
        self.mode: Optional[str] = None

    # -- public --------------------------------------------------------------

    def available_tools(self) -> List[Tool]:
        return [
            trans_to_function_tool(self.start_new_dashboard),
            trans_to_function_tool(self.bind_existing_dashboard),
            # ``save_query_template.sample_params`` is a free-form ``Dict[str, Any]``
            # — the keys are whatever the LLM declared in ``-- @datus-params`` and
            # the values are scalars / arrays. Strict mode would reject the
            # ``additionalProperties: true`` JSON schema this produces; we
            # validate keys + types ourselves once the call lands.
            trans_to_function_tool(self.save_query_template, strict_mode=False),
            trans_to_function_tool(self.validate_render),
        ]

    # -- intent declaration --------------------------------------------------

    def start_new_dashboard(self, slug: str, name: str, description: str) -> FuncToolResult:
        """
        Create a fresh dashboard directory at ``dashboards/<slug>/``, write its manifest, and bind it.

        The LLM picks the ``slug`` — it doubles as the on-disk directory
        name and as the stable identifier surfaced everywhere downstream
        (SaaS list pages, IDE explorer, backend routes). **Before calling
        this tool the LLM must ``glob('dashboards/*')`` and confirm the
        chosen slug doesn't collide** — this tool refuses to overwrite an
        existing directory.

        Args:
            slug: Lowercase ASCII identifier matching ``^[a-z0-9_]{1,80}$``.
                Becomes the directory name (``dashboards/<slug>/``). Pick
                something semantic and stable (e.g.
                ``revenue_overview``); do NOT include personal
                information or timestamps unless they're load-bearing
                for disambiguation.
            name: Human-readable display name (any language is fine —
                Chinese / mixed scripts welcome). Required, max 200 chars.
            description: One-paragraph description of what the
                dashboard tracks / answers. Surfaced in list pages and
                IDE explorers next to the name. Required, max 1000
                chars.

        Returns:
            FuncToolResult.result is a dict like::

                {
                    "dashboard_slug": "<slug>",
                    "dashboard_dir": "dashboards/<slug>",
                    "render_dir": "dashboards/<slug>/render",
                    "queries_dir": "dashboards/<slug>/queries",
                    "analysis_dir": "dashboards/<slug>/analysis",
                    "manifest_path": "dashboards/<slug>/manifest.json",
                    "mode": "new",
                }

            ``analysis/intent.md`` is seeded with the user's original prompt.
        """
        if not slug or not DASHBOARD_SLUG_RE.fullmatch(slug):
            return FuncToolResult(
                success=0,
                error=(
                    f"slug must match {DASHBOARD_SLUG_RE.pattern} (lowercase letters / digits / underscores, "
                    f"1–80 chars); got {slug!r}. Pick a semantic identifier; the LLM is responsible for "
                    "uniqueness within dashboards/."
                ),
            )
        if not name or not name.strip():
            return FuncToolResult(success=0, error="name must be a non-empty display name (any language).")
        if not description or not description.strip():
            return FuncToolResult(
                success=0,
                error="description must be a non-empty one-paragraph description of what the dashboard covers.",
            )
        candidate = self._project_root / "dashboards" / slug
        if candidate.exists():
            return FuncToolResult(
                success=0,
                error=(
                    f"dashboards/{slug}/ already exists. Pick a different slug — first `glob('dashboards/*')` "
                    "to see what's taken, or call `bind_existing_dashboard` if you meant to edit it."
                ),
            )
        try:
            manifest = ArtifactManifest(
                slug=slug,
                name=name.strip(),
                description=description.strip(),
                kind="dashboard",
                created_at=utc_now_iso(),
            )
        except Exception as exc:
            return FuncToolResult(success=0, error=f"Manifest validation failed: {exc}")
        return self._activate(slug, mode="new", create_dirs=True, manifest=manifest)

    def bind_existing_dashboard(self, dashboard_slug: str) -> FuncToolResult:
        """
        Switch the active dashboard to an existing one and bind subsequent saves there.

        Call this when the user asks to **modify / update / edit /
        append to** a specific named dashboard. ``save_query_template``
        overwrites same-named queries; ``write_file`` / ``edit_file`` /
        ``delete_file`` mutate ``render/`` in-place. Use ``read_file`` +
        ``glob`` to inspect the existing tree before mutating it.

        When the user references the dashboard by its display name rather
        than its slug (``"update the revenue overview dashboard"``), the
        LLM should first ``glob('dashboards/*/manifest.json')`` and read
        each manifest's ``name`` to find the matching slug, then call this
        tool with that slug.

        Args:
            dashboard_slug: target dashboard slug, e.g. ``"revenue_overview"``.
                Must match ``^[a-z0-9_]{1,80}$`` and the directory
                (including ``render/app.jsx``) must already exist under
                ``<project_root>/dashboards/``.

        Returns:
            FuncToolResult.result is a dict like::

                {
                    "dashboard_slug": "<slug>",
                    "dashboard_dir": "dashboards/<slug>",
                    "render_dir": "dashboards/<slug>/render",
                    "queries_dir": "dashboards/<slug>/queries",
                    "analysis_dir": "dashboards/<slug>/analysis",
                    "mode": "edit",
                }

            ``analysis/intent.md`` gets a new ``### [timestamp] mode: edit``
            section appended with the user's latest prompt.
        """
        if not dashboard_slug or not DASHBOARD_SLUG_RE.fullmatch(dashboard_slug):
            return FuncToolResult(
                success=0,
                error=f"dashboard_slug must match {DASHBOARD_SLUG_RE.pattern}; got {dashboard_slug!r}",
            )
        candidate = self._project_root / "dashboards" / dashboard_slug
        if not candidate.is_dir():
            return FuncToolResult(
                success=0,
                error=(
                    f"Dashboard directory not found: dashboards/{dashboard_slug}. "
                    "Use start_new_dashboard() if you intended to create a new dashboard."
                ),
            )
        if not (candidate / "render" / "app.jsx").is_file():
            return FuncToolResult(
                success=0,
                error=(
                    f"dashboards/{dashboard_slug}/render/app.jsx is missing — the dashboard is "
                    "incomplete. Cannot bind for editing."
                ),
            )
        return self._activate(dashboard_slug, mode="edit", create_dirs=False)

    def _activate(
        self,
        dashboard_slug: str,
        *,
        mode: str,
        create_dirs: bool,
        manifest: Optional[ArtifactManifest] = None,
    ) -> FuncToolResult:
        dashboard_dir = self._project_root / "dashboards" / dashboard_slug
        queries_dir = dashboard_dir / "queries"
        render_dir = dashboard_dir / "render"
        analysis_dir = dashboard_dir / "analysis"
        manifest_path = dashboard_dir / "manifest.json"
        if create_dirs:
            queries_dir.mkdir(parents=True, exist_ok=True)
            render_dir.mkdir(parents=True, exist_ok=True)
            analysis_dir.mkdir(parents=True, exist_ok=True)
        manifest_rel: Optional[str] = None
        if manifest is not None:
            try:
                _atomic_write_text(
                    manifest_path,
                    json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2) + "\n",
                )
            except OSError as exc:
                return FuncToolResult(success=0, error=f"Failed to write manifest.json: {exc}")
            manifest_rel = manifest_path.relative_to(self._project_root).as_posix()
        self.dashboard_slug = dashboard_slug
        self.dashboard_dir = dashboard_dir
        self.queries_dir = queries_dir
        self.render_dir = render_dir
        self.analysis_dir = analysis_dir
        self.mode = mode

        # Mirror of ReportArtifactTools._activate: append the raw user
        # prompt to analysis/intent.md as a best-effort log.
        intent_warning = append_intent_section(
            analysis_dir,
            user_message=self._user_message,
            mode=mode,
            timestamp=utc_now_iso(),
        )

        result: Dict[str, Any] = {
            "dashboard_slug": dashboard_slug,
            "dashboard_dir": f"dashboards/{dashboard_slug}",
            "render_dir": f"dashboards/{dashboard_slug}/render",
            "queries_dir": f"dashboards/{dashboard_slug}/queries",
            "analysis_dir": f"dashboards/{dashboard_slug}/analysis",
            "mode": mode,
        }
        if manifest_rel:
            result["manifest_path"] = manifest_rel
        if intent_warning:
            result["intent_warning"] = intent_warning
        return FuncToolResult(result=result)

    def _require_active(self, tool_name: str) -> Optional[FuncToolResult]:
        if self.dashboard_slug is None or self.dashboard_dir is None or self.queries_dir is None:
            return FuncToolResult(
                success=0,
                error=(
                    f"No active dashboard bound. Call start_new_dashboard(slug=..., name=..., description=...) "
                    f"to create one, or bind_existing_dashboard(dashboard_slug=...) to edit an "
                    f"existing one, before calling {tool_name}()."
                ),
            )
        return None

    # -- save_query_template --------------------------------------------------

    def save_query_template(
        self,
        name: str,
        sql_template: str,
        sample_params: Dict[str, Any],
        goal: str,
        hypothesis: str,
        uses: Optional[Dict[str, Any]] = None,
        caveats: str = "",
        datasource: str = "",
    ) -> FuncToolResult:
        """
        Persist a parameterized Jinja2 SQL template after a trial render,
        and the per-query brief sidecar.

        Args:
            name: Semantic slug for the template (e.g. "revenue_by_region").
                Matches ``^[a-z0-9_]{1,64}$``. Reused names overwrite the
                previous files (``.sql.j2`` / ``.params.json`` /
                ``.brief.json``).
            sql_template: Jinja2 SQL body whose **first non-blank line** is
                a ``-- @datus-params <name>:<type>[:optional], ...`` header
                declaring the parameters the body references. Bind values
                appear inside the body as ``:name``; structural conditionals
                use Jinja2 ``{% if %}`` / ``{% for %}`` blocks. Multi-statement
                input is rejected.
            sample_params: Values used for the trial render. Required params
                must be present; optional params may be omitted. The
                inferred columns / preview rows come from this trial.
            goal: One-line research question this template answers, e.g.
                "revenue trends sliced by region over selectable window".
                Becomes the trailing SQL comment so a human reading the
                ``.sql.j2`` can recover intent. Required. Not persisted
                separately — the brief file holds only hypothesis / uses
                / caveats.
            hypothesis: One-sentence concrete prediction the LLM expects
                this query to validate (e.g. "regional revenue diverges
                month-over-month, justifying drilldown"). Required and
                non-empty.
            uses: Optional ``{"metrics": [{"path": [...], "name": "..."}],
                "reference_sql": [...], "ext_knowledge": [...]}``. Each
                bucket lists subject-library assets this template draws on,
                identified by their ``path`` + ``name`` pair (the same two
                fields ``list_metrics`` / ``search_metrics`` /
                ``list_subject_tree`` return). Surfaced verbatim in
                ``analysis/subject_refs.json`` for the follow-up subagent,
                deduped on ``(path, name)``. Malformed entries (missing
                ``path`` or ``name``, legacy string-id form) are rejected
                immediately.
            caveats: Before deciding this field is empty, check the
                template against the same five-gotcha checklist the report
                subagent uses (JOIN type / hardcoded value lists / implicit
                filters / NULL handling / non-standard aggregation), plus
                one dashboard-specific extra: if a particular sample_params
                value silently picks a different code path (e.g. an empty
                array disables a filter), say so. Write one concise
                sentence per applicable point. Truly routine templates get
                an empty string — filler like "no caveats" is NOT
                acceptable.
            datasource: Logical datasource name. Empty string uses the default.

        Returns:
            FuncToolResult.result is a dict like::

                {
                    "name": "<slug>",
                    "sql_path": "dashboards/<id>/queries/<slug>.sql.j2",
                    "params_path": "dashboards/<id>/queries/<slug>.params.json",
                    "brief_path": "dashboards/<id>/queries/<slug>.brief.json",
                    "data_ref": "queries/<slug>",
                    "params": [{"name": "...", "type": "...", "required": true}, ...],
                    "sample_row_count": <int>,
                    "columns": [{"name": "...", "type": "..."}, ...],
                    "preview_rows": [{ ... }],
                }
        """
        not_bound = self._require_active("save_query_template")
        if not_bound is not None:
            return not_bound
        if not name or not QUERY_SLUG_RE.fullmatch(name):
            return FuncToolResult(
                success=0,
                error=f"name must match {QUERY_SLUG_RE.pattern}; got {name!r}",
            )
        if not sql_template or not sql_template.strip():
            return FuncToolResult(success=0, error="sql_template must not be empty")
        if len(sql_template.encode("utf-8")) > _MAX_TEMPLATE_BYTES:
            return FuncToolResult(
                success=0,
                error=f"sql_template exceeds the {_MAX_TEMPLATE_BYTES // 1024} KB limit. Trim or split it.",
            )
        if not isinstance(sample_params, dict):
            return FuncToolResult(
                success=0,
                error=f"sample_params must be a JSON object (dict); got {type(sample_params).__name__}.",
            )
        if not goal or not goal.strip():
            return FuncToolResult(
                success=0,
                error="goal must be a non-empty one-line research question.",
            )
        if not hypothesis or not hypothesis.strip():
            return FuncToolResult(
                success=0,
                error=(
                    "hypothesis must be a non-empty one-sentence concrete prediction this query validates. "
                    "If you don't have a hypothesis, skip the query."
                ),
            )
        try:
            uses_obj = coerce_uses_arg(uses)
        except ValueError as exc:
            return FuncToolResult(success=0, error=f"uses argument invalid: {exc}")

        # 1. Parse the -- @datus-params header
        try:
            params_decl = parse_datus_params_header(sql_template)
        except ValueError as exc:
            return FuncToolResult(success=0, error=str(exc))

        # 1b. Every declared param MUST appear as ``:name`` somewhere in the
        # SQL body. Without this check, a template like the one DeepSeek
        # produced for ``supply_perishable_pie`` — declaring ``start_date`` /
        # ``end_date`` in the header but never binding either in the body —
        # silently renders a static snapshot while the dashboard UI still
        # exposes a time picker, so the user changes filters and the chart
        # doesn't move. ``validate_render`` only enforces the render-tree
        # ``params`` keys vs. the header; body usage is on us.
        bind_names = extract_bind_names(sql_template)
        unbound = [p.name for p in params_decl if p.name not in bind_names]
        if unbound:
            sample = unbound[0]
            return FuncToolResult(
                success=0,
                error=(
                    f"Parameters declared in -- @datus-params but never bound "
                    f"as ':name' in the SQL body: {unbound}. The runtime would "
                    f"forward filter values that the SQL never reads, producing "
                    f"a static snapshot that silently ignores the dashboard's "
                    f"filter controls. Fix one of three ways: "
                    f"(a) reference each via ':{sample}' in the SQL body (e.g. "
                    f"`WHERE ordered_at >= :{sample}`); "
                    f"(b) drop just the unused entries from the -- @datus-params "
                    f"header so the runtime contract reflects what the query "
                    f"actually consumes; or "
                    f"(c) if the query takes no parameters at all (e.g. a static "
                    f"catalog rollup), shrink the header to the bare line "
                    f"`-- @datus-params` with nothing after it — that's the "
                    f"canonical way to declare zero parameters."
                ),
            )

        # 2. Validate sample_params against declared params
        decl_names = {p.name for p in params_decl}
        unknown = set(sample_params.keys()) - decl_names
        if unknown:
            return FuncToolResult(
                success=0,
                error=(
                    f"sample_params contains keys not declared in -- @datus-params: {sorted(unknown)}. "
                    "Update the header or remove the extra keys."
                ),
            )
        missing_required = [p.name for p in params_decl if p.required and p.name not in sample_params]
        if missing_required:
            return FuncToolResult(
                success=0,
                error=(
                    f"sample_params is missing required params: {missing_required}. "
                    "Provide a representative value for each required param so the trial render succeeds."
                ),
            )

        # 3. Render Jinja2 + substitute bind placeholders for the trial run
        try:
            rendered_sql = _render_template_for_trial(sql_template, params_decl, sample_params)
        except ValueError as exc:
            return FuncToolResult(success=0, error=str(exc))

        if not _looks_like_select(rendered_sql):
            return FuncToolResult(
                success=0,
                error=(
                    "save_query_template only accepts read-only SQL "
                    "(SELECT / WITH / SHOW / DESCRIBE / EXPLAIN). "
                    "After Jinja2 rendering, the template body did not begin with a read-only statement."
                ),
            )

        # 4. Execute via the connector
        connector = None
        try:
            connector = self._db_func_tool._get_connector(datasource or None)
        except Exception as exc:
            return FuncToolResult(success=0, error=f"Failed to resolve datasource {datasource!r}: {exc}")

        ds_label = datasource or getattr(self._db_func_tool, "_default_datasource", "") or "default"

        try:
            execute_result = connector.execute_query(rendered_sql, result_format="list")
        except Exception as exc:
            logger.exception("save_query_template trial execute crashed", extra={"name": name})
            return FuncToolResult(success=0, error=f"Trial query execution failed: {exc}")

        if not execute_result.success:
            return FuncToolResult(
                success=0,
                error=f"Trial query failed: {execute_result.error}",
            )

        rows_raw = execute_result.sql_return or []
        if not isinstance(rows_raw, list):
            return FuncToolResult(
                success=0,
                error="Unexpected result format from connector; expected list of dicts",
            )
        rows: List[Dict[str, Any]] = []
        column_names: List[str] = []
        for row in rows_raw:
            if not isinstance(row, dict):
                return FuncToolResult(
                    success=0,
                    error="Unexpected row format from connector; expected dict per row",
                )
            normalized = {k: _normalize_value(v) for k, v in row.items()}
            rows.append(normalized)
            for key in row.keys():
                if key not in column_names:
                    column_names.append(key)

        columns_meta: List[Dict[str, str]] = []
        for col_name in column_names:
            sample = [row.get(col_name) for row in rows[:200]]
            columns_meta.append({"name": col_name, "type": _infer_column_type(sample)})

        if not columns_meta:
            return FuncToolResult(
                success=0,
                error="Trial query returned no columns. Refine the SQL so at least one column is selected.",
            )

        # 5. Persist .sql.j2 + .params.json + .brief.json
        # Note: the legacy ``description`` slot in params.json maps to the
        # new ``goal`` arg — same semantics (one-line research question),
        # but the analysis-artifact contract names it ``goal`` everywhere.
        meta_payload = {
            "slug": name,
            "description": goal.strip(),
            "datasource": ds_label,
            "params": [p.model_dump() for p in params_decl],
            "columns": columns_meta,
            "sample_params": sample_params,
            "sample_row_count": len(rows),
            "saved_at": utc_now_iso(),
        }

        try:
            QueryTemplateMetaFile.model_validate(meta_payload)
        except Exception as exc:
            return FuncToolResult(
                success=0,
                error=f"Template metadata failed schema validation: {exc}",
            )

        meta_blob = json.dumps(meta_payload, ensure_ascii=False, indent=2, default=_normalize_value)
        if len(meta_blob.encode("utf-8")) > _MAX_PARAMS_META_BYTES:
            return FuncToolResult(
                success=0,
                error=f"params.json exceeds the {_MAX_PARAMS_META_BYTES // 1024} KB limit.",
            )

        sql_path = self.queries_dir / f"{name}.sql.j2"
        meta_path = self.queries_dir / f"{name}.params.json"
        brief_path = self.queries_dir / f"{name}.brief.json"

        header_parts: List[str] = [f"-- {goal.strip()}"]
        header_parts.append(f"-- saved at {meta_payload['saved_at']} for dashboard {self.dashboard_slug}")
        sql_text = sql_template.rstrip() + "\n\n" + "\n".join(header_parts) + "\n"

        try:
            _atomic_write_text(sql_path, sql_text)
            _atomic_write_text(meta_path, meta_blob)
        except OSError as exc:
            return FuncToolResult(success=0, error=f"Failed to persist template files: {exc}")

        brief_err = write_query_brief(
            self.queries_dir,
            name=name,
            hypothesis=hypothesis.strip(),
            uses=uses_obj,
            caveats=caveats.strip() if caveats else "",
        )
        if brief_err:
            return FuncToolResult(success=0, error=brief_err)

        manifest_warning = upsert_manifest_after_save(
            self.dashboard_dir / "manifest.json",
            datasource=ds_label,
            timestamp=meta_payload["saved_at"],
        )

        rel_sql = sql_path.relative_to(self._project_root).as_posix()
        rel_meta = meta_path.relative_to(self._project_root).as_posix()
        rel_brief = brief_path.relative_to(self._project_root).as_posix()

        result: Dict[str, Any] = {
            "name": name,
            "sql_path": rel_sql,
            "params_path": rel_meta,
            "brief_path": rel_brief,
            "data_ref": f"queries/{name}",
            "params": [p.model_dump() for p in params_decl],
            "sample_row_count": len(rows),
            "columns": columns_meta,
            "preview_rows": rows[:3],
        }
        if manifest_warning:
            result["manifest_warning"] = manifest_warning
        return FuncToolResult(result=result)

    # -- validate_render -----------------------------------------------------

    def validate_render(self) -> FuncToolResult:
        """
        Validate the assembled render/ tree. Terminal action of this subagent.

        Walks ``dashboards/<id>/render/`` and verifies:

        * ``render/app.jsx`` exists and contains an ``export default``.
        * Every ``useQuerySql('queries/<slug>', {params})`` literal across
          all files resolves to a saved template **and** the keys of
          ``params`` exactly match the template's declared parameters.
        * Every ``import`` / ``export ... from`` path is either a bare
          specifier in the allowed list or a relative path that resolves
          to a file under ``render/``.

        Returns:
            FuncToolResult.result on success::

                {
                    "app_jsx_path": "dashboards/<id>/render/app.jsx",
                    "render_files": ["render/app.jsx", "render/filters.jsx", ...],
                    "query_refs": ["queries/foo", "queries/bar"],
                    "warnings": ["render/legacy.jsx is unreachable from app.jsx"],
                }
        """
        not_bound = self._require_active("validate_render")
        if not_bound is not None:
            return not_bound

        # Manifest must exist before render-tree validation — it's part
        # of the artifact contract that the list pages / IDE rely on.
        manifest_path = self.dashboard_dir / "manifest.json"
        if not manifest_path.is_file():
            return FuncToolResult(
                success=0,
                error=(
                    f"dashboards/{self.dashboard_slug}/manifest.json is missing. A dashboard must "
                    "always have a manifest with name + description. Re-run start_new_dashboard "
                    "or restore the manifest from a previous version."
                ),
            )
        try:
            ArtifactManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
        except Exception as exc:
            return FuncToolResult(
                success=0,
                error=f"dashboards/{self.dashboard_slug}/manifest.json is corrupt or off-spec: {exc}",
            )

        if not self.render_dir.is_dir():
            return FuncToolResult(
                success=0,
                error=(
                    f"render/ directory missing under dashboards/{self.dashboard_slug}. "
                    "Write at least an app.jsx with write_file before calling validate_render."
                ),
            )

        all_files: List[Path] = sorted(
            [p for p in self.render_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jsx", ".js"}]
        )
        if not all_files:
            return FuncToolResult(
                success=0,
                error=(
                    "render/ has no .jsx / .js files. Write at least an app.jsx with "
                    "write_file before calling validate_render."
                ),
            )

        app_jsx_path = self.render_dir / "app.jsx"
        if not app_jsx_path.is_file():
            return FuncToolResult(
                success=0,
                error="render/app.jsx is required as the entry module but is missing.",
            )

        modules: Dict[str, Dict[str, Any]] = {}
        for path in all_files:
            rel = path.relative_to(self.render_dir).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
            except OSError as exc:
                return FuncToolResult(
                    success=0,
                    error=f"Failed to read render file {rel}: {exc}",
                )
            modules[_module_key(rel)] = {
                "rel": rel,
                "source": source,
                "imports": [],
            }

        # Index saved templates: slug → declared param names (and required-ness).
        templates: Dict[str, List[TemplateParamDecl]] = {}
        for meta_path in self.queries_dir.glob("*.params.json"):
            slug = meta_path.stem.removesuffix(".params") if meta_path.stem.endswith(".params") else meta_path.stem
            # Path.stem strips one suffix; ``.params.json`` leaves ``.params``.
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                meta_obj = QueryTemplateMetaFile.model_validate(payload)
                templates[meta_obj.slug] = meta_obj.params
            except Exception as exc:
                return FuncToolResult(
                    success=0,
                    error=f"queries/{meta_path.name} is corrupt or off-spec: {exc}",
                )
            # Make sure the .sql.j2 sibling exists.
            if not (self.queries_dir / f"{slug}.sql.j2").is_file():
                return FuncToolResult(
                    success=0,
                    error=f"queries/{slug}.params.json has no sibling {slug}.sql.j2 — re-run save_query_template.",
                )

        module_keys: Set[str] = set(modules.keys())
        issues: List[str] = []
        warnings: List[str] = []
        query_refs: Set[str] = set()

        for key, mod in modules.items():
            source = mod["source"]

            # ---- 1-arg useQuerySql calls (no params) — always rejected.
            for match in _USE_QUERY_SQL_NO_PARAMS_RE.finditer(source):
                issues.append(
                    f"render/{mod['rel']}: {match.group(0)!r} — useQuerySql in a dashboard requires "
                    "a second `params` argument (use `{}` when the template takes no params)."
                )

            # ---- 2-arg useQuerySql calls — slug + params keys
            for match in _USE_QUERY_SQL_LITERAL_RE.finditer(source):
                literal = match.group(1)
                params_literal = match.group(2)
                slug = extract_query_slug(literal)
                if slug is None:
                    issues.append(
                        f"render/{mod['rel']}: useQuerySql received an invalid literal sqlId "
                        f"{literal!r}. Use 'queries/<slug>' where <slug> matches ^[a-z0-9_]+$."
                    )
                    continue
                query_refs.add(f"queries/{slug}")
                decl = templates.get(slug)
                if decl is None:
                    issues.append(
                        f"render/{mod['rel']}: useQuerySql('queries/{slug}', ...) points to a template "
                        "not produced via save_query_template."
                    )
                    continue

                # Statically pull the keys out of the params object literal.
                literal_keys: Set[str] = set()
                for key_match in _PARAMS_KEY_RE.finditer(params_literal):
                    literal_keys.add((key_match.group(1) or key_match.group(2)).lower())
                for key_match in _PARAMS_SHORTHAND_KEY_RE.finditer(params_literal):
                    literal_keys.add(key_match.group(1).lower())

                # Spread / computed keys appear in the source but not in our
                # capture group. Detect them so we can warn (and skip the
                # strict key check, since the actual keys resolve at runtime).
                if "..." in params_literal:
                    warnings.append(
                        f"render/{mod['rel']}: useQuerySql('queries/{slug}', ...) uses spread/computed "
                        "params — the static params-key check is deferred to runtime."
                    )
                    continue

                declared_required = {p.name for p in decl if p.required}
                declared_all = {p.name for p in decl}

                missing = declared_required - literal_keys
                extra = literal_keys - declared_all
                if missing:
                    issues.append(
                        f"render/{mod['rel']}: useQuerySql('queries/{slug}', ...) is missing required "
                        f"params: {sorted(missing)}. Template declares "
                        f"{sorted(declared_all)}; literal has {sorted(literal_keys)}."
                    )
                if extra:
                    issues.append(
                        f"render/{mod['rel']}: useQuerySql('queries/{slug}', ...) passes unknown "
                        f"params: {sorted(extra)}. Template declares {sorted(declared_all)}."
                    )

            # ---- import / export … from paths
            for match in _IMPORT_PATH_RE.finditer(source):
                spec = match.group(1)
                if spec in ALLOWED_BARE_MODULES:
                    continue
                if spec.startswith("./") or spec.startswith("../"):
                    resolved = _resolve_relative_import(key, spec, module_keys)
                    if resolved is None:
                        issues.append(
                            f"render/{mod['rel']}: relative import {spec!r} does not resolve to a file under render/."
                        )
                    else:
                        mod["imports"].append(resolved)
                    continue
                issues.append(
                    f"render/{mod['rel']}: import {spec!r} is not allowed. Only bare specifiers "
                    f"{sorted(ALLOWED_BARE_MODULES)} or relative paths under render/ are allowed."
                )

        if not _DEFAULT_EXPORT_RE.search(modules["app"]["source"]):
            issues.append(
                "render/app.jsx must include an `export default` (the renderer mounts the "
                "default export as the dashboard's root component)."
            )

        if issues:
            return FuncToolResult(
                success=0,
                error="validate_render found "
                + ("1 issue:\n  - " if len(issues) == 1 else f"{len(issues)} issues:\n  - ")
                + "\n  - ".join(issues),
            )

        # Reachability from app.jsx via static imports.
        reachable: Set[str] = set()
        stack: List[str] = ["app"]
        while stack:
            k = stack.pop()
            if k in reachable:
                continue
            reachable.add(k)
            stack.extend(modules[k]["imports"])
        unreferenced = sorted(modules.keys() - reachable)
        warnings.extend(
            f"render/{modules[k]['rel']} is not imported by render/app.jsx (directly or transitively)"
            for k in unreferenced
        )

        return FuncToolResult(
            result={
                "app_jsx_path": app_jsx_path.relative_to(self._project_root).as_posix(),
                "manifest_path": manifest_path.relative_to(self._project_root).as_posix(),
                "render_files": [f"render/{modules[k]['rel']}" for k in sorted(modules.keys())],
                "query_refs": sorted(query_refs),
                "warnings": warnings,
            }
        )
