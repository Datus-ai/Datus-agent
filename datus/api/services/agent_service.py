"""
Stateless service for agent CRUD operations.

Handles listing, creating, and editing sub-agents. Builtin agents are resolved
from the BUILTIN_SUBAGENTS set; custom agents are persisted in agent.yml.
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from datus.api.models.agent_models import CreateAgentInput, EditAgentInput
from datus.api.models.base_models import Result
from datus.configuration.agent_config import AgentConfig
from datus.prompts.prompt_manager import PromptManager
from datus.schemas.artifact_manifest import ARTIFACT_SLUG_RE
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.func_tool.database import DBFuncTool
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.platform_doc_search import PlatformDocSearchTool
from datus.tools.func_tool.reference_template_tools import ReferenceTemplateTools
from datus.tools.func_tool.semantic_tools import SemanticTools
from datus.tools.func_tool.sub_agent_task_tool import BUILTIN_SUBAGENT_DESCRIPTIONS
from datus.utils.constants import HIDDEN_SYS_SUB_AGENTS, SYS_SUB_AGENTS
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Valid tool categories and their methods, derived from tool classes in datus-agent.
# Matches GenSQLAgenticNode._setup_tool_pattern() categories.

VALID_TOOL_METHODS: dict[str, set[str]] = {
    "db_tools": set(DBFuncTool.all_tools_name()),
    "context_search_tools": set(ContextSearchTools.all_tools_name()),
    "semantic_tools": set(SemanticTools.all_tools_name()),
    "reference_template_tools": set(ReferenceTemplateTools.all_tools_name()),
    "date_parsing_tools": {"parse_temporal_expressions"},
    "filesystem_tools": set(FilesystemFuncTool.all_tools_name()),
    "platform_doc_tools": set(PlatformDocSearchTool.all_tools_name()),
}

VALID_TOOL_CATEGORIES = set(VALID_TOOL_METHODS.keys())

BUILTIN_SUBAGENTS = SYS_SUB_AGENTS - HIDDEN_SYS_SUB_AGENTS

# Curated list of categories surfaced through GET /agent/use_tools' ``tool_types``
# block. Matches the SaaS editor's whitelist — ``platform_doc_tools`` is a
# valid tool (and stays in VALID_TOOL_METHODS for write-side validation) but is
# excluded from the editor picker.
_USER_FACING_TOOL_CATEGORIES: tuple[str, ...] = (
    "db_tools",
    "context_search_tools",
    "semantic_tools",
    "reference_template_tools",
    "date_parsing_tools",
    "filesystem_tools",
)

# Filesystem methods an ``ask_*`` agent may use. Mirrors the read-only intent
# documented in the ``ask_report`` / ``ask_dashboard`` entries of
# :data:`SUBAGENT_TOOL_REFERENCE`: the consultant reads ``analysis/`` and
# ``queries/`` to answer follow-ups but must never mutate the artifact.
_ASK_AGENT_FILESYSTEM_READ_ONLY: tuple[str, ...] = ("glob", "grep", "read_file")

# Per-agent-type editor whitelist. Previously the saas frontend
# (``tool-tree.ts`` ``enableToolGroup``) decided which categories the picker
# would render — only ``gen_sql`` and the catch-all else branch (effectively
# ``gen_report``) were handled explicitly, which hid filesystem_tools from
# ask_* even though those agents preselect ``filesystem_tools.read_file`` /
# ``glob`` / ``grep`` in their defaults. The API is now the single source of
# truth: each agent type returns exactly the categories the editor should
# surface, and the frontend can render whatever it receives.
_TOOL_CATEGORIES_BY_AGENT_TYPE: dict[str, tuple[str, ...]] = {
    "chat": _USER_FACING_TOOL_CATEGORIES,
    "gen_sql": (
        "db_tools",
        "semantic_tools",
        "context_search_tools",
    ),
    "gen_report": (
        "db_tools",
        "semantic_tools",
        "context_search_tools",
        "date_parsing_tools",
        "reference_template_tools",
    ),
    "ask_metrics": _USER_FACING_TOOL_CATEGORIES,
    "ask_report": (
        "db_tools",
        "semantic_tools",
        "context_search_tools",
        "reference_template_tools",
        "date_parsing_tools",
        "filesystem_tools",
    ),
    "ask_dashboard": (
        "db_tools",
        "semantic_tools",
        "context_search_tools",
        "reference_template_tools",
        "date_parsing_tools",
        "filesystem_tools",
    ),
}


def _build_tool_types(agent_type: str) -> dict[str, dict[str, list[str]]]:
    """Compose the ``tool_types`` block returned by ``GET /agent/use_tools``
    for ``agent_type``. Categories are restricted to the per-type whitelist
    in :data:`_TOOL_CATEGORIES_BY_AGENT_TYPE`; for artifact ``ask_*`` agents,
    ``filesystem_tools`` is further restricted to the read-only subset so
    the editor picker never surfaces ``write_file`` / ``edit_file`` as
    selectable options.
    """
    is_ask_agent = agent_type in {"ask_report", "ask_dashboard"}
    tool_types: dict[str, dict[str, list[str]]] = {}
    for category in _TOOL_CATEGORIES_BY_AGENT_TYPE[agent_type]:
        if is_ask_agent and category == "filesystem_tools":
            tool_types[category] = {"tools": list(_ASK_AGENT_FILESYSTEM_READ_ONLY)}
        else:
            tool_types[category] = {"tools": sorted(VALID_TOOL_METHODS[category])}
    return tool_types


# Per-agent-type tool reference. Matches the SaaS editor contract:
# ``default_tools`` are wildcard / specific patterns preselected for the type,
# ``tool_types`` is the curated catalog of categories the editor should
# surface for that type (built via :func:`_build_tool_types`).
SUBAGENT_TOOL_REFERENCE: dict[str, dict[str, Any]] = {
    "chat": {
        "default_tools": [
            "db_tools.*",
            "context_search_tools.*",
            "reference_template_tools.*",
            "date_parsing_tools.*",
            "filesystem_tools.*",
            "platform_doc_tools.*",
        ],
        "tool_types": _build_tool_types("chat"),
    },
    "gen_sql": {
        "default_tools": [
            "db_tools.*",
            "semantic_tools.*",
            "context_search_tools.*",
        ],
        "tool_types": _build_tool_types("gen_sql"),
    },
    "gen_report": {
        "default_tools": [
            "semantic_tools.*",
            "context_search_tools.list_subject_tree",
        ],
        "tool_types": _build_tool_types("gen_report"),
    },
    "ask_metrics": {
        "default_tools": [
            "context_search_tools.search_metrics",
            "context_search_tools.get_metrics",
            "semantic_tools.list_metrics",
            "semantic_tools.get_dimensions",
            "semantic_tools.query_metrics",
            "semantic_tools.attribution_analyze",
            "context_search_tools.list_subject_tree",
        ],
        "tool_types": _build_tool_types("ask_metrics"),
    },
    # ask_report / ask_dashboard: read-only follow-up consultant for a single
    # visual artifact. Default tools cover data exploration (db_tools read
    # methods, semantic / context_search / reference_template) plus the
    # read-side of filesystem so the LLM can ``glob`` / ``grep`` / ``read_file``
    # the artifact's ``analysis/`` and ``queries/`` directories. Writes are
    # excluded by omission — these agents must never mutate the artifact.
    "ask_report": {
        "default_tools": [
            "db_tools.list_databases",
            "db_tools.list_schemas",
            "db_tools.list_tables",
            "db_tools.describe_table",
            "db_tools.search_table",
            "db_tools.get_table_ddl",
            "db_tools.read_query",
            "semantic_tools.*",
            "context_search_tools.*",
            "reference_template_tools.*",
            "date_parsing_tools.*",
            "filesystem_tools.read_file",
            "filesystem_tools.glob",
            "filesystem_tools.grep",
        ],
        "tool_types": _build_tool_types("ask_report"),
    },
    "ask_dashboard": {
        "default_tools": [
            "db_tools.list_databases",
            "db_tools.list_schemas",
            "db_tools.list_tables",
            "db_tools.describe_table",
            "db_tools.search_table",
            "db_tools.get_table_ddl",
            "db_tools.read_query",
            "semantic_tools.*",
            "context_search_tools.*",
            "reference_template_tools.*",
            "date_parsing_tools.*",
            "filesystem_tools.read_file",
            "filesystem_tools.glob",
            "filesystem_tools.grep",
        ],
        "tool_types": _build_tool_types("ask_dashboard"),
    },
}


def sanitize_agentic_node_name(name: str) -> str:
    """Sanitize a sub-agent name for agentic_nodes keys and template filenames.

    Replaces every character outside ``[A-Za-z0-9_-]`` with ``_`` so the name is
    safe to use as a yaml key, dict key, or filesystem path component. ``None``
    or an empty string degrades to ``""``.
    """
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name or "")


def _read_description(node: dict) -> str:
    """Read the description field from a yaml ``agentic_nodes`` entry.

    The runtime (``sub_agent_task_tool``, ``agentic_node``, the wizard, etc.)
    persists this field as ``agent_description``. Older yaml files written by
    earlier versions of the API used ``description`` — fall back to that so
    existing configs keep working until the next edit migrates them.
    """
    return node.get("agent_description") or node.get("description") or ""


def _parse_tools(value: Any) -> list[str]:
    """Normalize the yaml ``tools`` field to a list of pattern strings.

    Accepts either a comma-separated string (the canonical yaml form, which
    the runtime in ``GenSQLAgenticNode.setup_tools`` calls ``str.split`` on)
    or a list, and trims surrounding whitespace from every entry. Empty
    entries are dropped.
    """
    if not value:
        return []
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def _format_csv(value: Any) -> str:
    """Render a list / tuple / string into the comma-separated yaml form.

    The runtime expects ``tools``, ``mcp``, and the ``scoped_context`` path
    fields (``catalogs``, ``subjects``, ``tables``, …) as comma-separated
    strings — ``GenSQLAgenticNode.setup_tools`` and ``ScopedContext.as_lists``
    both rely on ``str.split(",")`` to recover the entries. Persisting these
    as yaml lists silently breaks both call sites, so the API normalizes on
    write.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return ""
    cleaned = [str(item).strip() for item in items if str(item) and str(item).strip()]
    return ", ".join(cleaned)


def _parse_csv(value: Any) -> list[str]:
    """Inverse of :func:`_format_csv` — split the yaml form back into a list."""
    return _parse_tools(value)


def _strip_leading_slashes(value: Any) -> list[str]:
    """Trim and drop leading ``/`` from each entry in a path list.

    Catalog / subject entries arrive from the editor as absolute-style paths
    (``/Commerce/Orders/Average_Gross_Order_Value``), but the runtime stores
    them without the leading slash so ``ScopedContext.as_lists`` and the
    downstream lookups don't treat the slash as a separator. Normalizing on
    write keeps both the API contract and the on-disk shape consistent —
    ``["/A/B", "C"]`` → ``["A/B", "C"]``.
    """
    return [token.lstrip("/") for token in _parse_csv(value) if token.lstrip("/")]


# Keys inside ``scoped_context`` that hold subject-tree path entries. The API's
# flat ``subjects`` array is the union of both stores; the runtime stores them
# split because each store (metrics, reference SQL) owns its own scope filter.
_SUBJECT_BUCKET_KEYS: tuple[str, ...] = ("metrics", "sqls")


def _classify_subject_paths(
    agent_config: AgentConfig,
    subject_paths: list[str],
    datasource_id: Optional[str] = None,
) -> dict[str, list[str]]:
    """Bucket subject paths into ``metrics`` / ``sqls``.

    The API surfaces a single ``subjects`` array — dot-separated paths like
    ``Commerce.Orders.Average_Order_Value.average_gross_order_value`` — that's
    the merged union of all entries the editor's subject-tree exposes
    (Metrics, Reference SQLs, Knowledge — see
    ``ExplorerService.get_subject_list``). The runtime expects them split:
    ``ScopedContext.metrics`` / ``.sqls`` each drive an
    independent scope filter.

    For every input path:

    1. Split via ``split_reference_path`` (handles quoted segments).
    2. Resolve the parent subject node via ``SubjectTreeStore.get_node_by_path``.
    3. Probe the metric / reference-sql stores for an entry
       named ``parts[-1]`` under that node.
    4. Bucket on the first store that owns the name.

    Paths that don't resolve land in ``metrics`` so the user's selection is
    never silently dropped. If the storage layer can't be initialized at all
    (no datasource bound, registry unavailable, etc.) every path falls back
    to ``metrics`` and a warning is logged — the editor's input survives the
    round-trip even when the project hasn't bootstrapped its KB yet.
    """
    buckets: dict[str, list[str]] = {key: [] for key in _SUBJECT_BUCKET_KEYS}
    if not subject_paths:
        return buckets

    try:
        from datus.storage.metric.store import MetricRAG
        from datus.storage.reference_sql.store import ReferenceSqlRAG
        from datus.storage.registry import get_subject_tree_store
        from datus.utils.reference_paths import split_reference_path
    except ImportError:
        logger.warning(
            "Subject classification skipped — storage modules unavailable; routing all subjects to 'metrics'"
        )
        buckets["metrics"] = list(subject_paths)
        return buckets

    ds = datasource_id or getattr(agent_config, "current_datasource", None)
    if not ds:
        logger.warning(
            "Subject classification skipped — no datasource bound on AgentConfig; routing all subjects to 'metrics'"
        )
        buckets["metrics"] = list(subject_paths)
        return buckets

    try:
        subject_tree = get_subject_tree_store(project=agent_config.project_name)
        metric_storage = MetricRAG(agent_config, datasource_id=ds).storage
        sql_storage = ReferenceSqlRAG(agent_config, datasource_id=ds).reference_sql_storage
    except Exception:
        logger.warning("Subject classification storage init failed — routing all subjects to 'metrics'", exc_info=True)
        buckets["metrics"] = list(subject_paths)
        return buckets

    probes: tuple[tuple[str, Any], ...] = (
        ("metrics", metric_storage),
        ("sqls", sql_storage),
    )

    for path in subject_paths:
        parts = split_reference_path(path)
        if not parts:
            continue
        parent_path, name = parts[:-1], parts[-1]

        parent_node = None
        if parent_path:
            try:
                parent_node = subject_tree.get_node_by_path(parent_path)
            except Exception:
                parent_node = None
        node_id = parent_node.get("node_id") if isinstance(parent_node, dict) else None

        bucket: Optional[str] = None
        if node_id is not None:
            for candidate, storage in probes:
                try:
                    matched = storage.list_entries(node_id, name=name, limit=1)
                except Exception:
                    matched = None
                if matched:
                    bucket = candidate
                    break

        buckets[bucket or "metrics"].append(path)

    return buckets


def _merge_subjects_from_scoped_context(scoped_ctx: Optional[dict]) -> list[str]:
    """Flatten ``metrics`` / ``sqls`` back into one list.

    Inverse of :func:`_classify_subject_paths`. Stored entries are returned
    verbatim (canonical dot-separated form), with duplicates dropped while
    preserving insertion order.
    """
    if not isinstance(scoped_ctx, dict):
        return []
    merged: list[str] = []
    seen: set[str] = set()
    for key in _SUBJECT_BUCKET_KEYS:
        for token in _parse_csv(scoped_ctx.get(key)):
            if token and token not in seen:
                seen.add(token)
                merged.append(token)
    return merged


def _build_scoped_context(
    base: Optional[dict],
    *,
    datasource: Optional[str] = None,
    catalogs: Any = None,
    subject_buckets: Optional[dict[str, list[str]]] = None,
) -> Optional[dict]:
    """Merge API-level fields into a runtime-shaped ``scoped_context`` dict.

    ``base`` carries any existing scoped_context payload (yaml-loaded for
    edits, an explicit ``request.scoped_context`` for inputs that send one).

    ``ScopedContext`` (``datus/schemas/agent_models.py``) only defines
    ``datasource`` / ``tables`` / ``metrics`` / ``sqls``;
    the API's ``catalogs`` array is the editor's name for the same scope as
    runtime ``tables`` (catalog/database/schema/table identifiers consumed by
    ``ScopedFilterBuilder.build_table_filter``), so this helper writes
    ``catalogs`` into ``scoped_context.tables`` and never persists a
    non-runtime ``catalogs`` key. Stale ``catalogs`` keys from earlier
    versions of this API are dropped on write.

    ``datasource`` mirrors the wizard's behavior — saving a subagent always
    binds it to the active datasource so ``SubAgentConfig.is_in_datasource``
    can gate at runtime. Passing an empty string clears the binding;
    ``None`` leaves the existing value intact.

    ``subject_buckets`` (pre-classified by :func:`_classify_subject_paths`)
    are written to the runtime-visible ``metrics`` / ``sqls`` keys;
    passing a non-``None`` ``subject_buckets`` rewrites both bucket keys
    (an empty bucket clears its key), so the
    API contract is "the caller's ``subjects`` list is the new full scope."

    Returns ``None`` when the merged dict would be empty.
    """
    merged: dict = dict(base) if isinstance(base, dict) else {}

    if datasource is not None:
        if datasource:
            merged["datasource"] = datasource
        else:
            merged.pop("datasource", None)

    if catalogs is not None:
        # Drop any non-runtime ``catalogs`` key written by earlier API versions —
        # only ``tables`` is honored by ``ScopedFilterBuilder.build_table_filter``.
        merged.pop("catalogs", None)
        rendered = _format_csv(_strip_leading_slashes(catalogs))
        if rendered:
            merged["tables"] = rendered
        else:
            merged.pop("tables", None)

    if subject_buckets is not None:
        for key in _SUBJECT_BUCKET_KEYS:
            rendered = _format_csv(subject_buckets.get(key, []))
            if rendered:
                merged[key] = rendered
            else:
                merged.pop(key, None)

    return merged or None


def _utc_now_iso() -> str:
    """Current UTC time as ISO-8601 string with ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _normalize_created_at(value: Any) -> Optional[str]:
    """Coerce a yaml-loaded ``created_at`` value into ISO-8601 UTC with ``Z``.

    yaml may parse the field as a ``datetime`` or pass it through as a string.
    Returns ``None`` when the value is missing or unrecognized.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, str):
        return value
    return None


def _file_mtime_iso(path: Path) -> Optional[str]:
    """Return the file's mtime as ISO-8601 UTC with ``Z`` suffix, or None on error."""
    try:
        ts = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_tools(tools: list[str]) -> list[str]:
    """Validate tool patterns and return list of invalid ones.

    Valid formats:
      - "db_tools"              (exact category)
      - "db_tools.*"            (wildcard — all methods in category)
      - "db_tools.list_tables"  (specific method)
    """
    invalid = []
    for pattern in tools:
        pattern = pattern.strip()
        if not pattern:
            continue
        # Exact category match: "db_tools"
        if pattern in VALID_TOOL_CATEGORIES:
            continue
        if "." in pattern:
            category, method = pattern.split(".", 1)
            if category not in VALID_TOOL_CATEGORIES:
                invalid.append(pattern)
                continue
            # Wildcard: "db_tools.*"
            if method == "*":
                continue
            # Specific method: "db_tools.list_tables"
            if method not in VALID_TOOL_METHODS[category]:
                invalid.append(pattern)
                continue
        else:
            invalid.append(pattern)
    return invalid


def _validate_tools_for_agent_type(tools: list[str], agent_type: str) -> list[str]:
    """Reject tool patterns outside a type-specific catalog.

    Returns the offending patterns; an empty list means OK.

    The general :func:`_validate_tools` only confirms patterns are
    syntactically valid (category / method exists). Artifact ask agents also
    have a narrower read-only filesystem contract, so enforce that allowlist
    matching the ``tool_types`` catalog returned by ``GET /agent/use_tools``.
    """
    if agent_type not in {"ask_report", "ask_dashboard"}:
        return []
    catalog = SUBAGENT_TOOL_REFERENCE[agent_type]["tool_types"]
    rejected: list[str] = []
    for raw in tools:
        pattern = raw.strip()
        if not pattern:
            continue
        # ``"db_tools"`` and ``"db_tools.*"`` both mean "everything in
        # this category" — only OK if the agent's allowlist already
        # contains every method in that category (i.e. read-only by
        # construction).
        if "." not in pattern:
            category, method = pattern, "*"
        else:
            category, method = pattern.split(".", 1)
        if category not in catalog:
            rejected.append(pattern)
            continue
        allowed_methods = set(catalog[category]["tools"])
        if method == "*":
            if allowed_methods != VALID_TOOL_METHODS.get(category, set()):
                rejected.append(pattern)
            continue
        if method not in allowed_methods:
            rejected.append(pattern)
    return rejected


def _validate_ask_artifact_binding(
    request: CreateAgentInput,
    agent_config: AgentConfig,
    agentic_nodes: dict,
) -> Optional[Result]:
    """Validate an ``ask_report`` / ``ask_dashboard`` create request.

    Returns a failure :class:`Result` when the binding is invalid, ``None``
    when it's good to proceed. Checks, in order:

    1. ``artifact_slug`` is set and matches the slug pattern.
    2. Computed ``reports/<slug>`` / ``dashboards/<slug>`` directory exists
       under ``agent_config.project_root``.
    3. No other ``ask_*`` agent already binds the same (type, slug) — same-
       artifact uniqueness is enforced here (not at the DB layer) so the
       CLI path has the same guarantee the SaaS DB will have via partial
       unique index.
    """
    slug = (request.artifact_slug or "").strip()
    if not slug:
        return Result(
            success=False,
            errorCode="ARTIFACT_SLUG_REQUIRED",
            errorMessage=(
                f"artifact_slug is required when type is {request.type!r} "
                f"(the agent is bound to a specific visual artifact)."
            ),
        )
    if not ARTIFACT_SLUG_RE.fullmatch(slug):
        return Result(
            success=False,
            errorCode="INVALID_ARTIFACT_SLUG",
            errorMessage=f"artifact_slug must match {ARTIFACT_SLUG_RE.pattern}; got {slug!r}",
        )

    project_root = Path(getattr(agent_config, "project_root", "") or ".").resolve()
    kind_dir = "reports" if request.type == "ask_report" else "dashboards"
    expected_dir = project_root / kind_dir / slug
    artifact_dir = expected_dir.resolve()
    # Path-traversal defence: even though ARTIFACT_SLUG_RE blocks ``..``,
    # a symlink at ``<kind_dir>/<slug>`` could still redirect us elsewhere
    # (outside project_root, or to a sibling directory inside it the ask
    # agent should not be reading — including project_root itself).
    # Require the resolved path to match the unresolved expected location
    # verbatim — any symlink redirection produces a mismatch.
    if artifact_dir != expected_dir:
        return Result(
            success=False,
            errorCode="INVALID_ARTIFACT_SLUG",
            errorMessage=f"artifact path resolved outside expected location: {artifact_dir}",
        )
    if not artifact_dir.is_dir():
        return Result(
            success=False,
            errorCode="ARTIFACT_NOT_FOUND",
            errorMessage=f"{kind_dir}/{slug} does not exist under project root",
        )

    # Same-artifact uniqueness — only one ask_* agent per (type, slug).
    for existing_name, existing_entry in (agentic_nodes or {}).items():
        if not isinstance(existing_entry, dict):
            continue
        if existing_entry.get("type") != request.type:
            continue
        if existing_entry.get("artifact_slug") == slug:
            return Result(
                success=False,
                errorCode="ARTIFACT_ALREADY_BOUND",
                errorMessage=(
                    f"An {request.type} agent for artifact {slug!r} already exists "
                    f"(name: {existing_name!r}). Delete the existing one before "
                    "creating a new binding."
                ),
            )
    return None


def _save_agentic_nodes(agent_config: AgentConfig, nodes: dict) -> None:
    """Persist agentic_nodes back to the loaded ``agent.yml``.

    Routes through the :class:`ConfigurationManager` singleton so:

    - Writes land in the same yaml that was read at startup (``--config``
      path), not a synthetic ``{home}/agent.yml`` — the latter is the
      runtime-data home and may be different from the source config file.
    - The on-disk shape is preserved (production configs nest everything
      under ``agent:``); ``ConfigurationManager.save`` round-trips the
      wrapping correctly via ``{"agent": self.data}``.
    """
    from datus.configuration.agent_config_loader import configuration_manager

    cfg_mgr = configuration_manager()
    cfg_mgr.data["agentic_nodes"] = nodes
    cfg_mgr.save()


def _save_channels(agent_config: AgentConfig, channels: dict) -> None:
    """Persist the global ``channels`` map back to the loaded ``agent.yml``.

    Mirrors :func:`_save_agentic_nodes`: routes through the
    ``ConfigurationManager`` singleton so the write lands in the source config
    file (``--config`` path) and the ``agent:`` wrapping is round-tripped. An
    empty map drops the ``channels`` section entirely. The in-memory
    ``agent_config.channels_config`` is updated too so the running API process
    stays consistent — the separate gateway process reads channels once at
    startup and still needs a restart to pick up the change.
    """
    from datus.configuration.agent_config_loader import configuration_manager

    cfg_mgr = configuration_manager()
    if channels:
        cfg_mgr.data["channels"] = channels
    else:
        cfg_mgr.data.pop("channels", None)
    cfg_mgr.save()
    agent_config.channels_config = channels


class AgentService:
    """Service for Agent API operations.

    Handles agent management (CRUD) and subagent chat with SSE streaming.
    """

    def __init__(self):
        """Initialize AgentService."""
        pass

    @staticmethod
    def get_use_tools(agent_type: str) -> Result[dict]:
        """Return available tools for a given agent type.

        Response payload follows the SaaS editor contract:
        ``{"default_tools": [...], "tool_types": {category: {"tools": [...]}}}``.
        """
        if agent_type not in SUBAGENT_TOOL_REFERENCE:
            return Result(
                success=False,
                errorCode="INVALID_AGENT_TYPE",
                errorMessage=f"Unknown agent_type '{agent_type}'. Must be one of: {', '.join(SUBAGENT_TOOL_REFERENCE)}",
            )
        return Result(success=True, data=SUBAGENT_TOOL_REFERENCE[agent_type])

    async def get_agent(
        self,
        agent_id: str,
        agent_config: AgentConfig,
    ) -> Result[dict]:
        """Return agent configuration matching IAgentInfo."""

        # 1. Check builtin agents
        if agent_id in BUILTIN_SUBAGENTS:
            return Result(
                success=True,
                data={
                    "agent": {
                        "id": agent_id,
                        "name": agent_id,
                        "type": "builtin",
                        "description": BUILTIN_SUBAGENT_DESCRIPTIONS.get(agent_id, ""),
                        "created_at": None,
                        "tools": [],
                        "rules": [],
                        "catalogs": [],
                        "subjects": [],
                    }
                },
            )

        # 2. Query custom sub-agent from agent.yml (dict keyed by name, treated as id)
        agentic_nodes = agent_config.agentic_nodes or {}
        agent = agentic_nodes.get(agent_id)
        if not agent:
            return Result(success=False, errorCode="AGENT_NOT_FOUND", errorMessage=f"Agent '{agent_id}' not found")

        agent_type = agent.get("type", "gen_sql")
        created_at = _normalize_created_at(agent.get("created_at"))
        if not created_at:
            from datus.configuration.agent_config_loader import configuration_manager

            created_at = _file_mtime_iso(configuration_manager().config_path)

        # The API ``catalogs`` field maps to ``scoped_context.tables`` (the
        # runtime-honored key consumed by ``ScopedFilterBuilder.build_table_filter``).
        # ``subjects`` is recomposed from the runtime buckets
        # (``metrics`` / ``sqls``) — the inverse of the
        # save-side classification. Stored dot-form (wizard convention) is
        # converted to the API's slash-form on the way out.
        scoped_ctx = agent.get("scoped_context") if isinstance(agent.get("scoped_context"), dict) else {}
        catalogs = _strip_leading_slashes(scoped_ctx.get("tables"))
        subjects = _merge_subjects_from_scoped_context(scoped_ctx)

        return Result(
            success=True,
            data={
                "agent": {
                    "id": agent_id,
                    "name": agent_id,
                    "type": agent_type,
                    "description": _read_description(agent),
                    "created_at": created_at,
                    "tools": _parse_tools(agent.get("tools")),
                    "rules": agent.get("rules") or [],
                    "catalogs": catalogs,
                    "subjects": subjects,
                }
            },
        )

    async def list_agents(self, agent_config: AgentConfig) -> Result[dict]:
        """List all agents available for this project."""

        # 1. Builtin agents
        builtin = [
            {
                "id": name,
                "name": name,
                "type": "builtin",
                "description": BUILTIN_SUBAGENT_DESCRIPTIONS.get(name, ""),
            }
            for name in sorted(BUILTIN_SUBAGENTS)
        ]

        # 2. Custom sub-agents from agent.yml
        agentic_nodes = agent_config.agentic_nodes or {}
        custom = [
            {
                "id": name,
                "name": name,
                "type": node.get("type", "gen_sql"),
                "description": _read_description(node),
            }
            for name, node in sorted(agentic_nodes.items())
        ]

        return Result(success=True, data={"agents": builtin + custom})

    # Map sub-agent type to builtin prompt template base name
    _TYPE_TO_TEMPLATE = {
        "gen_sql": "gen_sql_system",
        "ask_metrics": "ask_metrics_system",
        "gen_report": "gen_report_system",
        "ask_report": "ask_report_system",
        "ask_dashboard": "ask_dashboard_system",
        "chat": "chat_system",
    }

    _prompt_manager = PromptManager()

    async def create_agent(
        self,
        request: CreateAgentInput,
        agent_config: AgentConfig,
    ) -> Result[dict]:
        """Create a new custom sub-agent."""

        # Validate tools
        if request.tools:
            invalid = _validate_tools(request.tools)
            if invalid:
                return Result(
                    success=False,
                    errorCode="INVALID_TOOLS",
                    errorMessage=f"Invalid tool(s): {', '.join(invalid)}. Valid categories: {', '.join(sorted(VALID_TOOL_CATEGORIES))}",
                )
            forbidden = _validate_tools_for_agent_type(request.tools, request.type or "")
            if forbidden:
                return Result(
                    success=False,
                    errorCode="TOOL_NOT_ALLOWED_FOR_AGENT_TYPE",
                    errorMessage=(
                        f"Tool(s) not allowed for agent type {request.type!r}: "
                        f"{', '.join(forbidden)}. ask_* agents are read-only consultants and "
                        "must not include filesystem write tools or wildcards that expand "
                        "beyond the read-only allowlist."
                    ),
                )

        # Check name not taken
        agentic_nodes = agent_config.agentic_nodes or {}
        if request.name in agentic_nodes or request.name in BUILTIN_SUBAGENTS:
            return Result(
                success=False,
                errorCode="AGENT_ALREADY_EXISTS",
                errorMessage=f"Agent '{request.name}' already exists",
            )

        # ask_report / ask_dashboard agents are bound to exactly one visual
        # artifact (a report or dashboard) — validate the binding before any
        # filesystem writes. The matching ``reports/<slug>`` or
        # ``dashboards/<slug>`` directory MUST already exist; absent slug or
        # missing artifact is a hard error so we never end up with a subagent
        # entry pointing at nothing.
        if request.type in {"ask_report", "ask_dashboard"}:
            ask_check = _validate_ask_artifact_binding(request, agent_config, agentic_nodes)
            if ask_check is not None:
                return ask_check

        # Create new agent entry (dict keyed by name, which acts as the id).
        # API field ``description`` is persisted as ``agent_description`` to
        # match what the runtime reads (sub_agent_task_tool / agentic_node /
        # the wizard all look up ``agent_description`` from agentic_nodes).
        # ``tools`` is rendered as the comma-separated yaml form expected by
        # ``GenSQLAgenticNode.setup_tools``; ``catalogs`` / ``subjects`` are
        # nested under ``scoped_context`` so a single block describes the
        # subagent's full reference scope.
        agent_entry = {
            "type": request.type or "gen_sql",
            "agent_description": request.description or "",
            "tools": _format_csv(request.tools),
            "rules": request.rules or [],
            "created_at": _utc_now_iso(),
        }
        # Bind the subagent to the active datasource (mirrors the wizard so
        # ``SubAgentConfig.is_in_datasource`` can gate task delegation at runtime).
        # ``request.datasource_id`` wins when set; otherwise fall back to the
        # AgentConfig's current datasource.
        datasource = request.datasource_id or getattr(agent_config, "current_datasource", "") or ""
        subject_buckets = (
            _classify_subject_paths(
                agent_config,
                list(request.subjects),
                datasource_id=datasource or None,
            )
            if request.subjects
            else None
        )
        scoped_ctx = _build_scoped_context(
            base=None,
            datasource=datasource,
            catalogs=request.catalogs,
            subject_buckets=subject_buckets,
        )
        if scoped_ctx:
            agent_entry["scoped_context"] = scoped_ctx
        if request.prompt_template:
            agent_entry["prompt_template"] = request.prompt_template
        if request.prompt_version:
            agent_entry["prompt_version"] = request.prompt_version
        # ask_* agents carry their bound artifact's slug directly on the
        # agentic_nodes entry — the node reads it via ``self.node_config``
        # without any wrapper. The SaaS backend stores the same value under
        # ``subagents.extra.artifact.slug`` and flattens it back to this
        # key in ``config_loader._build_agentic_nodes_dict`` so the two
        # backends are fully interchangeable from the runtime's view.
        if request.type in {"ask_report", "ask_dashboard"} and request.artifact_slug:
            agent_entry["artifact_slug"] = request.artifact_slug

        # Save to agent.yml
        agentic_nodes[request.name] = agent_entry
        _save_agentic_nodes(agent_config, agentic_nodes)

        # Copy the builtin prompt template to the project's template directory (non-fatal)
        try:
            self._copy_prompt_template(
                agent_type=request.type or "gen_sql",
                agent_name=request.name,
                version=request.prompt_version,
                agent_config=agent_config,
            )
        except Exception:
            logger.warning(f"Failed to copy prompt template for agent '{request.name}' (non-fatal)", exc_info=True)

        return Result(success=True, data={"name": request.name, "id": request.name})

    @staticmethod
    def _sanitize_path_component(value: str) -> str:
        """Sanitize a string for safe use as a path component (no traversal)."""
        # Take only the basename to strip any directory separators
        safe = Path(value.replace(" ", "_")).name
        # Reject empty or dot-only names
        if not safe or safe in (".", ".."):
            raise ValueError(f"Invalid path component: {value!r}")
        return safe

    def _copy_prompt_template(
        self,
        agent_type: str,
        agent_name: str,
        version: Optional[str],
        agent_config: AgentConfig,
    ) -> None:
        """Copy the builtin prompt template for the agent type to the workspace template dir."""
        template_base = self._TYPE_TO_TEMPLATE.get(agent_type, "gen_sql_system")
        safe_name = self._sanitize_path_component(agent_name)
        try:
            source_path = self._prompt_manager._get_template_path(template_base)
        except FileNotFoundError:
            logger.warning(f"Builtin template '{template_base}' not found, skipping copy")
            return

        safe_version = self._sanitize_path_component(version) if version else version
        template_dir = agent_config.path_manager.datus_home / "template"
        os.makedirs(template_dir, exist_ok=True)
        target_file = template_dir / f"{safe_name}_system_{safe_version}.j2"
        if not target_file.resolve().is_relative_to(template_dir.resolve()):
            raise ValueError(f"Path escapes template directory: {target_file}")
        if not target_file.exists():
            content = source_path.read_text(encoding="utf-8")
            target_file.write_text(content, encoding="utf-8")
            logger.info(f"Copied prompt template: {source_path.name} -> {target_file}")

    def _save_prompt_template(
        self,
        agent_name: str,
        version: Optional[str],
        content: str,
        agent_config: AgentConfig,
    ) -> None:
        """Write prompt template content to the project's template file."""
        if not content:
            return
        safe_name = self._sanitize_path_component(agent_name)
        resolved = self._sanitize_path_component(version or "1.0")
        template_dir = agent_config.path_manager.datus_home / "template"
        os.makedirs(template_dir, exist_ok=True)
        target_file = template_dir / f"{safe_name}_system_{resolved}.j2"
        if not target_file.resolve().is_relative_to(template_dir.resolve()):
            raise ValueError(f"Path escapes template directory: {target_file}")
        target_file.write_text(content, encoding="utf-8")
        logger.info(f"Saved prompt template: {target_file}")

    async def edit_agent(
        self,
        request: EditAgentInput,
        agent_config: AgentConfig,
    ) -> Result[dict]:
        """Edit an existing custom sub-agent."""

        # Validate tool syntax up-front (doesn't need the agent record).
        if request.tools:
            invalid = _validate_tools(request.tools)
            if invalid:
                return Result(
                    success=False,
                    errorCode="INVALID_TOOLS",
                    errorMessage=f"Invalid tool(s): {', '.join(invalid)}. Valid categories: {', '.join(sorted(VALID_TOOL_CATEGORIES))}",
                )

        # Find the agent (dict keyed by name, treated as id)
        agentic_nodes = agent_config.agentic_nodes or {}
        if request.id not in agentic_nodes:
            return Result(
                success=False,
                errorCode="AGENT_NOT_FOUND",
                errorMessage=f"Agent '{request.id}' not found",
            )

        agent = agentic_nodes[request.id]

        # Per-agent-type allowlist runs after the lookup so we know the
        # bound type. ask_* must stay read-only — reject any tool that
        # would expand reach beyond the documented read-only set.
        if request.tools:
            forbidden = _validate_tools_for_agent_type(request.tools, agent.get("type") or "")
            if forbidden:
                return Result(
                    success=False,
                    errorCode="TOOL_NOT_ALLOWED_FOR_AGENT_TYPE",
                    errorMessage=(
                        f"Tool(s) not allowed for agent type {agent.get('type')!r}: "
                        f"{', '.join(forbidden)}. ask_* agents are read-only consultants and "
                        "must not include filesystem write tools or wildcards that expand "
                        "beyond the read-only allowlist."
                    ),
                )

        # Channel bindings live in a separate top-level ``channels`` yaml
        # section (consumed by the gateway), so persist them independently of
        # the agentic_nodes update below — and before the no-op short-circuit,
        # so a channels-only edit still saves.
        if request.channels is not None:
            channel_error = self._apply_channel_binding(request, agent_config)
            if channel_error is not None:
                return channel_error

        # If prompt_template content is provided, save to template file
        prompt_content = request.prompt_template
        if prompt_content is not None:
            version = request.prompt_version or agent.get("prompt_version")
            try:
                self._save_prompt_template(
                    agent_name=request.id,
                    version=version,
                    content=prompt_content,
                    agent_config=agent_config,
                )
            except Exception:
                logger.warning(f"Failed to save prompt template for agent '{request.id}' (non-fatal)", exc_info=True)

        # Update only provided fields (name is the dict key and acts as id, so exclude it).
        # API ``description`` lands on the runtime-visible ``agent_description``
        # key; drop any legacy flat ``description`` left over from older edits
        # so the read path doesn't see two competing values.
        update_data = request.model_dump(exclude={"id", "name", "prompt_template", "channels"}, exclude_none=True)
        if "description" in update_data:
            update_data["agent_description"] = update_data.pop("description")
            agent.pop("description", None)

        # ``tools`` is persisted as the comma-separated yaml form expected by
        # ``GenSQLAgenticNode.setup_tools`` (which calls ``str.split(",")``).
        if "tools" in update_data:
            update_data["tools"] = _format_csv(update_data["tools"])

        # The API ``catalogs`` field maps to ``scoped_context.tables`` — that's
        # the runtime-honored key consumed by
        # ``ScopedFilterBuilder.build_table_filter``. ``subjects`` is *classified*
        # (via the metric / reference-sql stores) and split
        # across the runtime-visible ``metrics`` / ``sqls``
        # keys, since ``ScopedContext`` has no flat ``subjects`` field. Editing
        # any scope-related field also rewrites ``scoped_context.datasource`` to
        # the active datasource so ``SubAgentConfig.is_in_datasource`` agrees
        # with the saved binding. Pre-existing yaml-loaded scoped_context keys
        # survive the merge; top-level ``catalogs`` / ``subjects`` from older
        # edits are dropped so the read path can't see two competing copies.
        catalogs_input = update_data.pop("catalogs", None)
        subjects_input = update_data.pop("subjects", None)
        scope_touched = catalogs_input is not None or subjects_input is not None or "scoped_context" in update_data
        # Tracks deletions applied directly to ``agent`` (the live yaml dict)
        # rather than through ``update_data``. ``agent.update(update_data)``
        # below can't represent a key removal, so we have to bypass the
        # ``not update_data`` short-circuit and force ``_save_agentic_nodes``
        # to run when the only mutation was a pop.
        agent_dict_mutated = False
        if scope_touched:
            base_ctx: dict = {}
            existing = agent.get("scoped_context")
            if isinstance(existing, dict):
                base_ctx.update(existing)
            request_ctx = update_data.pop("scoped_context", None)
            if isinstance(request_ctx, dict):
                base_ctx.update(request_ctx)
            # Resolve the effective datasource *before* classification: if
            # ``agent_config.current_datasource`` is unset but the agent
            # already has a saved binding under ``scoped_context.datasource``,
            # the classifier must use the saved DS to look up entries in the
            # right metric / sql stores. Otherwise it would
            # fall back to "no datasource → all metrics" and silently
            # mis-bucket subjects against a binding that's about to be
            # re-persisted by ``_build_scoped_context``.
            datasource = getattr(agent_config, "current_datasource", "") or base_ctx.get("datasource") or ""
            subject_buckets = None
            if subjects_input is not None:
                subject_buckets = _classify_subject_paths(
                    agent_config,
                    list(subjects_input),
                    datasource_id=datasource or None,
                )
            merged = _build_scoped_context(
                base=base_ctx,
                datasource=datasource,
                catalogs=catalogs_input,
                subject_buckets=subject_buckets,
            )
            if merged:
                update_data["scoped_context"] = merged
            elif agent.pop("scoped_context", None) is not None:
                # Scope was fully cleared — record the deletion so the
                # subsequent save persists it instead of leaving the old
                # block on disk.
                agent_dict_mutated = True
            if agent.pop("catalogs", None) is not None:
                agent_dict_mutated = True
            if agent.pop("subjects", None) is not None:
                agent_dict_mutated = True

        if not update_data and prompt_content is None and not agent_dict_mutated:
            return Result(success=True, data={"name": request.id, "id": request.id})

        # Merge update data into the agent entry
        agent.update(update_data)

        # Save back to agent.yml
        _save_agentic_nodes(agent_config, agentic_nodes)

        return Result(success=True, data={"name": request.id, "id": request.id})

    def _apply_channel_binding(self, request: EditAgentInput, agent_config: AgentConfig):
        """Replace this agent's entries in the global ``channels`` map.

        The binding carries the full desired channel list for ``request.id``.
        We drop the agent's existing entries (matched by ``subagent_id``),
        keeping channels routed to other agents, then append the new ones and
        persist. Each entry takes the gateway's yaml shape::

            <name>:
              adapter: <type>
              enabled: <bool>
              subagent_id: <agent id>   # routes inbound messages to this agent
              extra: {<secret fields>}  # written verbatim, no encryption

        Returns a failure :class:`Result` on a bad entry, else ``None``.
        """
        binding = request.channels
        channels = dict(getattr(agent_config, "channels_config", None) or {})

        # Keep only channels NOT routed to this agent.
        channels = {
            key: value
            for key, value in channels.items()
            if not (isinstance(value, dict) and value.get("subagent_id") == request.id)
        }

        for channel in binding.channels or []:
            name = (channel.name or "").strip()
            if not name:
                return Result(
                    success=False,
                    errorCode="INVALID_CHANNEL_NAME",
                    errorMessage="Channel name cannot be empty",
                )
            # ``channels`` no longer holds this agent's own entries, so any name
            # collision here belongs to a different agent — reject instead of
            # silently clobbering its binding.
            existing = channels.get(name)
            if isinstance(existing, dict) and existing.get("subagent_id") != request.id:
                return Result(
                    success=False,
                    errorCode="CHANNEL_NAME_CONFLICT",
                    errorMessage=f"Channel name '{name}' is already bound to another sub-agent",
                )
            channels[name] = {
                "adapter": channel.type,
                "enabled": channel.enabled,
                "subagent_id": request.id,
                "extra": dict(channel.secrets or {}),
            }

        _save_channels(agent_config, channels)
        return None

    async def delete_agent(
        self,
        agent_id: str,
        agent_config: AgentConfig,
    ) -> Result[dict]:
        """Delete a custom sub-agent from ``agent.yml``.

        Builtin sub-agents are immutable and cannot be removed. The matching
        entry is popped from ``agentic_nodes`` and the yaml is rewritten;
        any prompt-template files this agent owns under
        ``{datus_home}/template/<name>_system_*.j2`` are removed on a
        best-effort basis (failures are logged, not raised — the yaml write
        is the source of truth for whether the agent exists).
        """

        if agent_id in BUILTIN_SUBAGENTS:
            return Result(
                success=False,
                errorCode="BUILTIN_AGENT_IMMUTABLE",
                errorMessage=f"Builtin agent '{agent_id}' cannot be deleted",
            )

        agentic_nodes = agent_config.agentic_nodes or {}
        if agent_id not in agentic_nodes:
            return Result(
                success=False,
                errorCode="AGENT_NOT_FOUND",
                errorMessage=f"Agent '{agent_id}' not found",
            )

        del agentic_nodes[agent_id]
        _save_agentic_nodes(agent_config, agentic_nodes)

        # Drop any channels routed to this agent so deleting it doesn't leave
        # dangling gateway bindings pointing at a non-existent sub-agent.
        channels = getattr(agent_config, "channels_config", None) or {}
        remaining = {
            key: value
            for key, value in channels.items()
            if not (isinstance(value, dict) and value.get("subagent_id") == agent_id)
        }
        if len(remaining) != len(channels):
            _save_channels(agent_config, remaining)

        try:
            self._delete_prompt_templates(agent_id, agent_config)
        except Exception:
            logger.warning(f"Failed to clean prompt templates for agent '{agent_id}' (non-fatal)", exc_info=True)

        return Result(success=True, data={"id": agent_id, "name": agent_id})

    def _delete_prompt_templates(self, agent_name: str, agent_config: AgentConfig) -> None:
        """Best-effort cleanup of ``{datus_home}/template/<name>_system_*.j2``.

        Mirrors the create path's template-copy behavior: ``create_agent``
        seeds one template per agent under the project's ``template`` dir,
        so delete sweeps every version that shares the sanitized agent name
        prefix. ``_sanitize_path_component`` only strips path separators —
        glob metacharacters (``*``, ``?``, ``[]``) survive — so we iterate
        ``iterdir()`` and match literally with ``startswith`` / ``endswith``
        instead of feeding ``safe_name`` into ``Path.glob``. Each match is
        re-checked to be relative to the resolved template dir, so a
        maliciously crafted ``agent_name`` cannot reach files outside it.
        """
        safe_name = self._sanitize_path_component(agent_name)
        template_dir = (agent_config.path_manager.datus_home / "template").resolve()
        if not template_dir.is_dir():
            return
        prefix = f"{safe_name}_system_"
        for path in template_dir.iterdir():
            if not path.is_file():
                continue
            if not (path.name.startswith(prefix) and path.name.endswith(".j2")):
                continue
            try:
                if not path.resolve().is_relative_to(template_dir):
                    continue
            except (OSError, ValueError):
                continue
            try:
                path.unlink()
                logger.info(f"Removed prompt template: {path}")
            except OSError:
                logger.warning(f"Failed to remove prompt template: {path}", exc_info=True)
