# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared base for the two ``ask_*`` follow-up subagents.

``AskReportAgenticNode`` and ``AskDashboardAgenticNode`` are read-only
follow-up consultants bound to **one specific visual artifact** (a
``reports/<slug>/`` or ``dashboards/<slug>/`` directory produced by the
matching ``gen_visual_*`` subagent). They reuse the conversational
plumbing of :class:`ChatAgenticNode` (sessions, memory, SSE, tool
permissions, etc.) and add three things:

1. **Artifact binding** — bind to one specific artifact via either of two
   sources: an in-memory ``artifact_blob`` injected into the agentic_nodes
   entry by the backend (a frozen ``{manifest, files}`` snapshot of the
   latest published version), or an on-disk ``reports/<slug>/`` /
   ``dashboards/<slug>/`` directory under ``project_root``. The blob
   source wins when present; the disk source remains the fallback for
   CLI runs and kinds that have not yet been wired through publish
   (currently ``ask_dashboard``). ``BLOB_REQUIRED = True`` on a subclass
   turns missing-blob into a hard failure rather than a disk fallback —
   used by ``ask_report`` where every live SaaS session must answer
   against the published artifact, not whatever happens to be on local
   disk.
2. **Constrained filesystem view** — override ``_make_filesystem_tool``
   so the LLM's ``read_file`` / ``glob`` / ``grep`` calls are anchored
   at the artifact root. Relative paths in prompts (``analysis/intent.md``,
   ``queries/<name>.json``) just work, and the LLM cannot accidentally
   peek into a sibling artifact or the global subject library through
   filesystem traversal. The blob source uses :class:`MemoryFilesystemFuncTool` (no disk
   touched); the disk source uses :class:`FilesystemFuncTool`.
3. **Artifact context injection** — load ``manifest.json`` plus
   ``analysis/intent.md`` once at node startup, and render the full
   artifact context preamble (manifest header, intent, subject scope,
   confirmed insights for reports, and a per-query catalog with
   brief + columns + sample/rows + SQL) into the system prompt.
   Earlier iterations only preloaded the manifest + intent and left
   everything else to ``read_file`` round-trips at turn time; the
   observed failure mode was the LLM issuing 8–10+ serial
   ``glob`` + ``read_file`` calls per follow-up before producing any
   output, even when the prompt could carry every sidecar directly.
   The renderer now inlines as much as fits under
   :data:`INLINE_CATALOG_BYTES_CAP` and degrades the long tail
   (sample rows instead of full, tighter SQL truncation) so the LLM
   has a complete grounding without paying read round-trips for it.

``suggested_questions.json`` is the one analysis file still kept out
of the inline preamble — it's surfaced via the detail API as UI
chips, and injecting its contents here would anchor the LLM toward a
fixed question set whenever the user types an open-ended follow-up.
The filename does appear in the layout-tree section, but only on a
``DO NOT read`` line so the original anti-anchor intent is enforced
by explicit instruction. The earlier ``interpretation.json`` preload
was removed along with the file itself.

Per-kind specialization (``ARTIFACT_KIND`` / template name / whether
``insights.json`` is expected / whether ``BLOB_REQUIRED``) lives in the
two concrete subclasses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.artifact_manifest import ARTIFACT_SLUG_RE
from datus.schemas.chat_agentic_node_models import ChatNodeInput
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# ── Inline-rendering thresholds ──────────────────────────────────────────────
# A query result's rows are inlined into the system prompt only when BOTH
# conditions hold:
#   1. row_count <= INLINE_ROW_LIMIT
#   2. JSON-serialized rows fit in INLINE_ROW_BYTES_LIMIT bytes
# Either threshold being exceeded triggers degraded mode: the catalog entry
# keeps the columns + a 2-row sample, and the full payload stays read-only
# behind ``read_file('queries/<slug>.json')``. The byte gate is what catches
# a 20-row result whose rows each carry a multi-KB text field — row count
# alone would let that through.
INLINE_ROW_LIMIT = 20
INLINE_ROW_BYTES_LIMIT = 4 * 1024

# SQL bodies > this many lines are truncated to the first N lines with a
# trailing "...(truncated)" marker. Report SQLs are typically <20 lines;
# anything beyond 40 is exceptional and the LLM can pay one read_file
# round-trip if it genuinely needs the full body.
INLINE_SQL_LINE_LIMIT = 40

# Soft cap on the total bytes the query-catalog section contributes to the
# system prompt. When approached the renderer degrades remaining entries
# (drop full rows -> drop caveats -> tighter SQL truncation). Header info
# (slug + size + hypothesis + columns) always survives so the LLM still
# knows what data exists; a follow-up ``read_file`` can fetch detail.
INLINE_CATALOG_BYTES_CAP = 64 * 1024

# Per-table column cap for the Table Schemas section. A wide fact table
# (200+ columns) inlined whole would dominate the prompt without payoff —
# the LLM only references a handful per follow-up. We truncate to the
# first N and tell the LLM to call ``describe_table()`` for the full list
# when it needs more.
INLINE_SCHEMA_COLS_PER_TABLE = 50

# Soft cap on the Table Schemas section as a whole. Sized so a typical
# report (2–5 tables × ~10–30 columns each) fits comfortably; a report
# referencing 20+ tables hits the cap and the renderer drops trailing
# tables with a "remaining omitted" marker so the LLM knows to fall back
# to ``describe_table``.
INLINE_SCHEMA_BYTES_CAP = 8 * 1024


def _compact_row(row: Any) -> str:
    """Render a single result row as one deterministic line.

    Used in the inline ``rows`` / ``sample`` blocks of the query catalog
    section. JSON-encodes scalars so units/quoting stay unambiguous (e.g.
    ``aov=29.19`` vs ``aov="29.19"``) and joins key/value pairs with the
    middle dot so the eye can scan a wide row without confusing commas
    inside string values with row-level separators.
    """
    if isinstance(row, dict):
        parts = [f"{k}={json.dumps(v, ensure_ascii=False, default=str)}" for k, v in row.items()]
        return " · ".join(parts)
    return json.dumps(row, ensure_ascii=False, default=str)


class BaseArtifactAskAgenticNode(ChatAgenticNode):
    """Shared lifecycle for ``ask_report`` / ``ask_dashboard`` nodes.

    Subclasses must set:

    * :pyattr:`NODE_NAME` — ``"ask_report"`` / ``"ask_dashboard"`` (used
      as the configured_node_name and prompt template root).
    * :pyattr:`ARTIFACT_KIND` — ``"report"`` / ``"dashboard"`` (rendered
      into the prompt context so the same partial branches on it).
    * :pyattr:`ARTIFACT_ROOT_DIR_NAME` — ``"reports"`` / ``"dashboards"``
      (directory under ``project_root`` where the bound slug lives).
    """

    NODE_NAME: ClassVar[str] = "ask_artifact"
    ARTIFACT_KIND: ClassVar[Literal["report", "dashboard"]] = "report"
    ARTIFACT_ROOT_DIR_NAME: ClassVar[str] = "reports"
    # When True, a missing ``artifact_blob`` in the agentic_nodes entry is a
    # fatal startup error rather than a signal to fall back to the on-disk
    # ``<kind>/<slug>/`` directory. Kinds whose backend publish flow always
    # produces a blob (currently ``ask_report``) set this to True so the
    # half-bound state (subagent exists, no published version) errors at init
    # instead of silently grounding the LLM against an unrelated on-disk
    # tree (or worse, the backend's own filesystem which won't have the
    # artifact at all). Kinds without a publish flow yet
    # (``ask_dashboard``) keep this False so the disk path stays available.
    BLOB_REQUIRED: ClassVar[bool] = False

    # Capability tool groups gated by the subagent's ``tools`` whitelist, mapped
    # to the node attribute that holds the built tool instance. Infrastructure
    # tools (artifact-anchored filesystem, ask_user, plan/todo, sub-agent
    # delegation) are deliberately ABSENT: a read-only consultant needs them to
    # read its bound artifact and converse regardless of the whitelist, so they
    # always survive pruning — mirroring GenSQLAgenticNode's always-on
    # filesystem + sub_agent_task + plan set. ``read_query`` & friends live
    # under ``db_tools`` and are dropped unless explicitly whitelisted.
    _WHITELIST_GROUP_ATTRS: ClassVar[Dict[str, str]] = {
        "db_tools": "db_func_tool",
        "context_search_tools": "context_search_tools",
        "semantic_tools": "semantic_tools",
        "reference_template_tools": "reference_template_tools",
        "date_parsing_tools": "date_parsing_tools",
        "platform_doc_tools": "_platform_doc_tool",
        "bash_tools": "bash_tool",
        "skills": "skill_func_tool",
    }

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[ChatNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
        node_name: Optional[str] = None,
        scope: Optional[str] = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        is_subagent: bool = False,
        session_id: Optional[str] = None,
    ) -> None:
        # Stash the subagent name BEFORE super().__init__() runs because
        # ChatAgenticNode hard-codes ``configured_node_name = "chat"`` and we
        # need our own (``node_name`` from agentic_nodes, e.g. "ask_xxx") so
        # template resolution + node_config lookup land on the right entry.
        self._configured_subagent_name = node_name or self.NODE_NAME

        # ChatAgenticNode never builds semantic tools, but an ask_* ``tools``
        # whitelist may request them (metric/dimension/attribution analysis).
        # Declare the slot before super().__init__() (which triggers
        # ``setup_tools``) so the whitelist pass can build it on demand and
        # ``_tool_category_map`` can reference it safely.
        self.semantic_tools = None

        # Resolve the artifact binding BEFORE super().__init__() because
        # ChatAgenticNode.__init__ calls ``setup_tools()`` synchronously,
        # which builds the filesystem tool — and that needs the artifact
        # root as its ``root_path`` to constrain the LLM's reach. Loading
        # the binding here means ``_make_filesystem_tool`` (overridden
        # below) sees ``self._artifact_root`` already set when super-init
        # calls it. Any failure is fatal — a half-bound ask agent must
        # never silently answer against the wrong artifact.
        self._artifact_slug: str = ""
        self._artifact_root: Optional[Path] = None
        self._artifact_manifest: Dict[str, Any] = {}
        self._artifact_intent_md: str = ""
        # Populated only when the agentic_nodes entry carries an
        # ``artifact_blob``. When set, the filesystem tool is wired through
        # :class:`MemoryFilesystemFuncTool` instead of the disk-backed
        # :class:`FilesystemFuncTool` and ``_artifact_root`` stays None.
        self._artifact_files: Optional[Dict[str, str]] = None
        self._resolve_artifact_binding_early(agent_config)
        self._load_artifact_anchor_files()

        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools,
            scope=scope,
            execution_mode=execution_mode,
            is_subagent=is_subagent,
            session_id=session_id,
        )

        # ChatAgenticNode.__init__ overwrites configured_node_name to "chat";
        # restore our own AFTER super-init so prompt resolution uses the
        # right template (e.g. "ask_report_system" via ``_TYPE_TO_TEMPLATE``).
        self.configured_node_name = self._configured_subagent_name

    # ── Configured node name ────────────────────────────────────────────

    def get_node_name(self) -> str:
        # ChatAgenticNode.__init__ hard-codes ``configured_node_name = "chat"``
        # which would otherwise make ``AgenticNode._parse_node_config`` look up
        # the wrong agentic_nodes entry during super().__init__(). We stash the
        # caller-supplied subagent name on ``_configured_subagent_name`` before
        # super-init so this getter can prefer it. After super-init we also
        # restore ``configured_node_name`` to the same value so any downstream
        # code reading the attribute directly (rather than via this method)
        # sees the right name too.
        name = getattr(self, "_configured_subagent_name", None)
        if name:
            return name
        return self.configured_node_name or self.NODE_NAME

    # ── Tool whitelist enforcement (honor SubAgent.tools) ───────────────

    def setup_tools(self) -> None:
        """Build the full chat tool surface, then constrain capability tools
        to the subagent's configured ``tools`` whitelist.

        ChatAgenticNode wires its capability tools (db_tools, context_search,
        …) unconditionally and never consults ``tools``. For an ask_*
        consultant that field is a hard whitelist — the agent must not reach a
        capability the operator did not grant (e.g. ``read_query`` when only
        semantic/context tools were configured). We reuse the base setup so all
        infrastructure (permission/skill managers, artifact-anchored
        filesystem, plan + sub-agent tooling) is built correctly, then apply
        the whitelist. An empty/absent whitelist leaves the inherited surface
        untouched (back-compat for ask agents created before tool scoping).
        """
        super().setup_tools()
        self._apply_tools_whitelist()

    def _rebuild_tools(self) -> None:
        """Re-apply the whitelist after any base rebuild.

        ``ChatAgenticNode._rebuild_tools`` (also reached via a mid-session
        datasource switch) repopulates ``self.tools`` from the full set of tool
        instances, which would silently re-expose pruned capabilities. Re-running
        the whitelist here keeps the constraint enforced for the life of the
        node. Safe during ``super().__init__`` because the slots it reads are
        initialised beforehand or accessed via ``getattr``.
        """
        super()._rebuild_tools()
        self._apply_tools_whitelist()

    def _tool_category_map(self) -> Dict[str, List[Any]]:
        """Register semantic tools under their canonical permission category.

        The chat base has no semantic tools so its map omits them; when the
        whitelist makes us build them, surface them here so PermissionHooks
        matches ``semantic_tools.*`` rules instead of the ``tools.*`` catch-all.
        """
        mapping = super()._tool_category_map()
        if getattr(self, "semantic_tools", None):
            mapping["semantic_tools"] = list(self.semantic_tools.available_tools())
        return mapping

    def _apply_tools_whitelist(self) -> None:
        """Constrain ``self.tools`` to the configured whitelist + infrastructure."""
        patterns = self._parse_tool_whitelist(self.node_config.get("tools"))
        if not patterns:
            # Unconfigured -> keep the inherited full surface (back-compat).
            return
        self._ensure_whitelisted_groups_present(patterns)
        self._prune_capability_tools(patterns)

    @staticmethod
    def _parse_tool_whitelist(tools_value: Any) -> List[str]:
        """Split the comma-separated ``tools`` field into trimmed patterns."""
        if not tools_value or not str(tools_value).strip():
            return []
        return [p.strip() for p in str(tools_value).split(",") if p.strip()]

    @staticmethod
    def _tool_matches_whitelist(group: str, tool_name: str, patterns: List[str]) -> bool:
        """True if ``group``/``tool_name`` is granted by any whitelist pattern.

        Accepts the same three shapes GenSQLAgenticNode does: ``group`` (whole
        group), ``group.*`` (wildcard) and ``group.<method>`` (single tool).
        """
        candidates = {group, f"{group}.*", f"{group}.{tool_name}"}
        return any(p in candidates for p in patterns)

    def _ensure_whitelisted_groups_present(self, patterns: List[str]) -> None:
        """Build + surface whitelisted capability groups the chat base omits.

        Only ``semantic_tools`` falls in this bucket today — ChatAgenticNode
        builds every other gated group. Built once, then re-surfaced into
        ``self.tools`` on every call so a post-rebuild whitelist pass does not
        lose it (``_rebuild_tools`` never re-adds semantic tools itself).
        """
        wants_semantic = any(p == "semantic_tools" or p.startswith("semantic_tools.") for p in patterns)
        if not wants_semantic:
            return
        if not getattr(self, "semantic_tools", None):
            try:
                from datus.tools.func_tool.semantic_tools import SemanticTools

                self.semantic_tools = SemanticTools(
                    agent_config=self.agent_config,
                    sub_agent_name=self.node_config.get("system_prompt"),
                    adapter_type=self.node_config.get("adapter_type", "metricflow"),
                )
            except Exception as exc:
                logger.error("%s: failed to build whitelisted semantic_tools: %s", self.get_node_name(), exc)
                return
        present = {t.name for t in self.tools}
        for tool in self.semantic_tools.available_tools():
            if tool.name not in present:
                self.tools.append(tool)

    def _capability_tool_groups(self) -> Dict[str, str]:
        """Map each built capability tool *name* -> its whitelist group label.

        Built only from the gated groups in ``_WHITELIST_GROUP_ATTRS``;
        infrastructure tools are intentionally excluded so they never appear
        here and therefore always survive pruning.
        """
        name_to_group: Dict[str, str] = {}
        for group, attr in self._WHITELIST_GROUP_ATTRS.items():
            inst = getattr(self, attr, None)
            if not inst:
                continue
            try:
                for tool in inst.available_tools():
                    name_to_group[tool.name] = group
            except Exception as exc:
                logger.warning("%s: cannot enumerate %s tools for whitelist: %s", self.get_node_name(), group, exc)
        return name_to_group

    def _prune_capability_tools(self, patterns: List[str]) -> None:
        """Drop every capability tool not granted by the whitelist.

        A tool is kept when it is NOT a gated capability (i.e. infrastructure)
        or when a whitelist pattern grants it. Idempotent — safe to call on
        every rebuild.
        """
        name_to_group = self._capability_tool_groups()
        kept: List[Any] = []
        dropped: List[str] = []
        for tool in self.tools:
            group = name_to_group.get(tool.name)
            if group is None or self._tool_matches_whitelist(group, tool.name, patterns):
                kept.append(tool)
            else:
                dropped.append(tool.name)
        self.tools = kept
        if dropped:
            logger.info(
                "%s tools whitelist applied: dropped=%s exposed=%s",
                self.get_node_name(),
                sorted(set(dropped)),
                sorted({t.name for t in kept}),
            )

    # ── Artifact binding resolution ─────────────────────────────────────

    def _resolve_artifact_binding_early(self, agent_config: Optional[AgentConfig]) -> None:
        """Resolve the artifact binding directly from the agentic_nodes entry.

        Called BEFORE ``super().__init__()`` runs, so we can't rely on
        ``self.node_config`` (set by AgenticNode init) or on
        ``self.agent_config`` (set by AgenticNode init). We read the raw
        ``agent_config.agentic_nodes[subagent_name]`` entry directly.

        Resolution order:

        1. If ``entry["artifact_blob"]`` is present, bind to the in-memory
           bundle (``{manifest, files}``). The filesystem tool then runs
           against :class:`MemoryFilesystemFuncTool` and ``_artifact_root`` stays None.
        2. Otherwise, if ``BLOB_REQUIRED`` is True, fail — the caller is
           contractually supposed to provide a blob for this kind.
        3. Otherwise, fall back to resolving the on-disk
           ``<project_root>/<kind>/<slug>/`` directory (legacy CLI flow and
           kinds without a backend publish path yet).

        Failures raise :class:`DatusException` — there is no useful default
        for a missing binding and we'd rather see a clear startup error
        than a runtime "I don't know which artifact you mean".
        """
        if agent_config is None or not getattr(agent_config, "agentic_nodes", None):
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={
                    "config_error": (
                        f"{self.NODE_NAME} requires an agent_config with a populated "
                        "agentic_nodes registry to resolve its artifact binding."
                    )
                },
            )
        entry = (agent_config.agentic_nodes or {}).get(self._configured_subagent_name)
        if not isinstance(entry, dict):
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={
                    "config_error": (
                        f"agentic_nodes entry {self._configured_subagent_name!r} not "
                        f"found (or not a dict). {self.NODE_NAME} cannot resolve its "
                        "artifact binding."
                    )
                },
            )
        slug = (entry.get("artifact_slug") or "").strip()
        if not slug:
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={
                    "config_error": (
                        f"{self.NODE_NAME} agent requires ``artifact_slug`` in its "
                        "agentic_nodes entry (SaaS path: subagents.extra.artifact.slug; "
                        "CLI path: yaml ``artifact_slug`` key)."
                    )
                },
            )
        if not ARTIFACT_SLUG_RE.fullmatch(slug):
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": (f"artifact_slug {slug!r} must match {ARTIFACT_SLUG_RE.pattern}")},
            )

        self._artifact_slug = slug

        # Path 1: in-memory blob from the agentic_nodes entry. Backend
        # populates this for ``ask_report`` from the latest VisualReportVersion
        # at config-build time. Reject obviously degenerate shapes (empty
        # dict, ``{"files": []}``, missing manifest) before binding so a
        # malformed blob ends up in the BLOB_REQUIRED / disk-fallback
        # branches below instead of silently binding to an empty
        # filesystem — without this, a half-bound report would answer
        # "File not found" to every read and look like a working agent.
        blob = entry.get("artifact_blob")
        if self._is_usable_blob(blob):
            self._bind_artifact_from_blob(blob)
            return

        if blob is not None:
            logger.warning(
                "%s artifact_blob present but unusable (type=%s, keys=%s); routing to BLOB_REQUIRED/disk fallback",
                self.NODE_NAME,
                type(blob).__name__,
                sorted(blob.keys()) if isinstance(blob, dict) else None,
            )

        if self.BLOB_REQUIRED:
            logger.error(
                "%s init failing: slug=%s has no usable artifact_blob and BLOB_REQUIRED=True",
                self.NODE_NAME,
                slug,
            )
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={
                    "config_error": (
                        f"{self.NODE_NAME} agent for slug {slug!r} has no "
                        "``artifact_blob`` in its agentic_nodes entry. The "
                        f"{self.ARTIFACT_KIND} has not been published yet — "
                        "publish it first so the latest version's artifact is "
                        "snapshotted into the subagent config."
                    )
                },
            )

        self._bind_artifact_from_disk(agent_config, slug)

    @staticmethod
    def _is_usable_blob(blob: Any) -> bool:
        """Return True only for blobs that carry real artifact content.

        The backend's wire shape is ``{manifest: {...}, files: [{path,
        content}, ...]}`` and a successful publish always populates both:
        ``manifest`` is required on the source ``VisualReportVersion`` and
        ``files`` covers the per-prefix allowlist (render/queries/analysis)
        which is non-empty for any artifact that passed the publish
        validator. So an empty dict, a ``files``-only blob with no
        manifest, or a blob with ``files: []`` is a degenerate/half-bound
        signal — treat it as a missing blob so the BLOB_REQUIRED branch
        fires for kinds that need it (rather than the agent silently
        binding to an empty filesystem and answering "File not found" to
        every read).
        """
        if not isinstance(blob, dict):
            return False
        manifest = blob.get("manifest")
        files = blob.get("files")
        return isinstance(manifest, dict) and bool(manifest) and isinstance(files, list) and bool(files)

    def _bind_artifact_from_blob(self, blob: Dict[str, Any]) -> None:
        """Bind to an in-memory ``{manifest, files}`` snapshot.

        Flattens the ``files: [{path, content}, ...]`` list into a dict
        keyed by slug-relative path so :class:`MemoryFilesystemFuncTool` can serve it
        directly. Non-dict / malformed entries are skipped silently — the
        wire format is owned by the backend and any drift should surface
        as missing files at read time rather than a hard init error.

        ``manifest.json`` is intentionally omitted from the backend's
        ``files[]`` (it's already carried structured at ``blob["manifest"]``
        to avoid duplication on the wire), but the LLM-facing tool surface
        advertises it as a readable file — the prompt preamble even prints
        ``manifest.json`` in the directory tree. To keep blob mode
        feature-parity with the disk-backed tool (and avoid an LLM-visible
        "File not found" the moment it follows the prompt), synthesize the
        entry back from the structured form.
        """
        manifest = blob.get("manifest")
        if isinstance(manifest, dict):
            self._artifact_manifest = manifest

        raw_files = blob.get("files")
        files: Dict[str, str] = {}
        if isinstance(raw_files, list):
            for entry in raw_files:
                if not isinstance(entry, dict):
                    continue
                path = entry.get("path")
                content = entry.get("content")
                if isinstance(path, str) and path and isinstance(content, str):
                    files[path] = content

        if "manifest.json" not in files and isinstance(manifest, dict):
            try:
                files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2)
            except TypeError:
                # Manifest carries something json can't encode (shouldn't
                # happen with the current Pydantic-derived shape, but stay
                # defensive). Init still succeeds; the LLM gets a clearly
                # empty placeholder rather than a "File not found".
                files["manifest.json"] = "{}"

        self._artifact_files = files
        logger.info(
            "%s bound from in-memory blob: slug=%s files=%d",
            self.NODE_NAME,
            self._artifact_slug,
            len(self._artifact_files),
        )

    def _bind_artifact_from_disk(self, agent_config: AgentConfig, slug: str) -> None:
        """Bind to the on-disk ``<project_root>/<kind>/<slug>/`` directory."""
        project_root_raw = getattr(agent_config, "project_root", None)
        if not project_root_raw:
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"{self.NODE_NAME} requires agent_config.project_root"},
            )
        project_root = Path(project_root_raw).resolve()
        expected_dir = project_root / self.ARTIFACT_ROOT_DIR_NAME / slug
        artifact_dir = expected_dir.resolve()

        # Path traversal defence — slug regex already blocks ``..`` literals,
        # but a symlink at ``<kind>/<slug>`` could still redirect us elsewhere
        # (outside project_root entirely, or to a sibling directory inside it
        # the ask agent should not be reading). Require the resolved path to
        # match the unresolved expected location verbatim — any symlink
        # redirection produces a mismatch.
        if artifact_dir != expected_dir:
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": (f"artifact path resolved outside expected location: {artifact_dir}")},
            )
        if not artifact_dir.is_dir():
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={
                    "config_error": (
                        f"{self.ARTIFACT_ROOT_DIR_NAME}/{slug} not found under "
                        f"project root {project_root}. Was the artifact deleted "
                        "after this subagent was created?"
                    )
                },
            )

        self._artifact_root = artifact_dir
        logger.info(
            "%s bound from on-disk artifact: slug=%s root=%s",
            self.NODE_NAME,
            self._artifact_slug,
            artifact_dir,
        )

    # ── Filesystem tool override ────────────────────────────────────────

    def _make_filesystem_tool(self, **kwargs):
        """Anchor the filesystem tool at the bound artifact.

        Two modes:

        * **Blob mode** (``self._artifact_files is not None``): return a
          :class:`MemoryFilesystemFuncTool` reading from the in-memory bundle. The disk is
          never touched, so concurrent writes to the on-disk source tree
          can't drift the answer mid-conversation, and the backend can
          serve ``ask_report`` even when it has no access to the IDE's
          filesystem.
        * **Disk mode** (``self._artifact_root is not None``): fall through
          to the base node's :class:`FilesystemFuncTool` with ``root_path``
          pinned to the artifact directory — preserves the original
          behaviour for CLI runs and kinds without a publish path.
        """
        if self._artifact_files is not None:
            from datus.tools.func_tool import MemoryFilesystemFuncTool

            logger.info(
                "%s filesystem tool wired to MemoryFilesystemFuncTool: slug=%s files=%d",
                self.NODE_NAME,
                self._artifact_slug,
                len(self._artifact_files),
            )
            # BaseTool absorbs unknown kwargs into tool_params — keeps
            # disk-mode-only kwargs from crashing init here.
            return MemoryFilesystemFuncTool(
                self._artifact_files,
                root_label=f"in-memory:{self._artifact_slug}",
                **kwargs,
            )

        # ``root_path`` is what gates the LLM's ``read_file`` / ``glob`` /
        # ``grep`` reach; passing it via kwargs ensures the policy layer
        # rejects any attempt to traverse outside this artifact.
        if "root_path" not in kwargs and self._artifact_root is not None:
            kwargs["root_path"] = str(self._artifact_root)
        return super()._make_filesystem_tool(**kwargs)

    # ── Anchor-file load (manifest + intent.md) ─────────────────────────

    def _load_artifact_anchor_files(self) -> None:
        """Load ``manifest.json`` + ``analysis/intent.md``.

        These are small (typically < 4KB total) and read once at node
        startup so the prompt template can render them directly. Other
        analysis files (insights, suggested_questions, subject_refs) are
        intentionally NOT preloaded — the LLM fetches them on demand
        with ``read_file`` to keep the per-turn system prompt small,
        and ``suggested_questions`` would also bias the LLM toward a
        fixed question set if it lived in the header.

        Missing / corrupt files degrade silently to empty values; the
        prompt template branches on emptiness. We log a warning so
        operators can investigate but never block the conversation.

        In blob mode the manifest was already populated by
        ``_bind_artifact_from_blob`` (parsed directly from the JSON
        structure rather than re-decoded from a string), so this method
        only needs to populate ``intent.md`` from the in-memory file map.
        """
        if self._artifact_files is not None:
            self._artifact_intent_md = self._artifact_files.get("analysis/intent.md", "")
            return

        if self._artifact_root is None:
            return

        manifest_path = self._artifact_root / "manifest.json"
        if manifest_path.is_file():
            try:
                self._artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to read %s: %s", manifest_path, exc)

        intent_path = self._artifact_root / "analysis" / "intent.md"
        if intent_path.is_file():
            try:
                self._artifact_intent_md = intent_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("Failed to read %s: %s", intent_path, exc)

    # ── Prompt context injection ────────────────────────────────────────

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> str:
        """Render the ask-* system prompt with artifact context added.

        Delegates to ``ChatAgenticNode._get_system_prompt`` for the
        heavy lifting (template lookup, skill XML injection, memory,
        language directive) and then prepends a markdown header block
        with the artifact's manifest fields and raw intent.md so the
        chat template's general copy ("You are the follow-up
        consultant…") already knows what it's talking about by the
        time the user's first message arrives.
        """
        # We can't simply override the template context dict the base
        # builds — ``prepare_template_context`` returns a fresh dict per
        # call. Instead, hook ``_finalize_system_prompt`` style: render
        # via parent, then prepend our artifact-context block so the
        # template-specific copy ("You are the follow-up consultant…")
        # already knows what it's talking about by the time the user's
        # first message arrives.
        #
        # The cleaner long-term fix is to let ``_get_system_prompt`` take
        # an extra context dict; for now this two-step approach keeps the
        # base class untouched.
        base_prompt = super()._get_system_prompt(conversation_summary, prompt_version)
        artifact_header = self._render_artifact_context_block()
        final_prompt = (artifact_header + "\n\n" + base_prompt) if artifact_header else base_prompt
        # Observability hook: real-world prompt growth after the
        # inline-rendering rework is hard to predict per artifact (varies
        # with insight count, query catalog size, table schema width).
        # Single grep-friendly line per turn so operators can spot
        # outliers without dragging through trace logs. Key=value form
        # makes ad-hoc parsing trivial.
        header_bytes = len(artifact_header.encode("utf-8")) if artifact_header else 0
        base_bytes = len(base_prompt.encode("utf-8"))
        final_bytes = len(final_prompt.encode("utf-8"))
        # ``count('\n') + 1`` matches what an LLM would see as "the
        # number of lines"; cheaper than splitting and we don't need
        # accuracy on trailing newlines.
        logger.info(
            "ask_artifact prompt assembled: node=%s kind=%s slug=%s lines=%d bytes=%d header_bytes=%d base_bytes=%d",
            self._configured_subagent_name,
            self.ARTIFACT_KIND,
            self._artifact_slug,
            final_prompt.count("\n") + 1,
            final_bytes,
            header_bytes,
            base_bytes,
        )
        return final_prompt

    def _render_artifact_context_block(self) -> str:
        """Build the artifact-context preamble prepended to the chat prompt.

        Composes per-section helper methods so each concern (header,
        intent, subject scope, insights, query catalog, layout, rules)
        stays self-contained and individually testable. The catalog
        section inlines as much of the artifact as fits under
        :data:`INLINE_CATALOG_BYTES_CAP` so the LLM rarely needs to
        ``read_file`` defensively — the trace this was tuned against
        was paying 10+ serial tool round-trips before producing any
        output, all loading content the prompt could carry directly.

        Hand-rolls markdown rather than a j2 template because the
        section helpers already encapsulate the structure and a
        template here would add indirection without saving lines.
        """
        if self._artifact_files is None and self._artifact_root is None:
            return ""

        # Sections produced in order; empty ones are silently dropped so
        # the rendered prompt stays clean for artifacts that don't have
        # insights / subject refs / queries (rare but real, e.g. a newly
        # created report with no save_query calls yet).
        sections: List[str] = [self._render_header_section()]
        for render in (
            self._render_intent_section,
            self._render_subject_scope_section,
            self._render_table_schemas_section,
            self._render_insights_section,
            self._render_query_catalog_section,
        ):
            block = render()
            if block:
                sections.append(block)
        sections.append(self._render_filesystem_layout_section())
        sections.append(self._render_behavioral_rules_section())
        return "\n\n".join(sections)

    # ── Unified artifact-file access ────────────────────────────────────

    def _read_artifact_file(self, rel_path: str) -> Optional[str]:
        """Return the contents of a slug-relative artifact file.

        Unifies blob mode (in-memory file map) and disk mode (rooted at
        ``self._artifact_root``) so the section renderers below don't
        each have to branch on the source. The path uses POSIX
        separators — same shape the LLM passes to ``read_file``.

        Returns ``None`` when the file is missing or unreadable so the
        caller can skip a section rather than raise mid-render. We log
        on disk-side ``OSError`` because it's the only signal an
        operator gets that the artifact tree is corrupted; missing
        files in blob mode are silently skipped (a degenerate blob is
        already caught at init by ``_is_usable_blob``).
        """
        if self._artifact_files is not None:
            return self._artifact_files.get(rel_path)
        if self._artifact_root is None:
            return None
        full = self._artifact_root / rel_path
        if not full.is_file():
            return None
        try:
            return full.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read %s during prompt render: %s", full, exc)
            return None

    def _query_slugs(self) -> List[str]:
        """Return query slugs in deterministic alphabetical order.

        Built from ``queries/*.brief.json`` because every saved query
        has a brief sidecar (``save_query`` writes them atomically with
        the SQL/result triple); SQL files without a brief shouldn't
        exist in a well-formed artifact and surfacing them would just
        clutter the catalog. Deterministic order is important for
        prompt caching — the same artifact must render the same catalog
        across turns.
        """
        prefix = "queries/"
        suffix = ".brief.json"
        slugs: List[str] = []
        if self._artifact_files is not None:
            for path in sorted(self._artifact_files):
                if path.startswith(prefix) and path.endswith(suffix):
                    slugs.append(path[len(prefix) : -len(suffix)])
            return slugs
        if self._artifact_root is None:
            return []
        queries_dir = self._artifact_root / "queries"
        if not queries_dir.is_dir():
            return []
        for path in sorted(queries_dir.glob("*.brief.json")):
            slugs.append(path.name[: -len(suffix)])
        return slugs

    def _load_query_bundle(self, slug: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str]]:
        """Return ``(brief, data, sql)`` for a query slug.

        ``data`` is the result snapshot for reports (``<slug>.json``)
        or the params declaration for dashboards (``<slug>.params.json``);
        ``sql`` is the SQL text (``<slug>.sql``) or template
        (``<slug>.sql.j2``). Any of the three may come back missing/
        partial — the catalog renderer falls back to whatever pieces
        are available so a single corrupt sidecar doesn't strand the
        whole section.
        """
        brief: Dict[str, Any] = {}
        brief_raw = self._read_artifact_file(f"queries/{slug}.brief.json")
        if brief_raw:
            try:
                parsed = json.loads(brief_raw)
                if isinstance(parsed, dict):
                    brief = parsed
            except json.JSONDecodeError as exc:
                logger.warning("Catalog render: brief.json for %s unreadable: %s", slug, exc)

        data: Optional[Dict[str, Any]] = None
        data_path = f"queries/{slug}.json" if self.ARTIFACT_KIND == "report" else f"queries/{slug}.params.json"
        data_raw = self._read_artifact_file(data_path)
        if data_raw:
            try:
                parsed = json.loads(data_raw)
                if isinstance(parsed, dict):
                    data = parsed
            except json.JSONDecodeError as exc:
                logger.warning("Catalog render: %s unreadable: %s", data_path, exc)

        sql_path = f"queries/{slug}.sql" if self.ARTIFACT_KIND == "report" else f"queries/{slug}.sql.j2"
        sql = self._read_artifact_file(sql_path)
        return brief, data, sql

    # ── Section renderers ──────────────────────────────────────────────

    def _render_header_section(self) -> str:
        """Top metadata block — artifact identity, source, key tables.

        Always rendered (caller already guarded against the "neither
        blob nor disk" case). Mirrors the original preamble field
        choices because downstream tests assert on specific labels
        ("Slug", "Tables referenced", "in-memory snapshot", etc.).
        """
        manifest = self._artifact_manifest or {}
        artifact_name = manifest.get("name") or self._artifact_slug
        artifact_description = manifest.get("description") or ""

        lines: List[str] = []
        lines.append(f"## Bound Artifact — {self.ARTIFACT_KIND.title()}: {artifact_name}")
        lines.append("")
        lines.append(f"- **Slug**: `{self._artifact_slug}`")
        if self._artifact_files is not None:
            lines.append(
                f"- **Source**: in-memory snapshot of the latest published version "
                f"({len(self._artifact_files)} files; filesystem tool anchored here)"
            )
        else:
            lines.append(f"- **Root**: `{self._artifact_root}` (anchors the filesystem tool)")
        if artifact_description:
            lines.append(f"- **Description**: {artifact_description}")
        if manifest.get("datasources"):
            lines.append(f"- **Datasources**: {', '.join(manifest['datasources'])}")
        if manifest.get("key_tables"):
            # Code-aggregated by finalize from the SQL bodies, not an
            # LLM claim — trustworthy as long as it's present. Surfacing
            # it here lets the LLM skip ``list_tables`` round-trips
            # when planning follow-up SQL.
            lines.append(f"- **Tables referenced**: {', '.join(manifest['key_tables'])}")
        return "\n".join(lines)

    def _render_intent_section(self) -> str:
        """User's original intent.md, verbatim.

        Returns "" when intent is empty so the section is skipped — a
        missing intent.md degrades gracefully (the manifest description
        already frames the artifact).
        """
        if not self._artifact_intent_md.strip():
            return ""
        return "### User's Original Intent (`analysis/intent.md`)\n\n" + self._artifact_intent_md.strip()

    def _render_subject_scope_section(self) -> str:
        """Subject-library assets the artifact was grounded in.

        Walks ``analysis/subject_refs.json`` (the code-aggregated dedup
        of every brief's ``uses`` block) and surfaces:

        * per-asset: kind, full subject path, name
        * reverse index: which query slugs reference each asset

        The reverse index is built fresh from every brief rather than
        from the subject_refs file itself because subject_refs only
        carries the dedup'd asset list, not the back-pointers. This
        lets the LLM answer "which queries use metric X?" without any
        file reads.

        Returns "" when subject_refs is missing/empty so a report that
        didn't draw on the subject library doesn't get a stub heading.
        """
        raw = self._read_artifact_file("analysis/subject_refs.json")
        if not raw:
            return ""
        try:
            refs = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Subject scope render: subject_refs.json unreadable: %s", exc)
            return ""
        if not isinstance(refs, dict):
            return ""

        # Reverse index: (kind, tuple(path), name) -> sorted list of
        # referencing query slugs. The path is part of the key because
        # ``get_metrics(path, name)`` / ``get_reference_sql(path, name)``
        # both take (path, name) and two different assets can legitimately
        # share a leaf ``name`` under different folders (e.g.
        # ``Commerce/Orders/aov`` vs ``Finance/Reporting/aov``). Keying
        # on name alone would conflate them and the "used by" list would
        # falsely show queries from the wrong asset.
        usage_by_asset: Dict[Tuple[str, Tuple[str, ...], str], List[str]] = {}
        for slug in self._query_slugs():
            brief_raw = self._read_artifact_file(f"queries/{slug}.brief.json")
            if not brief_raw:
                continue
            try:
                brief = json.loads(brief_raw)
            except json.JSONDecodeError:
                continue
            uses = brief.get("uses") if isinstance(brief, dict) else None
            if not isinstance(uses, dict):
                continue
            for kind_key in ("metrics", "reference_sql", "ext_knowledge"):
                for entry in uses.get(kind_key) or []:
                    if not isinstance(entry, dict):
                        continue
                    name = entry.get("name")
                    raw_path = entry.get("path")
                    if not isinstance(name, str) or not name:
                        continue
                    if not isinstance(raw_path, list) or not all(isinstance(p, str) for p in raw_path):
                        # Drop malformed brief entries entirely rather
                        # than fall back to a path-less key — letting
                        # one bad entry coalesce with valid ones would
                        # silently re-introduce the conflation we're
                        # fixing.
                        continue
                    usage_by_asset.setdefault((kind_key, tuple(raw_path), name), []).append(slug)

        body_lines: List[str] = []
        for kind_key, label in (
            ("metrics", "metric"),
            ("reference_sql", "sql"),
            ("ext_knowledge", "knowledge"),
        ):
            entries = refs.get(kind_key) or []
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name") or ""
                if not isinstance(name, str) or not name:
                    continue
                raw_path = entry.get("path") or []
                if isinstance(raw_path, list) and all(isinstance(p, str) for p in raw_path):
                    path_tuple = tuple(raw_path)
                    path_str = " > ".join(raw_path) if raw_path else "(no path)"
                else:
                    path_tuple = ()
                    path_str = "(no path)"
                used_by = sorted(set(usage_by_asset.get((kind_key, path_tuple, name), [])))
                line = f"- **{label}** `{path_str} > {name}`"
                if used_by:
                    line += f"\n  · used by: {', '.join(used_by)}"
                body_lines.append(line)

        if not body_lines:
            return ""

        header = [
            "### Subject Library Scope (`analysis/subject_refs.json`)",
            "",
            (
                "The artifact was grounded in the following subject-library "
                "assets. To fetch a canonical definition, call "
                "`get_metrics(path, name)` / `get_reference_sql(path, name)` "
                "/ `get_ext_knowledge(path, name)`:"
            ),
            "",
        ]
        return "\n".join(header + body_lines)

    def _render_table_schemas_section(self) -> str:
        """Inline ``analysis/key_tables_schema.json`` — snapshot of
        ``describe_table`` output for every ``manifest.key_tables`` entry.

        The snapshot lets the LLM plan follow-up SQL on the listed
        tables without paying ``describe_table`` round-trips. Critical
        prompt design choice: the intro **explicitly carves out** the
        cases where the LLM MUST still call ``describe_table`` — live
        schema drift (user asking about "current" / "latest" state),
        column names not in this list (typos / post-finalize
        additions), and tables outside ``manifest.key_tables``. Without
        the carve-out the LLM would treat the snapshot as authoritative
        forever and answer stale-schema questions confidently.

        Returns "" when the file is missing or empty so reports
        without a baked schema (older artifacts, dry runs) skip the
        section silently.
        """
        raw = self._read_artifact_file("analysis/key_tables_schema.json")
        if not raw:
            return ""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Table schemas render: key_tables_schema.json unreadable: %s", exc)
            return ""
        if not isinstance(data, dict):
            return ""
        tables = data.get("tables")
        if not isinstance(tables, list) or not tables:
            return ""

        intro = (
            "Columns of every table in `manifest.key_tables` at the time "
            "the artifact was finalized. **This is a SNAPSHOT — call "
            "`describe_table('<table>')` instead when the user explicitly "
            "asks about the LATEST / CURRENT schema, when they reference "
            "a column NOT in this list (could be a typo or a post-"
            "finalize addition), or when they ask about tables NOT in "
            "`manifest.key_tables`.** For everything else (writing "
            "follow-up SQL on the listed tables, explaining a column, "
            "planning a join between listed tables), use this snapshot "
            "directly — no `describe_table` round-trip needed."
        )

        lines: List[str] = ["### Table Schemas (`analysis/key_tables_schema.json`)", "", intro, ""]
        # Measure in UTF-8 bytes (not code points) since the cap is
        # phrased in bytes; a Chinese-heavy description (each CJK
        # codepoint ≈ 3 UTF-8 bytes) would otherwise silently fit
        # ~3× more content than INLINE_SCHEMA_BYTES_CAP intends. The
        # ``+1`` per line accounts for the joining newline appended at
        # render time by ``"\n".join(...)``.
        running_bytes = sum(len(line.encode("utf-8")) + 1 for line in lines)
        cap_reached = False
        for tbl in tables:
            if not isinstance(tbl, dict):
                continue
            entry_lines = self._render_table_schema_entry(tbl)
            entry_bytes = sum(len(line.encode("utf-8")) + 1 for line in entry_lines)
            if running_bytes + entry_bytes > INLINE_SCHEMA_BYTES_CAP:
                cap_reached = True
                break
            lines.extend(entry_lines)
            lines.append("")
            running_bytes += entry_bytes
        if cap_reached:
            lines.append(
                "_(schema section cap reached — remaining tables omitted; "
                "call `describe_table('<table>')` for any name in "
                "`manifest.key_tables` not shown above.)_"
            )
        return "\n".join(lines).rstrip()

    def _render_table_schema_entry(self, tbl: Dict[str, Any]) -> List[str]:
        """Render one table block: header + optional description + columns.

        Per-table ``error`` (populated when ``describe_table`` failed
        at finalize) surfaces as a "schema unavailable" hint with the
        exact remediation (call ``describe_table('<name>')``) so the
        LLM has a clear next step rather than guessing.
        """
        name = tbl.get("name") or "?"
        if not isinstance(name, str):
            name = str(name)
        description = tbl.get("description") or ""
        columns = tbl.get("columns") or []
        error = tbl.get("error")

        lines: List[str] = [f"#### `{name}`"]
        if description:
            lines.append(f"_(description: {description})_")
        if error:
            lines.append(f"_(schema unavailable: {error}; call `describe_table('{name}')` to fetch live schema.)_")
            return lines
        if not isinstance(columns, list):
            return lines

        # Truncate wide tables: keep the first N columns + a marker
        # pointing the LLM at ``describe_table`` for the rest.
        truncated = len(columns) > INLINE_SCHEMA_COLS_PER_TABLE
        cols_to_render = columns[:INLINE_SCHEMA_COLS_PER_TABLE] if truncated else columns
        for c in cols_to_render:
            if not isinstance(c, dict):
                continue
            cname = c.get("name")
            if not isinstance(cname, str) or not cname:
                continue
            ctype = c.get("type") or "?"
            comment = c.get("comment") or ""
            line = f"- `{cname}`: {ctype}"
            if comment:
                line += f"  -- {comment}"
            lines.append(line)
        if truncated:
            remaining = len(columns) - INLINE_SCHEMA_COLS_PER_TABLE
            lines.append(f"- _(... {remaining} more columns; call `describe_table('{name}')` for the full list.)_")
        return lines

    def _render_insights_section(self) -> str:
        """Inline ``analysis/insights.json`` (report-only).

        Confirmed findings are authoritative for the artifact — the
        LLM should be able to cite them by id without any file read.
        Dashboards have no equivalent (their templates have no static
        conclusions), so the section is empty for the dashboard kind
        and gets dropped by the orchestrator.
        """
        if self.ARTIFACT_KIND != "report":
            return ""
        raw = self._read_artifact_file("analysis/insights.json")
        if not raw:
            return ""
        try:
            insights = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Insights render: insights.json unreadable: %s", exc)
            return ""
        if not isinstance(insights, list) or not insights:
            return ""

        lines: List[str] = ["### Confirmed Findings (`analysis/insights.json`)", ""]
        for idx, insight in enumerate(insights, start=1):
            if not isinstance(insight, dict):
                continue
            iid = insight.get("id") or "?"
            title = insight.get("title") or "(no title)"
            confidence = insight.get("confidence")
            conf_str = f" _(conf {confidence:.2f})_" if isinstance(confidence, (int, float)) else ""
            summary = insight.get("summary") or ""
            evidence = insight.get("evidence_queries") or []
            lines.append(f"{idx}. **`{iid}`** — {title}{conf_str}")
            if summary:
                lines.append(f"   {summary}")
            if isinstance(evidence, list) and evidence:
                ev_strs = [f"`{e}`" for e in evidence if isinstance(e, str)]
                if ev_strs:
                    lines.append(f"   · evidence: {', '.join(ev_strs)}")
        return "\n".join(lines)

    def _render_query_catalog_section(self) -> str:
        """Per-query entries: brief + columns + sample/rows + SQL.

        Walks every query slug in deterministic order and produces a
        compact catalog entry. Each entry's size is tracked against
        :data:`INLINE_CATALOG_BYTES_CAP`; once the running total would
        exceed the cap, remaining entries are rendered in degraded
        mode (drop full rows, tighter SQL truncation, drop caveats)
        so the LLM still knows what data exists for every query, just
        with less detail for the long tail.

        Row inlining for report queries uses TWO thresholds at once:
        :data:`INLINE_ROW_LIMIT` (count) and :data:`INLINE_ROW_BYTES_LIMIT`
        (serialized byte size). Either being exceeded triggers degraded
        sampling (first 2 rows only). The byte gate matters when a
        20-row result carries a multi-KB text column per row — row
        count alone would silently inflate the prompt.
        """
        slugs = self._query_slugs()
        if not slugs:
            return ""

        if self.ARTIFACT_KIND == "report":
            data_label = "result snapshot"
            data_suffix = ".json"
            sql_suffix = ".sql"
            sql_word = "SQL"
        else:
            data_label = "params declaration"
            data_suffix = ".params.json"
            sql_suffix = ".sql.j2"
            sql_word = "SQL template"

        intro = (
            f"Each entry below summarizes one `queries/<slug>` triple "
            f"(brief + {data_label} + {sql_word}). When a query's full rows "
            f"or SQL body would blow the inline budget, only a small sample "
            f"is shown and the full payload remains read-only behind "
            f"`read_file('queries/<slug>{data_suffix}')` / "
            f"`read_file('queries/<slug>{sql_suffix}')`."
        )

        header: List[str] = ["### Query Catalog", "", intro, ""]

        body: List[str] = []
        # Cap is phrased in bytes (UTF-8) and the cap covers the whole
        # section, not just the body — count the header upfront so a
        # large intro doesn't silently leak past the cap, and use
        # ``.encode("utf-8")`` length on every line so non-ASCII content
        # (Chinese caveats, emoji in SQL comments, etc.) is sized
        # honestly.
        running_bytes = sum(len(line.encode("utf-8")) + 1 for line in header)
        degraded = False
        for slug in slugs:
            brief, data, sql = self._load_query_bundle(slug)
            entry_lines = self._render_query_catalog_entry(slug, brief, data, sql, degraded=degraded)
            entry_bytes = sum(len(line.encode("utf-8")) + 1 for line in entry_lines)
            if not degraded and running_bytes + entry_bytes > INLINE_CATALOG_BYTES_CAP:
                degraded = True
                entry_lines = self._render_query_catalog_entry(slug, brief, data, sql, degraded=True)
                entry_bytes = sum(len(line.encode("utf-8")) + 1 for line in entry_lines)
            body.extend(entry_lines)
            body.append("")
            running_bytes += entry_bytes

        return "\n".join(header + body).rstrip()

    def _render_query_catalog_entry(
        self,
        slug: str,
        brief: Dict[str, Any],
        data: Optional[Dict[str, Any]],
        sql: Optional[str],
        *,
        degraded: bool,
    ) -> List[str]:
        """Render one catalog entry. Caller stitches entries together.

        Degraded mode skips full-row inlining (always sample) and
        caveats, and tightens the SQL line cap. The header (slug + row
        count + hypothesis + columns) always survives so the LLM keeps
        a usable index of every query even when the catalog cap fires.
        """
        lines: List[str] = []

        # Header: slug + size descriptor.
        if self.ARTIFACT_KIND == "report" and isinstance(data, dict):
            rows = data.get("rows") or []
            row_count = data.get("row_count", len(rows))
            size_desc = f"{row_count} rows"
        elif self.ARTIFACT_KIND == "dashboard" and isinstance(data, dict):
            sample_row_count = data.get("sample_row_count", 0)
            size_desc = f"template · sample {sample_row_count} rows"
        else:
            size_desc = "no data file"
        lines.append(f"#### `{slug}` — {size_desc}")

        hypothesis = brief.get("hypothesis") or ""
        if hypothesis:
            lines.append(f"- **hypothesis**: {hypothesis}")

        # Subjects: short labels only — full paths live in the global
        # Subject Library Scope section so we don't repeat the path
        # once per query.
        uses = brief.get("uses") or {}
        subject_parts: List[str] = []
        if isinstance(uses, dict):
            for kind_key, label in (
                ("metrics", "metric"),
                ("reference_sql", "sql"),
                ("ext_knowledge", "knowledge"),
            ):
                for entry in uses.get(kind_key) or []:
                    if isinstance(entry, dict):
                        name = entry.get("name")
                        if isinstance(name, str) and name:
                            subject_parts.append(f"{label}:`{name}`")
        if subject_parts:
            lines.append(f"- **subjects**: {', '.join(subject_parts)}")

        caveats = brief.get("caveats") or ""
        if caveats and not degraded:
            lines.append(f"- **caveats**: {caveats}")

        # Columns block — always rendered when available since it's
        # cheap and key to "what data does this query produce".
        if isinstance(data, dict):
            cols = data.get("columns") or []
            col_strs: List[str] = []
            for c in cols:
                if isinstance(c, dict):
                    cname = c.get("name", "?")
                    ctype = c.get("type", "?")
                    col_strs.append(f"{cname}:{ctype}")
            if col_strs:
                lines.append(f"- **columns**: {', '.join(col_strs)}")

        # Rows (report) / sample params (dashboard).
        if self.ARTIFACT_KIND == "report" and isinstance(data, dict):
            rows = data.get("rows") or []
            row_count = data.get("row_count", len(rows))
            # Double-gate the inline: count AND serialized byte size.
            full_inline = False
            if rows and not degraded and row_count <= INLINE_ROW_LIMIT:
                payload = json.dumps(rows, ensure_ascii=False, default=str)
                if len(payload.encode("utf-8")) <= INLINE_ROW_BYTES_LIMIT:
                    full_inline = True
            if rows:
                if full_inline:
                    lines.append(f"- **rows** ({row_count}):")
                    for row in rows:
                        lines.append(f"    - {_compact_row(row)}")
                else:
                    show = min(2, len(rows))
                    lines.append(
                        f"- **sample** (first {show} of {row_count}; full data via `read_file('queries/{slug}.json')`):"
                    )
                    for row in rows[:show]:
                        lines.append(f"    - {_compact_row(row)}")
        elif self.ARTIFACT_KIND == "dashboard" and isinstance(data, dict):
            sample_params = data.get("sample_params") or {}
            if sample_params:
                lines.append("- **sample_params**: " + json.dumps(sample_params, ensure_ascii=False, default=str))

        # SQL body, truncated when long. Degraded mode tightens the cap
        # so the long tail of queries stays compact under the catalog
        # cap.
        if sql:
            lines.append("- **SQL**:")
            lines.append("  ```sql")
            sql_lines = sql.strip().splitlines()
            max_lines = (INLINE_SQL_LINE_LIMIT // 2) if degraded else INLINE_SQL_LINE_LIMIT
            if len(sql_lines) > max_lines:
                for line in sql_lines[:max_lines]:
                    lines.append(f"  {line}")
                remaining = len(sql_lines) - max_lines
                full_sql_suffix = ".sql" if self.ARTIFACT_KIND == "report" else ".sql.j2"
                lines.append(f"  -- ... ({remaining} more lines; read queries/{slug}{full_sql_suffix} for full body)")
            else:
                for line in sql_lines:
                    lines.append(f"  {line}")
            lines.append("  ```")

        return lines

    def _artifact_has_insights(self) -> bool:
        """Same emptiness check ``_render_insights_section`` uses.

        The layout-tree and ``loaded_list`` are claims to the LLM that
        a file is already in the prompt. If we unconditionally said so
        for every report — even those whose finalize LLM failed and
        never wrote ``insights.json`` — the LLM would believe the file
        is loaded and skip a legitimate ``read_file``. Mirror the
        renderer's exact gate so the tree only advertises insights
        when the section actually rendered them.
        """
        if self.ARTIFACT_KIND != "report":
            return False
        raw = self._read_artifact_file("analysis/insights.json")
        if not raw:
            return False
        try:
            insights = json.loads(raw)
        except json.JSONDecodeError:
            return False
        return isinstance(insights, list) and bool(insights)

    def _render_filesystem_layout_section(self) -> str:
        """Tell the LLM what's already inlined vs what still needs read_file.

        The directory tree mirrors the on-disk artifact layout but
        annotates each entry with "inlined above" vs "DO NOT read" vs
        "read on demand" so a model trying to be helpful by pre-fetching
        files immediately sees that defensive reads are not useful.
        """
        layout_root_label = self._artifact_root.name if self._artifact_root is not None else self.ARTIFACT_ROOT_DIR_NAME
        has_insights = self._artifact_has_insights()
        loaded_list = (
            "manifest.json, analysis/intent.md, "
            + ("analysis/insights.json, " if has_insights else "")
            + "analysis/subject_refs.json (when present), "
            "analysis/key_tables_schema.json (when present), and every "
            "`queries/*.brief.json` plus the inlined slice of "
            "`queries/*` data + SQL above"
        )
        lines: List[str] = [
            "### Artifact Filesystem Layout",
            "",
            (
                f"Filesystem tool anchored at the artifact root (relative paths "
                f"resolve under `{layout_root_label}/`). **The following are "
                f"already loaded into this prompt: {loaded_list}.** "
                f"**Do NOT `read_file` / `glob` them defensively.** Read on "
                f"demand only when the catalog above flags a query's data as "
                f"sampled or its SQL as truncated."
            ),
            "",
            "```",
            ".",
            "├── manifest.json                # inlined above",
            "├── analysis/",
            "│   ├── intent.md                # inlined above",
        ]
        # Only advertise insights when there's actually a populated
        # insights.json on disk / in the blob — see
        # :meth:`_artifact_has_insights`. Dashboards skip the line
        # unconditionally (no insights file by design).
        if has_insights:
            lines.append("│   ├── insights.json            # inlined above")
        lines.extend(
            [
                "│   ├── subject_refs.json        # inlined above (if present)",
                "│   ├── key_tables_schema.json   # inlined above (if present); snapshot only — describe_table() for live",
                "│   └── suggested_questions.json # UI chips — DO NOT read",
                "├── queries/                     # briefs always inlined; data + SQL inlined or sampled per catalog above",
                "└── render/                     # presentation tier — DO NOT READ",
                "```",
            ]
        )
        return "\n".join(lines)

    def _render_behavioral_rules_section(self) -> str:
        """Load-bearing rules that define the ask agent's role.

        Rule 1 is the load-bearing change in this rewrite: it forbids
        defensive pre-fetching of files that are already inlined above.
        Without this the LLM tends to ``glob`` + ``read_file`` every
        sidecar at turn start out of habit, paying many serial tool
        round-trips for content the prompt already carries.
        """
        # Same presence check the layout-tree uses (see
        # ``_artifact_has_insights``). Drives both rule 1's preamble
        # (which lists what was actually inlined) and rule 6's
        # report-only branch — without this, a report whose finalize
        # produced no insights would still have rule 6 claim
        # "insights.json is the authoritative findings record" and
        # rule 1 claim "confirmed insights" are inlined, both of
        # which would mislead the LLM.
        has_insights = self._artifact_has_insights()
        lines: List[str] = ["### Behavioral Rules (load-bearing)", ""]
        lines.append(
            "1. **Answer from the inlined context first**. The header above "
            "already contains the manifest, the original intent, the "
            "subject library scope, the table schemas snapshot, "
            + ("confirmed insights, " if has_insights else "")
            + "and a query catalog with hypothesis / caveats / columns / "
            "sample rows / SQL for every saved query. **Do NOT issue "
            "`glob` or `read_file` to pre-fetch anything already inlined "
            "above.** Re-read or re-fetch only when (a) the catalog "
            "explicitly flagged a query's data as sampled or its SQL as "
            "truncated, (b) the user asks about LIVE / CURRENT state "
            "the snapshot can't answer — fresh schema after a DDL "
            "change, live row counts, today's data (call "
            "`describe_table` / `execute_sql` for those), or (c) the "
            "inlined summary genuinely doesn't address the question. "
            "When you do, briefly say which file or tool and why."
        )
        lines.append(
            "2. **Do NOT regenerate the artifact**. You are read-only. If "
            "the user asks to add a chart, edit a panel, or rewrite the "
            f"{self.ARTIFACT_KIND}, direct them to the "
            f"`gen_visual_{self.ARTIFACT_KIND}` subagent."
        )
        lines.append(
            "3. **Cite by slug**. Refer to queries as ``queries/<name>`` "
            "and (report only) insights as ``insight:<id>`` so the UI "
            "can highlight / jump to them."
        )
        lines.append(
            "4. **Stay anchored to the original intent**. Flag when the "
            "user's new question genuinely shifts scope from the "
            f"original {self.ARTIFACT_KIND}'s coverage."
        )
        lines.append(
            "5. **Respect the data scope**. The Subject Library Scope "
            "section above (when present) lists the authoritative "
            "subject assets. Exploring outside that scope is OK if the "
            "user explicitly asks, but call it out in your answer."
        )
        if self.ARTIFACT_KIND == "dashboard":
            lines.append(
                "6. **Dashboard queries have no precomputed data**. The "
                "`queries/<slug>.sql.j2` files (inlined above) are "
                "templates; to answer quantitative questions, run an "
                "equivalent ad-hoc SQL via `execute_sql` within the "
                "dashboard's datasource scope, or use the params "
                "declaration to explain what user-controllable filters "
                "exist."
            )
        elif has_insights:
            lines.append(
                "6. **`insights.json` is the authoritative findings "
                "record**. Already inlined above; each insight has "
                "`evidence_queries[]` you can cross-reference."
            )
        else:
            # Report whose finalize LLM crashed (or whose insights list
            # was empty). Emit rule 6 in its insights-absent form so
            # the numbering stays 1–7 across kinds AND the LLM knows
            # not to claim non-existent findings as if it had read
            # them. Without this branch the rule is silently dropped
            # and the LLM sees rules 1..5,7 (jumping over 6) — a small
            # but persistent prompt-quality regression.
            lines.append(
                "6. **No confirmed-findings record for this report**. "
                "Finalize did not produce a populated `insights.json`. "
                "Ground answers in the query catalog above and cite "
                "individual queries by slug; do NOT claim insights "
                "that aren't in the prompt."
            )
        lines.append(
            "7. **No artifact mutations**. Filesystem write/edit/delete "
            "are not available to you and will be rejected — do not "
            "attempt them."
        )
        return "\n".join(lines)
