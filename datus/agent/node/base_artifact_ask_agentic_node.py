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
   ``analysis/intent.md`` once at node startup, and surface them to
   the prompt template so the LLM has a baseline grounding without
   paying ``read_file`` tool calls every turn.

The earlier ``interpretation.json`` preload was removed along with the
file itself — ``manifest.description`` covers framing and
``analysis/insights.json`` (read on demand by the LLM) covers the
substantive findings. Likewise, ``suggested_questions.json`` is **not**
preloaded into the prompt: it's surfaced via the detail API as UI
chips, but injecting it here would anchor the LLM toward a fixed
question set whenever the user types an open-ended follow-up.

Per-kind specialization (``ARTIFACT_KIND`` / template name / whether
``insights.json`` is expected / whether ``BLOB_REQUIRED``) lives in the
two concrete subclasses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, Literal, Optional

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.artifact_manifest import ARTIFACT_SLUG_RE
from datus.schemas.chat_agentic_node_models import ChatNodeInput
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


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
        if artifact_header:
            return artifact_header + "\n\n" + base_prompt
        return base_prompt

    def _render_artifact_context_block(self) -> str:
        """Build the artifact-context preamble prepended to the chat prompt.

        Hand-rolls a small markdown block rather than a separate j2
        template because the structure is dead simple, the inputs are
        already in memory, and a 30-line template adds more indirection
        than it saves.
        """
        if self._artifact_files is None and self._artifact_root is None:
            return ""

        manifest = self._artifact_manifest or {}
        artifact_name = manifest.get("name") or self._artifact_slug
        artifact_description = manifest.get("description") or ""

        lines: list[str] = []
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
            # Surface code-aggregated table list so the LLM can answer
            # "which tables does this report touch" / plan a follow-up
            # SQL without first ``list_tables`` / ``describe_table`` round-
            # trips. Code-generated by finalize from the SQL bodies, not
            # an LLM claim — trustworthy as long as it's present.
            lines.append(f"- **Tables referenced**: {', '.join(manifest['key_tables'])}")
        lines.append("")

        if self._artifact_intent_md.strip():
            lines.append("### User's Original Intent (`analysis/intent.md`)")
            lines.append("")
            lines.append(self._artifact_intent_md.strip())
            lines.append("")

        # File-system layout & usage hints — kept brief because the chat
        # template already documents the available tools; we just point
        # the LLM at what's under the bound artifact. We deliberately
        # do NOT list ``analysis/suggested_questions.json`` here — it
        # exists as UI chip data and including it would anchor the LLM
        # toward a fixed question set when the user asks something open.
        # ``analysis/subject_refs.json`` is also omitted from the static
        # tree because it's present-iff-non-empty; the LLM can ``glob``
        # for it if it cares.
        lines.append("### Artifact Filesystem Layout")
        lines.append("")
        layout_root_label = self._artifact_root.name if self._artifact_root is not None else self.ARTIFACT_ROOT_DIR_NAME
        lines.append(
            f"Your filesystem tools are anchored at the artifact root. "
            f"Relative paths resolve under `{layout_root_label}/`:"
        )
        lines.append("")
        lines.append("```")
        lines.append(".")
        lines.append("├── manifest.json")
        lines.append("├── analysis/")
        lines.append("│   ├── intent.md                 # raw user prompts (append-only)")
        if self.ARTIFACT_KIND == "report":
            lines.append("│   └── insights.json             # confirmed findings (report only)")
        lines.append("├── queries/")
        if self.ARTIFACT_KIND == "report":
            lines.append("│   ├── <name>.sql                # SQL text")
            lines.append("│   ├── <name>.json               # query result snapshot")
        else:
            lines.append("│   ├── <name>.sql.j2             # Jinja2 SQL template (params header)")
            lines.append("│   └── <name>.params.json        # declared params + sample columns")
        lines.append("│   └── <name>.brief.json         # hypothesis / uses / caveats")
        lines.append("└── render/                       # presentation tier — DO NOT READ")
        lines.append("```")
        lines.append("")

        # Behavioral rules — these are the load-bearing rules that define
        # the ask agent's role. They sit at the top of the prompt so the
        # LLM internalizes them before reading the chat template's general
        # tool documentation below.
        lines.append("### Behavioral Rules (load-bearing)")
        lines.append("")
        lines.append(
            "1. **Ground in existing analysis first**. Before running new SQL, "
            "try to answer from the anchor context above and from on-disk "
            "files via `read_file` / `glob` / `grep`. Only run new queries "
            "when the existing data genuinely doesn't cover the question — "
            "and when you do, briefly explain why."
        )
        lines.append(
            "2. **Do NOT regenerate the artifact**. You are read-only. If the "
            "user asks to add a chart, edit a panel, or rewrite the report, "
            f"direct them to the `gen_visual_{self.ARTIFACT_KIND}` subagent."
        )
        lines.append(
            "3. **Cite by slug**. Refer to queries as ``queries/<name>`` and "
            "(report only) insights as ``insight:<id>`` so the UI can "
            "highlight / jump to them."
        )
        lines.append(
            "4. **Stay anchored to the original intent**. Re-read the user "
            "prompts in `analysis/intent.md` before answering complex "
            "follow-ups; flag when the user's new question genuinely "
            "shifts scope from the original artifact's coverage."
        )
        lines.append(
            "5. **Respect the data scope**. If `analysis/subject_refs.json` "
            "exists (it's present iff at least one query declared a "
            "subject-library asset), it lists what the artifact originally "
            "drew on. Exploring outside that scope is OK if the user "
            "explicitly asks, but call it out in your answer."
        )
        if self.ARTIFACT_KIND == "dashboard":
            lines.append(
                "6. **Dashboard queries have no precomputed data**. The "
                "`queries/<name>.sql.j2` files are templates; to answer "
                "quantitative questions, run an equivalent ad-hoc SQL via "
                "`execute_sql` within the dashboard's datasource scope, or "
                "use the params declaration in `<name>.params.json` to "
                "explain what user-controllable filters exist."
            )
        else:
            lines.append(
                "6. **`insights.json` is the authoritative findings record**. "
                "Read it when the user's question touches on confirmed "
                "conclusions; each insight has `evidence_queries[]` that "
                "you can cross-reference."
            )
        lines.append(
            "7. **No artifact mutations**. Filesystem write/edit/delete are "
            "not available to you and will be rejected — do not attempt them."
        )

        return "\n".join(lines)
