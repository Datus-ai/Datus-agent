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

1. **Artifact binding** — read ``artifact_slug`` from the agentic_nodes
   config, resolve it under the project root, and fail loud if the
   directory is missing or escapes ``project_root``.
2. **Constrained filesystem view** — override ``_make_filesystem_tool``
   so the LLM's ``read_file`` / ``glob`` / ``grep`` calls are anchored
   at the artifact root. Relative paths in prompts (``analysis/intent.md``,
   ``queries/<name>.json``) just work, and the LLM cannot accidentally
   peek into a sibling artifact or the global subject library through
   filesystem traversal.
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
``insights.json`` is expected) lives in the two concrete subclasses.
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
        self._resolve_artifact_root_early(agent_config)
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

    def _resolve_artifact_root_early(self, agent_config: Optional[AgentConfig]) -> None:
        """Resolve the artifact binding directly from the agentic_nodes entry.

        Called BEFORE ``super().__init__()`` runs, so we can't rely on
        ``self.node_config`` (set by AgenticNode init) or on
        ``self.agent_config`` (set by AgenticNode init). We read the raw
        ``agent_config.agentic_nodes[subagent_name]`` entry directly.

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

        self._artifact_slug = slug
        self._artifact_root = artifact_dir

    # ── Filesystem tool override (cwd → artifact_root) ──────────────────

    def _make_filesystem_tool(self, **kwargs):
        """Anchor the filesystem tool at the artifact root.

        Overrides the base node's default which uses
        ``_resolve_workspace_root()``. By the time ``setup_tools`` calls
        this, ``_resolve_artifact_root()`` has already run during
        ``__init__``, so ``self._artifact_root`` is set.
        """
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
        """
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
        if self._artifact_root is None:
            return ""

        manifest = self._artifact_manifest or {}
        artifact_name = manifest.get("name") or self._artifact_slug
        artifact_description = manifest.get("description") or ""

        lines: list[str] = []
        lines.append(f"## Bound Artifact — {self.ARTIFACT_KIND.title()}: {artifact_name}")
        lines.append("")
        lines.append(f"- **Slug**: `{self._artifact_slug}`")
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
        lines.append(
            f"Your filesystem tools are anchored at the artifact root. "
            f"Relative paths resolve under `{self._artifact_root.name}/`:"
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
