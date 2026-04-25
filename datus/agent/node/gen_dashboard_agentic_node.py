# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""GenDashboardAgenticNode — BI dashboard / chart / dataset subagent.

Most plumbing (session management, validation retry loop, prompt rendering)
lives in :class:`DeliverableAgenticNode`. This subclass only declares the
BI-specific pieces: platform resolution, BI tool registration, dashboard-
workflow skill injection, and the domain result shape.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Literal, Optional

from datus.agent.node.deliverable_node import DeliverableAgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.gen_dashboard_agentic_node_models import GenDashboardNodeResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenDashboardAgenticNode(DeliverableAgenticNode):
    """BI dashboard / chart / dataset subagent.

    Registers a :class:`BIFuncTool` against the configured BI platform
    (Superset or Grafana) and injects the platform-specific dashboard skill via
    ``skills_to_inject``. Validator skills such as ``bi-validation`` are
    resolved by the validator registry and run through :class:`ValidationHook`.
    """

    NODE_NAME: ClassVar[str] = "gen_dashboard"
    NODE_TYPE: ClassVar[str] = NodeType.TYPE_GEN_DASHBOARD
    ACTION_TYPE: ClassVar[str] = "gen_dashboard_response"
    PROMPT_TEMPLATE: ClassVar[str] = "gen_dashboard_system"
    DEFAULT_MAX_TURNS: ClassVar[int] = 30
    # Dashboard skills are injected dynamically by ``_setup_dashboard_skills``
    # based on the configured BI platform — leave ``DEFAULT_SKILLS`` empty so
    # the base-class fallback does not over-expose unrelated platforms.
    DEFAULT_SKILLS: ClassVar[Optional[str]] = None
    # gen_dashboard only builds BI assets on top of tables that already exist
    # in the BI serving DB. Data preparation belongs to gen_job / scheduler.
    DEFAULT_SUBAGENTS: ClassVar[str] = ""

    def __init__(
        self,
        agent_config: AgentConfig,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
        scope: Optional[str] = None,
        is_subagent: bool = False,
    ):
        # Populated by ``_setup_domain_tools`` on success / ``_bi_setup_error``
        # on failure. Initialized here so the attributes exist when the base
        # ``setup_tools`` chain runs them during ``super().__init__``.
        self.bi_func_tool = None
        self._bi_setup_error: Optional[str] = None
        super().__init__(
            agent_config=agent_config,
            execution_mode=execution_mode,
            node_id=node_id,
            node_name=node_name,
            scope=scope,
            is_subagent=is_subagent,
        )

    # ── tool setup ────────────────────────────────────────────────────

    def setup_tools(self):
        """Setup dashboard tools and platform-specific skills.

        - **BIFuncTool** for read/write on the BI platform itself.
        - Platform dashboard skill for the configured BI platform.

        Filesystem / date tools are intentionally skipped — gen_dashboard
        doesn't write to the local FS. Subagent task tools are also skipped:
        data movement must be completed before this node is invoked.
        """
        if not self.agent_config:
            return
        self.tools = []
        self._setup_domain_tools()
        self._setup_dashboard_skills()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()
        logger.debug("Setup %d tools for %s", len(self.tools), self.NODE_NAME)

    def _setup_domain_tools(self) -> None:
        """Resolve the BI platform and register :class:`BIFuncTool`."""
        bi_platform = self._resolve_bi_platform()
        if not bi_platform:
            return
        try:
            from datus.tools.func_tool.bi_tools import BIFuncTool

            bi_func_tool = BIFuncTool(self.agent_config, bi_service=bi_platform)
            # Trigger adapter build + capability sniff before publishing the
            # instance so that a failing ``datus_bi_core`` import leaves
            # ``self.bi_func_tool`` as ``None``.
            tools = bi_func_tool.available_tools()
        except ImportError as e:
            logger.warning("BI adapter package not installed: %s", e)
            self._bi_setup_error = f"BI adapter package for '{bi_platform}' is not installed: {self._sanitize_error(e)}"
            return
        except Exception as e:
            logger.error("Failed to setup BI tools for '%s': %s", bi_platform, e, exc_info=True)
            self._bi_setup_error = f"Failed to initialize BI platform '{bi_platform}': {self._sanitize_error(e)}"
            return

        self.bi_func_tool = bi_func_tool
        self.tools.extend(tools)
        logger.info("BI tools initialized for platform '%s'", bi_platform)

    def _setup_dashboard_skills(self):
        """Inject only the platform-specific dashboard workflow skill.

        Only fires when BI tools are available — otherwise the skills would
        reference missing tools like ``create_dashboard`` / ``get_chart``.
        Validation skills are handled by :class:`ValidationHook`.
        """
        if not self.bi_func_tool:
            return
        bi_platform = self._resolve_bi_platform()
        if not bi_platform:
            return

        skills_to_inject = []
        tool_names = {tool.name for tool in self.bi_func_tool.available_tools()}
        has_write_tools = "create_dashboard" in tool_names or "create_chart" in tool_names
        if has_write_tools:
            skills_to_inject.append(f"{bi_platform}-dashboard")

        self.node_config["skills"] = self._merge_skill_patterns(
            self.node_config.get("skills"),
            skills_to_inject,
        )
        self._setup_skill_func_tools()

    def _resolve_bi_platform(self) -> Optional[str]:
        """Resolve which BI platform (``superset`` / ``grafana``) to target.

        Preference order:
        1. ``node_config["bi_platform"]`` — explicit override.
        2. Auto-detect when exactly one platform is declared in
           ``agent_config.dashboard_config``.
        """
        node_config = self.node_config or {}
        bi_platform = node_config.get("bi_platform")
        if bi_platform:
            return bi_platform

        dashboard_config = getattr(self.agent_config, "dashboard_config", {})
        if len(dashboard_config) == 1:
            platform = next(iter(dashboard_config))
            logger.debug("Auto-detected bi_platform '%s' from dashboard_config", platform)
            return platform

        if len(dashboard_config) > 1:
            logger.warning(
                "Multiple BI platforms configured (%s); set 'bi_platform' explicitly "
                "in gen_dashboard agentic_nodes config",
                list(dashboard_config.keys()),
            )
        return None

    @staticmethod
    def _sanitize_error(exc: BaseException) -> str:
        """Redact potential credentials before surfacing an error to the LLM.

        BI adapter failures frequently embed raw URLs or auth payloads
        (``401 Unauthorized at https://user:pass@host/...``). The system prompt
        surfaces ``_bi_setup_error`` to the model, so strip obvious
        user:pass@ segments and cap the length. Full error stays in logs.
        """
        import re

        msg = str(exc)
        # ``authorization`` is intentionally excluded from the dict/key=value
        # patterns so the dedicated Bearer/Basic regex below can capture the
        # full token including the scheme keyword. Otherwise the dict-style
        # regex eats ``"Bearer"`` / ``"Basic"`` and leaves the secret bare.
        sensitive_key = r"(password|passwd|token|api[_-]?key|secret)"
        # user:pass@host URL segments.
        msg = re.sub(r"://[^/\s@]+@", "://<redacted>@", msg)
        # HTTP ``Authorization: Bearer …`` / ``Basic …`` headers (first so the
        # dict-style regex below can't clobber the scheme keyword).
        msg = re.sub(
            r"(?i)\b(authorization\s*[:=]\s*)(bearer|basic)\s+\S+",
            r"\1\2 <redacted>",
            msg,
        )
        # key=value pairs (query strings, env-style dumps).
        msg = re.sub(
            rf"(?i)\b{sensitive_key}\s*=\s*[^\s,;&]+",
            lambda m: f"{m.group(1)}=<redacted>",
            msg,
        )
        # JSON / dict-style ``"password": "secret"``.
        colon_pattern = rf"""(?ix)(["']?{sensitive_key}["']?\s*:\s*)["']?[^"',}}\s]+["']?"""
        msg = re.sub(colon_pattern, lambda m: f"{m.group(1)}<redacted>", msg)
        if len(msg) > 300:
            msg = msg[:300] + "..."
        return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__

    def _tool_category_map(self) -> Dict[str, List[Any]]:
        """Declare categories so ``bi_tools.delete_*`` DENY rules fire.

        Without this mapping ``PermissionHooks`` falls back to the catch-all
        ``tools`` category for every BI tool, so no ``bi_tools.*`` rule can
        ever match — every profile (even ``normal``) silently lets
        ``delete_chart`` / ``delete_dataset`` through.
        """
        mapping = super()._tool_category_map()
        if self.bi_func_tool:
            mapping["bi_tools"] = list(self.bi_func_tool.available_tools())
        if self.ask_user_tool:
            mapping.setdefault("tools", []).extend(self.ask_user_tool.available_tools())
        return mapping

    # ── template context ──────────────────────────────────────────────

    def _prepare_template_context(self, user_input: Any = None) -> dict:
        """Render BI capability flags into the system-prompt template."""
        context: dict = {
            "native_tools": ", ".join([tool.name for tool in self.tools]) if self.tools else "None",
            "has_ask_user_tool": self.ask_user_tool is not None,
            "has_bi_tools": self.bi_func_tool is not None,
            "bi_setup_error": self._bi_setup_error,
        }
        if self.bi_func_tool:
            tool_names = {tool.name for tool in self.bi_func_tool.available_tools()}
            context["has_dashboard_write"] = "create_dashboard" in tool_names
            context["has_chart_write"] = "create_chart" in tool_names
            context["has_dataset_write"] = "create_dataset" in tool_names
            context["has_serving_db"] = self.bi_func_tool.serving_dataset_db is not None
        else:
            context["has_dashboard_write"] = False
            context["has_chart_write"] = False
            context["has_dataset_write"] = False
            context["has_serving_db"] = False
        context["bi_platform"] = self._resolve_bi_platform() or "unknown"
        return context

    # ── result construction ────────────────────────────────────────────

    def _make_success_result(
        self,
        *,
        success: bool,
        response_content: Any,
        tokens_used: int,
        validation_report: Optional[dict],
        last_successful_output: Optional[dict],
        blocked: bool,
    ) -> GenDashboardNodeResult:
        structured_result = None
        if last_successful_output and isinstance(last_successful_output, dict):
            result_data = last_successful_output.get("result")
            if isinstance(result_data, dict):
                structured_result = result_data

        # ``response_content`` may arrive as a dict (raw_output) — coerce to
        # string for the user-facing response field.
        response_str = ""
        if response_content:
            response_str = response_content if isinstance(response_content, str) else str(response_content)
        elif blocked:
            response_str = "Validation failed"

        return GenDashboardNodeResult(
            success=success,
            response=response_str,
            dashboard_result=structured_result,
            tokens_used=tokens_used,
            error=None if success else ("Validation blocked the run" if blocked else None),
            validation_report=validation_report,
        )

    def _make_error_result(
        self,
        *,
        error: str,
        action_history_manager: ActionHistoryManager,
    ) -> GenDashboardNodeResult:
        partial = self._collect_created_resources(action_history_manager)
        return GenDashboardNodeResult(
            success=False,
            error=error,
            response="Sorry, I encountered an error while processing your request.",
            dashboard_result=partial if partial else None,
            tokens_used=0,
        )

    # ── partial-resource collection (used on error) ───────────────────

    @staticmethod
    def _collect_created_resources(action_history_manager: ActionHistoryManager) -> dict:
        """Scan action history for resources created before a failure.

        Returns a dict summarising dashboards, charts, and datasets that were
        already created, so that a retry can clean up or reuse them.
        """
        created: dict = {"dashboards": [], "charts": [], "datasets": []}
        resource_tools = {
            "create_dashboard": "dashboards",
            "create_chart": "charts",
            "create_dataset": "datasets",
        }

        for action in action_history_manager.get_actions():
            if action.status != ActionStatus.SUCCESS:
                continue
            if action.action_type not in resource_tools:
                continue

            output = action.output
            if not isinstance(output, dict):
                continue

            result = output.get("result")
            if not isinstance(result, dict):
                continue

            resource_id = result.get("id")
            resource_name = result.get("name", "")
            if resource_id is not None:
                category = resource_tools[action.action_type]
                created[category].append({"id": resource_id, "name": resource_name})

        # Strip empty categories
        return {k: v for k, v in created.items() if v}
