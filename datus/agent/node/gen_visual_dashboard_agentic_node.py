# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenVisualDashboardAgenticNode — parameterized dashboard generation.

Companion to ``GenVisualReportAgenticNode``. Instead of pre-baked JSON
result files, this node produces a parameterized React-JSX dashboard
artifact under ``<project_root>/dashboards/<slug>/``:

* ``render/app.jsx`` — the React entry module the LLM authors (default
  export); it owns the filter state and imports the chart components.
* ``queries/<slug>.sql.j2`` + ``queries/<slug>.params.json`` — per-query
  Jinja2 SQL template plus its declared parameter metadata.
* ``manifest.json`` — ``{slug, name, description, kind, created_at}``.

At view time the query backend renders the template with user-selected
filter values and executes it live against the bound datasource — the
agent ``--web`` server exposes ``POST /api/v1/dashboard/query`` for
that, and the CLI HTML compile path below produces a
``dashboards/<slug>/index.html`` that drives the dashboard renderer
against the local agent server (or any configured backend).

Common machinery lives in :class:`BaseVisualArtifactAgenticNode`; this
file owns the dashboard-specific artifact wiring, result model, and the
CLI HTML compile path that needs a live query backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from datus.agent.node.base_visual_artifact_agentic_node import BaseVisualArtifactAgenticNode
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.schemas.gen_visual_dashboard_models import (
    GenVisualDashboardNodeInput,
    GenVisualDashboardNodeResult,
)
from datus.tools.func_tool.dashboard_artifact_tools import (
    DashboardArtifactTools,
    DashboardFilesystemFuncTool,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenVisualDashboardAgenticNode(
    BaseVisualArtifactAgenticNode[GenVisualDashboardNodeInput, GenVisualDashboardNodeResult]
):
    """
    Visual dashboard subagent.

    Sets up semantic / db / context-search tools plus the dashboard-specific
    ``DashboardArtifactTools`` (save_query_template / validate_render) and a
    hardened ``DashboardFilesystemFuncTool`` that denies direct writes to
    dashboard artifact paths.

    The LLM chooses the ``dashboard_slug`` on every fresh
    ``start_new_dashboard`` call; the system prompt directs it to
    ``glob('dashboards/*')`` first so the chosen slug doesn't collide with
    an existing one.
    """

    NODE_NAME = "gen_visual_dashboard"
    result_class = GenVisualDashboardNodeResult
    ARTIFACT_KIND = "dashboard"
    ARTIFACT_ROOT_DIR_NAME = "dashboards"
    FILESYSTEM_TOOL_CLS = DashboardFilesystemFuncTool
    QUERY_SAVE_ACTION_TYPE = "save_query_template"
    FALLBACK_TEMPLATE_NAME = "gen_visual_dashboard_system"

    def get_node_name(self) -> str:
        return self.configured_node_name or self.NODE_NAME

    # ────────── Convenience accessors ──────────

    @property
    def _active_dashboard_slug(self) -> Optional[str]:
        return self._active_artifact_slug

    @_active_dashboard_slug.setter
    def _active_dashboard_slug(self, value: Optional[str]) -> None:
        self._active_artifact_slug = value

    @property
    def dashboard_artifact_tools(self) -> Optional[DashboardArtifactTools]:
        return self.artifact_tools  # type: ignore[return-value]

    @dashboard_artifact_tools.setter
    def dashboard_artifact_tools(self, value: Optional[DashboardArtifactTools]) -> None:
        self.artifact_tools = value

    # ────────── Hooks the base class calls ──────────

    def _make_artifact_tools(self, user_input: GenVisualDashboardNodeInput) -> DashboardArtifactTools:
        return DashboardArtifactTools(
            agent_config=self.agent_config,
            db_func_tool=self.db_func_tool,
            user_message=getattr(user_input, "user_message", "") or "",
        )

    def _read_artifact_slug_from_tools(self) -> Optional[str]:
        tools = self.artifact_tools
        if tools is None:
            return None
        return getattr(tools, "dashboard_slug", None)

    def _finalize_artifact_success(
        self,
        *,
        user_input: GenVisualDashboardNodeInput,
        response_content: str,
        artifact_slug: Optional[str],
        app_jsx_rel_path: Optional[str],
        render_file_count: int,
        query_actions: List[ActionHistory],
        tokens_used: int,
        all_actions: List[ActionHistory],
        tool_calls: List[ActionHistory],
    ) -> GenVisualDashboardNodeResult:
        manifest = self._read_artifact_manifest(artifact_slug)
        mode = getattr(self.artifact_tools, "mode", None) if self.artifact_tools is not None else None
        return GenVisualDashboardNodeResult(
            success=app_jsx_rel_path is not None,
            response=response_content,
            dashboard_slug=artifact_slug,
            app_jsx_path=app_jsx_rel_path,
            render_file_count=render_file_count,
            template_count=len(query_actions),
            html_path=None,  # filled in by _post_validate_hook on success
            tokens_used=tokens_used,
            artifact_mode=mode,
            name=manifest.get("name"),
            description=manifest.get("description"),
            created_at=manifest.get("created_at"),
            action_history=[a.model_dump() for a in all_actions],
            execution_stats={
                "total_actions": len(all_actions),
                "tool_calls_count": len(tool_calls),
                "tools_used": sorted({a.action_type for a in tool_calls}),
                "total_tokens": tokens_used,
            },
        )

    def _finalize_artifact_error(self, exc: BaseException) -> GenVisualDashboardNodeResult:
        mode = getattr(self.artifact_tools, "mode", None) if self.artifact_tools is not None else None
        manifest = self._read_artifact_manifest(self._active_artifact_slug)
        return GenVisualDashboardNodeResult(
            success=False,
            error=str(exc),
            response="Sorry, I encountered an error while generating the visual dashboard.",
            dashboard_slug=self._active_artifact_slug,
            tokens_used=0,
            artifact_mode=mode,
            name=manifest.get("name"),
            description=manifest.get("description"),
            created_at=manifest.get("created_at"),
        )

    # ────────── CLI HTML compile ──────────

    def _post_validate_hook(
        self,
        artifact_slug: str,
        result: GenVisualDashboardNodeResult,
    ) -> Optional[ActionHistory]:
        """CLI-mode side effect: compile a standalone HTML wired against the
        ``datus --web`` dashboard-query endpoint, then emit a status action.

        Unlike the report subagent (which writes pre-executed JSON next to
        the JSX), the compiled dashboard HTML needs a live backend at view
        time — its ``DatusArtifact.initDashboard`` bootstrap issues
        ``POST <queryEndpoint>`` for every filter change. The status
        action therefore carries both the HTML path **and** the
        ``datus --web`` launch hint, so the user can copy-paste the
        command without scrolling back through the docs.

        Returns ``None`` when no HTML was compiled (non-CLI mode, missing
        render, etc.) — the base class then emits nothing.
        """
        html_rel_path = self._maybe_compile_html(artifact_slug)
        if not html_rel_path:
            return None
        result.html_path = html_rel_path
        project_root = Path(self.agent_config.project_root).resolve()
        html_abs_path = (project_root / html_rel_path).resolve()
        return self._build_html_path_action(artifact_slug, html_abs_path)

    def _build_html_path_action(self, dashboard_slug: str, html_abs_path: Path) -> ActionHistory:
        """Stream a status message naming the compiled dashboard + how to serve it.

        Carries:

        * the compiled file's absolute path, so the user can reopen it
          after closing the auto-opened browser tab;
        * the query endpoint URL the iframe will POST to (the same value
          baked into the HTML at compile time);
        * the ``datus --web`` command needed to bring the backend up.

        Uses the ``WORKFLOW`` role (the node status-message convention)
        so the TUI same-turn assistant-response dedup never drops it.
        """
        host = self._web_host()
        port = self._web_port()
        query_endpoint = self._query_endpoint(host, port)
        datasource = self._configured_datasource() or "<your_datasource>"
        web_command = f"datus --web --datasource {datasource}"
        if port != 8501:
            web_command += f" --port {port}"
        if host not in ("localhost", "127.0.0.1"):
            web_command += f" --host {host}"

        message = (
            f"Dashboard saved to {html_abs_path}\n"
            f"Live queries will be served from {query_endpoint}.\n"
            f"Start the query backend with:\n"
            f"  {web_command}\n"
            "Then open the HTML file in a browser (the CLI auto-opens it unless you passed --no-open-report)."
        )
        return ActionHistory.create_action(
            role=ActionRole.WORKFLOW,
            action_type="dashboard_html_path",
            messages=message,
            input_data={"dashboard_slug": dashboard_slug},
            output_data={
                "html_path": str(html_abs_path),
                "url": html_abs_path.as_uri(),
                "query_endpoint": query_endpoint,
                "datus_web_command": web_command,
                "host": host,
                "port": port,
            },
            status=ActionStatus.SUCCESS,
        )

    # ----------------------------------------------------------- CLI compile

    def _is_cli_mode(self) -> bool:
        """CLI deployments compile a standalone HTML; API/SaaS deployments don't."""
        if self.agent_config is None:
            return True
        deployment = getattr(self.agent_config, "deployment_mode", None) or getattr(self.agent_config, "run_mode", None)
        if isinstance(deployment, str):
            return deployment.lower() in {"", "cli", "interactive", "local"}
        # If filesystem_strict is True the node is being driven by the API gateway;
        # treat that as SaaS mode and skip the HTML compile.
        return not bool(getattr(self.agent_config, "filesystem_strict", False))

    def _maybe_compile_html(self, dashboard_slug: str) -> Optional[str]:
        if not self._is_cli_mode():
            return None
        try:
            from datus.agent.node.visual_artifact.dashboard_html_renderer import render_dashboard_html

            project_root = Path(self.agent_config.project_root).resolve()
            # ``dashboard_dist`` priority (highest first):
            #   1. ``agentic_nodes.gen_visual_dashboard.dashboard_dist`` in agent.yml
            #   2. ``--report-dist`` CLI flag (the bundle was unified into
            #      ``@datus/web-artifact-render`` so one flag covers both kinds)
            #   3. unpkg CDN (renderer default when ``dashboard_dist`` is None)
            node_dist = self.node_config.get("dashboard_dist")
            cli_override = getattr(self.agent_config, "report_dist_cli_override", None)
            dashboard_dist_value = node_dist or cli_override
            dashboard_dist = Path(dashboard_dist_value).expanduser() if dashboard_dist_value else None

            host = self._web_host()
            port = self._web_port()
            html_path = render_dashboard_html(
                project_root=project_root,
                dashboard_slug=dashboard_slug,
                dashboard_dist=dashboard_dist,
                query_endpoint=self._query_endpoint(host, port),
            )
            self._maybe_open_in_browser(html_path)
            return str(html_path.relative_to(project_root))
        except Exception as exc:
            logger.error("Failed to compile dashboard HTML for %s: %s", dashboard_slug, exc, exc_info=True)
            return None

    def _maybe_open_in_browser(self, html_path: Path) -> None:
        """Open the compiled dashboard in the system browser when the CLI opts in.

        Reuses the ``report_auto_open`` flag the CLI already sets — same
        bundle, same auto-open semantics, one knob. Background-thread so
        a slow platform launcher never blocks the agent's final action
        emission (mirrors :class:`GenVisualReportAgenticNode`).
        """
        if not bool(getattr(self.agent_config, "report_auto_open", False)):
            return
        try:
            import threading
            import webbrowser

            url = html_path.resolve().as_uri()

            def _open() -> None:
                try:
                    webbrowser.open(url)
                except Exception as exc:  # pragma: no cover — depends on the host env
                    logger.debug("webbrowser.open failed: %s", exc)

            threading.Thread(target=_open, daemon=True).start()
            logger.info("Opening dashboard in browser: %s", url)
        except Exception as exc:
            logger.debug("Failed to schedule browser open: %s", exc)

    # ----------------------------------------------------------- agent --web hints

    def _web_host(self) -> str:
        """Host the ``datus --web`` query backend listens on.

        Priority: ``agentic_nodes.gen_visual_dashboard.web_host`` in
        agent.yml → ``localhost`` (matches the CLI default).
        """
        return str(self.node_config.get("web_host") or "localhost")

    def _web_port(self) -> int:
        """Port the ``datus --web`` query backend listens on.

        Priority: ``agentic_nodes.gen_visual_dashboard.web_port`` in
        agent.yml → ``8501`` (matches the CLI default).
        """
        try:
            return int(self.node_config.get("web_port") or 8501)
        except (TypeError, ValueError):
            return 8501

    def _query_endpoint(self, host: str, port: int) -> str:
        """Full URL baked into the compiled HTML as the live-query target.

        Priority: explicit ``agentic_nodes.gen_visual_dashboard.query_endpoint``
        in agent.yml (lets ops point dashboards at a SaaS backend without
        rewriting any code) → derived from ``web_host`` / ``web_port``.
        """
        explicit = self.node_config.get("query_endpoint")
        if explicit:
            return str(explicit)
        return f"http://{host}:{port}/api/v1/dashboard/query"

    def _configured_datasource(self) -> Optional[str]:
        """Active datasource the user should pass to ``datus --web``.

        Best-effort, in priority order:
          1. ``agent_config.current_datasource`` — the live selection,
             which the ``/datasource <name>`` slash command and the
             ``--datasource`` CLI flag both feed into. Reflects what the
             current session is actually pointed at.
          2. ``agent_config.services.default_datasource`` — the YAML
             default, used when the user hasn't switched in this session.
          3. ``None`` (the message then surfaces a ``<your_datasource>``
             placeholder).
        """
        current = getattr(self.agent_config, "current_datasource", None)
        if isinstance(current, str) and current:
            return current
        services = getattr(self.agent_config, "services", None)
        if services is None:
            return None
        default = getattr(services, "default_datasource", None)
        if isinstance(default, str) and default:
            return default
        return None
