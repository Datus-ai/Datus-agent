# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenVisualReportAgenticNode — visualizable report generation.

Replacement track for the legacy ``gen_report`` subagent. Instead of
returning a Markdown blob, this node produces a React-JSX report artifact
under ``<project_root>/reports/<slug>/``:

* ``render/app.jsx`` — the React entry module the LLM authors (default export);
  it imports any additional ``render/*.jsx`` components the report needs.
* ``queries/<slug>.sql`` + ``queries/<slug>.json`` — per-query source + result.
* ``manifest.json`` — ``{slug, name, description, kind, created_at}``.

The artifact is consumed by:

* Datus-CLI — compiles to a self-contained ``index.html`` that embeds
  ``render/`` files + queries and loads them in a sandboxed iframe.
* Datus-SaaS — served by the backend ``GET /api/v1/report/detail`` endpoint
  and rendered dynamically by ``@datus/web-common/modules/report`` (also
  iframe-sandboxed).

See ``Datus-saas/docs/gen-report-artifact.md`` for the full contract this
node enforces. Common machinery (tool setup, prompt rendering, the LLM
loop, action-history extraction) lives in
:class:`BaseVisualArtifactAgenticNode`; this file owns the
report-specific artifact wiring, result model, and the CLI HTML compile
path that has no analogue in dashboard mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from datus.agent.node.base_visual_artifact_agentic_node import BaseVisualArtifactAgenticNode
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.schemas.gen_visual_report_models import (
    GenVisualReportNodeInput,
    GenVisualReportNodeResult,
)
from datus.tools.func_tool import (
    ReportArtifactTools,
    ReportFilesystemFuncTool,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenVisualReportAgenticNode(BaseVisualArtifactAgenticNode[GenVisualReportNodeInput, GenVisualReportNodeResult]):
    """
    Visual report subagent.

    Sets up semantic / db / context-search tools plus the report-specific
    ``ReportArtifactTools`` (save_query / validate_render) and a hardened
    ``ReportFilesystemFuncTool`` that denies direct writes to report
    artifact paths.

    The LLM chooses the ``report_slug`` on every fresh ``start_new_report``
    call; the system prompt directs it to ``glob('reports/*')`` first so the
    chosen slug doesn't collide with an existing one.
    """

    NODE_NAME = "gen_visual_report"
    result_class = GenVisualReportNodeResult
    ARTIFACT_KIND = "report"
    ARTIFACT_ROOT_DIR_NAME = "reports"
    FILESYSTEM_TOOL_CLS = ReportFilesystemFuncTool
    QUERY_SAVE_ACTION_TYPE = "save_query"
    FALLBACK_TEMPLATE_NAME = "gen_visual_report_system"

    # ------------------------------------------------------------------ name

    def get_node_name(self) -> str:
        return self.configured_node_name or self.NODE_NAME

    # ────────── Convenience accessors ──────────

    @property
    def _active_report_slug(self) -> Optional[str]:
        return self._active_artifact_slug

    @_active_report_slug.setter
    def _active_report_slug(self, value: Optional[str]) -> None:
        self._active_artifact_slug = value

    @property
    def report_artifact_tools(self) -> Optional[ReportArtifactTools]:
        return self.artifact_tools  # type: ignore[return-value]

    @report_artifact_tools.setter
    def report_artifact_tools(self, value: Optional[ReportArtifactTools]) -> None:
        self.artifact_tools = value

    # ────────── Hooks the base class calls ──────────

    def _make_artifact_tools(self, user_input: GenVisualReportNodeInput) -> ReportArtifactTools:
        return ReportArtifactTools(
            agent_config=self.agent_config,
            db_func_tool=self.db_func_tool,
            user_message=getattr(user_input, "user_message", "") or "",
        )

    def _read_artifact_slug_from_tools(self) -> Optional[str]:
        tools = self.artifact_tools
        if tools is None:
            return None
        return getattr(tools, "report_slug", None)

    def _finalize_artifact_success(
        self,
        *,
        user_input: GenVisualReportNodeInput,
        response_content: str,
        artifact_slug: Optional[str],
        app_jsx_rel_path: Optional[str],
        render_file_count: int,
        query_actions: List[ActionHistory],
        tokens_used: int,
        all_actions: List[ActionHistory],
        tool_calls: List[ActionHistory],
    ) -> GenVisualReportNodeResult:
        manifest = self._read_artifact_manifest(artifact_slug)
        mode = getattr(self.artifact_tools, "mode", None) if self.artifact_tools is not None else None
        return GenVisualReportNodeResult(
            success=app_jsx_rel_path is not None,
            response=response_content,
            report_slug=artifact_slug,
            app_jsx_path=app_jsx_rel_path,
            render_file_count=render_file_count,
            html_path=None,  # filled in by _post_validate_hook on success
            query_count=len(query_actions),
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

    def _finalize_artifact_error(self, exc: BaseException) -> GenVisualReportNodeResult:
        mode = getattr(self.artifact_tools, "mode", None) if self.artifact_tools is not None else None
        manifest = self._read_artifact_manifest(self._active_artifact_slug)
        return GenVisualReportNodeResult(
            success=False,
            error=str(exc),
            response="Sorry, I encountered an error while generating the visual report.",
            report_slug=self._active_artifact_slug,
            tokens_used=0,
            artifact_mode=mode,
            name=manifest.get("name"),
            description=manifest.get("description"),
            created_at=manifest.get("created_at"),
        )

    def _post_validate_hook(self, artifact_slug: str, result: GenVisualReportNodeResult) -> Optional[ActionHistory]:
        """CLI-mode side effect: compile a standalone HTML and open it.

        Dashboard mode has no analogue (its queries are templates that
        must run against a live datasource), so this stays in the report
        subclass and the base class skips the hook in dashboard mode.

        Returns a stream message carrying the compiled HTML's absolute path
        so the user can reopen the report after closing the auto-opened
        browser tab; ``None`` when no HTML was compiled (non-CLI mode).
        """
        html_rel_path = self._maybe_compile_html(artifact_slug)
        if not html_rel_path:
            return None
        result.html_path = html_rel_path
        project_root = Path(self.agent_config.project_root).resolve()
        html_abs_path = (project_root / html_rel_path).resolve()
        return self._build_html_path_action(artifact_slug, html_abs_path)

    def _build_html_path_action(self, report_slug: str, html_abs_path: Path) -> ActionHistory:
        """Stream a status message naming the compiled report's absolute path.

        The CLI auto-opens the report in a browser; once the user closes that
        tab they otherwise have no record of where the artifact lives. Emitting
        the absolute path into the action stream keeps it in the scrollback for
        later reference. Uses the ``WORKFLOW`` role (the node status-message
        convention) so the TUI same-turn assistant-response dedup never drops it.
        """
        message = (
            f"Report saved to {html_abs_path} — open this file in a browser to view it again after closing the tab."
        )
        return ActionHistory.create_action(
            role=ActionRole.WORKFLOW,
            action_type="report_html_path",
            messages=message,
            input_data={"report_slug": report_slug},
            output_data={"html_path": str(html_abs_path), "url": html_abs_path.as_uri()},
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

    def _maybe_compile_html(self, report_slug: str) -> Optional[str]:
        if not self._is_cli_mode():
            return None
        try:
            from datus.agent.node.visual_artifact.report_html_renderer import render_report_html

            project_root = Path(self.agent_config.project_root).resolve()
            # ``report_dist`` priority (highest first):
            #   1. ``--report-dist`` CLI flag (stashed on agent_config by
            #      DatusCLI / PrintModeRunner as ``report_dist_cli_override``)
            #   2. ``agentic_nodes.gen_visual_report.report_dist`` in agent.yml
            #   3. unpkg CDN (renderer default when ``report_dist`` is None)
            cli_override = getattr(self.agent_config, "report_dist_cli_override", None)
            report_dist_value = cli_override or self.node_config.get("report_dist")
            report_dist = Path(report_dist_value).expanduser() if report_dist_value else None
            html_path = render_report_html(
                project_root=project_root,
                report_slug=report_slug,
                report_dist=report_dist,
            )
            self._maybe_open_in_browser(html_path)
            return str(html_path.relative_to(project_root))
        except Exception as exc:
            logger.error("Failed to compile report HTML for %s: %s", report_slug, exc, exc_info=True)
            return None

    def _maybe_open_in_browser(self, html_path: Path) -> None:
        """Open the compiled report in the system browser when the CLI opts in.

        Gated on ``agent_config.report_auto_open`` which ``DatusCLI`` sets to
        ``True`` for the interactive REPL (unless the user passes
        ``--no-open-report``) and ``False`` for print mode. Mirrors the
        background-thread pattern in ``datus.cli.web.chatbot`` so a slow
        platform launcher never blocks the agent's final action emission.
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
            logger.info("Opening report in browser: %s", url)
        except Exception as exc:
            logger.debug("Failed to schedule browser open: %s", exc)
