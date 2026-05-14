# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenVisualReportAgenticNode — visualizable report generation.

Replacement track for the legacy ``gen_report`` subagent. Instead of
returning a Markdown blob, this node produces a React-JSX report artifact
under ``<project_root>/reports/<id>/``:

* ``main.jsx`` — the React component the LLM authored (default export).
* ``queries/<slug>.sql`` + ``queries/<slug>.json`` — per-query source + result.

The artifact is consumed by:

* Datus-CLI — compiles to a self-contained ``index.html`` that embeds
  ``main.jsx`` + queries and loads them in a sandboxed iframe.
* Datus-SaaS — served by the backend ``GET /api/v1/report/detail`` endpoint
  and rendered dynamically by ``@datus/web-common/modules/report`` (also
  iframe-sandboxed).

See ``Datus-saas/docs/gen-report-artifact.md`` for the full contract this
node enforces.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.cli.execution_state import ExecutionInterrupted
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.gen_visual_report_models import (
    REPORT_ID_RE,
    GenVisualReportNodeInput,
    GenVisualReportNodeResult,
)
from datus.tools.func_tool import (
    ContextSearchTools,
    DBFuncTool,
    ReportArtifactTools,
    ReportFilesystemFuncTool,
)
from datus.tools.func_tool.semantic_tools import SemanticTools
from datus.utils.loggings import get_logger
from datus.utils.message_utils import MessagePart, build_structured_content

logger = get_logger(__name__)


# Inline scan for ``rpt_<id>`` mentions in the user prompt. The result is
# fed to the LLM as an awareness hint (not used to auto-bind) so the model
# can decide between editing the existing report and producing a new one
# that references it. Validation against the strict ``REPORT_ID_RE``
# happens after the dir-exists check.
_REPORT_ID_INLINE_RE = re.compile(r"(?:(?<=[^a-z0-9])|^)rpt_[a-z0-9][a-z0-9_-]{0,80}")


def _detect_referenced_report_ids(user_message: str, project_root: Path) -> List[str]:
    """Return all on-disk report ids mentioned in ``user_message``.

    Used only to inform the LLM ("the user mentioned these existing
    reports") — the model decides whether to edit them via
    ``bind_existing_report`` or use them as references for a fresh
    report via ``start_new_report``. Deduplicates while preserving
    first-mention order so the hint reads naturally.
    """
    reports_root = project_root / "reports"
    if not reports_root.is_dir():
        return []
    seen: set[str] = set()
    found: List[str] = []
    for match in _REPORT_ID_INLINE_RE.finditer(user_message.lower()):
        candidate = match.group(0)
        if candidate in seen or not REPORT_ID_RE.fullmatch(candidate):
            continue
        candidate_dir = reports_root / candidate
        if candidate_dir.is_dir() and (candidate_dir / "main.jsx").is_file():
            seen.add(candidate)
            found.append(candidate)
    return found


class GenVisualReportAgenticNode(AgenticNode):
    """
    Visual report subagent.

    Sets up semantic / db / context-search tools plus the report-specific
    ``ReportArtifactTools`` (save_query / save_main_jsx) and a hardened
    ``ReportFilesystemFuncTool`` that denies direct writes to report
    artifact paths.

    A fresh ``report_id`` is allocated on every ``execute_stream`` call so
    repeated runs against the same node produce independent artifacts.
    """

    NODE_NAME = "gen_visual_report"

    DEFAULT_TOOLS = "semantic_tools.*, db_tools.*, context_search_tools.list_subject_tree"

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[GenVisualReportNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
        node_name: Optional[str] = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        scope: Optional[str] = None,
        is_subagent: bool = False,
    ):
        self.execution_mode = execution_mode
        self.configured_node_name = node_name

        self.max_turns = 40
        if agent_config and hasattr(agent_config, "agentic_nodes") and node_name in agent_config.agentic_nodes:
            cfg = agent_config.agentic_nodes[node_name]
            if isinstance(cfg, dict):
                self.max_turns = cfg.get("max_turns", 40)

        # Tool attributes must exist before the parent constructor calls
        # ``_get_system_prompt`` indirectly via skill setup.
        self.db_func_tool: Optional[DBFuncTool] = None
        self.semantic_tools: Optional[SemanticTools] = None
        self.context_search_tools: Optional[ContextSearchTools] = None
        self.filesystem_func_tool: Optional[ReportFilesystemFuncTool] = None
        self.report_artifact_tools: Optional[ReportArtifactTools] = None
        self._active_report_id: Optional[str] = None

        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools or [],
            mcp_servers={},
            scope=scope,
            is_subagent=is_subagent,
        )

        self.setup_tools()

        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()

        logger.debug(
            "GenVisualReportAgenticNode tools: %d - %s",
            len(self.tools),
            [t.name for t in self.tools],
        )

    # ------------------------------------------------------------------ name

    def get_node_name(self) -> str:
        return self.configured_node_name or self.NODE_NAME

    # ------------------------------------------------------------ tool setup

    def setup_tools(self) -> None:
        if not self.agent_config:
            return

        self.tools = []
        config_value = self.node_config.get("tools") or self.DEFAULT_TOOLS
        for pattern in (p.strip() for p in config_value.split(",") if p.strip()):
            self._setup_tool_pattern(pattern)

        # Always provide the hardened filesystem tool — the node needs it
        # both for second-round main.jsx edits (read_file the existing
        # file, rewrite, save_main_jsx) and for general exploration.
        if not self.filesystem_func_tool:
            self._setup_filesystem_tools()

        self._setup_sub_agent_task_tool()
        if self.sub_agent_task_tool:
            self.tools.extend(self.sub_agent_task_tool.available_tools())

        logger.info("setup_tools done: %d tools - %s", len(self.tools), [t.name for t in self.tools])

    def _make_filesystem_tool(self, **kwargs):  # type: ignore[override]
        """Swap in :class:`ReportFilesystemFuncTool` for write-protection on artifact paths."""
        from datus.configuration.inherited_memory_overrides import get_inherited_memory

        root_path = kwargs.pop("root_path", None) or self._resolve_workspace_root()
        datus_home = kwargs.pop("datus_home", None)
        if datus_home is None and self.agent_config is not None:
            path_manager = getattr(self.agent_config, "path_manager", None)
            if path_manager is not None:
                try:
                    datus_home = str(path_manager.datus_home)
                except Exception:
                    datus_home = None
        strict = kwargs.pop("strict", None)
        if strict is None:
            strict = self._resolve_filesystem_strict()
        current_node = kwargs.pop("current_node", None) or self.get_node_name()
        inherited_memory_node = kwargs.pop("inherited_memory_node", None)
        if inherited_memory_node is None:
            inherited_memory_node = get_inherited_memory(current_node)
        return ReportFilesystemFuncTool(
            root_path=root_path,
            current_node=current_node,
            datus_home=datus_home,
            strict=strict,
            inherited_memory_node=inherited_memory_node,
            **kwargs,
        )

    def _setup_tool_pattern(self, pattern: str) -> None:
        try:
            if pattern.endswith(".*"):
                base = pattern[:-2]
                if base == "semantic_tools":
                    self._setup_semantic_tools()
                elif base == "db_tools":
                    self._setup_db_tools()
                elif base == "context_search_tools":
                    self._setup_context_search_tools()
                elif base == "filesystem_tools":
                    self._setup_filesystem_tools()
                else:
                    logger.warning("Unknown tool type: %s", base)
                return

            if pattern == "semantic_tools":
                self._setup_semantic_tools()
            elif pattern == "db_tools":
                self._setup_db_tools()
            elif pattern == "context_search_tools":
                self._setup_context_search_tools()
            elif pattern == "filesystem_tools":
                self._setup_filesystem_tools()
            elif "." in pattern:
                tool_type, method_name = pattern.split(".", 1)
                self._setup_specific_tool_method(tool_type, method_name)
            else:
                logger.warning("Unknown tool pattern: %s", pattern)
        except Exception as exc:
            logger.error("Failed to setup tool pattern %r: %s", pattern, exc)

    def _setup_db_tools(self) -> None:
        try:
            self.db_func_tool = DBFuncTool(
                agent_config=self.agent_config,
                sub_agent_name=self.node_config.get("system_prompt"),
            )
            self.tools.extend(self.db_func_tool.available_tools())
        except Exception as exc:
            logger.error("Failed to setup db tools: %s", exc)

    def _setup_semantic_tools(self) -> None:
        try:
            adapter_type = self.node_config.get("adapter_type", "metricflow")
            self.semantic_tools = SemanticTools(
                agent_config=self.agent_config,
                sub_agent_name=self.node_config.get("system_prompt"),
                adapter_type=adapter_type,
            )
            self.tools.extend(self.semantic_tools.available_tools())
        except Exception as exc:
            logger.error("Failed to setup semantic tools: %s", exc)

    def _setup_context_search_tools(self) -> None:
        try:
            self.context_search_tools = ContextSearchTools(
                self.agent_config, sub_agent_name=self.node_config.get("system_prompt")
            )
            self.tools.extend(self.context_search_tools.available_tools())
        except Exception as exc:
            logger.error("Failed to setup context search tools: %s", exc)

    def _setup_filesystem_tools(self) -> None:
        try:
            self.filesystem_func_tool = self._make_filesystem_tool()
            self.tools.extend(self.filesystem_func_tool.available_tools())
        except Exception as exc:
            logger.error("Failed to setup filesystem tools: %s", exc)

    def _setup_specific_tool_method(self, tool_type: str, method_name: str) -> None:
        try:
            if tool_type == "semantic_tools":
                if not self.semantic_tools:
                    self.semantic_tools = SemanticTools(
                        agent_config=self.agent_config,
                        sub_agent_name=self.node_config.get("system_prompt"),
                        adapter_type=self.node_config.get("adapter_type", "metricflow"),
                    )
                tool_instance = self.semantic_tools
            elif tool_type == "db_tools":
                if not self.db_func_tool:
                    self.db_func_tool = DBFuncTool(
                        agent_config=self.agent_config,
                        sub_agent_name=self.node_config.get("system_prompt"),
                    )
                tool_instance = self.db_func_tool
            elif tool_type == "context_search_tools":
                if not self.context_search_tools:
                    self.context_search_tools = ContextSearchTools(
                        self.agent_config, sub_agent_name=self.node_config.get("system_prompt")
                    )
                tool_instance = self.context_search_tools
            elif tool_type == "filesystem_tools":
                if not self.filesystem_func_tool:
                    self.filesystem_func_tool = self._make_filesystem_tool()
                tool_instance = self.filesystem_func_tool
            else:
                logger.warning("Unknown tool type: %s", tool_type)
                return

            if hasattr(tool_instance, method_name):
                from datus.tools.func_tool import trans_to_function_tool

                self.tools.append(trans_to_function_tool(getattr(tool_instance, method_name)))
            else:
                logger.warning("Method %r not found in %s", method_name, tool_type)
        except Exception as exc:
            logger.error("Failed to setup %s.%s: %s", tool_type, method_name, exc)

    # ---------------------------------------------- prompt + message helpers

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> str:
        context: Dict[str, Any] = {
            "has_semantic_tools": bool(self.semantic_tools),
            "has_db_tools": bool(self.db_func_tool),
            "has_context_search_tools": bool(self.context_search_tools),
            "has_ask_user_tool": self.ask_user_tool is not None,
            "has_task_tool": bool(self.sub_agent_task_tool),
            "agent_config": self.agent_config,
            "conversation_summary": conversation_summary,
            "report_id": self._active_report_id,
            "rules": self.node_config.get("rules", []),
            "agent_description": self.node_config.get("agent_description", ""),
        }

        if self.agent_config:
            from datus.utils.node_utils import build_datasource_prompt_context

            context.update(build_datasource_prompt_context(self.agent_config))
            context["db_name"] = context.get("datasource")

        from datus.utils.time_utils import get_default_current_date

        context["current_date"] = get_default_current_date(None)

        version = None if prompt_version in (None, "") else str(prompt_version)
        system_prompt_name = self.node_config.get("system_prompt") or self.get_node_name()
        template_name = f"{system_prompt_name}_system"

        from datus.prompts.prompt_manager import get_prompt_manager

        pm = get_prompt_manager(agent_config=self.agent_config)
        try:
            base_prompt = pm.render_template(template_name=template_name, version=version, **context)
        except FileNotFoundError:
            logger.warning(
                "Template %r missing, falling back to gen_visual_report_system",
                system_prompt_name,
            )
            base_prompt = pm.render_template(template_name="gen_visual_report_system", version=version, **context)

        return self._finalize_system_prompt(base_prompt)

    def _build_enhanced_message(self, user_input: GenVisualReportNodeInput) -> str:
        parts: List[str] = []
        if user_input.catalog:
            parts.append(f"Catalog: {user_input.catalog}")
        if user_input.database:
            parts.append(f"Database context: {user_input.database}")
        if user_input.db_schema:
            parts.append(f"Schema: {user_input.db_schema}")

        if self.agent_config and getattr(self.agent_config, "project_root", None):
            project_root = Path(self.agent_config.project_root).resolve()
            referenced = _detect_referenced_report_ids(user_input.user_message, project_root)
            if referenced:
                ref_list = ", ".join(referenced)
                parts.append(
                    "The user's message references existing report id(s) on disk: "
                    f"{ref_list}. Decide whether they want to EDIT one of these in place "
                    "(call bind_existing_report) or PRODUCE A NEW report that draws on "
                    "them as references (call start_new_report and use the filesystem "
                    "read tool on reports/<id>/main.jsx and reports/<id>/queries/* "
                    "to learn from them). When unclear, default to start_new_report."
                )

        if parts:
            return build_structured_content(
                [
                    MessagePart(type="enhanced", content=chr(10).join(parts)),
                    MessagePart(type="user", content=user_input.user_message),
                ]
            )
        return user_input.user_message

    # ------------------------------------------------ artifact tools wiring

    def _prepare_report_artifacts(self, user_input: GenVisualReportNodeInput) -> None:
        """Wire the artifact tools into ``self.tools`` without binding a report id.

        The LLM decides between ``start_new_report`` (create) and
        ``bind_existing_report`` (edit) at execution time; we just make
        both tools available here. ``_active_report_id`` stays ``None``
        until the LLM commits to one or the other.
        """
        if not self.agent_config or not getattr(self.agent_config, "project_root", None):
            raise ValueError("agent_config.project_root is required for gen_visual_report")
        if not self.db_func_tool:
            # save_query needs a connector; fail loud rather than silently produce a no-op tool.
            raise ValueError(
                "gen_visual_report requires db_tools to be configured (DEFAULT_TOOLS includes db_tools.*)."
            )

        # Tools start unbound — the LLM picks new vs edit via the two
        # intent-declaration tools below.
        self._active_report_id = None
        self.report_artifact_tools = ReportArtifactTools(
            agent_config=self.agent_config,
            db_func_tool=self.db_func_tool,
        )
        self.tools.extend(self.report_artifact_tools.available_tools())

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

    def _maybe_compile_html(self, report_id: str) -> Optional[str]:
        if not self._is_cli_mode():
            return None
        try:
            from datus.agent.node.report_html_renderer import render_report_html

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
                report_id=report_id,
                report_dist=report_dist,
            )
            self._maybe_open_in_browser(html_path)
            return str(html_path.relative_to(project_root))
        except Exception as exc:
            logger.error("Failed to compile report HTML for %s: %s", report_id, exc, exc_info=True)
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

    # ----------------------------------------------------------- execution

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()
        if not self.input:
            raise ValueError("Visual report input not set. Provide GenVisualReportNodeInput via setup_input().")
        user_input: GenVisualReportNodeInput = self.input

        # Bind artifact tools for this run (regenerates report_id every call).
        self._prepare_report_artifacts(user_input)

        action = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type=self.get_node_name(),
            messages=f"User: {user_input.user_message}",
            input_data=user_input.model_dump(),
            status=ActionStatus.PROCESSING,
        )
        action_history_manager.add_action(action)
        yield action

        try:
            await self._auto_compact()
            session, conversation_summary = self._get_or_create_session()
            prompt_version = getattr(user_input, "prompt_version", None) or self.node_config.get("prompt_version")
            system_instruction = self._get_system_prompt(conversation_summary, prompt_version)
            enhanced_message = self._build_enhanced_message(user_input)

            response_content = ""
            tokens_used = 0

            async for stream_action in self.model.generate_with_tools_stream(
                prompt=enhanced_message,
                tools=self.tools,
                mcp_servers=self.mcp_servers,
                instruction=system_instruction,
                max_turns=self.max_turns,
                session=session,
                action_history_manager=action_history_manager,
                agent_name=self.get_node_name(),
                interrupt_controller=self.interrupt_controller,
            ):
                if stream_action.status == ActionStatus.SUCCESS and stream_action.output:
                    if isinstance(stream_action.output, dict):
                        response_content = (
                            stream_action.output.get("content")
                            or stream_action.output.get("response")
                            or stream_action.output.get("raw_output")
                            or response_content
                        )
                yield stream_action

            all_actions = action_history_manager.get_actions()
            tool_calls = [a for a in all_actions if a.role == ActionRole.TOOL and a.status == ActionStatus.SUCCESS]

            for past in reversed(all_actions):
                if past.role == ActionRole.ASSISTANT and isinstance(past.output, dict):
                    usage = past.output.get("usage") or {}
                    if isinstance(usage, dict) and usage.get("total_tokens"):
                        tokens_used = int(usage["total_tokens"])
                        break

            # Find the most recent successful main.jsx finalizer. Either
            # save_main_jsx (one-shot) or append_jsx_chunk(position='last')
            # writes the file; the result envelope of the call that did the
            # write has main_jsx_path set. Earlier failed calls have it null
            # and are skipped.
            query_actions = [a for a in tool_calls if a.action_type == "save_query"]
            main_jsx_rel_path: Optional[str] = None
            for action in reversed(tool_calls):
                if action.action_type not in ("save_main_jsx", "append_jsx_chunk"):
                    continue
                candidate = self._extract_artifact_result_field(action, "main_jsx_path")
                if candidate:
                    main_jsx_rel_path = candidate
                    break

            # The LLM picked the active report by calling start_new_report or
            # bind_existing_report; the tools instance owns the resulting id.
            if self.report_artifact_tools is not None and self.report_artifact_tools.report_id:
                self._active_report_id = self.report_artifact_tools.report_id

            html_rel_path: Optional[str] = None
            if main_jsx_rel_path and self._active_report_id:
                html_rel_path = self._maybe_compile_html(self._active_report_id)

            result = GenVisualReportNodeResult(
                success=main_jsx_rel_path is not None,
                response=response_content,
                report_id=self._active_report_id,
                main_jsx_path=main_jsx_rel_path,
                html_path=html_rel_path,
                query_count=len(query_actions),
                tokens_used=tokens_used,
                action_history=[a.model_dump() for a in all_actions],
                execution_stats={
                    "total_actions": len(all_actions),
                    "tool_calls_count": len(tool_calls),
                    "tools_used": sorted({a.action_type for a in tool_calls}),
                    "total_tokens": tokens_used,
                },
            )
            if main_jsx_rel_path is None:
                if self._active_report_id is None:
                    result.error = (
                        "Run finished without binding a report. The LLM must call either "
                        "start_new_report(...) or bind_existing_report(...) before producing "
                        "the artifact."
                    )
                else:
                    result.error = "save_main_jsx was never called — the report artifact is incomplete."

            self.actions.extend(all_actions)

            summary_messages = (
                f"Visual report generated: reports/{self._active_report_id}/main.jsx"
                if main_jsx_rel_path
                else "Visual report run finished without main.jsx."
            )
            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type=f"{self.get_node_name()}_response",
                messages=summary_messages,
                input_data=user_input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS if main_jsx_rel_path else ActionStatus.FAILED,
            )
            action_history_manager.add_action(final_action)
            yield final_action

        except ExecutionInterrupted:
            raise
        except Exception as exc:
            logger.error("%s execution error: %s", self.get_node_name(), exc, exc_info=True)
            error_result = GenVisualReportNodeResult(
                success=False,
                error=str(exc),
                response="Sorry, I encountered an error while generating the visual report.",
                report_id=self._active_report_id,
                tokens_used=0,
            )
            action_history_manager.update_current_action(
                status=ActionStatus.FAILED,
                output=error_result.model_dump(),
                messages=f"Error: {exc}",
            )
            error_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="error",
                messages=f"{self.get_node_name()} failed: {exc}",
                input_data=user_input.model_dump(),
                output_data=error_result.model_dump(),
                status=ActionStatus.FAILED,
            )
            action_history_manager.add_action(error_action)
            yield error_action

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _find_artifact_tool_call(actions: List[ActionHistory], tool_name: str) -> Optional[ActionHistory]:
        for a in reversed(actions):
            if a.action_type == tool_name:
                return a
        return None

    @staticmethod
    def _extract_artifact_result_field(action: ActionHistory, field: str) -> Optional[str]:
        """
        Walk the recorded tool output structure looking for ``result[field]``.

        Tool outputs land in :pyattr:`ActionHistory.output` under a few possible
        shapes depending on which dispatcher recorded them — see the agent
        framework's tool harness and the test harness in ``mock_llm_model``.
        ``FuncToolResult`` is always serialized as ``{success, error, result}``,
        so we recursively scan for that envelope.
        """
        output = action.output
        if not isinstance(output, dict):
            return None

        def _scan(obj: Any) -> Optional[str]:
            if isinstance(obj, dict):
                if field in obj and isinstance(obj[field], str):
                    return obj[field]
                for key in ("result", "raw_output", "output", "data"):
                    if key in obj:
                        found = _scan(obj[key])
                        if found:
                            return found
                # Fallback: scan all values.
                for value in obj.values():
                    found = _scan(value)
                    if found:
                        return found
            elif isinstance(obj, str):
                try:
                    parsed = json.loads(obj)
                except (TypeError, json.JSONDecodeError):
                    return None
                return _scan(parsed)
            return None

        return _scan(output)
