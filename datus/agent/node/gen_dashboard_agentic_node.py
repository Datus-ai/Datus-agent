# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenDashboardAgenticNode implementation for BI dashboard operations.

This module provides a specialized implementation of AgenticNode focused on
BI dashboard creation, management, and visualization (Superset, Grafana).
Only BI tools + ask_user are included — no DB tools exposed.
"""

from typing import Any, AsyncGenerator, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.cli.execution_state import ExecutionInterrupted
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.gen_dashboard_agentic_node_models import GenDashboardNodeResult
from datus.utils.loggings import get_logger
from datus.utils.message_utils import MessagePart, build_structured_content

logger = get_logger(__name__)


class GenDashboardAgenticNode(AgenticNode):
    """
    BI dashboard agentic node.

    This node provides dashboard management capabilities with:
    - BI tools for dashboard/chart/dataset CRUD (via BIFuncTool)
    - Source DB connector for write_query (not exposed as tools)
    - Ask-user tool for interactive confirmation
    - Session-based conversation management
    """

    NODE_NAME = "gen_dashboard"

    def __init__(
        self,
        agent_config: AgentConfig,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
        is_subagent: bool = False,
    ):
        self.execution_mode = execution_mode
        # Support custom node_name for alias subagents (e.g. my_dashboard: {node_class: gen_dashboard})
        self._configured_node_name = node_name or self.NODE_NAME

        self.max_turns = 30
        config_key = self._configured_node_name
        if agent_config and hasattr(agent_config, "agentic_nodes") and config_key in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[config_key]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 30)

        from datus.configuration.node_type import NodeType

        super().__init__(
            node_id=node_id or f"{self.NODE_NAME}_node",
            description=f"BI dashboard node: {self.NODE_NAME}",
            node_type=NodeType.TYPE_GEN_DASHBOARD,
            input_data=None,
            agent_config=agent_config,
            tools=[],
            mcp_servers={},
            is_subagent=is_subagent,
        )

        self.bi_func_tool = None
        self._read_connector = None
        self.ask_user_tool = None
        self.setup_tools()

    def get_node_name(self) -> str:
        return self._configured_node_name

    # ── Tool Setup ──────────────────────────────────────────────────────

    def setup_tools(self):
        if not self.agent_config:
            return

        self.tools = []
        self._setup_db_read_connector()
        self._setup_bi_tools()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()

        logger.debug(f"Setup {len(self.tools)} tools for {self.NODE_NAME}: {[tool.name for tool in self.tools]}")

    def _setup_db_read_connector(self):
        """Create a source DB connector for write_query support (no tools exposed)."""
        try:
            from datus.tools.db_tools.db_manager import db_manager_instance

            db_manager = db_manager_instance(self.agent_config.namespaces)
            conn = db_manager.get_conn(self.agent_config.current_database, self.agent_config.current_database)
            self._read_connector = conn
        except Exception as e:
            logger.warning(f"No source DB connector for write_query: {e}")
            self._read_connector = None

    def _setup_bi_tools(self):
        """Setup BI tools based on bi_platform config.

        Resolution order for bi_platform:
        1. self.node_config["bi_platform"] (explicit in gen_dashboard agentic_nodes config)
        2. Auto-detect: if exactly one platform in agent_config.dashboard_config, use it
        """
        try:
            bi_platform = self._resolve_bi_platform()
            if not bi_platform:
                return

            dash_cfg = getattr(self.agent_config, "dashboard_config", {}).get(bi_platform)
            if not dash_cfg:
                logger.warning(f"bi_platform '{bi_platform}' configured but no dashboard config found")
                return

            from datus_bi_core import AuthParam, adapter_registry

            adapter_cls = adapter_registry.get(bi_platform)
            if not adapter_cls:
                logger.warning(f"No BI adapter registered for platform '{bi_platform}'")
                return

            api_url = dash_cfg.api_url
            auth_params = AuthParam(
                username=dash_cfg.username,
                password=dash_cfg.password,
                api_key=dash_cfg.api_key,
                extra=dash_cfg.extra or {},
            )
            # Derive dialect from dataset_db config
            dialect = ""
            if dash_cfg.dataset_db:
                dialect = dash_cfg.dataset_db.get("dialect", "")
                if not dialect:
                    ds_uri = dash_cfg.dataset_db.get("uri", "")
                    if ds_uri:
                        try:
                            from sqlalchemy.engine.url import make_url

                            dialect = make_url(ds_uri).get_backend_name()
                        except Exception:
                            pass

            adapter = adapter_cls(api_base_url=api_url, auth_params=auth_params, dialect=dialect)

            dataset_db_uri = ""
            dataset_db_schema = ""
            datasource_name = ""
            if dash_cfg.dataset_db:
                dataset_db_uri = dash_cfg.dataset_db.get("uri", "")
                dataset_db_schema = dash_cfg.dataset_db.get("schema", "")
                datasource_name = dash_cfg.dataset_db.get("datasource_name", "")

            from datus.tools.func_tool.bi_tools import BIFuncTool

            self.bi_func_tool = BIFuncTool(
                adapter,
                dataset_db_uri=dataset_db_uri,
                dataset_db_schema=dataset_db_schema,
                read_connector=self._read_connector,
                datasource_name=datasource_name,
            )
            self.tools.extend(self.bi_func_tool.available_tools())
            logger.info(f"BI tools initialized for platform '{bi_platform}'")
        except ImportError as e:
            logger.warning(f"BI adapter package not installed: {e}")
        except Exception as e:
            logger.error(f"Failed to setup BI tools: {e}")

    def _resolve_bi_platform(self) -> Optional[str]:
        """Resolve which BI platform to use."""
        # 1. Explicit config on this node
        node_config = self.node_config or {}
        bi_platform = node_config.get("bi_platform")
        if bi_platform:
            return bi_platform

        # 2. Auto-detect: single platform in dashboard_config
        dashboard_config = getattr(self.agent_config, "dashboard_config", {})
        if len(dashboard_config) == 1:
            platform = next(iter(dashboard_config))
            logger.debug(f"Auto-detected bi_platform '{platform}' from dashboard_config")
            return platform

        if len(dashboard_config) > 1:
            logger.warning(
                f"Multiple BI platforms configured ({list(dashboard_config.keys())}), "
                "set 'bi_platform' explicitly in gen_dashboard agentic_nodes config"
            )

        return None

    # ── System Prompt ───────────────────────────────────────────────────

    def _prepare_template_context(self) -> dict:
        """Build template context with BI capability flags."""
        context: dict[str, Any] = {}
        context["native_tools"] = ", ".join([tool.name for tool in self.tools]) if self.tools else "None"
        context["has_ask_user_tool"] = self.ask_user_tool is not None

        # BI capability flags for template
        if self.bi_func_tool:
            tool_names = {tool.name for tool in self.bi_func_tool.available_tools()}
            context["has_dashboard_write"] = "create_dashboard" in tool_names
            context["has_chart_write"] = "create_chart" in tool_names
            context["has_dataset_write"] = "create_dataset" in tool_names
            context["has_write_query"] = "write_query" in tool_names
        else:
            context["has_dashboard_write"] = False
            context["has_chart_write"] = False
            context["has_dataset_write"] = False
            context["has_write_query"] = False

        bi_platform = self._resolve_bi_platform()
        context["bi_platform"] = bi_platform or "unknown"

        return context

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        template_context: Optional[dict] = None,
        prompt_version: Optional[str] = None,
    ) -> str:
        version = prompt_version or self.node_config.get("prompt_version")
        template_name = f"{self.NODE_NAME}_system"

        try:
            template_vars = {
                "agent_config": self.agent_config,
                "conversation_summary": conversation_summary,
            }
            if template_context:
                template_vars.update(template_context)

            from datus.prompts.prompt_manager import get_prompt_manager

            base_prompt = get_prompt_manager(agent_config=self.agent_config).render_template(
                template_name=template_name, version=version, **template_vars
            )
            return self._finalize_system_prompt(base_prompt)

        except FileNotFoundError:
            logger.warning(f"Template '{template_name}' not found, using inline fallback")
            return self._finalize_system_prompt(self._fallback_system_prompt(template_context or {}))
        except Exception as e:
            logger.error(f"Template loading error for '{template_name}': {e}")
            return self._finalize_system_prompt(self._fallback_system_prompt(template_context or {}))

    def _fallback_system_prompt(self, context: dict) -> str:
        """Inline fallback prompt when template is not found."""
        platform = context.get("bi_platform", "unknown")
        return (
            f"You are a BI dashboard specialist working with {platform}.\n\n"
            "Available tools are listed in your tool definitions. "
            "For creating dashboards, follow this workflow:\n"
            "1. Use write_query to materialize data into the dashboard database (if needed)\n"
            "2. Use create_dataset to register the data in the BI platform\n"
            "3. Use create_chart with appropriate chart type, metrics, and dimensions\n"
            "4. Use create_dashboard to create the dashboard\n"
            "5. Use add_chart_to_dashboard to add charts to the dashboard\n\n"
            "For read operations, use list_dashboards, get_dashboard, list_charts, list_datasets."
        )

    # ── Execution ───────────────────────────────────────────────────────

    async def execute_stream(
        self,
        action_history_manager: Optional[ActionHistoryManager] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        if self.input is None:
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(ErrorCode.COMMON_FIELD_REQUIRED, message_args={"field_name": "input"})

        user_input = self.input

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
            session = None
            conversation_summary = None
            if self.execution_mode == "interactive":
                await self._auto_compact()
                session, conversation_summary = self._get_or_create_session()

            template_context = self._prepare_template_context()
            system_instruction = self._get_system_prompt(
                conversation_summary, template_context, prompt_version=user_input.prompt_version
            )

            enhanced_message = user_input.user_message
            if user_input.database:
                enhanced_message = build_structured_content(
                    [
                        MessagePart(type="enhanced", content=f"Context: database={user_input.database}"),
                        MessagePart(type="user", content=user_input.user_message),
                    ]
                )

            response_content = ""
            last_successful_output = None

            async for stream_action in self.model.generate_with_tools_stream(
                prompt=enhanced_message,
                tools=self.tools,
                mcp_servers=self.mcp_servers,
                instruction=system_instruction,
                max_turns=self.max_turns,
                session=session,
                action_history_manager=action_history_manager,
                hooks=None,
                agent_name=self.get_node_name(),
                interrupt_controller=self.interrupt_controller,
            ):
                yield stream_action

                if stream_action.status == ActionStatus.SUCCESS and stream_action.output:
                    if isinstance(stream_action.output, dict):
                        last_successful_output = stream_action.output
                        raw_output = stream_action.output.get("raw_output", "")
                        if isinstance(raw_output, dict):
                            response_content = str(raw_output)
                        elif raw_output:
                            response_content = str(raw_output)

            if not response_content and last_successful_output:
                raw_output = last_successful_output.get("raw_output", "")
                if isinstance(raw_output, dict):
                    response_content = str(raw_output)
                elif raw_output:
                    response_content = str(raw_output)
                else:
                    response_content = str(last_successful_output)

            tokens_used = 0
            if self.execution_mode == "interactive":
                final_actions = action_history_manager.get_actions()
                for act in reversed(final_actions):
                    if act.role == "assistant":
                        if act.output and isinstance(act.output, dict):
                            usage_info = act.output.get("usage", {})
                            if usage_info and isinstance(usage_info, dict) and usage_info.get("total_tokens"):
                                tokens_used = usage_info.get("total_tokens", 0)
                                if tokens_used > 0:
                                    break

            # Extract structured result from tool outputs
            structured_result = None
            if last_successful_output and isinstance(last_successful_output, dict):
                result_data = last_successful_output.get("result")
                if isinstance(result_data, dict):
                    structured_result = result_data

            result = GenDashboardNodeResult(
                success=True,
                response=response_content,
                dashboard_result=structured_result,
                tokens_used=int(tokens_used),
            )

            self.actions.extend(action_history_manager.get_actions())

            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="gen_dashboard_response",
                messages=f"{self.get_node_name()} interaction completed successfully",
                input_data=user_input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(final_action)
            yield final_action

        except ExecutionInterrupted:
            raise

        except Exception as e:
            logger.error(f"{self.get_node_name()} execution error: {e}")

            error_result = GenDashboardNodeResult(
                success=False,
                error=str(e),
                response="Sorry, I encountered an error while processing your request.",
                tokens_used=0,
            )

            error_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="error",
                messages=f"{self.get_node_name()} interaction failed: {str(e)}",
                input_data=user_input.model_dump(),
                output_data=error_result.model_dump(),
                status=ActionStatus.FAILED,
            )
            action_history_manager.add_action(error_action)
            yield error_action
