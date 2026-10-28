# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/bootstrap-bi`` compatibility shortcut for ``dashboard-bootstrap``.

Dashboard bootstrap orchestration lives in the bundled
``dashboard-bootstrap`` skill. The slash command only injects a deterministic
request into the standard chat pipeline, so plugin discovery, selection,
confirmation, export, and Knowledge Base construction remain skill/LLM driven.

The legacy ``_run_plan`` helpers remain temporarily for API compatibility and
targeted migration tests, but the user-facing command no longer invokes them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from rich.console import Console

from datus.cli.bootstrap_bi_picker import BootstrapBiPlan
from datus.cli.bootstrap_bi_streams import (
    BiBuildState,
    stream_bi_metadata,
    stream_bi_reference_sql,
    stream_bi_semantic_model,
)
from datus.cli.bootstrap_bi_subagents import (
    build_sub_agent_name,
    dedupe_values,
    qualify_table_names,
    stream_bi_save_subagents,
)
from datus.cli.bootstrap_subagent import message_action
from datus.cli.cli_styles import print_error
from datus.cli.skill_command_utils import render_skill_prompt
from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import configuration_manager
from datus.schemas.action_history import ActionHistory, ActionStatus
from datus.schemas.agent_models import ScopedContext
from datus.utils.constants import SYS_SUB_AGENTS
from datus.utils.sub_agent_manager import SubAgentManager
from datus.utils.traceable_utils import optional_traceable

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

_DASHBOARD_BOOTSTRAP_PROMPT = (
    "Bootstrap reference SQL and metrics from a BI dashboard by following the "
    "`dashboard-bootstrap` skill. "
    'Call `load_skill(skill_name="dashboard-bootstrap")` first and execute its steps in order. '
    "Use the selected BI plugin for dashboard discovery and SQL export, and use the builtin "
    "agents named by the skill to build context. Do not use the legacy bootstrap picker, "
    "or legacy BI streams. Let the skill decide whether its optional final subagent step is "
    "available."
    "{user_context}"
)


class BootstrapBiCommands:
    """Bind point for the ``/bootstrap-bi`` REPL slash command."""

    def __init__(
        self,
        agent_config: "AgentConfig | DatusCLI",
        console: Optional[Console] = None,
    ) -> None:
        self.cli: Optional["DatusCLI"] = None
        if hasattr(agent_config, "agent_config"):
            self.cli = agent_config
            self.agent_config = agent_config.agent_config
            self.console = console or agent_config.console
            self._configuration_manager = getattr(agent_config, "configuration_manager", None)
        else:
            self.agent_config = agent_config
            self.console = console or Console(log_path=False)
            self._configuration_manager = None

    @optional_traceable(name="bootstrap_bi")
    def cmd(self, args: str = "") -> None:
        """Delegate ``/bootstrap-bi`` to the skill-aware chat pipeline."""
        chat_commands = getattr(self.cli, "chat_commands", None) if self.cli else None
        if chat_commands is None:
            print_error(
                self.console,
                "Chat is not initialized — /bootstrap-bi relies on the chat pipeline.",
                prefix=False,
            )
            return

        chat_commands.execute_chat_command(
            render_skill_prompt(_DASHBOARD_BOOTSTRAP_PROMPT, args),
            plan_mode=getattr(self.cli, "plan_mode_active", False),
            subagent_name=None,
        )

    async def _run_plan(self, plan: BootstrapBiPlan, actions: List[ActionHistory]) -> None:
        # Header context.
        actions.append(message_action(f"Dashboard: {plan.dashboard.name} (id={plan.dashboard.id})"))
        actions.append(
            message_action(
                f"Selected {len(plan.chart_selections_ref)}/{len(plan.chart_selections_metrics)} chart(s); "
                f"{len(plan.assembled.tables or [])} table(s); pool_size={plan.pool_size}"
            )
        )

        if not plan.chart_selections_ref and not plan.chart_selections_metrics:
            actions.append(message_action("No charts selected. Aborting.", status=ActionStatus.FAILED))
            return

        if not getattr(self.agent_config, "current_datasource", ""):
            actions.append(message_action("No datasource set; skipping sub-agent build.", status=ActionStatus.FAILED))
            return

        sub_agent_name = build_sub_agent_name(plan.options.platform, plan.dashboard.name or "")
        if sub_agent_name in SYS_SUB_AGENTS:
            actions.append(
                message_action(
                    f"'{sub_agent_name}' is reserved for built-in sub-agents.",
                    status=ActionStatus.FAILED,
                )
            )
            return

        # Resolve catalog/database/schema once (depends on cli_context).
        catalog, database, schema = self._resolve_default_table_context()
        table_names = qualify_table_names(
            dedupe_values([t for t in (plan.assembled.tables or []) if t]),
            self.agent_config,
            catalog=catalog,
            database=database,
            schema=schema,
        )

        state = BiBuildState(table_names=table_names)

        # 1. Metadata
        async for action in stream_bi_metadata(
            self.agent_config,
            table_names=table_names,
            pool_size=plan.pool_size,
        ):
            actions.append(action)

        # 2. Reference SQL
        if plan.assembled.reference_sqls:
            async for action in stream_bi_reference_sql(
                self.agent_config,
                reference_sqls=plan.assembled.reference_sqls,
                platform=plan.options.platform,
                dashboard_name=plan.dashboard.name or "",
                pool_size=plan.pool_size,
                state=state,
            ):
                actions.append(action)

        # 3. Unified semantic model and metric authoring.
        if plan.assembled.metric_sqls:
            async for action in stream_bi_semantic_model(
                self.agent_config,
                sqls=plan.assembled.metric_sqls,
                platform=plan.options.platform,
                dashboard_name=plan.dashboard.name or "",
                state=state,
            ):
                actions.append(action)

            if not state.semantic_ok:
                actions.append(
                    message_action(
                        "Unified semantic modeling failed; generated metrics are unavailable.",
                        status=ActionStatus.FAILED,
                    )
                )
                return

        # 4. Build ScopedContext and persist the two sub-agent yamls.
        scoped = self._build_scoped_context(state)
        if scoped is None:
            actions.append(
                message_action(
                    "No scoped context derived; skipping sub-agent save.",
                    status=ActionStatus.FAILED,
                )
            )
            return

        manager = SubAgentManager(
            configuration_manager=self._configuration_manager or configuration_manager(),
            datasource=self.agent_config.current_datasource,
            agent_config=self.agent_config,
        )

        async for action in stream_bi_save_subagents(
            self.agent_config,
            sub_agent_name=sub_agent_name,
            description=plan.dashboard.description or plan.dashboard.name or "",
            scoped_context=scoped,
            sub_agent_manager=manager,
            cli_ref=self.cli,
        ):
            actions.append(action)

        actions.append(message_action("Sub-Agent build successful."))

    def _resolve_default_table_context(self) -> tuple[str, str, str]:
        catalog = ""
        database = ""
        schema = ""

        cli_context = getattr(self.cli, "cli_context", None) if self.cli else None
        if cli_context:
            catalog = (cli_context.current_catalog or "").strip()
            database = (cli_context.current_db_name or "").strip()
            schema = (cli_context.current_schema or "").strip()

        if not (catalog and database and schema):
            try:
                db_config = self.agent_config.current_db_config(self.agent_config.current_datasource)
            except Exception:
                db_config = None
            if db_config is not None:
                catalog = catalog or (db_config.catalog or "")
                database = database or (db_config.database or "")
                schema = schema or (db_config.schema or "")
        return catalog, database, schema

    @staticmethod
    def _build_scoped_context(state: BiBuildState) -> Optional[ScopedContext]:
        if not (state.table_names or state.ref_sqls or state.metrics):
            return None
        return ScopedContext(
            tables=",".join(state.table_names) if state.table_names else None,
            sqls=",".join(state.ref_sqls) if state.ref_sqls else None,
            metrics=",".join(state.metrics) if state.metrics else None,
        )


__all__ = ["BootstrapBiCommands"]
