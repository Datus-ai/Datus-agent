# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/bootstrap-bi`` slash-command driver.

Mirrors :class:`bootstrap_commands.BootstrapCommands`:

1. Run :class:`BootstrapBiPicker` (multi-stage TUI + BI adapter IO) to
   produce a :class:`BootstrapBiPlan`.
2. Build a unified streaming pipeline (one ``actions`` list, one
   :class:`InlineStreamingContext`).
3. Drive the four ``stream_bi_*`` async generators in order, threading
   :class:`BiBuildState` through them so the metrics stream can be
   gated on the semantic-model validation result.
4. Build the final :class:`ScopedContext` and persist the two
   sub-agent yaml files via :func:`stream_bi_save_subagents`.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, List, Optional

from rich.console import Console

from datus.cli.action_display import ActionHistoryDisplay
from datus.cli.bootstrap_bi_picker import BootstrapBiPicker, BootstrapBiPlan
from datus.cli.bootstrap_bi_streams import (
    BiBuildState,
    stream_bi_metadata,
    stream_bi_metrics,
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
from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import configuration_manager
from datus.schemas.action_history import ActionHistory, ActionStatus
from datus.schemas.agent_models import ScopedContext
from datus.utils.constants import SYS_SUB_AGENTS
from datus.utils.loggings import get_logger
from datus.utils.sub_agent_manager import SubAgentManager
from datus.utils.traceable_utils import optional_traceable

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


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
        try:
            plan = BootstrapBiPicker(self.agent_config, self.console, cli=self.cli).run()
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Cancelled.[/]")
            return
        except Exception as exc:
            logger.error("Failed to drive bootstrap-bi picker", exc_info=True)
            self.console.print(f"[bold red]Error:[/] {exc}")
            return

        if plan is None:
            self.console.print("\n[yellow]Cancelled.[/]")
            return

        actions: List[ActionHistory] = []
        live_state = getattr(self.cli, "live_state", None) if self.cli else None
        display = ActionHistoryDisplay(self.console, live_state=live_state)
        ctx = display.display_streaming_actions(actions=actions)

        original_console = self.console
        silent = Console(file=open(os.devnull, "w"), force_terminal=False, log_path=False)
        try:
            with ctx:
                self.console = silent
                asyncio.run(self._run_plan(plan, actions))
        finally:
            self.console = original_console
            try:
                silent.file.close()
            except Exception:
                pass
            try:
                plan.adapter.close()
            except Exception:
                pass

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

        # 3. Semantic model (gates metrics).
        if plan.assembled.metric_sqls:
            async for action in stream_bi_semantic_model(
                self.agent_config,
                sqls=plan.assembled.metric_sqls,
                platform=plan.options.platform,
                dashboard_name=plan.dashboard.name or "",
                state=state,
            ):
                actions.append(action)

            # 4. Metrics — only if validation passed.
            if state.semantic_ok:
                async for action in stream_bi_metrics(
                    self.agent_config,
                    sqls=plan.assembled.metric_sqls,
                    platform=plan.options.platform,
                    dashboard_name=plan.dashboard.name or "",
                    state=state,
                ):
                    actions.append(action)
            else:
                actions.append(
                    message_action(
                        "Skipping metrics generation due to semantic model failure",
                        status=ActionStatus.FAILED,
                    )
                )

        # 5. Build ScopedContext and persist the two sub-agent yamls.
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
