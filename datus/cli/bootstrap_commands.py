# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/bootstrap`` slash-command driver.

Pipeline:

1. Run :class:`BootstrapApp` (TUI) → :class:`BootstrapPlan`.
2. Build a unified streaming pipeline (one ``actions`` list, one
   :class:`InlineStreamingContext`).
3. For each enabled task, drive the matching ``stream_*`` async
   generator and append every yielded :class:`ActionHistory` into the
   shared list. The streaming daemon renders task-tool subagent groups
   identical to chat's ``task(…)`` output.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, AsyncGenerator, List, Optional

from rich.console import Console

from datus.cli.action_display import ActionHistoryDisplay
from datus.cli.bootstrap_app import BootstrapApp, BootstrapPlan, TaskSpec
from datus.cli.bootstrap_streams import (
    stream_metadata,
    stream_metrics,
    stream_reference_sql,
    stream_reference_template,
    stream_semantic_model,
)
from datus.cli.bootstrap_subagent import message_action
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionStatus
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


class BootstrapCommands:
    """Bind point for the ``/bootstrap`` REPL slash command."""

    def __init__(
        self,
        agent_config: AgentConfig | "DatusCLI",
        console: Optional[Console] = None,
    ) -> None:
        self.cli: Optional["DatusCLI"] = None
        if hasattr(agent_config, "agent_config"):
            self.cli = agent_config
            self.agent_config = agent_config.agent_config
            self.console = console or agent_config.console
        else:
            self.agent_config = agent_config
            self.console = console or Console(log_path=False)

    @optional_traceable(name="bootstrap")
    def cmd(self, args: str = "") -> None:
        plan = self._run_picker()
        if plan is None:
            self.console.print("\n[yellow]Cancelled.[/]")
            return

        actions: List[ActionHistory] = []
        # Pass through the TUI's ``live_state`` (when running inside the
        # REPL Application) so the streaming daemon routes its pinned
        # region through DatusApp's shared Live instead of trying to
        # start a second ``rich.Live`` — which would raise
        # ``LiveError: Only one live display may be active at once``.
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

    # ── picker ─────────────────────────────────────────────────────────

    def _run_picker(self) -> Optional[BootstrapPlan]:
        default_ds = getattr(self.agent_config, "current_datasource", "") or ""
        app = BootstrapApp(self.console, datasource_default=str(default_ds))
        tui_app = getattr(self.cli, "tui_app", None) if self.cli else None
        if tui_app is not None and getattr(tui_app, "_loop", None) is not None:
            return tui_app.run_wizard(app.build_embedded_panel)
        return app.run()

    # ── async runner ───────────────────────────────────────────────────

    async def _run_plan(self, plan: BootstrapPlan, actions: List[ActionHistory]) -> None:
        spec = plan.task
        actions.append(message_action(f"Running bootstrap task: {spec.name}"))
        try:
            async for action in self._stream_for(spec):
                actions.append(action)
        except Exception as exc:
            logger.error("stream_%s failed: %s", spec.name, exc, exc_info=True)
            actions.append(message_action(f"{spec.name} failed: {exc}", status=ActionStatus.FAILED))
        actions.append(message_action("Bootstrap finished."))

    def _stream_for(self, spec: TaskSpec) -> AsyncGenerator[ActionHistory, None]:
        o = spec.options
        bm = o.get("build_mode", "incremental")
        ds = o.get("datasource", "") or ""
        pool = int(o.get("pool_size", 3))
        subject_tree = _split_csv(o.get("subject_tree"))

        if spec.name == "metadata":
            return stream_metadata(self.agent_config, datasource=ds, build_mode=bm)
        if spec.name == "reference_sql":
            return stream_reference_sql(
                self.agent_config,
                datasource=ds,
                sql_dir=o.get("sql_dir", ""),
                pool_size=pool,
                build_mode=bm,
                subject_tree=subject_tree,
            )
        if spec.name == "reference_template":
            return stream_reference_template(
                self.agent_config,
                datasource=ds,
                template_dir=o.get("template_dir", ""),
                pool_size=pool,
                build_mode=bm,
                subject_tree=subject_tree,
            )
        if spec.name == "semantic_model":
            return stream_semantic_model(
                self.agent_config,
                datasource=ds,
                success_story=o.get("success_story", ""),
                build_mode=bm,
            )
        if spec.name == "metrics":
            return stream_metrics(
                self.agent_config,
                datasource=ds,
                success_story=o.get("success_story", ""),
                pool_size=pool,
                build_mode=bm,
                subject_tree=subject_tree,
            )

        async def _empty():
            return
            yield  # pragma: no cover

        return _empty()


def _split_csv(value: Any) -> Optional[List[str]]:
    if not value:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [s.strip() for s in str(value).split(",") if s.strip()] or None


__all__ = ["BootstrapCommands"]
