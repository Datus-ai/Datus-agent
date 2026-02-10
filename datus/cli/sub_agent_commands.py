# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from rich.syntax import Syntax
from rich.table import Table

from datus.cli.sub_agent_wizard import run_wizard
from datus.schemas.agent_models import SubAgentConfig
from datus.storage.sub_agent_kb_bootstrap import SUPPORTED_COMPONENTS
from datus.utils.constants import SYS_SUB_AGENTS
from datus.utils.json_utils import to_pretty_str
from datus.utils.loggings import get_logger
from datus.utils.sub_agent_manager import SubAgentManager

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


class SubAgentCommands:
    def __init__(self, cli_instance: "DatusCLI"):
        self.cli_instance: "DatusCLI" = cli_instance
        self._sub_agent_manager: Optional[SubAgentManager] = None

    @property
    def sub_agent_manager(self) -> SubAgentManager:
        if self._sub_agent_manager is None:
            self._sub_agent_manager = SubAgentManager(
                configuration_manager=self.cli_instance.configuration_manager,
                namespace=self.cli_instance.agent_config.current_namespace,
                agent_config=self.cli_instance.agent_config,
            )
        return self._sub_agent_manager

    def _refresh_agent_config(self):
        """Refresh in-memory agent configuration after updates."""
        try:
            if hasattr(self.cli_instance.agent_config, "agentic_nodes"):
                self.cli_instance.agent_config.agentic_nodes = self.sub_agent_manager.list_agents()
            # Also update available_subagents set for command parsing
            if hasattr(self.cli_instance, "available_subagents"):
                self.cli_instance.available_subagents = set(SYS_SUB_AGENTS)
                if self.cli_instance.agent_config.agentic_nodes:
                    self.cli_instance.available_subagents.update(
                        name for name in self.cli_instance.agent_config.agentic_nodes.keys() if name != "chat"
                    )
            # Refresh the SubagentCompleter so autocomplete reflects the change
            if hasattr(self.cli_instance, "subagent_completer"):
                self.cli_instance.subagent_completer.refresh()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to refresh in-memory agent config: %s", exc)

    def cmd(self, args: str):
        """Main entry point for .subagent commands."""
        parts = args.strip().split()
        if not parts:
            self._show_help()
            return

        command = parts[0].lower()
        cmd_args = parts[1:]

        if command == "add":
            self._cmd_add_agent()
        elif command == "list":
            self._list_agents()
        elif command == "remove":
            if not cmd_args:
                self.cli_instance.console.print(
                    "[bold red]Error:[/] Agent name is required for remove.", style="bold red"
                )
                return
            self._remove_agent(cmd_args[0])
        elif command == "update":
            if not cmd_args:
                self.cli_instance.console.print(
                    "[bold red]Error:[/] Agent name is required for update.", style="bold red"
                )
                return
            self._cmd_update_agent(cmd_args[0])
        elif command == "bootstrap":
            self._cmd_bootstrap(cmd_args)

        else:
            self._show_help()

    def _show_help(self):
        self.cli_instance.console.print("Usage: .subagent [add|list|remove|update|bootstrap] [args]", style="bold cyan")
        self.cli_instance.console.print(" - [bold]add[/]: Launch the interactive wizard to add a new agent.")
        self.cli_instance.console.print(" - [bold]list[/]: List all configured sub-agents.")
        self.cli_instance.console.print(" - [bold]remove <agent_name>[/]: Remove a configured sub-agent.")
        self.cli_instance.console.print(
            " - [bold]bootstrap <agent_name>[/]: Build scoped knowledge base "
            "[dim](--components metadata,metrics,reference_sql --plan to simulate)[/]"
        )

    def _cmd_add_agent(self):
        """Handles the .subagent add command by launching the wizard."""
        self._do_update_agent()

    def _cmd_update_agent(self, sub_agent_name):
        if sub_agent_name in SYS_SUB_AGENTS:
            self.cli_instance.console.print(
                f"[bold red]Error:[/] System sub-agent '[cyan]{sub_agent_name}[/]' cannot be modified."
            )
            return
        existing = self.sub_agent_manager.get_agent(sub_agent_name)
        if existing is None:
            self.cli_instance.console.print("[bold red]Error:[/] Agent not found.")
            return
        self._do_update_agent(existing, original_name=sub_agent_name)

    def _cmd_bootstrap(self, args: List[str]):
        parser = argparse.ArgumentParser(
            prog=".subagent bootstrap",
            description="Bootstrap scoped knowledge base for a sub-agent.",
        )
        parser.add_argument("name", help="Sub-agent name")
        parser.add_argument(
            "--components",
            default=",".join(SUPPORTED_COMPONENTS),
            help=f"Comma-separated components to build ({', '.join(SUPPORTED_COMPONENTS)})",
        )
        parser.add_argument(
            "--plan",
            action="store_true",
            help="Simulate bootstrap and display planned records without writing.",
        )
        try:
            parsed = parser.parse_args(args)
        except SystemExit:
            return

        components = self._normalize_components(parsed.components)
        if components is None:
            return

        existing = self.sub_agent_manager.get_agent(parsed.name)
        if existing is None:
            self.cli_instance.console.print(f"[bold red]Error:[/] Agent '[cyan]{parsed.name}[/]' not found.")
            return

        config = SubAgentConfig.model_validate(existing)
        self._execute_bootstrap(config, components, parsed.plan)

    def _list_agents(self):
        """Lists all configured sub-agents from agent.yml."""
        agents = self.sub_agent_manager.list_agents()
        if not agents:
            self.cli_instance.console.print("No sub-agents configured.", style="yellow")
            return
        show_agents: List[SubAgentConfig] = []
        # filter by namespace
        for _, agent in agents.items():
            agent = SubAgentConfig.model_validate(agent)
            if (
                not agent.has_scoped_context()
                or agent.scoped_context.namespace == self.cli_instance.agent_config.current_namespace
            ):
                show_agents.append(agent)

        table = Table(title="Configured Sub-Agents")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Scoped Context", style="cyan", min_width=20, max_width=60)
        table.add_column("Scoped KB", style="green", min_width=20, max_width=80)
        table.add_column("Tools", style="magenta", min_width=30, max_width=80)
        table.add_column("MCPs", style="green", min_width=30, max_width=80)
        table.add_column("Rules", style="blue")

        for config in show_agents:
            scoped_context = self._format_scoped_context(config.scoped_context)
            scoped_kb = config.scoped_kb_path or "—"
            tools = config.tools or ""
            mcps = config.mcp or ""
            rules = config.rules
            table.add_row(
                config.system_prompt,
                scoped_context,
                scoped_kb,
                tools,
                mcps,
                Syntax("\n".join(f"- {item}" for item in rules), "markdown"),
            )

        self.cli_instance.console.print(table)

    def _normalize_components(self, value: Optional[str]) -> Optional[List[str]]:
        if not value:
            return list(SUPPORTED_COMPONENTS)
        components = [part.strip().lower() for part in value.split(",") if part.strip()]
        if not components:
            return list(SUPPORTED_COMPONENTS)
        invalid = [comp for comp in components if comp not in SUPPORTED_COMPONENTS]
        if invalid:
            self.cli_instance.console.print(
                f"[bold red]Unsupported components:[/] {', '.join(invalid)}\n"
                f"Supported components: {', '.join(SUPPORTED_COMPONENTS)}"
            )
            return None
        return components

    def _execute_bootstrap(
        self,
        config: SubAgentConfig,
        components: Optional[List[str]],
        plan: bool,
    ):
        selected_components = components or list(SUPPORTED_COMPONENTS)
        strategy = "plan" if plan else "overwrite"
        result = self.sub_agent_manager.bootstrap_agent(
            config,
            components=selected_components,
            strategy=strategy,
        )
        self._render_bootstrap_report(result)

        if strategy != "plan" and any(r.status == "success" for r in result.results):
            try:
                self.sub_agent_manager.save_agent(config, previous_name=config.system_prompt)
            except Exception as exc:
                self.cli_instance.console.print(f"[bold red]Failed to persist scoped KB path:[/] {exc}")
                logger.error("Failed to persist scoped KB for '%s': %s", config.system_prompt, exc)
            else:
                self._refresh_agent_config()

        return result

    def _render_bootstrap_report(self, result):
        table = Table(title="Scoped KB Bootstrap Summary", show_header=True)
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="magenta")
        table.add_column("Message", style="white", overflow="fold")
        status_styles = {"success": "green", "error": "red", "skipped": "yellow", "plan": "cyan"}

        for component_result in result.results:
            color = status_styles.get(component_result.status, "white")
            status_label = component_result.status.upper()
            table.add_row(
                component_result.component,
                f"[{color}]{status_label}[/]",
                component_result.message,
            )

        self.cli_instance.console.print(table)
        path_prefix = "Simulated scoped directory" if result.strategy == "plan" else "Scoped directory"
        self.cli_instance.console.print(f"{path_prefix}: [cyan]{result.storage_path}[/]")

        for component_result in result.results:
            if not component_result.details:
                continue
            if result.strategy != "plan" and component_result.status == "success":
                missing = component_result.details.get("missing")
                invalid = component_result.details.get("invalid")
                if not missing and not invalid:
                    continue
            pretty = to_pretty_str(component_result.details)
            self.cli_instance.console.print(Syntax(pretty, "json"))

    @staticmethod
    def _format_scoped_context(value: Any) -> Union[str, Syntax]:
        """Pretty print scoped context for table display."""
        if not value:
            return ""

        if isinstance(value, (Syntax, str)):
            return value

        if not isinstance(value, dict):
            return str(value)

        lines: List[str] = []
        for key in ("tables", "metrics", "sqls"):
            lines.append(f"{key}: {value.get(key)}")

        if not lines:
            return ""

        return Syntax("\n".join(lines), "yaml", word_wrap=True)

    def _remove_agent(self, agent_name: str):
        """Removes a sub-agent's configuration from agent.yml."""
        if agent_name in SYS_SUB_AGENTS:
            self.cli_instance.console.print(
                f"[bold red]Error:[/] System sub-agent '[cyan]{agent_name}[/]' cannot be removed."
            )
            return
        removed = False
        try:
            removed = self.sub_agent_manager.remove_agent(agent_name)
        except Exception as exc:
            self.cli_instance.console.print(f"[bold red]Error removing agent:[/] {exc}")
            logger.error("Failed to remove agent '%s': %s", agent_name, exc)
            return
        if not removed:
            self.cli_instance.console.print(
                f"[bold red]Error:[/] Agent '[bold cyan]{agent_name}[/]' not found.", style="bold red"
            )
            return
        self.cli_instance.console.print(f"- Removed agent '[bold green]{agent_name}[/]' from configuration.")
        self._refresh_agent_config()

    def _do_update_agent(
        self, data: Optional[Union[SubAgentConfig, Dict[str, Any]]] = None, original_name: Optional[str] = None
    ):
        if original_name and original_name in SYS_SUB_AGENTS:
            self.cli_instance.console.print(
                f"[bold red]Error:[/] System sub-agent '[cyan]{original_name}[/]' cannot be modified."
            )
            return
        try:
            result = run_wizard(self.cli_instance, data)
        except Exception as e:
            self.cli_instance.console.print(f"[bold red]An error occurred while running the wizard:[/] {e}")
            logger.error(f"Sub-agent wizard failed: {e}")
            return
        if result is None:
            self.cli_instance.console.print(
                f"Agent cancelled {'creation' if not data else 'modification'}.", style="yellow"
            )
            return
        if original_name is None and data is not None:
            if isinstance(data, SubAgentConfig):
                original_name = data.system_prompt
            elif isinstance(data, dict):
                original_name = data.get("system_prompt")
        agent_name = result.system_prompt
        if agent_name in SYS_SUB_AGENTS:
            self.cli_instance.console.print(
                f"[bold red]Error:[/] '{agent_name}' is reserved for built-in sub-agents and cannot be used."
            )
            return
        try:
            save_result = self.sub_agent_manager.save_agent(result, previous_name=original_name)
        except Exception as exc:
            self.cli_instance.console.print(f"[bold red]Failed to persist sub-agent:[/] {exc}")
            logger.error("Failed to persist sub-agent '%s': %s", agent_name, exc)
            return
        changed = save_result.get("changed", True)
        kb_action = save_result.get("kb_action")

        if not changed:
            self.cli_instance.console.print("[yellow]No changes detected; skipping save.[/]")
            return

        self._refresh_agent_config()

        config_path = save_result.get("config_path")
        prompt_path = save_result.get("prompt_path")
        if config_path:
            self.cli_instance.console.print(f"- Updated configuration file: [cyan]{config_path}[/]")
        if prompt_path:
            self.cli_instance.console.print(f"- Created prompt template: [cyan]{prompt_path}[/]")
        if kb_action == "cleared":
            self.cli_instance.console.print(
                "- Cleared scoped knowledge base for previous configuration.", style="yellow"
            )

        self.cli_instance.console.print(
            f"[bold green]Sub-agent {agent_name} {'created' if not data else 'modified'} successfully.[/]"
        )

        if result.has_scoped_context() and kb_action != "cleared":
            if result.scoped_kb_path:
                self.cli_instance.console.print(
                    f"[bold green]Scoped KB exists at {result.scoped_kb_path}, will be rebuilt.[/]"
                )

            with self.cli_instance.console.status("[bold green]Building scoped KB...[/]"):
                self._execute_bootstrap(result, None, plan=False)
