from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from datus.cli.sub_agent_wizard import run_wizard
from datus.schemas.agent_models import SubAgentConfig
from datus.utils.loggings import get_logger
from datus.utils.sub_agent_manager import SubAgentManager

if TYPE_CHECKING:
    from datus.cli import DatusCLI

logger = get_logger(__name__)
console = Console()


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
            )
        return self._sub_agent_manager

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
                console.print("[bold red]Error:[/] Agent name is required for remove.", style="bold red")
                return
            self._remove_agent(cmd_args[0])
        elif command == "update":
            if not cmd_args:
                console.print("[bold red]Error:[/] Agent name is required for update.", style="bold red")
                return
            self._cmd_update_agent(cmd_args[0])

        else:
            self._show_help()

    def _show_help(self):
        console.print("Usage: .subagent [add|list|remove|update] [args]", style="bold cyan")
        console.print(" - [bold]add[/]: Launch the interactive wizard to add a new agent.")
        console.print(" - [bold]list[/]: List all configured sub-agents.")
        console.print(" - [bold]remove <agent_name>[/]: Remove a configured sub-agent.")

    def _cmd_add_agent(self):
        """Handles the .subagent add command by launching the wizard."""
        self._do_update_agent()

    def _cmd_update_agent(self, sub_agent_name):
        existing = self.sub_agent_manager.get_agent(sub_agent_name)
        if existing is None:
            console.print("[bold red]Error:[/] Agent not found.")
            return
        self._do_update_agent(existing, original_name=sub_agent_name)

    def _list_agents(self):
        """Lists all configured sub-agents from agent.yml."""
        agents = self.sub_agent_manager.list_agents()
        if not agents:
            console.print("No sub-agents configured.", style="yellow")
            return

        table = Table(title="Configured Sub-Agents")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Scoped Context", style="cyan", min_width=20, max_width=60)
        table.add_column("Tools", style="magenta", min_width=30, max_width=80)
        table.add_column("MCPs", style="green", min_width=30, max_width=80)
        table.add_column("Rules", style="blue")

        for name, config in agents.items():
            scoped_context = self._format_scoped_context(config.get("scoped_context"))
            tools = config.get("tools") or ""
            mcps = config.get("mcp") or ""
            rules = config.get("rules", [])
            table.add_row(
                name, scoped_context, tools, mcps, Syntax("\n".join(f"- {item}" for item in rules), "markdown")
            )

        console.print(table)

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
        removed = False
        try:
            removed = self.sub_agent_manager.remove_agent(agent_name)
        except Exception as exc:
            console.print(f"[bold red]Error removing agent:[/] {exc}")
            logger.error("Failed to remove agent '%s': %s", agent_name, exc)
            return
        if not removed:
            console.print(f"[bold red]Error:[/] Agent '[bold cyan]{agent_name}[/]' not found.", style="bold red")
            return
        console.print(f"- Removed agent '[bold green]{agent_name}[/]' from configuration.")

    def _do_update_agent(
        self, data: Optional[Union[SubAgentConfig, Dict[str, Any]]] = None, original_name: Optional[str] = None
    ):
        try:
            result = run_wizard(self.cli_instance, data)
        except Exception as e:
            console.print(f"[bold red]An error occurred while running the wizard:[/] {e}")
            logger.error(f"Sub-agent wizard failed: {e}")
            return
        if result is None:
            console.print(f"Agent cancelled {'creation' if not data else 'modification'}.", style="yellow")
            return
        if original_name is None and data is not None:
            if isinstance(data, SubAgentConfig):
                original_name = data.system_prompt
            elif isinstance(data, dict):
                original_name = data.get("system_prompt")
        agent_name = result.system_prompt
        try:
            save_result = self.sub_agent_manager.save_agent(result, previous_name=original_name)
        except Exception as exc:
            console.print(f"[bold red]Failed to persist sub-agent:[/] {exc}")
            logger.error("Failed to persist sub-agent '%s': %s", agent_name, exc)
            return

        config_path = save_result.get("config_path")
        prompt_path = save_result.get("prompt_path")
        if config_path:
            console.print(f"- Updated configuration file: [cyan]{config_path}[/]")
        if prompt_path:
            console.print(f"- Created prompt template: [cyan]{prompt_path}[/]")
        console.print(f"[bold green]Sub-agent {agent_name} {'created' if not data else 'modified'} successfully.[/]")
