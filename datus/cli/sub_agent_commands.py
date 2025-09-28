import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from datus.cli.sub_agent_wizard import run_wizard
from datus.prompts.prompt_manager import PromptManager
from datus.prompts.sub_agent_prompt_template import render_template
from datus.schemas.agent_models import SubAgentConfig
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli import DatusCLI

logger = get_logger(__name__)
console = Console()


class SubAgentCommands:
    def __init__(self, cli_instance: "DatusCLI"):
        self.cli_instance: "DatusCLI" = cli_instance

    @property
    def prompt_manager(self) -> PromptManager:
        from datus.prompts.prompt_manager import prompt_manager

        return prompt_manager

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
        if (
            "agentic_nodes" in self.cli_instance.configuration_manager.data
            and sub_agent_name in self.cli_instance.configuration_manager.data["agentic_nodes"]
        ):
            self._do_update_agent(self.cli_instance.configuration_manager.data["agentic_nodes"][sub_agent_name])
        else:
            console.print("[bold red]Error:[/] Agent not found.")

    def _update_agent_yml(self, config: SubAgentConfig):
        """Updates the agent.yml file with the new agent's configuration."""
        agent_name = config.system_prompt
        agent_config: Dict[str, Any] = {
            "system_prompt": agent_name,
            "prompt_version": config.prompt_version,
            "prompt_language": config.prompt_language,
            "tools": config.tools,
            "mcp": config.mcp,
            "rules": list(config.rules or []),
        }

        if config.scoped_context:
            scoped_context = config.scoped_context.model_dump(exclude_none=True)
            if scoped_context:
                agent_config["scoped_context"] = scoped_context

        # Add the new agent config
        self.cli_instance.configuration_manager.update_item(
            "agentic_nodes", {agent_name: agent_config}, delete_old_key=True
        )
        console.print(f"- Updated configuration file: [cyan]{self.cli_instance.configuration_manager.config_path}[/]")

    def _create_prompt_template(self, config: SubAgentConfig):
        """Creates the .j2 prompt template file."""
        agent_name = config.system_prompt
        version = config.prompt_version
        lang = config.prompt_language

        # Only add language to filename if it's not the default 'en'
        lang_suffix = f"_{lang}" if lang != "en" else ""

        template_filename = f"{agent_name}{lang_suffix}_{version}.j2"

        # TODO: This path should be made more robust
        template_path = os.path.join("datus", "prompts", "prompt_templates", template_filename)

        template_content = render_template(self.cli_instance.agent_config.current_namespace, config)

        try:
            self.prompt_manager.save(agent_name, version=version, prompt_content=template_content)
            console.print(f"- Created prompt template: [cyan]{template_path}[/]")
        except IOError as e:
            console.print(f"[bold red]Error creating template file:[/] {e}")
            logger.error(f"Failed to write template file {template_path}: {e}")

    def _list_agents(self):
        """Lists all configured sub-agents from agent.yml."""
        agents = self.cli_instance.configuration_manager.get("agentic_nodes", {})
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
            scoped_context = config.get("scoped_context", {})
            if scoped_context:
                scoped_context = ""
            else:
                scoped_context = ""
            tools = config.get("tools") or ""
            mcps = config.get("mcp") or ""
            rules = config.get("rules", [])
            table.add_row(
                name, scoped_context, tools, mcps, Syntax("\n".join(f"- {item}" for item in rules), "markdown")
            )

        console.print(table)

    def _remove_agent(self, agent_name: str):
        """Removes a sub-agent's configuration from agent.yml."""
        agents = self.cli_instance.configuration_manager.get("agentic_nodes", {})
        if agent_name not in agents:
            console.print(f"[bold red]Error:[/] Agent '[bold cyan]{agent_name}[/]' not found.", style="bold red")
            return

        # Remove from config
        del self.cli_instance.configuration_manager.data["agentic_nodes"][agent_name]
        self.cli_instance.configuration_manager.save()
        console.print(f"- Removed agent '[bold green]{agent_name}[/]' from configuration.")

        console.print("[yellow]Note:[/] The associated .j2 template file was not removed.", style="yellow")

    def _do_update_agent(self, data: Optional[Union[SubAgentConfig, Dict[str, Any]]] = None):
        try:
            # from datus.cli.sub_agent_wizard2 import run_add_wizard
            result = run_wizard(self.cli_instance, data)
        except Exception as e:
            console.print(f"[bold red]An error occurred while running the wizard:[/] {e}")
            logger.error(f"Sub-agent wizard failed: {e}")
            return

        if result is None:
            console.print("Agent creation cancelled.", style="yellow")
            return
        agent_name = result.system_prompt
        # 1. Update agent.yml configuration
        self._update_agent_yml(result)

        # 2. Create the .j2 prompt template file
        self._create_prompt_template(result)
        is_created = not data
        console.print(f"[bold green]Successfully {'create' if is_created else 'update'} agent '{agent_name}'.[/]")
