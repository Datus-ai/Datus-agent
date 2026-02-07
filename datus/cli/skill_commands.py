# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Skill marketplace commands for the Datus CLI.
Follows the same pattern as MCPCommands.
"""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from rich.table import Table

from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


class SkillCommands:
    """Handles all skill marketplace commands."""

    def __init__(self, cli_instance: "DatusCLI"):
        self.cli = cli_instance
        self.console = cli_instance.console

    def _get_skill_manager(self):
        """Get the SkillManager from the agent if available, else create a standalone one."""
        if self.cli.agent and hasattr(self.cli.agent, "skill_manager") and self.cli.agent.skill_manager:
            return self.cli.agent.skill_manager

        # Standalone: create from config
        from datus.tools.skill_tools.skill_config import SkillConfig
        from datus.tools.skill_tools.skill_manager import SkillManager

        skills_conf = {}
        if hasattr(self.cli, "agent_config") and self.cli.agent_config:
            skills_conf = getattr(self.cli.agent_config, "skills", {}) or {}
            if not isinstance(skills_conf, dict):
                skills_conf = {}

        config = SkillConfig.from_dict(skills_conf)
        return SkillManager(config=config)

    def cmd_skill(self, args: str):
        """Dispatch .skill subcommands."""
        args = args.strip()
        if not args or args == "help":
            self._show_usage()
        elif args == "list":
            self.cmd_skill_list()
        elif args.startswith("search"):
            self.cmd_skill_search(args[6:].strip())
        elif args.startswith("install"):
            self.cmd_skill_install(args[7:].strip())
        elif args.startswith("publish"):
            self.cmd_skill_publish(args[7:].strip())
        elif args.startswith("info"):
            self.cmd_skill_info(args[4:].strip())
        elif args.startswith("update"):
            self.cmd_skill_update()
        elif args.startswith("remove"):
            self.cmd_skill_remove(args[6:].strip())
        else:
            self.console.print(f"[red]Unknown skill command: {args}[/red]")
            self._show_usage()

    def cmd_skill_list(self):
        """List locally installed skills in a Rich table."""
        manager = self._get_skill_manager()
        skills = manager.list_all_skills()

        if not skills:
            self.console.print("[yellow]No skills installed locally.[/]")
            return

        table = Table(title="Installed Skills", show_header=True, header_style="bold green")
        table.add_column("Name", style="cyan")
        table.add_column("Version")
        table.add_column("Source")
        table.add_column("Tags")
        table.add_column("Description", max_width=40)

        for skill in skills:
            source = skill.source or "local"
            tags = ", ".join(skill.tags) if skill.tags else ""
            table.add_row(
                skill.name,
                skill.version or "-",
                source,
                tags,
                (skill.description[:37] + "...") if len(skill.description) > 40 else skill.description,
            )

        self.console.print(table)

    def cmd_skill_search(self, query: str):
        """Search marketplace for skills."""
        if not query:
            self.console.print("[yellow]Usage: .skill search <query>[/]")
            return

        manager = self._get_skill_manager()
        self.console.print(f"[dim]Searching marketplace for '{query}'...[/]")

        results = manager.search_marketplace(query=query)

        if not results:
            self.console.print("[yellow]No skills found.[/]")
            return

        table = Table(title=f"Marketplace Results for '{query}'", show_header=True, header_style="bold green")
        table.add_column("Name", style="cyan")
        table.add_column("Version")
        table.add_column("Owner")
        table.add_column("Tags")
        table.add_column("Description", max_width=40)

        for skill in results:
            tags = ", ".join(skill.get("tags", []))
            desc = skill.get("description", "")
            table.add_row(
                skill.get("name", ""),
                skill.get("latest_version", "-"),
                skill.get("owner", "-"),
                tags,
                (desc[:37] + "...") if len(desc) > 40 else desc,
            )

        self.console.print(table)

    def cmd_skill_install(self, args: str):
        """Install skill from marketplace: .skill install <name> [version]"""
        parts = args.strip().split()
        if not parts:
            self.console.print("[yellow]Usage: .skill install <name> [version][/]")
            return

        name = parts[0]
        version = parts[1] if len(parts) > 1 else "latest"

        manager = self._get_skill_manager()
        self.console.print(f"[dim]Installing {name}@{version} from marketplace...[/]")

        ok, msg = manager.install_from_marketplace(name, version)
        if ok:
            self.console.print(f"[bold green]Success:[/] {msg}")
        else:
            self.console.print(f"[bold red]Error:[/] {msg}")

    def cmd_skill_publish(self, args: str):
        """Publish local skill to marketplace: .skill publish <path> [--owner <name>]"""
        parts = args.strip().split()
        if not parts:
            self.console.print("[yellow]Usage: .skill publish <path> [--owner <name>][/]")
            return

        skill_dir = parts[0]
        owner = ""
        if "--owner" in parts:
            idx = parts.index("--owner")
            if idx + 1 < len(parts):
                owner = parts[idx + 1]

        manager = self._get_skill_manager()
        self.console.print(f"[dim]Publishing skill from {skill_dir}...[/]")

        ok, msg = manager.publish_to_marketplace(skill_dir, owner=owner)
        if ok:
            self.console.print(f"[bold green]Success:[/] {msg}")
        else:
            self.console.print(f"[bold red]Error:[/] {msg}")

    def cmd_skill_info(self, name: str):
        """Show skill details (local + remote)."""
        if not name:
            self.console.print("[yellow]Usage: .skill info <name>[/]")
            return

        manager = self._get_skill_manager()

        # Local info
        local_skill = manager.get_skill(name)
        if local_skill:
            self.console.print(f"[bold green]Local Skill:[/] {local_skill.name}")
            self.console.print(f"  Description: {local_skill.description}")
            self.console.print(f"  Version: {local_skill.version or 'unversioned'}")
            self.console.print(f"  Location: {local_skill.location}")
            self.console.print(f"  Tags: {', '.join(local_skill.tags) if local_skill.tags else 'none'}")
            self.console.print(f"  Source: {local_skill.source or 'local'}")
            if local_skill.license:
                self.console.print(f"  License: {local_skill.license}")
        else:
            self.console.print(f"[dim]Not installed locally. Checking marketplace...[/]")

        # Remote info
        try:
            client = manager._get_marketplace_client()
            remote = client.get_skill_info(name)
            self.console.print(f"\n[bold cyan]Marketplace Info:[/] {remote.get('name')}")
            self.console.print(f"  Latest Version: {remote.get('latest_version', '-')}")
            self.console.print(f"  Owner: {remote.get('owner', '-')}")
            self.console.print(f"  Promoted: {remote.get('promoted', False)}")
            self.console.print(f"  Usage Count: {remote.get('usage_count', 0)}")
            versions = remote.get("versions", [])
            if versions:
                self.console.print(f"  Versions: {', '.join(v.get('version', '?') for v in versions)}")
        except Exception:
            if not local_skill:
                self.console.print(f"[yellow]Skill '{name}' not found locally or in marketplace.[/]")

    def cmd_skill_update(self):
        """Update all marketplace-installed skills to latest."""
        manager = self._get_skill_manager()
        skills = manager.list_all_skills()
        marketplace_skills = [s for s in skills if s.source == "marketplace"]

        if not marketplace_skills:
            self.console.print("[yellow]No marketplace-installed skills to update.[/]")
            return

        updated = 0
        for skill in marketplace_skills:
            self.console.print(f"[dim]Checking {skill.name}...[/]")
            try:
                client = manager._get_marketplace_client()
                remote = client.get_skill_info(skill.name)
                remote_version = remote.get("latest_version", "")
                if remote_version and remote_version != skill.version:
                    ok, msg = manager.install_from_marketplace(skill.name, remote_version)
                    if ok:
                        self.console.print(f"  [green]Updated {skill.name} to {remote_version}[/]")
                        updated += 1
                    else:
                        self.console.print(f"  [red]Failed: {msg}[/]")
                else:
                    self.console.print(f"  [dim]Already up to date[/]")
            except Exception as e:
                self.console.print(f"  [red]Error checking {skill.name}: {e}[/]")

        self.console.print(f"\n[bold]{updated} skill(s) updated.[/]")

    def cmd_skill_remove(self, name: str):
        """Remove a locally installed skill."""
        if not name:
            self.console.print("[yellow]Usage: .skill remove <name>[/]")
            return

        manager = self._get_skill_manager()
        skill = manager.get_skill(name)

        if not skill:
            self.console.print(f"[yellow]Skill '{name}' not found locally.[/]")
            return

        # Remove from registry
        removed = manager.registry.remove_skill(name)
        if removed:
            # Optionally delete files if marketplace-installed
            if skill.source == "marketplace":
                skill_path = skill.location
                if skill_path.exists():
                    shutil.rmtree(str(skill_path), ignore_errors=True)
                    self.console.print(f"[dim]Deleted files at {skill_path}[/]")
            self.console.print(f"[bold green]Removed skill '{name}'[/]")
        else:
            self.console.print(f"[bold red]Failed to remove skill '{name}'[/]")

    def _show_usage(self):
        """Show .skill command usage."""
        self.console.print("[bold]Skill Marketplace Commands:[/]")
        cmds = [
            (".skill list", "List locally installed skills"),
            (".skill search <query>", "Search skills in marketplace"),
            (".skill install <name> [version]", "Install skill from marketplace"),
            (".skill publish <path>", "Publish local skill to marketplace"),
            (".skill info <name>", "Show skill details"),
            (".skill update", "Update all marketplace skills to latest"),
            (".skill remove <name>", "Remove a locally installed skill"),
        ]
        for cmd, desc in cmds:
            self.console.print(f"  {cmd:<35} {desc}")
