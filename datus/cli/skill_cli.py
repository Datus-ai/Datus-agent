# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Non-REPL skill CLI handler for `datus skill <subcommand>` usage.
"""

import getpass

import httpx
from rich.console import Console
from rich.table import Table

from datus.tools.skill_tools.marketplace_auth import clear_token, save_token
from datus.tools.skill_tools.skill_config import SkillConfig
from datus.tools.skill_tools.skill_manager import SkillManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)
console = Console()


def _get_manager(args) -> SkillManager:
    """Create a SkillManager from CLI args."""
    config_kwargs = {}
    if getattr(args, "marketplace", None):
        config_kwargs["marketplace_url"] = args.marketplace

    config = SkillConfig(**config_kwargs) if config_kwargs else SkillConfig()
    return SkillManager(config=config)


def run_skill_command(args) -> int:
    """Dispatch skill subcommand."""
    subcmd = args.subcommand
    skill_args = getattr(args, "skill_args", [])
    manager = _get_manager(args)

    if subcmd == "login":
        marketplace_url = getattr(args, "marketplace", "") or manager.config.marketplace_url
        _cmd_login(marketplace_url, args)
    elif subcmd == "logout":
        marketplace_url = getattr(args, "marketplace", "") or manager.config.marketplace_url
        _cmd_logout(marketplace_url)
    elif subcmd == "list":
        _cmd_list(manager)
    elif subcmd == "search":
        query = " ".join(skill_args) if skill_args else ""
        _cmd_search(manager, query)
    elif subcmd == "install":
        if not skill_args:
            console.print("[red]Usage: datus skill install <name> [version][/]")
            return 1
        name = skill_args[0]
        version = skill_args[1] if len(skill_args) > 1 else "latest"
        _cmd_install(manager, name, version)
    elif subcmd == "publish":
        if not skill_args:
            console.print("[red]Usage: datus skill publish <path>[/]")
            return 1
        path = skill_args[0]
        owner = getattr(args, "owner", "")
        _cmd_publish(manager, path, owner)
    elif subcmd == "info":
        if not skill_args:
            console.print("[red]Usage: datus skill info <name>[/]")
            return 1
        _cmd_info(manager, skill_args[0])
    elif subcmd == "update":
        _cmd_update(manager)
    elif subcmd == "remove":
        if not skill_args:
            console.print("[red]Usage: datus skill remove <name>[/]")
            return 1
        _cmd_remove(manager, skill_args[0])

    return 0


def _cmd_login(marketplace_url: str, args) -> None:
    """Authenticate with the Town Marketplace and save the JWT token."""
    email = getattr(args, "email", None) or input("Email: ")
    password = getattr(args, "password", None) or getpass.getpass("Password: ")

    login_url = f"{marketplace_url.rstrip('/')}/api/auth/login"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(login_url, json={"email": email, "password": password})
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                console.print(f"[red]Login failed ({resp.status_code}): {detail}[/]")
                return

            # Token is returned as a cookie named 'town_token'
            token = resp.cookies.get("town_token")
            if not token:
                # Fallback: check JSON body
                body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                token = body.get("access_token") or body.get("token")
            if not token:
                console.print("[red]Login succeeded but no token was returned.[/]")
                return

            save_token(token, marketplace_url, email)
            console.print(f"[green]Login successful![/] Token saved for {marketplace_url}")
    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to {login_url}[/]")
    except Exception as exc:
        console.print(f"[red]Login error: {exc}[/]")


def _cmd_logout(marketplace_url: str) -> None:
    """Clear saved credentials for the marketplace."""
    if clear_token(marketplace_url):
        console.print(f"[green]Logged out from {marketplace_url}[/]")
    else:
        console.print(f"[yellow]No saved credentials for {marketplace_url}[/]")


def _cmd_list(manager: SkillManager):
    skills = manager.list_all_skills()
    if not skills:
        console.print("[yellow]No skills installed locally.[/]")
        return

    table = Table(title="Installed Skills", show_header=True, header_style="bold green")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Source")
    table.add_column("Tags")

    for s in skills:
        table.add_row(
            s.name,
            s.version or "-",
            s.source or "local",
            ", ".join(s.tags) if s.tags else "",
        )
    console.print(table)


def _cmd_search(manager: SkillManager, query: str):
    if not query:
        console.print("[yellow]Usage: datus skill search <query>[/]")
        return
    console.print(f"[dim]Searching for '{query}'...[/]")
    results = manager.search_marketplace(query=query)
    if not results:
        console.print("[yellow]No results.[/]")
        return
    for s in results:
        console.print(f"  [cyan]{s.get('name')}[/] v{s.get('latest_version', '?')} — {s.get('description', '')}")


def _cmd_install(manager: SkillManager, name: str, version: str):
    console.print(f"[dim]Installing {name}@{version}...[/]")
    ok, msg = manager.install_from_marketplace(name, version)
    if ok:
        console.print(f"[green]{msg}[/]")
    else:
        console.print(f"[red]{msg}[/]")


def _cmd_publish(manager: SkillManager, path: str, owner: str):
    console.print(f"[dim]Publishing from {path}...[/]")
    ok, msg = manager.publish_to_marketplace(path, owner=owner)
    if ok:
        console.print(f"[green]{msg}[/]")
    else:
        console.print(f"[red]{msg}[/]")


def _cmd_info(manager: SkillManager, name: str):
    local = manager.get_skill(name)
    if local:
        console.print(f"[bold]Local:[/] {local.name} v{local.version or '?'} ({local.source or 'local'})")
        console.print(f"  {local.description}")
    try:
        client = manager._get_marketplace_client()
        remote = client.get_skill_info(name)
        console.print(f"[bold]Marketplace:[/] {remote['name']} v{remote.get('latest_version', '?')}")
        console.print(f"  Owner: {remote.get('owner', '-')}  Promoted: {remote.get('promoted', False)}")
    except Exception:
        if not local:
            console.print(f"[yellow]Skill '{name}' not found.[/]")


def _cmd_update(manager: SkillManager):
    skills = [s for s in manager.list_all_skills() if s.source == "marketplace"]
    if not skills:
        console.print("[yellow]No marketplace skills to update.[/]")
        return
    for s in skills:
        console.print(f"[dim]Checking {s.name}...[/]")
        ok, msg = manager.install_from_marketplace(s.name)
        if ok:
            console.print("  [green]Updated[/]")
        else:
            console.print(f"  [red]Failed to update {s.name}: {msg}[/]")


def _cmd_remove(manager: SkillManager, name: str):
    if manager.registry.remove_skill(name):
        console.print(f"[green]Removed {name}[/]")
    else:
        console.print(f"[yellow]Skill '{name}' not found.[/]")
