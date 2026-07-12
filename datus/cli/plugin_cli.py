# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Non-REPL handler for the ``datus plugin`` management subcommand.

``datus plugin install|uninstall|list|info|enable|disable`` manages the
installed ``datus.plugins`` packages and this project's activation. Handled
outside the REPL (like ``datus upgrade``) so it works even when a plugin is
misconfigured, and so it is never gated by a plugin's own ``enabled`` flag —
you must be able to run ``datus plugin enable`` on a disabled plugin.

Install sources: a PyPI requirement (``datus-foo-plugin``), a wheel
(``./dist/foo-1.0-py3-none-any.whl``), or a local directory (``./foo``).
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from rich.console import Console

from datus.cli import plugin_service as svc
from datus.cli._render_utils import build_row_table
from datus.cli.cli_styles import (
    SYM_CHECK,
    SYM_CROSS,
    print_error,
    print_info,
    print_status,
    print_success,
    print_warning,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datus plugin",
        description="Manage installed datus plugins and their per-project activation.",
    )
    sub = parser.add_subparsers(dest="command")

    p_install = sub.add_parser("install", help="Install a plugin from PyPI, a wheel, or a local directory.")
    p_install.add_argument("source", help="PyPI requirement, path to a .whl, or a local project directory.")
    p_install.add_argument(
        "-e", "--editable", action="store_true", help="Editable install (only for a local source tree)."
    )

    p_uninstall = sub.add_parser("uninstall", help="Uninstall the package registering a plugin.")
    p_uninstall.add_argument("name", help="Plugin name (the `datus <name>` subcommand).")

    sub.add_parser("list", help="List installed plugins with their configured profiles and activation.")

    p_info = sub.add_parser("info", help="Show details for one installed plugin.")
    p_info.add_argument("name", help="Plugin name.")

    p_enable = sub.add_parser("enable", help="Activate a plugin for this project.")
    p_enable.add_argument("name", help="Plugin name.")
    p_enable.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        help="Restrict to this profile (repeatable). Omit to activate all profiles.",
    )

    p_disable = sub.add_parser("disable", help="Deactivate a plugin for this project.")
    p_disable.add_argument("name", help="Plugin name.")

    return parser


def _load_agent_config(console: Console):
    """Best-effort agent config load; returns ``None`` (with a note) on failure."""
    try:
        from datus.configuration.agent_config_loader import load_agent_config

        return load_agent_config()
    except Exception as exc:  # noqa: BLE001 - listing/enable should degrade, not crash
        print_warning(console, f"Could not load agent config: {exc}")
        return None


def run_plugin_command(argv: List[str]) -> int:
    """Entry point for ``datus plugin``. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    console = Console()

    if not args.command:
        parser.print_help()
        return 0
    if args.command == "install":
        return _cmd_install(console, args)
    if args.command == "uninstall":
        return _cmd_uninstall(console, args)
    if args.command == "list":
        return _cmd_list(console)
    if args.command == "info":
        return _cmd_info(console, args)
    if args.command in ("enable", "disable"):
        return _cmd_activation(console, args)
    parser.print_help()
    return 2


def _cmd_install(console: Console, args: argparse.Namespace) -> int:
    print_info(console, f"Installing plugin from {args.source} ...")
    result = svc.install(args.source, editable=args.editable)
    if not result.ok:
        print_status(console, "Install failed.", ok=False)
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-10:]
        for line in tail:
            console.print(f"  [dim]{line}[/]")
        print_error(console, result.error or "unknown error")
        return 1
    if result.new_plugins:
        print_status(console, f"Installed. New plugin(s): {', '.join(result.new_plugins)}", ok=True)
        print_info(console, "Configure a profile with `/plugins` or activate with `datus plugin enable <name>`.")
    else:
        print_warning(
            console,
            "Install succeeded but the package registered no `datus.plugins` entry point "
            "— it may not be a datus plugin.",
        )
    return 0


def _cmd_uninstall(console: Console, args: argparse.Namespace) -> int:
    result = svc.uninstall(args.name)
    if not result.ok:
        print_error(console, result.error or "uninstall failed")
        return 1
    print_status(console, f"Uninstalled plugin `{result.plugin}` (package {result.package}).", ok=True)
    return 0


def _cmd_list(console: Console) -> int:
    agent_config = _load_agent_config(console)
    plugins = svc.list_plugins(agent_config)
    if not plugins:
        print_info(console, "No plugins installed. Install one with `datus plugin install <source>`.")
        return 0
    rows = []
    for p in plugins:
        if p.active is None:
            active = "-"
        elif p.active:
            active = SYM_CHECK + ("" if p.active_profiles is None else f" ({', '.join(p.active_profiles)})")
        else:
            active = SYM_CROSS
        rows.append(
            {
                "name": p.name,
                "package": p.package or "-",
                "version": p.version or "-",
                "profiles": ", ".join(p.profiles) if p.profiles else "-",
                "active": active,
            }
        )
    table = build_row_table(rows, title="Installed plugins")
    if table is not None:
        console.print(table)
    return 0


def _cmd_info(console: Console, args: argparse.Namespace) -> int:
    agent_config = _load_agent_config(console)
    plugins = {p.name: p for p in svc.list_plugins(agent_config)}
    info = plugins.get(args.name)
    if info is None:
        print_error(console, f"No installed plugin named `{args.name}`.")
        return 1

    from datus.plugins.registry import plugin_config_schema

    console.print(f"[bold]{info.name}[/]")
    console.print(f"  package: {info.package or '-'} {info.version}")
    console.print(f"  entry:   {info.entry or '-'}")
    if info.active is not None:
        state = "active" if info.active else "inactive"
        scope = "all profiles" if info.active_profiles is None else ", ".join(info.active_profiles) or "none"
        console.print(f"  project: {state} ({scope})")
    console.print(f"  profiles configured: {', '.join(info.profiles) if info.profiles else '(none)'}")

    schema = plugin_config_schema(info.name)
    if schema:
        console.print("  config schema:")
        for field_spec in schema:
            flags = []
            if field_spec.get("required"):
                flags.append("required")
            if field_spec.get("secret"):
                flags.append("secret")
            suffix = f" ({', '.join(flags)})" if flags else ""
            console.print(f"    - {field_spec['name']}{suffix}: {field_spec.get('description', '')}", markup=False)
    return 0


def _cmd_activation(console: Console, args: argparse.Namespace) -> int:
    agent_config = _load_agent_config(console)
    if agent_config is None:
        print_error(console, "Cannot change activation without an agent config.")
        return 3

    from datus.plugins.registry import plugin_entry_point_exists

    if not plugin_entry_point_exists(args.name):
        print_error(console, f"No installed plugin named `{args.name}`. Run `datus plugin list`.")
        return 1

    enable = args.command == "enable"
    profiles: Optional[List[str]] = getattr(args, "profiles", None) if enable else None
    try:
        agent_config.set_plugin_activation(
            args.name,
            enabled=enable,
            active_profiles=profiles,
            clear_profiles=enable and not profiles,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean error
        print_error(console, f"Failed to update activation: {exc}")
        return 1

    if enable:
        scope = "all profiles" if not profiles else ", ".join(profiles)
        print_success(console, f"Activated `{args.name}` for this project ({scope}).")
    else:
        print_success(console, f"Deactivated `{args.name}` for this project.")
    return 0


__all__ = ["run_plugin_command"]
