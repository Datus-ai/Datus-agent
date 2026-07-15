# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Non-REPL handler for the ``datus plugin`` management subcommand.

``datus plugin install|uninstall|list|info|enable|disable|pack|export|upgrade``
manages plugins installed under ``~/.datus/plugins/`` and this project's
activation. Handled outside the REPL (like ``datus upgrade``) so it works even
when a plugin is misconfigured, and so it is never gated by a plugin's own
``enabled`` flag — you must be able to run ``datus plugin enable`` on a disabled
plugin.

Install sources use a ``{type}:{src}`` prefix; the type is optional and defaults
to ``pip``: ``pip:`` (PyPI requirement, the default when no prefix is given),
``src:`` (local project directory), ``whl:`` (local wheel), ``git:`` (git
repository), or ``zip:`` (offline wheelhouse bundle built by ``datus plugin
pack``). ``export`` reproduces a distributable ``.zip`` from an installed
plugin, and ``upgrade`` re-fetches it via its original method.
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

    p_install = sub.add_parser("install", help="Install a plugin from a '[{type}:]{src}' source.")
    p_install.add_argument(
        "source",
        help="'[{type}:]{src}' — <req> or pip:<req> (default) / src:<dir> / whl:<file.whl> / "
        "git:<url> / zip:<bundle.zip>.",
    )
    p_install.add_argument(
        "--force",
        action="store_true",
        help="Replace an already-installed plugin (and skip a bundle's python/platform check).",
    )

    p_pack = sub.add_parser("pack", help="Build a distributable wheelhouse .zip from a plugin source. Needs network.")
    p_pack.add_argument("source", nargs="?", default=".", help="Plugin project directory (default: current dir).")
    p_pack.add_argument(
        "--with-deps",
        action="store_true",
        dest="with_deps",
        help="Bundle every dependency wheel for a fully offline install (default: plugin wheel only).",
    )
    p_pack.add_argument("-o", "--output", default=".", help="Output directory for the .zip (default: current dir).")

    p_export = sub.add_parser("export", help="Export an installed plugin as a distributable .zip.")
    p_export.add_argument("name", help="Plugin name (the `datus <name>` subcommand).")
    p_export.add_argument("-o", "--output", default=".", help="Output directory for the .zip (default: current dir).")

    p_upgrade = sub.add_parser("upgrade", help="Re-install a plugin from its recorded source.")
    p_upgrade.add_argument("name", help="Plugin name.")

    p_uninstall = sub.add_parser("uninstall", help="Remove an installed plugin's directory.")
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
    if args.command == "pack":
        return _cmd_pack(console, args)
    if args.command == "export":
        return _cmd_export(console, args)
    if args.command == "upgrade":
        return _cmd_upgrade(console, args)
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


def _print_tail(console: Console, result) -> None:
    """Print the last few lines of a failed subprocess for context.

    Subprocess output is untrusted (it may contain Rich markup tags that would
    otherwise alter or hide terminal output), so it is rendered with markup
    disabled and the dim style applied through the style argument.
    """
    tail = (getattr(result, "stderr", "") or getattr(result, "stdout", "") or "").strip().splitlines()[-10:]
    for line in tail:
        console.print(f"  {line}", style="dim", markup=False)


def _cmd_install(console: Console, args: argparse.Namespace) -> int:
    print_info(console, f"Installing plugin from {args.source} ...")
    result = svc.install(args.source, force=args.force)
    if not result.ok:
        print_status(console, "Install failed.", ok=False)
        _print_tail(console, result)
        print_error(console, result.error or "unknown error")
        return 1
    print_status(console, f"Installed `{result.name}` {result.version} into {result.plugin_dir}.", ok=True)
    print_info(console, f"Configure a profile with `/plugins` or activate with `datus plugin enable {result.name}`.")
    return 0


def _cmd_pack(console: Console, args: argparse.Namespace) -> int:
    from datus.cli import plugin_pack

    scope = "with dependencies" if args.with_deps else "plugin wheel only"
    print_info(console, f"Packing plugin bundle from {args.source} ({scope}) ... (requires network)")
    result = plugin_pack.pack(args.source, out_dir=args.output, with_deps=args.with_deps)
    if not result.ok:
        print_status(console, "Pack failed.", ok=False)
        _print_tail(console, result)
        print_error(console, result.error or "unknown error")
        return 1
    print_status(
        console,
        f"Built {result.bundle_path} ({result.wheel_count} wheel(s), plugin `{result.plugin_name}`).",
        ok=True,
    )
    print_info(console, "Install it with `datus plugin install zip:<bundle>.zip`.")
    return 0


def _cmd_export(console: Console, args: argparse.Namespace) -> int:
    print_info(console, f"Exporting plugin `{args.name}` ...")
    result = svc.export(args.name, out_dir=args.output)
    if not result.ok:
        print_error(console, result.error or "export failed")
        return 1
    print_status(console, f"Exported `{result.name}` to {result.out_path}.", ok=True)
    print_info(console, "Install it elsewhere with `datus plugin install zip:<bundle>.zip`.")
    return 0


def _cmd_upgrade(console: Console, args: argparse.Namespace) -> int:
    print_info(console, f"Upgrading plugin `{args.name}` ...")
    result = svc.upgrade(args.name)
    if not result.ok:
        print_status(console, "Upgrade failed.", ok=False)
        _print_tail(console, result)
        print_error(console, result.error or "unknown error")
        return 1
    if result.old_version and result.new_version and result.old_version != result.new_version:
        print_status(console, f"Upgraded `{result.name}` {result.old_version} → {result.new_version}.", ok=True)
    else:
        print_status(console, f"Re-installed `{result.name}` ({result.new_version or 'unchanged'}).", ok=True)
    return 0


def _cmd_uninstall(console: Console, args: argparse.Namespace) -> int:
    result = svc.uninstall(args.name)
    if not result.ok:
        print_error(console, result.error or "uninstall failed")
        return 1
    suffix = f" (package {result.package})" if result.package else ""
    print_status(console, f"Uninstalled plugin `{result.plugin}`{suffix}.", ok=True)
    return 0


def _cmd_list(console: Console) -> int:
    agent_config = _load_agent_config(console)
    plugins = svc.list_plugins(agent_config)
    if not plugins:
        print_info(console, "No plugins installed. Install one with `datus plugin install '{type}:{src}'`.")
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
                "source": p.install_type or p.source or "-",
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

    from datus.plugins.registry import load_plugin_manifest, plugin_config_schema

    console.print(f"[bold]{info.name}[/]")
    manifest = load_plugin_manifest(info.name)
    if manifest is not None and manifest.description:
        console.print(f"  {manifest.description}", markup=False)
    console.print(f"  package: {info.package or '-'} {info.version}")
    console.print(f"  entry:   {info.entry or '-'}")
    source_label = info.install_type or info.source or "-"
    console.print(f"  installed via: {source_label}")
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

    from datus.plugins import store
    from datus.plugins.registry import plugin_entry_point_exists

    # A managed plugin's directory must be on sys.path before the entry-point
    # probe can see it (metadata-only; no plugin code is imported). Plugins
    # mounted through ``agent.plugin_paths`` may equally be inactive (and thus
    # off sys.path), so those directories are injected too — a disabled plugin
    # must stay re-enable-able.
    if store.plugin_dir(args.name).is_dir():
        store.activate_name(args.name)
    else:
        store.activate_paths(getattr(agent_config, "plugin_paths", None))

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
