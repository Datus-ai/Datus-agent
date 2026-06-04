# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Non-REPL handler for the ``datus upgrade`` / ``datus update`` subcommand.

Lists the ``datus-*`` packages installed in the active interpreter and
upgrades them to the latest release in a single ``uv``/``pip`` run. Lives
outside the REPL so it works even when the installed CLI is broken — the
``main()`` entry point intercepts the subcommand before building the app.
"""

from __future__ import annotations

import argparse
from typing import List

from rich.console import Console

from datus.cli import upgrade_service as svc
from datus.cli.cli_styles import print_error, print_info, print_status, print_warning
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="datus upgrade",
        description="Upgrade datus-agent and installed datus-* adapter packages to the latest version.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report the installed packages and the latest datus-agent version; do not install.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Reserved for non-interactive use; currently a no-op (the command never prompts).",
    )
    return parser.parse_args(argv)


def _print_installed(console: Console, packages: List[svc.DatusPackage]) -> None:
    print_info(console, "Installed datus packages:")
    for pkg in packages:
        suffix = "  [dim](editable/source — will be skipped)[/]" if pkg.editable else ""
        console.print(f"  [cyan]{pkg.name}[/] {pkg.version}{suffix}")


def run_upgrade_command(argv: List[str]) -> int:
    """Entry point for ``datus upgrade``. Returns a process exit code."""
    args = _parse_args(argv)
    console = Console()

    packages = svc.enumerate_datus_packages()
    _print_installed(console, packages)

    if args.check:
        latest = svc.get_latest_version(force=True)
        current = next((p.version for p in packages if p.name == svc.DISTRIBUTION_NAME), "")
        if latest is None:
            print_warning(console, "Could not reach PyPI to determine the latest datus-agent version.")
        elif latest == current:
            print_info(console, f"datus-agent is up to date ({current}).")
        else:
            newer = svc.newer_version_available(current) if current else latest
            if newer:
                print_warning(console, f"A newer datus-agent is available: {current} -> {latest}. Run `datus upgrade`.")
            else:
                print_info(console, f"Latest datus-agent on PyPI is {latest} (installed {current}).")
        return 0

    upgradable = [p for p in packages if not p.editable]
    if not upgradable:
        print_warning(
            console,
            "All datus packages are editable/source installs; nothing to upgrade via pip. "
            "Update your checkout with git instead.",
        )
        return 0

    print_info(console, f"Upgrading {len(upgradable)} package(s) to the latest version...")
    result = svc.upgrade_packages(packages)

    if result.skipped_editable:
        print_warning(console, "Skipped editable/source installs: " + ", ".join(result.skipped_editable))

    if result.ok:
        # Clear the version-check cache so the startup hint clears next launch.
        try:
            svc._write_cache(svc._cache_path(), None)
        except Exception:  # pragma: no cover - cache is best-effort
            pass
        if not result.changes:
            print_status(console, "All datus packages are already up to date.", ok=True)
            return 0
        print_status(console, "Upgrade complete.", ok=True)
        for name, old, new in result.changes:
            console.print(f"  [green]{name}[/] {old} -> {new}")
        print_info(console, "Restart datus for the new version(s) to take effect.")
        return 0

    print_status(console, "Upgrade failed.", ok=False)
    tail = (result.stderr or result.stdout or "").strip().splitlines()[-10:]
    for line in tail:
        console.print(f"  [dim]{line}[/]")
    print_error(console, result.error or "unknown error")
    return 1


__all__ = ["run_upgrade_command"]
