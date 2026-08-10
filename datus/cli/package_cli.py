# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``datus package`` — interactive wizard for exporting a self-contained project zip.

All parameters are collected interactively (per the design decision in
``DatusPackage-review.md`` §3): the only flag is ``-y/--yes``, which skips
the wizard and packages everything with defaults — the non-TTY / scripting
escape hatch. Pure build logic lives in :mod:`datus.cli.package_builder`;
this module owns argparse, the step flow, and console output only.

Exit codes: ``0`` success, ``1`` build failure or user abort, ``2`` usage
error or non-interactive terminal without ``--yes``, ``3`` no agent config.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from rich.console import Console

if TYPE_CHECKING:
    from datus.cli.package_builder import PackageOptions, PackageResult

from datus.cli._cli_utils import confirm_prompt, prompt_input, select_choice, select_multi_choice
from datus.cli.cli_styles import print_error, print_info, print_status, print_warning


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datus package",
        description="Pack this project into a self-contained zip (interactive).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="skip the wizard: package everything with defaults (for scripts / non-TTY)",
    )
    return parser


def _is_interactive() -> bool:
    """Both streams must be TTYs — mirrors ``service_bootstrap._is_interactive``."""
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, OSError):
        return False


def run_package_command(argv: List[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    console = Console()

    from datus.cli import package_builder as pb

    raw = pb.load_raw_agent_config()
    if raw is None:
        print_error(console, "No agent configuration found (conf/agent.yml or ~/.datus/conf/agent.yml).")
        return 3

    root = Path.cwd()
    if args.yes:
        options = pb.PackageOptions(root=root, assume_yes=True)
    else:
        if not _is_interactive():
            print_error(console, "datus package is interactive; run it in a terminal or pass --yes for defaults.")
            return 2
        try:
            options = _run_wizard(console, pb, raw, root)
        except KeyboardInterrupt:
            print_warning(console, "Aborted.")
            return 1
        if options is None:
            print_warning(console, "Aborted.")
            return 1

    result = pb.build_package(options)
    return _report_result(console, result)


# --------------------------------------------------------------------------- #
# Wizard                                                                      #
# --------------------------------------------------------------------------- #


def _run_wizard(console: Console, pb: ModuleType, raw: Dict, root: Path) -> Optional["PackageOptions"]:
    """Linear step flow; returns ``PackageOptions`` or ``None`` on abort."""
    project_name = pb.resolve_effective_project_name(root, raw)
    print_info(console, f"Packaging project {project_name!r} from {root}")

    output = _step_output_path(console, pb, root, project_name)
    if output is None:
        return None

    include, exclude = _step_file_scope(console)

    subagents = _step_multi(
        console,
        "Subagents",
        {name: f"{name} — {desc}" if desc else name for name, desc in pb.list_subagents(raw).items()},
    )
    skill_dirs = pb.list_skills(root)
    skills = _step_multi(
        console,
        "Skills",
        {name: f"{name} ({'project' if _is_under(path, root) else 'global'})" for name, path in skill_dirs.items()},
    )
    metrics = _step_multi(
        console,
        "Metric datasources",
        {ds: f"subject/semantic_models/{ds}" for ds in pb.list_metric_datasources(root)},
    )
    # Reference SQL: selection drives the receiver's KB rebuild, not staging —
    # every .sql directory ships either way. Preselect only the conventional
    # corpora so a migrations/init directory isn't bootstrapped by accident.
    reference_sql_dirs = pb.list_reference_sql_dirs(root)
    reference_sql = _step_multi(
        console,
        "Reference SQL to rebuild into the receiver's KB",
        {name: f"{name}/ (files ship regardless; selecting adds a bootstrap-kb step)" for name in reference_sql_dirs},
        default_selected=pb.select_reference_sql(root, None),
        allow_empty=True,
    )
    plugins = _step_multi(console, "Plugins", pb.list_packageable_plugins(root))
    reports = _step_multi(
        console,
        "Reports",
        {slug: f"reports/{slug}" for slug in pb.list_artifact_slugs(root, "report")},
    )
    dashboards = _step_multi(
        console,
        "Dashboards",
        {slug: f"dashboards/{slug}" for slug in pb.list_artifact_slugs(root, "dashboard")},
    )
    report_dist = _step_report_dist(console, raw) if reports else None

    options = pb.PackageOptions(
        root=root,
        output=output,
        include=tuple(include),
        exclude=tuple(exclude),
        subagents=tuple(subagents),
        skills=tuple(skills),
        metrics=tuple(metrics),
        reference_sql=tuple(reference_sql),
        plugins=tuple(plugins),
        reports=tuple(reports),
        dashboards=tuple(dashboards),
        report_dist=report_dist,
    )
    if not _step_summary(console, options, project_name):
        return None
    return options


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _step_output_path(console: Console, pb: ModuleType, root: Path, project_name: str) -> Optional[Path]:
    default = str(pb.default_output_path(root, project_name))
    while True:
        answer = prompt_input(console, "Output zip path", default=default).strip()
        if not answer:
            answer = default
        candidate = Path(answer).expanduser()
        if candidate.suffix != ".zip":
            print_warning(console, "Output path must end with .zip")
            continue
        if candidate.exists() and not confirm_prompt(console, f"{candidate} exists — overwrite?", default=False):
            continue
        return candidate


def _step_file_scope(console: Console) -> Tuple[List[str], List[str]]:
    if confirm_prompt(console, "Package all project files (recommended)?", default=True):
        return [], []
    include = _prompt_patterns(console, "Include regex patterns (comma-separated, empty = all)")
    exclude = _prompt_patterns(console, "Exclude regex patterns (comma-separated, empty = none)")
    return include, exclude


def _prompt_patterns(console: Console, message: str) -> List[str]:
    while True:
        answer = prompt_input(console, message, default="").strip()
        if not answer:
            return []
        patterns = [part.strip() for part in answer.split(",") if part.strip()]
        try:
            for pattern in patterns:
                re.compile(pattern)
        except re.error as exc:
            print_warning(console, f"Invalid regex {pattern!r}: {exc}")
            continue
        return patterns


def _step_multi(
    console: Console,
    label: str,
    choices: Dict[str, str],
    *,
    default_selected: Optional[List[str]] = None,
    allow_empty: bool = False,
) -> List[str]:
    """One multi-select screen.

    Empty categories are skipped silently. An empty selection (which is
    also what Ctrl+C produces) gets an explicit confirmation so a stray
    interrupt can't silently drop a whole category; declining re-runs
    the step. ``allow_empty`` suppresses that guard for steps where
    selecting nothing is a normal, lossless choice.
    """
    if not choices:
        return []
    preselected = list(choices) if default_selected is None else [c for c in choices if c in default_selected]
    while True:
        print_info(console, f"{label}: Space toggles, 'a' toggles all, Enter confirms")
        selected = select_multi_choice(console, choices, default_selected=preselected)
        if selected or allow_empty:
            return selected
        if confirm_prompt(console, f"Package no {label.lower()} at all — continue?", default=False):
            return []


def _step_report_dist(console: Console, raw: Dict) -> Optional[Path]:
    choice = select_choice(
        console,
        {
            "cdn": "No local dist — report index.html loads its viewer from the CDN (default)",
            "dist": "Bundle the web-artifact-render dist so index.html opens via file://",
        },
        default="cdn",
    )
    if choice != "dist":
        return None
    from datus.agent.node.visual_artifact._artifact_html_renderer import _resolve_dist

    configured = ""
    nodes = raw.get("agentic_nodes")
    if isinstance(nodes, dict):
        gen_report = nodes.get("gen_visual_report")
        if isinstance(gen_report, dict) and isinstance(gen_report.get("report_dist"), str):
            configured = gen_report["report_dist"]
    while True:
        answer = prompt_input(
            console, "Path to the web-artifact-render dist directory (empty = skip)", default=configured
        ).strip()
        if not answer:
            print_info(console, "Skipping the local dist; reports will use the CDN.")
            return None
        resolved = _resolve_dist(Path(answer))
        if resolved is None:
            print_warning(console, f"{answer} is not a valid dist (needs index.css + index.umd.js)")
            continue
        return resolved


def _step_summary(console: Console, options: "PackageOptions", project_name: str) -> bool:
    from datus.cli._render_utils import build_row_table

    def _fmt(values: Optional[Tuple[str, ...]]) -> str:
        if values is None:
            return "(all)"
        return ", ".join(values) if values else "(none)"

    rows = [
        {"item": "Project", "value": project_name},
        {"item": "Output", "value": str(options.output)},
        {"item": "Include patterns", "value": ", ".join(options.include) or "(all files)"},
        {"item": "Exclude patterns", "value": ", ".join(options.exclude) or "(none)"},
        {"item": "Subagents", "value": _fmt(options.subagents)},
        {"item": "Skills", "value": _fmt(options.skills)},
        {"item": "Metric datasources", "value": _fmt(options.metrics)},
        {"item": "Reference SQL", "value": _fmt(options.reference_sql)},
        {"item": "Plugins", "value": _fmt(options.plugins)},
        {"item": "Reports", "value": _fmt(options.reports)},
        {"item": "Dashboards", "value": _fmt(options.dashboards)},
        {"item": "Report dist", "value": str(options.report_dist) if options.report_dist else "CDN (not bundled)"},
    ]
    table = build_row_table(rows, title="Package summary", columns=[("item", "Item"), ("value", "Value")])
    if table is not None:
        console.print(table)
    return confirm_prompt(console, "Build the package now?", default=True)


# --------------------------------------------------------------------------- #
# Result reporting                                                            #
# --------------------------------------------------------------------------- #


def _report_result(console: Console, result: "PackageResult") -> int:
    for warning in result.warnings:
        print_warning(console, warning)
    if not result.ok:
        print_status(console, "Package build failed.", ok=False)
        for finding in result.secret_findings:
            print_error(
                console, f"secret detected: {finding.arcname} ({finding.locator}, {finding.kind})", prefix=False
            )
        if result.secret_findings:
            print_info(
                console,
                "Remove the credential from the source config (use ${VAR} placeholders) or exclude the file, then retry.",
            )
        if result.error:
            print_error(console, result.error)
        return 1

    print_status(console, f"Package built: {result.zip_path}", ok=True)
    print_info(console, f"{result.file_count} files, {result.total_bytes / (1024 * 1024):.1f} MB uncompressed")
    env_vars = sorted({binding.var for binding in result.env_vars})
    if env_vars:
        print_info(console, "Receiver must export: " + ", ".join(env_vars))
        print_info(console, "The generated README.md lists where each variable is used.")
    return 0


__all__ = ["run_package_command"]
