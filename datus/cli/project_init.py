# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""First-run wizard that produces ``./.datus/config.yml``.

Runs only in REPL mode when the project-level overlay does not yet
exist.  The wizard is intentionally minimal: it asks the user to pick
one of the LLM models already defined in the shared ``agent.yml`` and
one of the datasources under ``agent.services.datasources``, plus an
optional ``project_name``.  Everything else (provider configuration,
API keys, datasource URIs, etc.) still lives in the base file — this
overlay is just "which one do I want for this project".
"""

import os
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt

from datus.cli._cli_utils import select_choice
from datus.configuration.agent_config import (
    AgentConfig,
    _normalize_project_name,
    _validate_project_name,
)
from datus.configuration.project_config import (
    ProjectOverride,
    project_config_path,
    save_project_override,
)
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def run_project_init(base_config: AgentConfig, cwd: Optional[str] = None) -> ProjectOverride:
    """Interactively collect the three override values and persist them.

    The wizard refuses to run when the base ``agent.yml`` is missing
    models or datasources — there is nothing meaningful for the user to
    pick in that case, and silently writing an empty overlay would mask
    the real problem (a broken global config).
    """
    console = Console()

    if not base_config.models:
        raise DatusException(
            code=ErrorCode.COMMON_CONFIG_ERROR,
            message_args={
                "config_error": (
                    "Base agent.yml has no 'agent.models' defined. Use /model inside the CLI to set up LLM providers."
                )
            },
        )
    if not base_config.services.datasources:
        raise DatusException(
            code=ErrorCode.COMMON_CONFIG_ERROR,
            message_args={
                "config_error": (
                    "Base agent.yml has no 'agent.services.datasources' defined. "
                    "Use /datasource inside the CLI to add at least one datasource."
                )
            },
        )

    console.print()
    console.print("[bold cyan]First-run project setup[/]")
    console.print(f"[dim]Writing project-level overrides to {project_config_path(cwd)}[/]")
    console.print(
        "[dim]This file pins target / default_datasource / project_name for this project; "
        "everything else comes from the shared agent.yml.[/]"
    )
    console.print()

    console.print("[bold]- Select LLM model (from agent.yml):[/]")
    model_choices = {name: name for name in base_config.models.keys()}
    target_default = base_config.target if base_config.target in model_choices else next(iter(model_choices))
    target = select_choice(console, model_choices, default=target_default)
    if not target:
        target = target_default

    console.print()
    console.print("[bold]- Select default datasource (from agent.yml):[/]")
    db_choices = {name: f"{name}  ({cfg.type})" for name, cfg in base_config.services.datasources.items()}
    db_default = (
        base_config.services.default_datasource
        if base_config.services.default_datasource in db_choices
        else next(iter(db_choices))
    )
    default_datasource = select_choice(console, db_choices, default=db_default)
    if not default_datasource:
        default_datasource = db_default

    console.print()
    project_name_default = _normalize_project_name(cwd or os.getcwd())
    while True:
        raw_project_name = Prompt.ask(
            "- Project name (used for sessions/data shard)",
            default=project_name_default,
        ).strip()
        if not raw_project_name:
            raw_project_name = project_name_default
        try:
            project_name = _validate_project_name(raw_project_name)
            break
        except DatusException as e:
            console.print(f"[red]{e}[/]")
            console.print("[dim]Try again with a name matching [A-Za-z0-9_.-] (no spaces/slashes).[/]")

    override = ProjectOverride(
        target=target,
        default_datasource=default_datasource,
        project_name=project_name if project_name != project_name_default else None,
    )
    written = save_project_override(override, cwd=cwd)
    console.print()
    console.print(f"[green]Saved project config:[/] {written}")
    console.print(f"[dim]  target={target}  default_datasource={default_datasource}  project_name={project_name}[/]")
    console.print()
    return override
