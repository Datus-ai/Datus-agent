# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Bounded REPL entry point for the bundled ``init`` skill.

The actual project-initialization logic lives in the ``init`` skill bundled
at ``datus/resources/skills/init/SKILL.md`` and loaded directly from the
package (no copy to ``~/.datus/skills`` required). Users can override it by
dropping a same-named SKILL.md into ``./.datus/skills/init/`` (project-level)
or ``~/.datus/skills/init/`` (user-level).

``/init`` is the lightweight pass: it statically analyzes the project's SQL
into the persisted lineage graph (``upsert_lineage``), reads the human-written docs, checks names
against the database schema without querying table data, and writes the ``AGENTS.md`` project map plus
per-domain ``./knowledge/*.md`` files.
It deliberately stops short of the expensive vector-indexed stores
(``semantic_models`` / ``metrics`` / ``reference_sql``) — those are built by
the ``build-kb`` skill behind ``/build-kb`` (see ``build_kb_commands.py``).

The host records current-scope coverage in memory. One ordinary chat run
follows the skill, using pagination and mid-turn compaction as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from datus.cli.cli_styles import print_error
from datus.cli.init_context import InitContext
from datus.cli.skill_command_utils import render_skill_prompt
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


_INIT_PROMPT = (
    "Initialize this project workspace by following the `init` skill. "
    "The complete init skill is supplied in the system instructions for this invocation. "
    "Read existing AGENTS.md, relevant knowledge and lineage for context. "
    "If present, read relevant entries in knowledge/pending-questions.md as unconfirmed questions, not facts. "
    "Update only the current scope and preserve existing knowledge and user content."
    "{user_context}"
)


class InitCommands:
    """Handler for the ``/init`` slash command."""

    def __init__(self, cli: "DatusCLI"):
        self.cli = cli
        self.console = cli.console

    def cmd_init(self, args: str) -> None:
        """Dispatch ``/init`` — delegate to the chat pipeline.

        Any text after ``/init`` is forwarded verbatim as extra goal/scope
        hints the skill folds into its inferred context.
        """
        chat_commands = getattr(self.cli, "chat_commands", None)
        if chat_commands is None:
            print_error(
                self.console,
                "Chat is not initialized — /init relies on the chat pipeline.",
                prefix=False,
            )
            return

        run = None
        try:
            scope = args.strip()
            config = self.cli.agent_config
            context = self.cli.cli_context
            node_config = config.sub_agent_config("chat") or {}
            root = Path(node_config.get("workspace_root") or config.project_root)
            binding = {
                "project": config.project_name,
                "datasource": config.current_datasource or "",
                "catalog": getattr(context, "current_catalog", "") or "",
                "database": getattr(context, "current_db_name", "") or "",
                "schema_name": getattr(context, "current_schema", "") or "",
            }
            if binding["datasource"]:
                db_config = config.current_db_config()
                binding["dialect"] = db_config.type
                for target, field in (("catalog", "catalog"), ("database", "database"), ("schema_name", "schema")):
                    binding[target] = binding[target] or getattr(db_config, field, "") or ""
            run = InitContext(root, binding, scope)
            prompt = render_skill_prompt(_INIT_PROMPT, run.state["scope"])
            chat_commands.execute_chat_command(
                prompt,
                plan_mode=getattr(self.cli, "plan_mode_active", False),
                subagent_name=None,
                init_context=run,
            )
        except (OSError, ValueError) as exc:
            print_error(self.console, f"Cannot run init: {exc}", prefix=False)
        finally:
            if run is not None:
                run.close()
                progress = run.progress()
                self.console.print(
                    f"Init: {progress['completed']}/{progress['selected']} selected tables completed "
                    f"({progress['status']}; current scope).",
                    markup=False,
                )
                self.console.print(
                    f"Modified files: {len(run.state['outputs'])}. "
                    f"Uncovered: {run.state.get('uncovered') or 'Database scope outside the selected core tables.'}",
                    markup=False,
                )
                self.console.print(
                    f"Structure checks succeeded: {progress.get('structure_checked', 0)}. "
                    "See the reply for findings, unknowns and unfinished work.",
                    markup=False,
                )
                if progress["stop_reason"]:
                    self.console.print(f"Stopped: {progress['stop_reason']}", markup=False)
