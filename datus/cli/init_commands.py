# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/init`` slash command — chat shortcut for the bundled ``init`` skill.

The actual project-initialization logic lives in the ``init`` skill bundled
at ``datus/resources/skills/init/SKILL.md`` and loaded directly from the
package (no copy to ``~/.datus/skills`` required). Users can override it by
dropping a same-named SKILL.md into ``./.datus/skills/init/`` (project-level)
or ``~/.datus/skills/init/`` (user-level). The skill walks the agent through
``ask_user`` → ``filesystem_tools.list`` → ``db_tools.list_tables`` →
``filesystem_tools.write_file`` to produce ``AGENTS.md``.

Rather than reimplement that flow in Python, ``/init`` injects a
deterministic chat message that tells the active agent to load and follow
the ``init`` skill. The standard chat streaming pipeline (action stream,
Ctrl+O verbose, ESC interrupt, multi-turn refinement) renders the result
for free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datus.cli.cli_styles import print_error
from datus.cli.skill_command_utils import render_skill_prompt
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


_INIT_PROMPT = (
    "Initialize this project workspace by following the `init` skill. "
    'Call `load_skill(skill_name="init")` first and execute its steps in order. '
    "The skill infers the project goal and in-scope datasources and surfaces them "
    "in the Generation Manifest for my confirmation. "
    "If `AGENTS.md` already exists, confirm before overwriting it."
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
        hints the skill folds into its inferred context and manifest.
        """
        chat_commands = getattr(self.cli, "chat_commands", None)
        if chat_commands is None:
            print_error(
                self.console,
                "Chat is not initialized — /init relies on the chat pipeline.",
                prefix=False,
            )
            return

        chat_commands.execute_chat_command(
            render_skill_prompt(_INIT_PROMPT, args),
            plan_mode=getattr(self.cli, "plan_mode_active", False),
            subagent_name=None,
        )
