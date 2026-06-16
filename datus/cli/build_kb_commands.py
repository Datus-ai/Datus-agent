# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/build-kb`` slash command — chat shortcut for the bundled ``build-kb`` skill.

The heavy knowledge-base construction lives in the ``build-kb`` skill bundled
at ``datus/resources/skills/build-kb/SKILL.md`` and loaded directly from the
package (no copy to ``~/.datus/skills`` required). Users can override it by
dropping a same-named SKILL.md into ``./.datus/skills/build-kb/`` (project-level)
or ``~/.datus/skills/build-kb/`` (user-level).

``build-kb`` is the heavy companion to the lightweight ``/init``: where ``/init``
writes the AGENTS.md inventory plus the cheap file-based stores (``knowledge`` /
``memory``), ``/build-kb`` scans, explores, and generates the vector-indexed
stores (``semantic_models`` / ``metrics`` / ``reference_sql``), then refreshes
AGENTS.md's KB index. It accepts an optional free-text file/table scope appended
after the command, which the skill parses to limit what it scans and generates.

Rather than reimplement that flow in Python, ``/build-kb`` injects a
deterministic chat message that tells the active agent to load and follow the
``build-kb`` skill. The standard chat streaming pipeline (action stream,
Ctrl+O verbose, ESC interrupt, multi-turn refinement) renders the result for
free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datus.cli.cli_styles import print_error
from datus.cli.skill_command_utils import render_skill_prompt
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


_BUILD_KB_PROMPT = (
    "Build the project's knowledge base by following the `build-kb` skill. "
    'Call `load_skill(skill_name="build-kb")` first and execute its steps in order. '
    "If `AGENTS.md` already exists, reuse its inventory and only refresh the KB index."
    "{user_context}"
)


class BuildKbCommands:
    """Handler for the ``/build-kb`` slash command."""

    def __init__(self, cli: "DatusCLI"):
        self.cli = cli
        self.console = cli.console

    def cmd_build_kb(self, args: str) -> None:
        """Dispatch ``/build-kb`` — delegate to the chat pipeline.

        Any text after ``/build-kb`` is forwarded verbatim as the file/table
        scope hints the skill parses in its Step 0 (e.g.
        ``/build-kb orders + order_items tables and queries/*.sql, sales domain only``).
        """
        chat_commands = getattr(self.cli, "chat_commands", None)
        if chat_commands is None:
            print_error(
                self.console,
                "Chat is not initialized — /build-kb relies on the chat pipeline.",
                prefix=False,
            )
            return

        chat_commands.execute_chat_command(
            render_skill_prompt(_BUILD_KB_PROMPT, args),
            plan_mode=getattr(self.cli, "plan_mode_active", False),
            subagent_name=None,
        )
