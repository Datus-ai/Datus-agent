# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/session-summarize`` and ``/memory-organize`` slash commands.

Both are chat shortcuts for bundled persistence skills, mirroring ``/init``
(:mod:`datus.cli.init_commands`). The real logic lives in the skills bundled
at ``datus/resources/skills/session-summarize/SKILL.md`` and
``datus/resources/skills/memory-organization/SKILL.md`` — loaded directly from
the package and overridable by dropping a same-named ``SKILL.md`` into
``./.datus/skills/<name>/`` (project) or ``~/.datus/skills/<name>/`` (user).

Rather than reimplement those flows in Python, each command injects a
deterministic chat message telling the active agent to load and follow the
matching skill. The standard chat streaming pipeline (action stream, Ctrl+O
verbose, ESC interrupt, multi-turn refinement) renders the result for free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datus.cli.cli_styles import print_error
from datus.cli.skill_command_utils import render_skill_prompt
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


_SESSION_SUMMARIZE_PROMPT = (
    "Summarize the current chat session by following the `session-summarize` skill. "
    'Call `load_skill(skill_name="session-summarize")` first and execute its steps in order: '
    "harvest the candidates worth persisting, classify each via storage-classify, then present "
    "the Summary Manifest and STOP for my confirmation before generating or writing anything."
    "{user_context}"
)


_MEMORY_ORGANIZE_PROMPT = (
    "Audit and reorganize all persistent stores by following the `memory-organization` skill. "
    'Call `load_skill(skill_name="memory-organization")` first and execute its steps in order: '
    "inventory every store, analyze them against storage-classify rules, then present a "
    "Remediation Plan and STOP for my confirmation before applying any fix. If nothing needs "
    "fixing, report that and stop."
    "{user_context}"
)


class SessionSummarizeCommands:
    """Handler for the ``/session-summarize`` slash command."""

    def __init__(self, cli: "DatusCLI"):
        self.cli = cli
        self.console = cli.console

    def cmd_session_summarize(self, args: str) -> None:
        """Dispatch ``/session-summarize`` — delegate to the chat pipeline.

        Any text after the command is forwarded verbatim as focus/refinement
        hints the skill folds into its Summary Manifest.
        """
        chat_commands = getattr(self.cli, "chat_commands", None)
        if chat_commands is None:
            print_error(
                self.console,
                "Chat is not initialized — /session-summarize relies on the chat pipeline.",
                prefix=False,
            )
            return

        chat_commands.execute_chat_command(
            render_skill_prompt(_SESSION_SUMMARIZE_PROMPT, args),
            plan_mode=getattr(self.cli, "plan_mode_active", False),
            subagent_name=None,
        )


class MemoryOrganizeCommands:
    """Handler for the ``/memory-organize`` slash command."""

    def __init__(self, cli: "DatusCLI"):
        self.cli = cli
        self.console = cli.console

    def cmd_memory_organize(self, args: str) -> None:
        """Dispatch ``/memory-organize`` — delegate to the chat pipeline.

        Any text after the command is forwarded verbatim as focus/refinement
        hints the skill folds into its Remediation Plan.
        """
        chat_commands = getattr(self.cli, "chat_commands", None)
        if chat_commands is None:
            print_error(
                self.console,
                "Chat is not initialized — /memory-organize relies on the chat pipeline.",
                prefix=False,
            )
            return

        chat_commands.execute_chat_command(
            render_skill_prompt(_MEMORY_ORGANIZE_PROMPT, args),
            plan_mode=getattr(self.cli, "plan_mode_active", False),
            subagent_name=None,
        )
