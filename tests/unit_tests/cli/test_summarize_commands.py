# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/summarize_commands.py.

``/session-summarize`` and ``/memory-organize`` are thin wrappers that inject a
deterministic chat message asking the agent to follow the matching bundled
skill. The test surface is correspondingly small: argument validation,
missing-chat fallback, and the exact prompt + plan_mode propagation handed to
``execute_chat_command``.
"""

from unittest.mock import MagicMock, patch


def _build_cli(*, plan_mode: bool = False, with_chat: bool = True):
    """Minimal stand-in for ``DatusCLI`` exposing only what the handlers read."""
    cli = MagicMock()
    cli.console = MagicMock()
    cli.plan_mode_active = plan_mode
    if with_chat:
        cli.chat_commands = MagicMock()
    else:
        cli.chat_commands = None
    return cli


class TestSessionSummarizeArgumentHandling:
    """`/session-summarize` accepts an optional free-text description."""

    def test_blank_sends_canonical_prompt(self):
        from datus.cli.skill_command_utils import render_skill_prompt
        from datus.cli.summarize_commands import (
            _SESSION_SUMMARIZE_PROMPT,
            SessionSummarizeCommands,
        )

        cli = _build_cli()
        sc = SessionSummarizeCommands(cli)

        with patch("datus.cli.summarize_commands.print_error") as mock_err:
            sc.cmd_session_summarize("   ")  # whitespace-only is treated as empty

        mock_err.assert_not_called()
        cli.chat_commands.execute_chat_command.assert_called_once()
        args, _ = cli.chat_commands.execute_chat_command.call_args
        assert args[0] == render_skill_prompt(_SESSION_SUMMARIZE_PROMPT, "")
        assert "{user_context}" not in args[0]

    def test_argument_appended_as_context_block(self):
        from datus.cli.summarize_commands import SessionSummarizeCommands

        cli = _build_cli()
        sc = SessionSummarizeCommands(cli)

        sc.cmd_session_summarize("only keep the datasource conventions")

        cli.chat_commands.execute_chat_command.assert_called_once()
        args, _ = cli.chat_commands.execute_chat_command.call_args
        assert "`session-summarize` skill" in args[0]
        assert "only keep the datasource conventions" in args[0]
        assert "Additional context from the user" in args[0]
        assert "{user_context}" not in args[0]


class TestSessionSummarizeChatDispatch:
    """Successful path: forward the canonical prompt to the chat pipeline."""

    def test_forwards_prompt_to_chat(self):
        from datus.cli.skill_command_utils import render_skill_prompt
        from datus.cli.summarize_commands import (
            _SESSION_SUMMARIZE_PROMPT,
            SessionSummarizeCommands,
        )

        cli = _build_cli()
        sc = SessionSummarizeCommands(cli)

        sc.cmd_session_summarize("")

        cli.chat_commands.execute_chat_command.assert_called_once()
        args, _ = cli.chat_commands.execute_chat_command.call_args
        assert args[0] == render_skill_prompt(_SESSION_SUMMARIZE_PROMPT, "")
        # Skill must be referenced explicitly so the model picks the right one.
        assert "`session-summarize` skill" in args[0]
        assert "STOP" in args[0]

    def test_propagates_plan_mode_active_flag(self):
        from datus.cli.summarize_commands import SessionSummarizeCommands

        cli = _build_cli(plan_mode=True)
        sc = SessionSummarizeCommands(cli)

        sc.cmd_session_summarize("")

        kwargs = cli.chat_commands.execute_chat_command.call_args.kwargs
        assert kwargs.get("plan_mode") is True
        assert kwargs.get("subagent_name") is None

    def test_default_plan_mode_is_false(self):
        from datus.cli.summarize_commands import SessionSummarizeCommands

        cli = _build_cli(plan_mode=False)
        sc = SessionSummarizeCommands(cli)

        sc.cmd_session_summarize("")

        kwargs = cli.chat_commands.execute_chat_command.call_args.kwargs
        assert kwargs.get("plan_mode") is False


class TestSessionSummarizeMissingChat:
    """Defensive: surface a clear error when chat hasn't initialised yet."""

    def test_errors_when_chat_commands_missing(self):
        from datus.cli.summarize_commands import SessionSummarizeCommands

        cli = _build_cli(with_chat=False)
        sc = SessionSummarizeCommands(cli)

        with patch("datus.cli.summarize_commands.print_error") as mock_err:
            sc.cmd_session_summarize("")

        mock_err.assert_called_once()


class TestMemoryOrganizeArgumentHandling:
    """`/memory-organize` accepts an optional free-text description."""

    def test_blank_sends_canonical_prompt(self):
        from datus.cli.skill_command_utils import render_skill_prompt
        from datus.cli.summarize_commands import (
            _MEMORY_ORGANIZE_PROMPT,
            MemoryOrganizeCommands,
        )

        cli = _build_cli()
        mc = MemoryOrganizeCommands(cli)

        with patch("datus.cli.summarize_commands.print_error") as mock_err:
            mc.cmd_memory_organize("  ")

        mock_err.assert_not_called()
        cli.chat_commands.execute_chat_command.assert_called_once()
        args, _ = cli.chat_commands.execute_chat_command.call_args
        assert args[0] == render_skill_prompt(_MEMORY_ORGANIZE_PROMPT, "")
        assert "{user_context}" not in args[0]

    def test_argument_appended_as_context_block(self):
        from datus.cli.summarize_commands import MemoryOrganizeCommands

        cli = _build_cli()
        mc = MemoryOrganizeCommands(cli)

        mc.cmd_memory_organize("dedupe the knowledge files first")

        cli.chat_commands.execute_chat_command.assert_called_once()
        args, _ = cli.chat_commands.execute_chat_command.call_args
        assert "`memory-organization` skill" in args[0]
        assert "dedupe the knowledge files first" in args[0]
        assert "Additional context from the user" in args[0]
        assert "{user_context}" not in args[0]


class TestMemoryOrganizeChatDispatch:
    """Successful path: forward the canonical prompt to the chat pipeline."""

    def test_forwards_prompt_to_chat(self):
        from datus.cli.skill_command_utils import render_skill_prompt
        from datus.cli.summarize_commands import (
            _MEMORY_ORGANIZE_PROMPT,
            MemoryOrganizeCommands,
        )

        cli = _build_cli()
        mc = MemoryOrganizeCommands(cli)

        mc.cmd_memory_organize("")

        cli.chat_commands.execute_chat_command.assert_called_once()
        args, _ = cli.chat_commands.execute_chat_command.call_args
        assert args[0] == render_skill_prompt(_MEMORY_ORGANIZE_PROMPT, "")
        assert "`memory-organization` skill" in args[0]
        assert "STOP" in args[0]

    def test_propagates_plan_mode_active_flag(self):
        from datus.cli.summarize_commands import MemoryOrganizeCommands

        cli = _build_cli(plan_mode=True)
        mc = MemoryOrganizeCommands(cli)

        mc.cmd_memory_organize("")

        kwargs = cli.chat_commands.execute_chat_command.call_args.kwargs
        assert kwargs.get("plan_mode") is True
        assert kwargs.get("subagent_name") is None


class TestMemoryOrganizeMissingChat:
    """Defensive: surface a clear error when chat hasn't initialised yet."""

    def test_errors_when_chat_commands_missing(self):
        from datus.cli.summarize_commands import MemoryOrganizeCommands

        cli = _build_cli(with_chat=False)
        mc = MemoryOrganizeCommands(cli)

        with patch("datus.cli.summarize_commands.print_error") as mock_err:
            mc.cmd_memory_organize("")

        mock_err.assert_called_once()
