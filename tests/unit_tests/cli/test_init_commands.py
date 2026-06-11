# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/init_commands.py — the /init slash handler.

`/init` is a thin wrapper that injects a deterministic chat message asking
the agent to follow the bundled ``init`` skill. The test surface is
correspondingly small: argument validation, missing-chat fallback, and the
exact prompt + plan_mode propagation handed to ``execute_chat_command``.
"""

from unittest.mock import MagicMock, patch


def _build_cli(*, plan_mode: bool = False, with_chat: bool = True):
    """Minimal stand-in for ``DatusCLI`` exposing only what InitCommands reads."""
    cli = MagicMock()
    cli.console = MagicMock()
    cli.plan_mode_active = plan_mode
    if with_chat:
        cli.chat_commands = MagicMock()
    else:
        cli.chat_commands = None
    return cli


class TestCmdInitArgumentHandling:
    """`/init` accepts an optional free-text description appended to the prompt."""

    def test_blank_sends_canonical_prompt(self):
        from datus.cli.init_commands import _INIT_PROMPT, InitCommands
        from datus.cli.skill_command_utils import render_skill_prompt

        cli = _build_cli()
        ic = InitCommands(cli)

        with patch("datus.cli.init_commands.print_error") as mock_err:
            ic.cmd_init("   ")  # whitespace-only is treated as empty

        mock_err.assert_not_called()
        cli.chat_commands.execute_chat_command.assert_called_once()
        args, _ = cli.chat_commands.execute_chat_command.call_args
        # No placeholder leaks; identical to the empty-args render.
        assert args[0] == render_skill_prompt(_INIT_PROMPT, "")
        assert "{user_context}" not in args[0]

    def test_argument_appended_as_context_block(self):
        from datus.cli.init_commands import InitCommands

        cli = _build_cli()
        ic = InitCommands(cli)

        ic.cmd_init("sales analytics warehouse, focus on the orders domain")

        cli.chat_commands.execute_chat_command.assert_called_once()
        args, _ = cli.chat_commands.execute_chat_command.call_args
        # The base instruction survives and the description is forwarded verbatim.
        assert "`init` skill" in args[0]
        assert "sales analytics warehouse, focus on the orders domain" in args[0]
        assert "Additional context from the user" in args[0]
        assert "{user_context}" not in args[0]


class TestCmdInitChatDispatch:
    """Successful path: forward the canonical prompt to the chat pipeline."""

    def test_forwards_init_prompt_to_chat(self):
        from datus.cli.init_commands import _INIT_PROMPT, InitCommands
        from datus.cli.skill_command_utils import render_skill_prompt

        cli = _build_cli()
        ic = InitCommands(cli)

        ic.cmd_init("")

        cli.chat_commands.execute_chat_command.assert_called_once()
        args, kwargs = cli.chat_commands.execute_chat_command.call_args
        # First positional argument is the chat message.
        assert args[0] == render_skill_prompt(_INIT_PROMPT, "")
        # Skill must be referenced explicitly so the model picks the right one.
        assert "`init` skill" in args[0]
        assert "AGENTS.md" in args[0]

    def test_propagates_plan_mode_active_flag(self):
        from datus.cli.init_commands import InitCommands

        cli = _build_cli(plan_mode=True)
        ic = InitCommands(cli)

        ic.cmd_init("")

        kwargs = cli.chat_commands.execute_chat_command.call_args.kwargs
        assert kwargs.get("plan_mode") is True
        assert kwargs.get("subagent_name") is None

    def test_default_plan_mode_is_false(self):
        from datus.cli.init_commands import InitCommands

        cli = _build_cli(plan_mode=False)
        ic = InitCommands(cli)

        ic.cmd_init("")

        kwargs = cli.chat_commands.execute_chat_command.call_args.kwargs
        assert kwargs.get("plan_mode") is False


class TestCmdInitMissingChat:
    """Defensive: surface a clear error when chat hasn't initialised yet."""

    def test_errors_when_chat_commands_missing(self):
        from datus.cli.init_commands import InitCommands

        cli = _build_cli(with_chat=False)
        ic = InitCommands(cli)

        with patch("datus.cli.init_commands.print_error") as mock_err:
            ic.cmd_init("")

        mock_err.assert_called_once()
        # No AttributeError leaked.
