# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus.gateway.commands."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from datus.gateway.commands import (
    _COMMAND_REGISTRY,
    ChatCommand,
    CommandContext,
    HelpCommand,
    NewSessionCommand,
    VerboseCommand,
    list_commands,
    match_command,
    register_builtin_commands,
    register_command,
)
from datus.gateway.models import InboundMessage, OutboundMessage, Verbose

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inbound(text="hello") -> InboundMessage:
    return InboundMessage(
        channel_id="ch1",
        sender_id="user1",
        conversation_id="conv1",
        message_id="msg1",
        text=text,
    )


class _DummyCommand(ChatCommand):
    @property
    def names(self):
        return ["/ping", "/pong"]

    @property
    def description(self):
        return "A test command"

    async def execute(self, ctx: CommandContext) -> None:
        pass


@pytest.fixture(autouse=True)
def _clear_registry():
    """Reset the command registry before each test."""
    _COMMAND_REGISTRY.clear()
    yield
    _COMMAND_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCommandRegistry:
    def test_register_and_match(self):
        cmd = _DummyCommand()
        register_command(cmd)
        result = match_command("/ping")
        assert result is not None
        assert result.command is cmd
        assert result.args == ""

    def test_match_variants(self):
        cmd = _DummyCommand()
        register_command(cmd)
        assert match_command("/pong").command is cmd
        assert match_command("ping").command is cmd
        assert match_command("PING").command is cmd
        assert match_command("  /Pong  ").command is cmd

    def test_match_returns_none_for_unknown(self):
        assert match_command("/unknown") is None
        assert match_command("") is None

    def test_match_with_args(self):
        cmd = _DummyCommand()
        register_command(cmd)
        result = match_command("/ping foo bar")
        assert result is not None
        assert result.command is cmd
        assert result.args == "foo bar"

    def test_match_with_args_strips_whitespace(self):
        cmd = _DummyCommand()
        register_command(cmd)
        result = match_command("/ping   baz  ")
        assert result.args == "baz"

    def test_full_text_match_takes_priority(self):
        """Full text match (no args) takes priority over first-word match."""
        cmd = _DummyCommand()
        register_command(cmd)
        # "/ping" as exact full text should match with empty args
        result = match_command("/ping")
        assert result.args == ""

    def test_list_commands_deduplicates(self):
        cmd = _DummyCommand()
        register_command(cmd)
        commands = list_commands()
        assert len(commands) == 1
        assert commands[0] is cmd

    def test_register_builtin_commands(self):
        register_builtin_commands()
        assert match_command("/new") is not None
        assert match_command("/reset") is not None
        assert match_command("/verbose") is not None
        assert match_command("/help") is not None
        # /new and /reset should resolve to the same command instance
        assert match_command("/new").command is match_command("/reset").command


class TestNewSessionCommand:
    @pytest.mark.asyncio
    async def test_execute_clears_session_and_replies(self):
        cmd = NewSessionCommand()
        msg = _make_inbound("/new")

        adapter = MagicMock()
        adapter.send_message = AsyncMock(return_value="bot1")

        bridge = MagicMock()
        bridge.clear_session = MagicMock()

        ctx = CommandContext(msg=msg, adapter=adapter, bridge=bridge)
        await cmd.execute(ctx)

        bridge.clear_session.assert_called_once_with(msg)
        adapter.send_message.assert_called_once()

        sent: OutboundMessage = adapter.send_message.call_args.args[0]
        assert "Session cleared" in sent.text
        assert sent.channel_id == "ch1"
        assert sent.conversation_id == "conv1"


class TestVerboseCommand:
    @pytest.fixture
    def cmd(self):
        return VerboseCommand()

    @pytest.fixture
    def adapter(self):
        a = MagicMock()
        a.send_message = AsyncMock(return_value="bot1")
        return a

    @pytest.fixture
    def bridge(self):
        b = MagicMock()
        b.get_verbose = MagicMock(return_value=Verbose.ON)
        b.set_verbose = MagicMock()
        return b

    @pytest.mark.asyncio
    async def test_no_args_shows_current_level(self, cmd, adapter, bridge):
        msg = _make_inbound("/verbose")
        ctx = CommandContext(msg=msg, adapter=adapter, bridge=bridge, args="")
        await cmd.execute(ctx)

        bridge.get_verbose.assert_called_once_with(msg)
        sent: OutboundMessage = adapter.send_message.call_args.args[0]
        assert "brief" in sent.text.lower()

    @pytest.mark.asyncio
    async def test_quiet_sets_off(self, cmd, adapter, bridge):
        msg = _make_inbound("/verbose quiet")
        ctx = CommandContext(msg=msg, adapter=adapter, bridge=bridge, args="quiet")
        await cmd.execute(ctx)

        bridge.set_verbose.assert_called_once_with(msg, Verbose.OFF)

    @pytest.mark.asyncio
    async def test_brief_sets_on(self, cmd, adapter, bridge):
        msg = _make_inbound("/verbose brief")
        ctx = CommandContext(msg=msg, adapter=adapter, bridge=bridge, args="brief")
        await cmd.execute(ctx)

        bridge.set_verbose.assert_called_once_with(msg, Verbose.ON)

    @pytest.mark.asyncio
    async def test_detail_sets_full(self, cmd, adapter, bridge):
        msg = _make_inbound("/verbose detail")
        ctx = CommandContext(msg=msg, adapter=adapter, bridge=bridge, args="detail")
        await cmd.execute(ctx)

        bridge.set_verbose.assert_called_once_with(msg, Verbose.FULL)

    @pytest.mark.asyncio
    async def test_alias_off(self, cmd, adapter, bridge):
        ctx = CommandContext(msg=_make_inbound("/verbose off"), adapter=adapter, bridge=bridge, args="off")
        await cmd.execute(ctx)
        bridge.set_verbose.assert_called_once_with(ctx.msg, Verbose.OFF)

    @pytest.mark.asyncio
    async def test_alias_full(self, cmd, adapter, bridge):
        ctx = CommandContext(msg=_make_inbound("/verbose full"), adapter=adapter, bridge=bridge, args="full")
        await cmd.execute(ctx)
        bridge.set_verbose.assert_called_once_with(ctx.msg, Verbose.FULL)

    @pytest.mark.asyncio
    async def test_case_insensitive(self, cmd, adapter, bridge):
        ctx = CommandContext(msg=_make_inbound("/verbose QUIET"), adapter=adapter, bridge=bridge, args="QUIET")
        await cmd.execute(ctx)
        bridge.set_verbose.assert_called_once_with(ctx.msg, Verbose.OFF)

    @pytest.mark.asyncio
    async def test_invalid_arg_shows_error(self, cmd, adapter, bridge):
        ctx = CommandContext(msg=_make_inbound("/verbose xyz"), adapter=adapter, bridge=bridge, args="xyz")
        await cmd.execute(ctx)

        bridge.set_verbose.assert_not_called()
        sent: OutboundMessage = adapter.send_message.call_args.args[0]
        assert "xyz" in sent.text
        assert "quiet" in sent.text.lower()


class TestHelpCommand:
    @pytest.mark.asyncio
    async def test_lists_all_commands(self):
        register_builtin_commands()
        cmd = HelpCommand()
        msg = _make_inbound("/help")

        adapter = MagicMock()
        adapter.send_message = AsyncMock(return_value="bot1")
        bridge = MagicMock()

        ctx = CommandContext(msg=msg, adapter=adapter, bridge=bridge)
        await cmd.execute(ctx)

        sent: OutboundMessage = adapter.send_message.call_args.args[0]
        assert "/new" in sent.text
        assert "/verbose" in sent.text
        assert "/help" in sent.text

    @pytest.mark.asyncio
    async def test_includes_descriptions(self):
        register_builtin_commands()
        cmd = HelpCommand()
        msg = _make_inbound("/help")

        adapter = MagicMock()
        adapter.send_message = AsyncMock(return_value="bot1")
        bridge = MagicMock()

        ctx = CommandContext(msg=msg, adapter=adapter, bridge=bridge)
        await cmd.execute(ctx)

        sent: OutboundMessage = adapter.send_message.call_args.args[0]
        # Check that descriptions from builtins appear
        assert "Reset the current conversation session" in sent.text
        assert "Show all available commands" in sent.text
