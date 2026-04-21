# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Extensible chat command abstraction for the Datus Gateway IM gateway."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from datus.gateway.channel.base import ChannelAdapter
from datus.gateway.models import InboundMessage, OutboundMessage, Verbose

if TYPE_CHECKING:
    from datus.gateway.bridge import ChannelBridge


@dataclass
class CommandMatch:
    """Result of matching user text to a command, with optional arguments."""

    command: ChatCommand
    args: str = ""


@dataclass
class CommandContext:
    """Lightweight data bag passed to command handlers."""

    msg: InboundMessage
    adapter: ChannelAdapter
    bridge: ChannelBridge
    args: str = ""


class ChatCommand(ABC):
    """Base class for slash commands handled before the agentic loop."""

    @property
    @abstractmethod
    def names(self) -> Sequence[str]:
        """Canonical command names, e.g. ``["/new", "/reset"]``."""
        ...

    @property
    def description(self) -> str:
        return ""

    @abstractmethod
    async def execute(self, ctx: CommandContext) -> None: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_COMMAND_REGISTRY: Dict[str, ChatCommand] = {}


def _normalize(name: str) -> str:
    return name.lstrip("/").lower()


def register_command(cmd: ChatCommand) -> None:
    """Register *cmd* under all its name variants (with and without ``/``, lowercase)."""
    for name in cmd.names:
        key = _normalize(name)
        _COMMAND_REGISTRY[key] = cmd


def match_command(text: str) -> Optional[CommandMatch]:
    """Return a ``CommandMatch`` for *text*, or ``None``.

    Tries full-text match first (backward compat for ``/new``, ``/reset``, etc.),
    then falls back to first-word match for commands with arguments.
    """
    stripped = text.strip()
    normalized = _normalize(stripped)
    # Full-text match first
    cmd = _COMMAND_REGISTRY.get(normalized)
    if cmd is not None:
        return CommandMatch(command=cmd, args="")
    # First-word match for commands with arguments — preserve original case in args
    first_word, _, _ = normalized.partition(" ")
    cmd = _COMMAND_REGISTRY.get(first_word)
    if cmd is not None:
        # Split original text to preserve case of arguments
        _, _, raw_remainder = stripped.partition(" ")
        return CommandMatch(command=cmd, args=raw_remainder.strip())
    return None


def list_commands() -> List[ChatCommand]:
    """Return a deduplicated list of registered commands."""
    seen_ids: set[int] = set()
    result: List[ChatCommand] = []
    for cmd in _COMMAND_REGISTRY.values():
        if id(cmd) not in seen_ids:
            seen_ids.add(id(cmd))
            result.append(cmd)
    return result


# ---------------------------------------------------------------------------
# Built-in commands
# ---------------------------------------------------------------------------


class NewSessionCommand(ChatCommand):
    """Reset the current conversation session."""

    @property
    def names(self) -> Sequence[str]:
        return ["/new", "/reset", "/clear"]

    @property
    def description(self) -> str:
        return "Reset the current conversation session"

    async def execute(self, ctx: CommandContext) -> None:
        ctx.bridge.clear_session(ctx.msg)
        reply = OutboundMessage(
            channel_id=ctx.msg.channel_id,
            conversation_id=ctx.msg.conversation_id,
            thread_id=ctx.msg.thread_id,
            text="Session cleared.",
        )
        await ctx.adapter.send_message(reply)


class VerboseCommand(ChatCommand):
    """Toggle verbosity level for the current conversation.

    - quiet: thinking + final output only (no tool calls)
    - brief: thinking + tool summaries + final output
    - detail: thinking + tool params/results + final output
    """

    _ALIAS_MAP = {
        "quiet": Verbose.OFF,
        "off": Verbose.OFF,
        "brief": Verbose.ON,
        "on": Verbose.ON,
        "detail": Verbose.FULL,
        "full": Verbose.FULL,
    }

    @property
    def names(self) -> Sequence[str]:
        return ["/verbose"]

    @property
    def description(self) -> str:
        return "Set verbosity: /verbose [quiet|brief|detail]"

    async def execute(self, ctx: CommandContext) -> None:
        if not ctx.args:
            current = ctx.bridge.get_verbose(ctx.msg)
            reply = OutboundMessage(
                channel_id=ctx.msg.channel_id,
                conversation_id=ctx.msg.conversation_id,
                thread_id=ctx.msg.thread_id,
                text=f"Current verbosity: `{current.value}`",
            )
            await ctx.adapter.send_message(reply)
            return

        level = self._ALIAS_MAP.get(ctx.args.lower())
        if level is None:
            valid = ", ".join(sorted(self._ALIAS_MAP.keys()))
            reply = OutboundMessage(
                channel_id=ctx.msg.channel_id,
                conversation_id=ctx.msg.conversation_id,
                thread_id=ctx.msg.thread_id,
                text=f"Unknown verbosity level `{ctx.args}`. Valid options: {valid}",
            )
            await ctx.adapter.send_message(reply)
            return

        ctx.bridge.set_verbose(ctx.msg, level)
        reply = OutboundMessage(
            channel_id=ctx.msg.channel_id,
            conversation_id=ctx.msg.conversation_id,
            thread_id=ctx.msg.thread_id,
            text=f"Verbosity set to `{level.value}`",
        )
        await ctx.adapter.send_message(reply)


class HelpCommand(ChatCommand):
    """Show all available commands."""

    @property
    def names(self) -> Sequence[str]:
        return ["/help"]

    @property
    def description(self) -> str:
        return "Show all available commands"

    async def execute(self, ctx: CommandContext) -> None:
        lines: list[str] = ["**Available commands:**"]
        for cmd in list_commands():
            names_str = ", ".join(f"`{n}`" for n in cmd.names)
            desc = cmd.description
            lines.append(f"- {names_str} — {desc}" if desc else f"- {names_str}")
        reply = OutboundMessage(
            channel_id=ctx.msg.channel_id,
            conversation_id=ctx.msg.conversation_id,
            thread_id=ctx.msg.thread_id,
            text="\n".join(lines),
        )
        await ctx.adapter.send_message(reply)


def register_builtin_commands() -> None:
    """Register all built-in commands. Safe to call multiple times."""
    register_command(NewSessionCommand())
    register_command(VerboseCommand())
    register_command(HelpCommand())
