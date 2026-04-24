# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Single source of truth for Datus-CLI slash commands.

The registry drives three consumers:

* :attr:`DatusCLI.commands` — maps ``/name`` (and aliases) to the handler bound
  at REPL construction.
* :class:`SlashCommandCompleter` — renders the prompt-toolkit completion menu
  with ``display_meta`` pulled from :attr:`SlashSpec.summary`.
* :meth:`DatusCLI._cmd_help` — emits the grouped help text.

Adding a command: append a :class:`SlashSpec` below, then wire the matching
handler in ``DatusCLI._build_slash_handler_map`` so the registry integrity
test (``tests/unit_tests/cli/test_slash_registry.py``) keeps passing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

GROUP_ORDER: tuple[str, ...] = (
    "session",
    "metadata",
    "context",
    "agent",
    "system",
)


GROUP_TITLES: dict[str, str] = {
    "session": "Session",
    "metadata": "Metadata",
    "context": "Context",
    "agent": "Agents",
    "system": "System",
}


@dataclass(frozen=True)
class SlashSpec:
    """Metadata for one top-level slash command."""

    name: str
    summary: str
    group: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    hidden: bool = False


SLASH_COMMANDS: tuple[SlashSpec, ...] = (
    # session
    SlashSpec("help", "Display help for all slash commands", "session"),
    SlashSpec("exit", "Exit the CLI", "session", aliases=("quit",)),
    SlashSpec("clear", "Clear console and chat session", "session"),
    SlashSpec("chat_info", "Show current chat session information", "session"),
    SlashSpec("compact", "Compact chat session by summarizing history", "session"),
    SlashSpec("resume", "List and resume a previous chat session", "session"),
    SlashSpec("rewind", "Rewind current session to a specific turn", "session"),
    # metadata
    SlashSpec("databases", "List all databases", "metadata"),
    SlashSpec("database", "Switch the current database", "metadata"),
    SlashSpec("tables", "List all tables", "metadata"),
    SlashSpec("schemas", "List all schemas or show schema details", "metadata"),
    SlashSpec("schema", "Switch the current schema", "metadata"),
    SlashSpec("table_schema", "Show table field details", "metadata"),
    SlashSpec("indexes", "Show indexes for a table", "metadata"),
    # context
    SlashSpec("catalog", "Display database catalog explorer", "context"),
    SlashSpec("subject", "Display semantic models, metrics, and references", "context"),
    # agent
    SlashSpec("agent", "Open the unified agent manager (built-in overrides / default)", "agent"),
    SlashSpec("subagent", "Open the unified agent manager (custom sub-agents)", "agent"),
    SlashSpec("datasource", "Manage datasources (list/add/edit/delete/switch)", "agent"),
    SlashSpec("language", "Set or show the response language for model outputs", "agent"),
    # system
    SlashSpec("mcp", "Manage MCP servers (list/add/remove/check/call/filter)", "system"),
    SlashSpec("skill", "Manage skills and marketplace (list/install/publish/...)", "system"),
    SlashSpec("bootstrap-bi", "Extract BI dashboard assets for sub-agent context", "system"),
    SlashSpec("model", "Switch LLM provider/model", "system", aliases=("models",)),
    SlashSpec("effort", "Set reasoning effort (off|minimal|low|medium|high)", "system"),
    SlashSpec("services", "List configured service platforms and their read-only methods", "system"),
    SlashSpec(
        "profile",
        "Switch the permission profile (normal / auto / dangerous)",
        "system",
    ),
)


_BY_TOKEN: dict[str, SlashSpec] = {}
for _spec in SLASH_COMMANDS:
    _BY_TOKEN[_spec.name] = _spec
    for _alias in _spec.aliases:
        _BY_TOKEN[_alias] = _spec


def iter_visible() -> Iterable[SlashSpec]:
    """Yield every non-hidden spec in declaration order."""

    for spec in SLASH_COMMANDS:
        if not spec.hidden:
            yield spec


def lookup(token: str) -> SlashSpec | None:
    """Resolve a bare token (e.g. ``"help"`` or ``"quit"``) to its spec.

    Returns ``None`` when the token is unknown. The caller is responsible for
    stripping the leading ``/`` before passing the token in.
    """

    return _BY_TOKEN.get(token)


def all_tokens() -> tuple[str, ...]:
    """Return every callable token (canonical names + aliases)."""

    return tuple(_BY_TOKEN.keys())
