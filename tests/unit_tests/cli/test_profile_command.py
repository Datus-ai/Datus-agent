# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for the /profile slash command handler.

Uses a minimal CLI stub + InteractionBroker mock so we exercise handler
logic without spinning up prompt_toolkit's event loop.
"""

from unittest.mock import MagicMock

import pytest

from datus.tools.permission.permission_manager import PermissionManager


class _FakeBroker:
    """Async stand-in for ``InteractionBroker.request``."""

    def __init__(self, scripted_responses):
        self.scripted_responses = list(scripted_responses)
        self.requests = []

    async def request(self, contents, choices, default_choices=None):
        self.requests.append({"contents": contents, "choices": choices, "default": default_choices})

        async def _callback(_msg):
            return None

        choice = self.scripted_responses.pop(0)
        return choice, _callback


class _FakeCLI:
    """Minimal CLI surface exercised by ``_cmd_profile_async``."""

    def __init__(self, broker, manager, agent_config):
        self.broker = broker
        self.console = MagicMock()
        self.agent_config = agent_config
        self.active_profile = agent_config.active_profile_name
        self.chat_commands = MagicMock()
        self.chat_commands.current_node = MagicMock()
        self.chat_commands.current_node.permission_manager = manager


def _make_agent_config(profile: str = "normal"):
    from datus.configuration.agent_config import AgentConfig

    cfg = AgentConfig.__new__(AgentConfig)
    cfg.active_profile_name = "normal"  # pre-seed
    cfg._raw_permissions = {}
    cfg.permissions_config = cfg._init_permissions_config({"profile": profile})
    return cfg


@pytest.mark.asyncio
async def test_profile_switch_to_auto():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="normal")
    manager.approve_for_session("db_tools", "execute_ddl")
    agent_config = _make_agent_config("normal")
    broker = _FakeBroker(scripted_responses=["auto"])
    cli = _FakeCLI(broker, manager, agent_config)

    await DatusCLI._cmd_profile_async(cli, "")

    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"
    # Session approvals cleared by switch_profile
    assert manager._session_approvals == {}
    # agent_config rebuilt so future nodes see the new profile
    assert agent_config.active_profile_name == "auto"


@pytest.mark.asyncio
async def test_profile_switch_dangerous_requires_confirmation():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="normal")
    agent_config = _make_agent_config("normal")
    # First dialog: user picks 'dangerous'. Second dialog: confirms 'enable'.
    broker = _FakeBroker(scripted_responses=["dangerous", "enable"])
    cli = _FakeCLI(broker, manager, agent_config)

    await DatusCLI._cmd_profile_async(cli, "")

    assert cli.active_profile == "dangerous"
    assert manager.active_profile == "dangerous"
    # Two separate broker requests: selection + confirmation
    assert len(broker.requests) == 2


@pytest.mark.asyncio
async def test_profile_switch_dangerous_cancelled():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="auto")
    agent_config = _make_agent_config("auto")
    broker = _FakeBroker(scripted_responses=["dangerous", "cancel"])
    cli = _FakeCLI(broker, manager, agent_config)

    await DatusCLI._cmd_profile_async(cli, "")

    # Profile unchanged
    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"


@pytest.mark.asyncio
async def test_profile_dialog_cancel_keeps_current():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="auto")
    agent_config = _make_agent_config("auto")
    broker = _FakeBroker(scripted_responses=["cancel"])
    cli = _FakeCLI(broker, manager, agent_config)

    await DatusCLI._cmd_profile_async(cli, "")

    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"


@pytest.mark.asyncio
async def test_profile_select_same_profile_is_noop():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="auto")
    manager.approve_for_session("db_tools", "execute_ddl")
    agent_config = _make_agent_config("auto")
    broker = _FakeBroker(scripted_responses=["auto"])
    cli = _FakeCLI(broker, manager, agent_config)

    await DatusCLI._cmd_profile_async(cli, "")

    # No switch → approvals preserved, no_secondary_request
    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"
    assert manager._session_approvals  # not cleared
    assert len(broker.requests) == 1  # no second dialog


@pytest.mark.asyncio
async def test_profile_every_dangerous_transition_reconfirms():
    """Spec decision #5: every session transition into dangerous must confirm,
    not just the first. User goes normal → dangerous (confirmed) → auto →
    dangerous again — the second entry must also require confirmation."""
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="normal")
    agent_config = _make_agent_config("normal")
    # Sequence: [normal → dangerous/enable] → [dangerous → auto] → [auto → dangerous/enable]
    broker = _FakeBroker(scripted_responses=["dangerous", "enable", "auto", "dangerous", "enable"])
    cli = _FakeCLI(broker, manager, agent_config)

    await DatusCLI._cmd_profile_async(cli, "")
    assert cli.active_profile == "dangerous"
    await DatusCLI._cmd_profile_async(cli, "")
    assert cli.active_profile == "auto"
    await DatusCLI._cmd_profile_async(cli, "")
    assert cli.active_profile == "dangerous"

    # Three switches total: first and third each needed two broker requests;
    # middle (auto) needed one. 2 + 1 + 2 = 5.
    assert len(broker.requests) == 5
