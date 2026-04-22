# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for the /profile slash command handler.

Mocks the dialog primitive so we exercise handler logic without a real UI.
"""

from unittest.mock import MagicMock, patch

from datus.tools.permission.permission_manager import PermissionManager


class _FakeCLI:
    """Minimal CLI surface for /profile handler tests."""

    def __init__(self, manager, agent_config):
        self.console = MagicMock()
        self.agent_config = agent_config
        self.active_profile = agent_config.active_profile_name
        self.chat_commands = MagicMock()
        self.chat_commands.current_node = MagicMock()
        self.chat_commands.current_node.permission_manager = manager


def _make_agent_config(profile: str = "normal"):
    from datus.configuration.agent_config import AgentConfig

    cfg = AgentConfig.__new__(AgentConfig)
    cfg.active_profile_name = "normal"
    cfg._raw_permissions = {}
    cfg.permissions_config = cfg._init_permissions_config({"profile": profile})
    return cfg


def _patch_dialog(return_values):
    """Patch the profile selection dialog to return scripted values in order.

    The _cmd_profile implementation delegates to module-level helpers
    ``datus.cli.repl._ask_profile_choice`` and
    ``datus.cli.repl._ask_dangerous_confirmation``, so patching these
    module-level names works regardless of the type of the ``self`` stub.
    """
    primary = return_values.get("primary", [])
    confirm = return_values.get("confirm", [])
    return (
        patch("datus.cli.repl._ask_profile_choice", side_effect=primary),
        patch(
            "datus.cli.repl._ask_dangerous_confirmation",
            side_effect=confirm,
        ),
    )


def test_profile_switch_to_auto():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="normal")
    manager.approve_for_session("db_tools", "execute_ddl")
    agent_config = _make_agent_config("normal")
    cli = _FakeCLI(manager, agent_config)

    p_primary, p_confirm = _patch_dialog({"primary": ["auto"]})
    with p_primary, p_confirm:
        DatusCLI._cmd_profile(cli, "")

    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"
    assert manager._session_approvals == {}
    assert agent_config.active_profile_name == "auto"


def test_profile_switch_dangerous_requires_confirmation():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="normal")
    agent_config = _make_agent_config("normal")
    cli = _FakeCLI(manager, agent_config)

    p_primary, p_confirm = _patch_dialog({"primary": ["dangerous"], "confirm": [True]})
    with p_primary, p_confirm:
        DatusCLI._cmd_profile(cli, "")

    assert cli.active_profile == "dangerous"
    assert manager.active_profile == "dangerous"


def test_profile_switch_dangerous_cancelled():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="auto")
    agent_config = _make_agent_config("auto")
    cli = _FakeCLI(manager, agent_config)

    p_primary, p_confirm = _patch_dialog({"primary": ["dangerous"], "confirm": [False]})
    with p_primary, p_confirm:
        DatusCLI._cmd_profile(cli, "")

    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"


def test_profile_dialog_cancel_keeps_current():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="auto")
    agent_config = _make_agent_config("auto")
    cli = _FakeCLI(manager, agent_config)

    p_primary, p_confirm = _patch_dialog({"primary": [None]})
    with p_primary, p_confirm:
        DatusCLI._cmd_profile(cli, "")

    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"


def test_profile_select_same_profile_is_noop():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="auto")
    manager.approve_for_session("db_tools", "execute_ddl")
    agent_config = _make_agent_config("auto")
    cli = _FakeCLI(manager, agent_config)

    p_primary, p_confirm = _patch_dialog({"primary": ["auto"]})
    with p_primary, p_confirm:
        DatusCLI._cmd_profile(cli, "")

    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"
    assert manager._session_approvals  # not cleared


def test_profile_every_dangerous_transition_reconfirms():
    """Spec decision #5: every session transition into dangerous must confirm."""
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="normal")
    agent_config = _make_agent_config("normal")
    cli = _FakeCLI(manager, agent_config)

    # Sequence: normal→dangerous(confirmed), dangerous→auto, auto→dangerous(confirmed)
    primary_sequence = ["dangerous", "auto", "dangerous"]
    confirm_sequence = [True, True]

    p_primary, p_confirm = _patch_dialog({"primary": primary_sequence, "confirm": confirm_sequence})
    with p_primary, p_confirm:
        DatusCLI._cmd_profile(cli, "")
        assert cli.active_profile == "dangerous"
        DatusCLI._cmd_profile(cli, "")
        assert cli.active_profile == "auto"
        DatusCLI._cmd_profile(cli, "")
        assert cli.active_profile == "dangerous"


def test_profile_no_current_node_still_works():
    """If no active chat node exists, /profile should still rebuild
    agent_config and self.active_profile (future nodes inherit)."""
    from datus.cli.repl import DatusCLI

    agent_config = _make_agent_config("normal")

    class _NoNodeCLI:
        def __init__(self):
            self.console = MagicMock()
            self.agent_config = agent_config
            self.active_profile = agent_config.active_profile_name
            self.chat_commands = MagicMock()
            self.chat_commands.current_node = None

    cli = _NoNodeCLI()

    p_primary, p_confirm = _patch_dialog({"primary": ["auto"]})
    with p_primary, p_confirm:
        DatusCLI._cmd_profile(cli, "")

    assert cli.active_profile == "auto"
    assert agent_config.active_profile_name == "auto"
