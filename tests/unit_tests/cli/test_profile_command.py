# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for the /profile slash command handler.

Injects stub picker callables onto the CLI stub so we exercise handler
logic without spinning up prompt_toolkit.
"""

from unittest.mock import MagicMock

from datus.tools.permission.permission_manager import PermissionManager


class _FakeCLI:
    """Minimal CLI surface for /profile handler tests.

    Exposes picker callables as instance attributes; ``_cmd_profile``
    reads them via ``self._run_profile_picker(current)`` /
    ``self._run_dangerous_confirm()`` which finds them on the instance
    before falling through to the class method.
    """

    def __init__(self, manager, agent_config, profile_responses, confirm_responses=None):
        self.console = MagicMock()
        self.agent_config = agent_config
        self.active_profile = agent_config.active_profile_name
        self.chat_commands = MagicMock()
        self.chat_commands.current_node = MagicMock()
        self.chat_commands.current_node.permission_manager = manager

        self._profile_responses = list(profile_responses)
        self._confirm_responses = list(confirm_responses or [])
        self.picker_calls = 0
        self.confirm_calls = 0

    # Instance-level overrides that _cmd_profile will find via normal
    # attribute lookup before hitting DatusCLI's class methods.
    def _run_profile_picker(self, current):
        self.picker_calls += 1
        return self._profile_responses.pop(0)

    def _run_dangerous_confirm(self):
        self.confirm_calls += 1
        return self._confirm_responses.pop(0)


def _make_agent_config(profile: str = "normal"):
    from datus.configuration.agent_config import AgentConfig

    cfg = AgentConfig.__new__(AgentConfig)
    cfg.active_profile_name = "normal"
    cfg._raw_permissions = {}
    cfg.permissions_config = cfg._init_permissions_config({"profile": profile})
    return cfg


def test_profile_switch_to_auto():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="normal")
    manager.approve_for_session("db_tools", "execute_ddl")
    agent_config = _make_agent_config("normal")
    cli = _FakeCLI(manager, agent_config, profile_responses=["auto"])

    DatusCLI._cmd_profile(cli, "")

    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"
    assert manager._session_approvals == {}
    assert agent_config.active_profile_name == "auto"


def test_profile_switch_dangerous_requires_confirmation():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="normal")
    agent_config = _make_agent_config("normal")
    cli = _FakeCLI(
        manager,
        agent_config,
        profile_responses=["dangerous"],
        confirm_responses=[True],
    )

    DatusCLI._cmd_profile(cli, "")

    assert cli.active_profile == "dangerous"
    assert manager.active_profile == "dangerous"
    assert cli.picker_calls == 1
    assert cli.confirm_calls == 1


def test_profile_switch_dangerous_cancelled():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="auto")
    agent_config = _make_agent_config("auto")
    cli = _FakeCLI(
        manager,
        agent_config,
        profile_responses=["dangerous"],
        confirm_responses=[False],
    )

    DatusCLI._cmd_profile(cli, "")

    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"


def test_profile_dialog_cancel_keeps_current():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="auto")
    agent_config = _make_agent_config("auto")
    cli = _FakeCLI(manager, agent_config, profile_responses=[None])

    DatusCLI._cmd_profile(cli, "")

    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"


def test_profile_select_same_profile_is_noop():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="auto")
    manager.approve_for_session("db_tools", "execute_ddl")
    agent_config = _make_agent_config("auto")
    cli = _FakeCLI(manager, agent_config, profile_responses=["auto"])

    DatusCLI._cmd_profile(cli, "")

    assert cli.active_profile == "auto"
    assert manager.active_profile == "auto"
    assert manager._session_approvals  # not cleared


def test_profile_every_dangerous_transition_reconfirms():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="normal")
    agent_config = _make_agent_config("normal")
    cli = _FakeCLI(
        manager,
        agent_config,
        profile_responses=["dangerous", "auto", "dangerous"],
        confirm_responses=[True, True],
    )

    DatusCLI._cmd_profile(cli, "")
    assert cli.active_profile == "dangerous"
    DatusCLI._cmd_profile(cli, "")
    assert cli.active_profile == "auto"
    DatusCLI._cmd_profile(cli, "")
    assert cli.active_profile == "dangerous"

    assert cli.confirm_calls == 2  # first and third dangerous transitions


def test_profile_no_current_node_still_works():
    from datus.cli.repl import DatusCLI

    agent_config = _make_agent_config("normal")

    class _NoNodeCLI(_FakeCLI):
        def __init__(self):
            self.console = MagicMock()
            self.agent_config = agent_config
            self.active_profile = agent_config.active_profile_name
            self.chat_commands = MagicMock()
            self.chat_commands.current_node = None
            self._profile_responses = ["auto"]
            self._confirm_responses = []
            self.picker_calls = 0
            self.confirm_calls = 0

    cli = _NoNodeCLI()
    DatusCLI._cmd_profile(cli, "")

    assert cli.active_profile == "auto"
    assert agent_config.active_profile_name == "auto"


def test_run_profile_picker_delegates_to_picker_app(monkeypatch):
    """``_run_profile_picker`` hands selection off to ``ProfilePickerApp``
    and respects the outer TUI's input-suspension contract when present.
    """
    from datus.cli.repl import DatusCLI

    calls = {"run": 0}

    class _FakePicker:
        def __init__(self, console, current):
            assert current == "normal"

        def run(self):
            calls["run"] += 1
            return "dangerous"

    monkeypatch.setattr("datus.cli.profile_picker_app.ProfilePickerApp", _FakePicker)

    class _CLIStub:
        console = MagicMock()
        tui_app = None

    result = DatusCLI._run_profile_picker(_CLIStub(), "normal")
    assert result == "dangerous"
    assert calls["run"] == 1


def test_run_dangerous_confirm_delegates_to_confirm_app(monkeypatch):
    """``_run_dangerous_confirm`` is a thin wrapper over ``DangerousConfirmApp``."""
    from datus.cli.repl import DatusCLI

    class _FakeConfirm:
        def __init__(self, console):
            pass

        def run(self):
            return True

    monkeypatch.setattr("datus.cli.profile_picker_app.DangerousConfirmApp", _FakeConfirm)

    class _CLIStub:
        console = MagicMock()
        tui_app = None

    assert DatusCLI._run_dangerous_confirm(_CLIStub()) is True


def test_run_profile_picker_suspends_tui_when_present(monkeypatch):
    """When a TUI is active, the picker runs inside ``suspend_input()``.

    Otherwise picker output races the outer Application's rendering and
    breaks stdin exclusivity.
    """
    from contextlib import contextmanager

    from datus.cli.repl import DatusCLI

    suspend_entries = {"n": 0}

    @contextmanager
    def _fake_suspend():
        suspend_entries["n"] += 1
        yield

    class _FakeTUI:
        def suspend_input(self):
            return _fake_suspend()

    class _FakePicker:
        def __init__(self, console, current):
            pass

        def run(self):
            return "auto"

    monkeypatch.setattr("datus.cli.profile_picker_app.ProfilePickerApp", _FakePicker)

    class _CLIStub:
        console = MagicMock()
        tui_app = _FakeTUI()

    DatusCLI._run_profile_picker(_CLIStub(), "normal")
    assert suspend_entries["n"] == 1


def test_profile_malformed_rules_fails_closed_to_normal():
    """``/profile dangerous`` with a broken rules list must refuse to apply.

    Mirrors startup's fail-closed: expanding permissions while silently
    dropping restrictive user overrides is the worst-case outcome.
    """
    from datus.cli.repl import DatusCLI

    # Start on ``auto``, user rules in ``_raw_permissions`` are malformed
    # (a non-string pattern), so ``build_effective_config`` raises.
    from datus.configuration.agent_config import AgentConfig

    agent_config = AgentConfig.__new__(AgentConfig)
    agent_config.active_profile_name = "auto"
    agent_config._raw_permissions = {
        "profile": "auto",
        "rules": [{"tool": "db_tools", "pattern": "x", "permission": "not_a_valid_level"}],
    }
    agent_config.permissions_config = None  # overwritten below

    manager = PermissionManager(active_profile="auto")
    cli = _FakeCLI(manager, agent_config, profile_responses=["dangerous"], confirm_responses=[True])

    DatusCLI._cmd_profile(cli, "")

    # Fail-closed: we don't land on ``dangerous`` — we land on ``normal``.
    assert cli.active_profile == "normal"
    assert agent_config.active_profile_name == "normal"


def test_profile_picker_returns_none_is_noop():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="normal")
    cli = _FakeCLI(manager, _make_agent_config("normal"), profile_responses=[None])

    DatusCLI._cmd_profile(cli, "")

    # Cancellation path: active profile unchanged, no confirm prompt asked.
    assert cli.active_profile == "normal"
    assert cli.confirm_calls == 0


def test_profile_unknown_selection_is_noop():
    from datus.cli.repl import DatusCLI

    manager = PermissionManager(active_profile="normal")
    cli = _FakeCLI(manager, _make_agent_config("normal"), profile_responses=["bogus"])

    DatusCLI._cmd_profile(cli, "")

    # Unknown profile name hits the guard, prints error, doesn't switch.
    assert cli.active_profile == "normal"
