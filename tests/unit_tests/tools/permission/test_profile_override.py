# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CI-level tests for apply_profile_override.

The helper is what carries a per-request permission profile onto one node's
manager — the API path cannot put it on the shared AgentConfig, and the
subagent path has no other way to learn what its parent is running under.
Its return value is the contract callers branch on, so every path asserts it.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datus.tools.permission.profile_override import apply_profile_override


def _agent_config(raw_permissions=None):
    return SimpleNamespace(active_profile_name="normal", _raw_permissions=raw_permissions or {})


def _manager(active_profile="normal", switch_side_effect=None):
    manager = MagicMock()
    manager.active_profile = active_profile
    if switch_side_effect is not None:
        manager.switch_profile.side_effect = switch_side_effect
    return manager


class TestApplyProfileOverride:
    def test_noop_without_a_profile_name(self):
        manager = _manager()
        assert apply_profile_override(manager, _agent_config(), None) is False
        manager.switch_profile.assert_not_called()

    def test_noop_without_a_manager(self):
        # Workflow nodes skip the skill setup and never get one.
        assert apply_profile_override(None, _agent_config(), "dangerous") is False

    def test_noop_when_the_profile_is_already_active(self):
        manager = _manager(active_profile="dangerous")
        assert apply_profile_override(manager, _agent_config(), "dangerous") is False
        manager.switch_profile.assert_not_called()

    def test_switches_without_user_overrides(self):
        manager = _manager()
        assert apply_profile_override(manager, _agent_config(), "auto") is True
        manager.switch_profile.assert_called_once_with("auto", user_overrides=None)

    def test_switches_with_user_overrides_built_from_agent_yml(self):
        from datus.tools.permission.permission_config import PermissionConfig

        manager = _manager()
        raw = {"rules": [{"tool": "db_tools", "pattern": "*", "permission": "ask"}]}

        assert apply_profile_override(manager, _agent_config(raw), "dangerous") is True

        args, kwargs = manager.switch_profile.call_args
        assert args == ("dangerous",)
        assert isinstance(kwargs["user_overrides"], PermissionConfig)

    def test_drops_the_profile_key_from_the_raw_rules(self):
        # ``profile`` names the posture; it is not itself a rule.
        manager = _manager()
        raw = {"profile": "normal"}

        assert apply_profile_override(manager, _agent_config(raw), "auto") is True
        manager.switch_profile.assert_called_once_with("auto", user_overrides=None)

    def test_raises_when_the_user_rules_cannot_be_rebuilt(self, monkeypatch):
        """Fail closed: the bare profile base can be broader than the yaml."""
        from datus.utils.exceptions import DatusException, ErrorCode

        manager = _manager()

        def _explode(*_args, **_kwargs):
            raise ValueError("malformed rule")

        monkeypatch.setattr("datus.tools.permission.profiles.build_user_overrides", _explode)

        with pytest.raises(DatusException, match="permission_mode='auto'") as excinfo:
            apply_profile_override(manager, _agent_config({"rules": [{"bad": "shape"}]}), "auto")
        assert excinfo.value.code == ErrorCode.COMMON_CONFIG_ERROR
        assert isinstance(excinfo.value.__cause__, ValueError)
        manager.switch_profile.assert_not_called()

    def test_swallows_a_failing_switch(self):
        """The manager keeps its original profile, which is the safe side."""
        manager = _manager(switch_side_effect=RuntimeError("unknown profile"))

        assert apply_profile_override(manager, _agent_config(), "nope") is False
