# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for permission profile loading in AgentConfig."""

from datus.configuration.agent_config import AgentConfig
from datus.tools.permission.permission_config import PermissionLevel


def _make_config(permissions_raw):
    """Build a bare AgentConfig exercising only ``_init_permissions_config``.

    Real AgentConfig.__init__ requires substantial YAML; for these tests we
    instantiate with ``__new__`` and call the helper directly. If the
    project adds a richer fixture helper later, prefer that.
    """
    cfg = AgentConfig.__new__(AgentConfig)
    cfg.active_profile_name = "normal"  # pre-seed so loader can overwrite
    cfg.permissions_config = cfg._init_permissions_config(permissions_raw or {})
    return cfg


def test_missing_permissions_yields_normal_profile():
    cfg = _make_config(None)
    assert cfg.active_profile_name == "normal"
    # Normal profile has explicit read allows
    assert any(r.tool == "db_tools" and r.pattern == "read_query" for r in cfg.permissions_config.rules)


def test_empty_permissions_yields_normal_profile():
    cfg = _make_config({})
    assert cfg.active_profile_name == "normal"
    assert cfg.permissions_config.default_permission == PermissionLevel.ASK


def test_profile_field_selects_auto():
    cfg = _make_config({"profile": "auto"})
    assert cfg.active_profile_name == "auto"
    # Auto has the workspace write allows
    assert any(
        r.tool == "filesystem_tools"
        and r.pattern == "write_file"
        and PermissionLevel(r.permission) == PermissionLevel.ALLOW
        for r in cfg.permissions_config.rules
    )


def test_dangerous_profile_loads():
    cfg = _make_config({"profile": "dangerous"})
    assert cfg.active_profile_name == "dangerous"
    assert cfg.permissions_config.default_permission == PermissionLevel.ALLOW


def test_user_rules_layered_on_profile_base():
    """User's permissions.rules should be appended after profile rules,
    so last-match-wins lets users override."""
    cfg = _make_config(
        {
            "profile": "auto",
            "rules": [
                {"tool": "db_tools", "pattern": "execute_ddl", "permission": "deny"},
            ],
        }
    )
    rules = cfg.permissions_config.rules
    # The Auto base has execute_ddl ASK; user rule must appear after it.
    auto_idx = next(
        i
        for i, r in enumerate(rules)
        if r.tool == "db_tools" and r.pattern == "execute_ddl" and PermissionLevel(r.permission) == PermissionLevel.ASK
    )
    user_idx = next(
        i
        for i, r in enumerate(rules)
        if r.tool == "db_tools" and r.pattern == "execute_ddl" and PermissionLevel(r.permission) == PermissionLevel.DENY
    )
    assert user_idx > auto_idx, "user rule must be appended after profile base"


def test_invalid_profile_falls_back_to_normal(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="datus.configuration.agent_config"):
        cfg = _make_config({"profile": "yolo"})
    assert cfg.active_profile_name == "normal"
    assert any("Invalid profile" in rec.message or "yolo" in rec.message for rec in caplog.records)


def test_raw_permissions_stashed_for_runtime_switch():
    """/profile must be able to rebuild effective config without re-reading YAML."""
    raw = {
        "profile": "auto",
        "rules": [
            {"tool": "db_tools", "pattern": "execute_ddl", "permission": "deny"},
        ],
    }
    cfg = _make_config(raw)
    assert cfg._raw_permissions == raw


def test_user_rules_preserve_profile_default_when_no_explicit_default():
    """User writing rules without ``default:`` must NOT clobber profile default.

    Spec decision #3: user rules layer on top; changing default_permission
    requires the user to opt in explicitly.
    """
    cfg = _make_config(
        {
            "profile": "auto",
            "rules": [
                {"tool": "db_tools", "pattern": "execute_ddl", "permission": "deny"},
            ],
        }
    )
    assert cfg.permissions_config.default_permission == PermissionLevel.ASK


def test_user_explicit_default_wins_over_profile():
    """If user writes ``default:`` explicitly, that's an opt-in override."""
    cfg = _make_config(
        {
            "profile": "normal",
            "default": "allow",
            "rules": [],
        }
    )
    assert cfg.permissions_config.default_permission == PermissionLevel.ALLOW


def test_user_explicit_default_permission_key_also_wins():
    """``default_permission`` alias must work same as ``default``."""
    cfg = _make_config(
        {
            "profile": "normal",
            "default_permission": "deny",
        }
    )
    assert cfg.permissions_config.default_permission == PermissionLevel.DENY


def test_malformed_user_rules_falls_back_to_base(caplog):
    """Malformed permissions.rules should not crash startup — fall back
    to profile base with a warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="datus.configuration.agent_config"):
        cfg = _make_config(
            {
                "profile": "normal",
                "rules": [{"tool": "db_tools", "pattern": "x", "permission": "not_a_valid_level"}],
            }
        )
    # Fell back to base normal — default still ASK, rules match normal's count
    from datus.tools.permission.profiles import NORMAL

    assert cfg.permissions_config.default_permission == PermissionLevel.ASK
    assert len(cfg.permissions_config.rules) == len(NORMAL.rules)
    assert any(
        "Invalid permissions.rules" in rec.message or "permissions.rules" in rec.message for rec in caplog.records
    )


def test_non_mapping_permissions_falls_back_to_normal(caplog):
    """A list/scalar in ``permissions`` must not crash the loader.

    ``dict(raw)`` and ``raw.get("profile")`` both raise on a non-mapping, so
    the loader has to detect the bad shape before calling them. Expect a
    fall-back to ``normal`` and a warning that names the received type.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="datus.configuration.agent_config"):
        cfg = _make_config(["not", "a", "mapping"])
    assert cfg.active_profile_name == "normal"
    assert cfg.permissions_config.default_permission == PermissionLevel.ASK
    assert any("expected mapping" in rec.message or "list" in rec.message for rec in caplog.records)


def test_non_string_profile_field_falls_back_to_normal(caplog):
    """``permissions.profile: 42`` is not a valid profile name.

    ``get_profile`` would raise on the numeric value; catch it upstream and
    log a warning.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="datus.configuration.agent_config"):
        cfg = _make_config({"profile": 42})
    assert cfg.active_profile_name == "normal"
    assert any(
        "Invalid permissions.profile" in rec.message or "Falling back to 'normal'" in rec.message
        for rec in caplog.records
    )
