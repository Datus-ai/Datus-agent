# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.plugins`` dispatch + ``--profile``/``--config`` split."""

from typing import Any, Optional

from datus.cli import main as cli_main
from datus.cli.main import _dispatch_plugin_command, _split_plugin_globals

# ── _split_plugin_globals ────────────────────────────────────────────────────


def test_split_consumes_leading_profile_and_config():
    profile, config, rest = _split_plugin_globals(
        ["--profile", "staging", "--config", "a.yml", "dags", "list", "--json"]
    )
    assert profile == "staging"
    assert config == "a.yml"
    assert rest == ["dags", "list", "--json"]


def test_split_equals_form():
    profile, config, rest = _split_plugin_globals(["--profile=prod", "--config=x.yml", "dags", "trigger", "d"])
    assert profile == "prod"
    assert config == "x.yml"
    assert rest == ["dags", "trigger", "d"]


def test_split_stops_at_first_command_token():
    profile, config, rest = _split_plugin_globals(["dags", "--profile", "ignored"])
    assert profile is None and config is None
    assert rest == ["dags", "--profile", "ignored"]


def test_split_no_globals():
    assert _split_plugin_globals(["dags", "list"]) == (None, None, ["dags", "list"])


# ── _dispatch_plugin_command ─────────────────────────────────────────────────


class _StubPlugin:
    """Records how datus constructed and drove it."""

    last: dict = {}

    def __init__(self, profile: Optional[dict] = None) -> None:
        _StubPlugin.last["profile"] = profile

    def run_cli(self, argv: list[str]) -> int:
        _StubPlugin.last["argv"] = argv
        return 7


class _StubConfig:
    def __init__(self, profile: dict) -> None:
        self._profile = profile
        self.requested: dict = {}

    def get_plugin_profile(self, name: str, profile: Optional[str] = None) -> dict:
        self.requested["name"] = name
        self.requested["profile"] = profile
        return self._profile


def _patch_dispatch(
    monkeypatch: Any,
    *,
    plugin_cls: type,
    profile_dict: dict,
    load_calls: Optional[list] = None,
) -> "_StubConfig":
    monkeypatch.setattr("datus.plugins.registry.plugin_entry_point_exists", lambda name: True)
    monkeypatch.setattr("datus.plugins.registry.load_plugin_class", lambda name: plugin_cls)

    stub_cfg = _StubConfig(profile_dict)

    def fake_load_agent_config(**kwargs):
        if load_calls is not None:
            load_calls.append(kwargs)
        return stub_cfg

    monkeypatch.setattr("datus.configuration.agent_config_loader.load_agent_config", fake_load_agent_config)
    return stub_cfg


def test_dispatch_constructs_plugin_with_resolved_profile(monkeypatch):
    _StubPlugin.last = {}
    profile = {"name": "prod", "api_base_url": "http://h"}
    stub_cfg = _patch_dispatch(monkeypatch, plugin_cls=_StubPlugin, profile_dict=profile)

    rc = _dispatch_plugin_command(["hello", "--profile", "prod", "dags", "list"])

    assert rc == 7  # run_cli's return code, coerced to int
    assert _StubPlugin.last["profile"] == profile
    assert _StubPlugin.last["argv"] == ["dags", "list"]  # globals stripped
    assert stub_cfg.requested == {"name": "hello", "profile": "prod"}


def test_dispatch_forwards_config_path(monkeypatch):
    _StubPlugin.last = {}
    load_calls = []
    _patch_dispatch(monkeypatch, plugin_cls=_StubPlugin, profile_dict={}, load_calls=load_calls)

    _dispatch_plugin_command(["hello", "--config", "/tmp/agent.yml", "version"])

    assert load_calls == [{"config": "/tmp/agent.yml"}]


def test_dispatch_no_config_flag_omits_config_kwarg(monkeypatch):
    _StubPlugin.last = {}
    load_calls = []
    _patch_dispatch(monkeypatch, plugin_cls=_StubPlugin, profile_dict={}, load_calls=load_calls)

    _dispatch_plugin_command(["hello", "version"])

    assert load_calls == [{}]  # no config → datus default resolution


def test_dispatch_unknown_plugin_returns_none(monkeypatch):
    monkeypatch.setattr("datus.plugins.registry.plugin_entry_point_exists", lambda name: False)
    assert _dispatch_plugin_command(["mystery", "x"]) is None


def test_dispatch_broken_plugin_load_falls_through(monkeypatch):
    """An entry point that exists but fails to load must fall through (None)."""
    _patch_dispatch(monkeypatch, plugin_cls=_StubPlugin, profile_dict={})
    monkeypatch.setattr("datus.plugins.registry.load_plugin_class", lambda name: None)
    assert _dispatch_plugin_command(["hello", "version"]) is None


def test_dispatch_flag_only_returns_none():
    assert _dispatch_plugin_command(["--web"]) is None
    assert _dispatch_plugin_command([]) is None


def test_dispatch_reserved_name_returns_none():
    for reserved in cli_main._RESERVED_SUBCOMMANDS:
        assert _dispatch_plugin_command([reserved, "x"]) is None


def test_dispatch_config_error_returns_3(monkeypatch):
    monkeypatch.setattr("datus.plugins.registry.plugin_entry_point_exists", lambda name: True)
    monkeypatch.setattr("datus.plugins.registry.load_plugin_class", lambda name: _StubPlugin)

    def boom(**kwargs):
        raise RuntimeError("bad config")

    monkeypatch.setattr("datus.configuration.agent_config_loader.load_agent_config", boom)
    assert _dispatch_plugin_command(["hello", "dags", "list"]) == 3


def test_dispatch_plugin_run_error_returns_1(monkeypatch):
    class _Boom:
        def __init__(self, profile=None):
            pass

        def run_cli(self, argv):
            raise RuntimeError("plugin blew up")

    _patch_dispatch(monkeypatch, plugin_cls=_Boom, profile_dict={})
    assert _dispatch_plugin_command(["hello", "dags", "list"]) == 1


def test_dispatch_refused_when_plugins_disabled(monkeypatch, capsys):
    _StubPlugin.last = {}
    stub_cfg = _patch_dispatch(monkeypatch, plugin_cls=_StubPlugin, profile_dict={"name": "prod"})
    stub_cfg.plugins_enabled = False

    rc = _dispatch_plugin_command(["hello", "dags", "list"])

    assert rc == 3
    assert "plugins are disabled" in capsys.readouterr().err
    # Neither profile resolution nor the plugin itself ran.
    assert stub_cfg.requested == {}
    assert _StubPlugin.last == {}


def test_dispatch_disabled_never_imports_plugin(monkeypatch):
    """With plugins disabled the plugin package must not even be loaded —
    ``ep.load()`` runs arbitrary module-level code."""
    stub_cfg = _patch_dispatch(monkeypatch, plugin_cls=_StubPlugin, profile_dict={})
    stub_cfg.plugins_enabled = False

    def must_not_load(name):
        raise AssertionError("load_plugin_class must not be called when plugins are disabled")

    monkeypatch.setattr("datus.plugins.registry.load_plugin_class", must_not_load)
    assert _dispatch_plugin_command(["hello", "dags", "list"]) == 3


def test_dispatch_run_cli_none_maps_to_zero(monkeypatch):
    class _NoneRc:
        def __init__(self, profile=None):
            pass

        def run_cli(self, argv):
            return None

    _patch_dispatch(monkeypatch, plugin_cls=_NoneRc, profile_dict={})
    assert _dispatch_plugin_command(["hello", "version"]) == 0


def _rc_plugin(rc_value: Any) -> type:
    class _Rc:
        def __init__(self, profile: Optional[dict] = None) -> None:
            pass

        def run_cli(self, argv: list[str]) -> Any:
            return rc_value

    return _Rc


def test_dispatch_run_cli_non_int_rc_does_not_crash(monkeypatch):
    """A legacy handler returning a non-int (e.g. 'ok') must not crash a
    successful run with ValueError."""
    _patch_dispatch(monkeypatch, plugin_cls=_rc_plugin("ok"), profile_dict={})
    assert _dispatch_plugin_command(["hello", "version"]) == 0


def test_dispatch_run_cli_bool_rc_maps_to_exit_semantics(monkeypatch):
    _patch_dispatch(monkeypatch, plugin_cls=_rc_plugin(True), profile_dict={})
    assert _dispatch_plugin_command(["hello", "version"]) == 0
    _patch_dispatch(monkeypatch, plugin_cls=_rc_plugin(False), profile_dict={})
    assert _dispatch_plugin_command(["hello", "version"]) == 1
