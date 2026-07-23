# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.plugins`` dispatch + ``--profile``/``--config`` split."""

from pathlib import Path
from typing import Any, Optional

from datus.cli import main as cli_main
from datus.cli.main import _dispatch_plugin_command, _split_plugin_globals
from datus.plugins.base import PluginManifest
from datus.plugins.runtime_context import RUNTIME_CONTEXT_ENV, PluginRuntimeContext

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

_MANIFEST = PluginManifest(name="hello", package_dir=Path("/tmp/hello"), cli="datus_plugin_hello.cli:main")
_NO_CLI_MANIFEST = PluginManifest(name="hello", package_dir=Path("/tmp/hello"))


class _StubCli:
    """Records how datus invoked the declared cli entry."""

    last: dict = {}

    def __init__(self, rc: Any = 7):
        self._rc = rc

    def __call__(self, argv: list, profile: dict) -> Any:
        _StubCli.last["argv"] = argv
        _StubCli.last["profile"] = profile
        return self._rc


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
    profile_dict: dict,
    cli_func: Any = None,
    manifest: Optional[PluginManifest] = _MANIFEST,
    load_calls: Optional[list] = None,
) -> "_StubConfig":
    _StubCli.last = {}
    monkeypatch.setattr("datus.plugins.registry.plugin_entry_point_exists", lambda name: True)
    monkeypatch.setattr("datus.plugins.registry.load_plugin_manifest", lambda name: manifest)
    resolved = cli_func if cli_func is not None else _StubCli()
    monkeypatch.setattr("datus.plugins.registry.resolve_code_ref", lambda ref, name: resolved)

    stub_cfg = _StubConfig(profile_dict)

    def fake_load_agent_config(**kwargs):
        if load_calls is not None:
            load_calls.append(kwargs)
        return stub_cfg

    monkeypatch.setattr("datus.configuration.agent_config_loader.load_agent_config", fake_load_agent_config)
    return stub_cfg


def test_dispatch_calls_cli_with_resolved_profile(monkeypatch):
    profile = {"name": "prod", "api_base_url": "http://h"}
    stub_cfg = _patch_dispatch(monkeypatch, profile_dict=profile)

    rc = _dispatch_plugin_command(["hello", "--profile", "prod", "dags", "list"])

    assert rc == 7  # the cli entry's return code, coerced to int
    assert _StubCli.last["profile"] == profile
    assert _StubCli.last["argv"] == ["dags", "list"]  # globals stripped
    assert stub_cfg.requested == {"name": "hello", "profile": "prod"}


def test_dispatch_forwards_config_path(monkeypatch):
    load_calls = []
    _patch_dispatch(monkeypatch, profile_dict={}, load_calls=load_calls)

    _dispatch_plugin_command(["hello", "--config", "/tmp/agent.yml", "version"])

    assert load_calls == [{"config": "/tmp/agent.yml"}]


def test_dispatch_no_config_flag_omits_config_kwarg(monkeypatch):
    load_calls = []
    _patch_dispatch(monkeypatch, profile_dict={}, load_calls=load_calls)

    _dispatch_plugin_command(["hello", "version"])

    assert load_calls == [{}]  # no config → datus default resolution


def test_dispatch_unknown_plugin_returns_none(monkeypatch):
    monkeypatch.setattr("datus.plugins.registry.plugin_entry_point_exists", lambda name: False)
    assert _dispatch_plugin_command(["mystery", "x"]) is None


def test_dispatch_injects_plugin_paths_when_unmanaged(monkeypatch, tmp_path):
    """Without a managed dir, ``agent.plugin_paths`` mounts are injected before the probe."""
    _patch_dispatch(monkeypatch, profile_dict={})
    monkeypatch.setattr("datus.plugins.store.plugin_dir", lambda name: tmp_path / "absent" / name)
    seen: dict = {}

    def fake_get_plugin_paths(config_file=""):
        seen["config"] = config_file
        return ["/mnt/plugins/hello"]

    monkeypatch.setattr("datus.configuration.agent_config_loader.get_plugin_paths", fake_get_plugin_paths)
    activated: list = []
    monkeypatch.setattr("datus.plugins.store.activate_paths", lambda paths: activated.append(paths) or [])

    rc = _dispatch_plugin_command(["hello", "--config", "/tmp/agent.yml", "version"])

    assert rc == 7
    assert activated == [["/mnt/plugins/hello"]]
    assert seen["config"] == "/tmp/agent.yml"  # the --config global reaches the pre-load read


def test_dispatch_managed_dir_skips_plugin_paths(monkeypatch, tmp_path):
    """A managed plugin's own directory wins; configured mounts are not read."""
    _patch_dispatch(monkeypatch, profile_dict={})
    managed = tmp_path / "plugins" / "hello"
    managed.mkdir(parents=True)
    monkeypatch.setattr("datus.plugins.store.plugin_dir", lambda name: managed)
    monkeypatch.setattr("datus.plugins.store.activate_name", lambda name: False)

    def fail_activate_paths(paths):
        raise AssertionError("plugin_paths must not be consulted")

    monkeypatch.setattr("datus.plugins.store.activate_paths", fail_activate_paths)

    assert _dispatch_plugin_command(["hello", "version"]) == 7


def test_dispatch_missing_manifest_falls_through(monkeypatch):
    """An entry point that exists but has no usable manifest must fall through
    (None) so a same-named legacy handler still gets its chance."""
    _patch_dispatch(monkeypatch, profile_dict={}, manifest=None)
    assert _dispatch_plugin_command(["hello", "version"]) is None


def test_dispatch_manifest_without_cli_returns_2(monkeypatch, capsys):
    _patch_dispatch(monkeypatch, profile_dict={}, manifest=_NO_CLI_MANIFEST)
    rc = _dispatch_plugin_command(["hello", "version"])
    assert rc == 2
    # Collapse whitespace: print_error renders through Rich, which may soft-wrap
    # the styled line to the (test) console width.
    assert "declares no CLI command" in " ".join(capsys.readouterr().err.split())


def test_dispatch_unloadable_cli_ref_returns_1(monkeypatch, capsys):
    _patch_dispatch(monkeypatch, profile_dict={})
    monkeypatch.setattr("datus.plugins.registry.resolve_code_ref", lambda ref, name: None)
    rc = _dispatch_plugin_command(["hello", "version"])
    assert rc == 1
    assert "could not be loaded" in " ".join(capsys.readouterr().err.split())


def test_dispatch_flag_only_returns_none():
    assert _dispatch_plugin_command(["--web"]) is None
    assert _dispatch_plugin_command([]) is None


def test_dispatch_reserved_name_returns_none():
    for reserved in cli_main._RESERVED_SUBCOMMANDS:
        assert _dispatch_plugin_command([reserved, "x"]) is None


def test_dispatch_config_error_returns_3(monkeypatch):
    monkeypatch.setattr("datus.plugins.registry.plugin_entry_point_exists", lambda name: True)
    monkeypatch.setattr("datus.plugins.registry.load_plugin_manifest", lambda name: _MANIFEST)

    def boom(**kwargs):
        raise RuntimeError("bad config")

    monkeypatch.setattr("datus.configuration.agent_config_loader.load_agent_config", boom)
    assert _dispatch_plugin_command(["hello", "dags", "list"]) == 3


def test_dispatch_cli_error_returns_1(monkeypatch):
    def boom_cli(argv, profile):
        raise RuntimeError("plugin blew up")

    _patch_dispatch(monkeypatch, profile_dict={}, cli_func=boom_cli)
    assert _dispatch_plugin_command(["hello", "dags", "list"]) == 1


def test_dispatch_refused_when_plugins_disabled(monkeypatch, capsys):
    stub_cfg = _patch_dispatch(monkeypatch, profile_dict={"name": "prod"})
    stub_cfg.plugins_enabled = False

    rc = _dispatch_plugin_command(["hello", "dags", "list"])

    assert rc == 3
    assert "plugins are disabled" in " ".join(capsys.readouterr().err.split())
    # Neither profile resolution nor the plugin itself ran.
    assert stub_cfg.requested == {}
    assert _StubCli.last == {}


def test_dispatch_refused_when_plugin_inactive(monkeypatch, capsys):
    """A plugin the project's ``plugins:`` whitelist does not enable is refused
    (its own CLI), even though the master switch is on."""
    stub_cfg = _patch_dispatch(monkeypatch, profile_dict={"name": "prod"})
    stub_cfg.plugin_active = lambda name: False

    rc = _dispatch_plugin_command(["hello", "dags", "list"])

    assert rc == 3
    assert "not active for this project" in " ".join(capsys.readouterr().err.split())
    # Neither profile resolution nor the plugin itself ran.
    assert stub_cfg.requested == {}
    assert _StubCli.last == {}


def test_dispatch_allowed_when_plugin_active(monkeypatch):
    stub_cfg = _patch_dispatch(monkeypatch, profile_dict={"name": "prod"})
    stub_cfg.plugin_active = lambda name: True

    rc = _dispatch_plugin_command(["hello", "dags", "list"])

    assert rc == 7
    assert _StubCli.last["argv"] == ["dags", "list"]


def test_dispatch_disabled_never_imports_plugin(monkeypatch):
    """With plugins disabled the plugin package must not even be imported —
    ``resolve_code_ref`` runs arbitrary module-level code."""
    stub_cfg = _patch_dispatch(monkeypatch, profile_dict={})
    stub_cfg.plugins_enabled = False

    def must_not_resolve(ref, name):
        raise AssertionError("resolve_code_ref must not be called when plugins are disabled")

    monkeypatch.setattr("datus.plugins.registry.resolve_code_ref", must_not_resolve)
    assert _dispatch_plugin_command(["hello", "dags", "list"]) == 3


def test_dispatch_cli_none_rc_maps_to_zero(monkeypatch):
    _patch_dispatch(monkeypatch, profile_dict={}, cli_func=_StubCli(rc=None))
    assert _dispatch_plugin_command(["hello", "version"]) == 0


def test_dispatch_cli_non_int_rc_does_not_crash(monkeypatch):
    """A handler returning a non-int (e.g. 'ok') must not crash a successful
    run with ValueError."""
    _patch_dispatch(monkeypatch, profile_dict={}, cli_func=_StubCli(rc="ok"))
    assert _dispatch_plugin_command(["hello", "version"]) == 0


def test_dispatch_cli_bool_rc_maps_to_exit_semantics(monkeypatch):
    _patch_dispatch(monkeypatch, profile_dict={}, cli_func=_StubCli(rc=True))
    assert _dispatch_plugin_command(["hello", "version"]) == 0
    _patch_dispatch(monkeypatch, profile_dict={}, cli_func=_StubCli(rc=False))
    assert _dispatch_plugin_command(["hello", "version"]) == 1


# ── managed runtime context ──────────────────────────────────────────────────


def test_runtime_context_bypasses_file_loader(monkeypatch):
    load_calls = []
    stub_cfg = _patch_dispatch(monkeypatch, profile_dict={"from": "file"}, load_calls=load_calls)
    runtime_profile = {"name": "tenant-a", "token": "secret-a"}
    monkeypatch.setenv(
        RUNTIME_CONTEXT_ENV,
        PluginRuntimeContext(plugin_name="hello", profile=runtime_profile).encode(),
    )

    rc = _dispatch_plugin_command(["hello", "--profile", "tenant-a", "version"])

    assert rc == 7
    assert load_calls == []
    assert stub_cfg.requested == {}
    assert _StubCli.last["profile"] == runtime_profile
    assert _StubCli.last["argv"] == ["version"]


def test_runtime_context_rejects_config_override(monkeypatch, capsys):
    _patch_dispatch(monkeypatch, profile_dict={})
    monkeypatch.setenv(
        RUNTIME_CONTEXT_ENV,
        PluginRuntimeContext(plugin_name="hello", profile={}).encode(),
    )

    assert _dispatch_plugin_command(["hello", "--config", "/tmp/other.yml", "version"]) == 3
    assert "--config" in " ".join(capsys.readouterr().err.split())


def test_runtime_context_rejects_config_flag_without_value(monkeypatch, capsys):
    _patch_dispatch(monkeypatch, profile_dict={})
    monkeypatch.setenv(
        RUNTIME_CONTEXT_ENV,
        PluginRuntimeContext(plugin_name="hello", profile={}).encode(),
    )

    assert _dispatch_plugin_command(["hello", "--config"]) == 3
    assert "--config" in " ".join(capsys.readouterr().err.split())


def test_runtime_context_mismatch_fails_closed(monkeypatch, capsys):
    load_calls = []
    _patch_dispatch(monkeypatch, profile_dict={}, load_calls=load_calls)
    monkeypatch.setenv(
        RUNTIME_CONTEXT_ENV,
        PluginRuntimeContext(plugin_name="other", profile={}).encode(),
    )

    assert _dispatch_plugin_command(["hello", "version"]) == 3
    assert load_calls == []
    assert "not `hello`" in " ".join(capsys.readouterr().err.split())


def test_runtime_context_missing_plugin_does_not_fall_through(monkeypatch, capsys):
    monkeypatch.setattr("datus.plugins.registry.plugin_entry_point_exists", lambda name: False)
    monkeypatch.setattr("datus.plugins.store.activate_paths", lambda paths: [])
    monkeypatch.setenv(
        RUNTIME_CONTEXT_ENV,
        PluginRuntimeContext(plugin_name="hello", profile={}).encode(),
    )

    assert _dispatch_plugin_command(["hello", "version"]) == 3
    assert "not installed or discoverable" in " ".join(capsys.readouterr().err.split())


def test_malformed_runtime_context_does_not_load_file(monkeypatch):
    load_calls = []
    _patch_dispatch(monkeypatch, profile_dict={}, load_calls=load_calls)
    monkeypatch.setenv(RUNTIME_CONTEXT_ENV, "v1.not-base64!")

    assert _dispatch_plugin_command(["hello", "version"]) == 3
    assert load_calls == []
