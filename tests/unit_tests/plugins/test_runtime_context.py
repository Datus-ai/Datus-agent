# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for managed plugin CLI runtime-context bridging."""

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from datus.plugins import runtime_context
from datus.tools.func_tool.bash_tool import BashExecutionContext, BashTool


class _Config:
    plugins_enabled = True
    plugin_paths = []
    config_mutable = False

    def __init__(self, profile=None, active=True):
        self.profile = profile or {"name": "prod", "token": "tenant-secret"}
        self.active = active
        self.requested = None

    def plugin_active(self, name):
        return self.active

    def get_plugin_profile(self, name, profile=None):
        self.requested = (name, profile)
        return dict(self.profile)


def test_context_round_trip_unicode():
    original = runtime_context.PluginRuntimeContext(
        plugin_name="hello",
        profile={"name": "生产", "token": "s3cr3t"},
        plugin_path="/opt/plugins/hello",
    )
    decoded = runtime_context.PluginRuntimeContext.decode(original.encode(), expected_plugin="hello")
    assert decoded == original


def test_context_rejects_wrong_plugin_and_malformed_value():
    encoded = runtime_context.PluginRuntimeContext("hello", {}).encode()
    with pytest.raises(runtime_context.PluginRuntimeContextError, match="not `other`"):
        runtime_context.PluginRuntimeContext.decode(encoded, expected_plugin="other")
    with pytest.raises(runtime_context.PluginRuntimeContextError, match="Malformed"):
        runtime_context.PluginRuntimeContext.decode("v1.not-base64!")


def test_context_rejects_non_json_profile():
    with pytest.raises(runtime_context.PluginRuntimeContextError, match="not JSON-serializable"):
        runtime_context.PluginRuntimeContext("hello", {"bad": object()}).encode()


def test_context_rejects_payload_over_size_limit():
    profile = {"value": "x" * runtime_context.MAX_RUNTIME_CONTEXT_SIZE}
    with pytest.raises(runtime_context.PluginRuntimeContextError, match="exceeds 64 KiB"):
        runtime_context.PluginRuntimeContext("hello", profile).encode()


def test_prepare_plain_command_resolves_explicit_profile(monkeypatch, tmp_path):
    plugin_dir = tmp_path / "hello"
    plugin_dir.mkdir()
    monkeypatch.setattr(runtime_context, "_resolve_plugin_path", lambda config, name: plugin_dir)
    config = _Config()

    prepared = runtime_context.prepare_plugin_invocation(
        "datus hello --profile staging greet Alice",
        config,
    )

    assert prepared is not None
    assert config.requested == ("hello", "staging")
    assert prepared.sandbox_read_dirs == [str(plugin_dir)]
    decoded = runtime_context.PluginRuntimeContext.decode(
        prepared.env[runtime_context.RUNTIME_CONTEXT_ENV],
        expected_plugin="hello",
    )
    assert decoded.profile["token"] == "tenant-secret"
    assert runtime_context.RUNTIME_CONTEXT_ENV in prepared.command
    assert "tenant-secret" not in prepared.command


def test_prepare_pipeline_injects_only_datus_segment(monkeypatch, tmp_path):
    plugin_dir = tmp_path / "hello"
    plugin_dir.mkdir()
    monkeypatch.setattr(runtime_context, "_resolve_plugin_path", lambda config, name: plugin_dir)

    prepared = runtime_context.prepare_plugin_invocation(
        "printf input | datus hello run | grep ok",
        _Config(),
    )

    assert prepared is not None
    segments = prepared.command.split(" | ")
    assert runtime_context.RUNTIME_CONTEXT_ENV in segments[1]
    assert runtime_context.RUNTIME_CONTEXT_ENV not in segments[2]


@pytest.mark.parametrize(
    "command",
    [
        "datus hello run && echo done",
        "echo before; datus hello run",
        "datus hello run > out.txt",
        "datus hello run |& grep ok",
        "datus hello run | datus hello inspect",
        "(datus hello run)",
        "echo $(datus hello run)",
        'echo "$(datus hello run)"',
        "echo `datus hello run`",
    ],
)
def test_prepare_rejects_unsupported_managed_shell_forms(command):
    with pytest.raises(runtime_context.PluginRuntimeContextError):
        runtime_context.prepare_plugin_invocation(command, _Config())


def test_prepare_rejects_managed_config_override(monkeypatch):
    monkeypatch.setattr(runtime_context, "_resolve_plugin_path", lambda config, name: None)
    with pytest.raises(runtime_context.PluginRuntimeContextError, match="--config"):
        runtime_context.prepare_plugin_invocation("datus hello --config tenant.yml run", _Config())
    with pytest.raises(runtime_context.PluginRuntimeContextError, match="--config"):
        runtime_context.prepare_plugin_invocation("datus hello --config", _Config())


def test_prepare_non_datus_command_is_ignored():
    assert runtime_context.prepare_plugin_invocation("printf hello | grep ell", _Config()) is None


def test_bash_pipeline_scopes_context_to_plugin_segment(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    plugin_dir = tmp_path / "hello"
    plugin_dir.mkdir()
    fake_datus = workspace / "datus"
    fake_datus.write_text(
        "#!/bin/bash\n"
        "python -c 'import os; "
        'print("plugin=" + str(bool(os.environ.get("DATUS_PLUGIN_RUNTIME_CONTEXT"))))\'\n',
        encoding="utf-8",
    )
    fake_datus.chmod(0o755)
    monkeypatch.setattr(runtime_context, "_resolve_plugin_path", lambda config, name: plugin_dir)

    config = _Config()

    def provider(command):
        prepared = runtime_context.prepare_plugin_invocation(command, config)
        if prepared is None:
            return None
        return BashExecutionContext(
            command=prepared.command,
            env=prepared.env,
            sandbox_read_dirs=prepared.sandbox_read_dirs,
        )

    tool = BashTool(
        workspace_root=str(workspace),
        allowed_patterns=["*"],
        extra_env={"PATH": f"{workspace}{os.pathsep}{os.environ.get('PATH', '')}"},
        execution_context_provider=provider,
    )
    result = tool.bash(
        "datus hello run | "
        "python -c 'import os,sys; "
        "print(sys.stdin.read().strip()); "
        'print("sibling=" + str(bool(os.environ.get("DATUS_PLUGIN_RUNTIME_CONTEXT"))))\''
    )

    assert result.success == 1
    assert "plugin=True" in result.result
    assert "sibling=False" in result.result


def test_bash_provider_does_not_mutate_parent_environment(monkeypatch, tmp_path):
    plugin_dir = tmp_path / "hello"
    plugin_dir.mkdir()
    monkeypatch.setattr(runtime_context, "_resolve_plugin_path", lambda config, name: plugin_dir)
    before = os.environ.get(runtime_context.RUNTIME_CONTEXT_ENV)

    def provider(command):
        prepared = runtime_context.prepare_plugin_invocation(command, _Config())
        assert prepared is not None
        return BashExecutionContext(prepared.command, prepared.env, prepared.sandbox_read_dirs)

    tool = BashTool(
        workspace_root=str(tmp_path),
        allowed_patterns=["*"],
        execution_context_provider=provider,
    )
    # No real datus execution is needed to prove the provider did not mutate
    # the parent; use the child env builder directly with the prepared value.
    context = provider("datus hello run")
    child_env = tool._get_safe_env(context.env)
    assert runtime_context.RUNTIME_CONTEXT_ENV in child_env
    assert os.environ.get(runtime_context.RUNTIME_CONTEXT_ENV) == before


def test_concurrent_tenants_keep_profiles_isolated_on_redirect_path(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    plugin_dir = tmp_path / "hello"
    plugin_dir.mkdir()
    fake_datus = workspace / "datus"
    fake_datus.write_text(
        "#!/usr/bin/env python3\n"
        "import base64\n"
        "import json\n"
        "import os\n"
        "\n"
        'value = os.environ["DATUS_PLUGIN_RUNTIME_CONTEXT"]\n'
        'payload = value.split(".", 1)[1]\n'
        'data = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))\n'
        'print(data["profile"]["token"])\n',
        encoding="utf-8",
    )
    fake_datus.chmod(0o755)
    monkeypatch.delenv(runtime_context.RUNTIME_CONTEXT_ENV, raising=False)
    monkeypatch.setattr(runtime_context, "_resolve_plugin_path", lambda config, name: plugin_dir)

    def run_for_tenant(token):
        config = _Config(profile={"name": token, "token": token})

        def provider(command):
            prepared = runtime_context.prepare_plugin_invocation(command, config)
            assert prepared is not None
            return BashExecutionContext(prepared.command, prepared.env, prepared.sandbox_read_dirs)

        tool = BashTool(
            workspace_root=str(workspace),
            allowed_patterns=["*"],
            extra_env={"PATH": f"{workspace}{os.pathsep}{os.environ.get('PATH', '')}"},
            output_dir_provider=lambda: output_dir,
            execution_context_provider=provider,
        )
        return tool.bash("datus hello show-profile")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_for_tenant, ["tenant-a-secret", "tenant-b-secret"]))

    assert [result.success for result in results] == [1, 1]
    assert [result.result.strip() for result in results] == [
        "tenant-a-secret",
        "tenant-b-secret",
    ]
    assert runtime_context.RUNTIME_CONTEXT_ENV not in os.environ
