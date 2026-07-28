# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for managed plugin CLI runtime-context bridging."""

import os
import re
from concurrent.futures import ThreadPoolExecutor

import pytest

from datus.plugins import runtime_context
from datus.tools.func_tool.bash_tool import BashExecutionContext, BashTool

_CTX = runtime_context.RUNTIME_CONTEXT_ENV


def _normalized_command(prepared):
    """Rewritten command with the random shell variable name replaced by ``V``."""
    return re.sub(r"__datus_plugin_ctx_[0-9a-f]{32}", "V", prepared.command)


def _expected(body: str) -> str:
    """The prologue every rewritten command carries, followed by ``body``."""
    return f'V="${{{_CTX}}}"; unset {_CTX}; ' + body


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

    assert isinstance(prepared, runtime_context.PreparedPluginInvocation)
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

    assert isinstance(prepared, runtime_context.PreparedPluginInvocation)
    segments = prepared.command.split(" | ")
    assert runtime_context.RUNTIME_CONTEXT_ENV in segments[1]
    assert runtime_context.RUNTIME_CONTEXT_ENV not in segments[2]


@pytest.mark.parametrize(
    "command,expected_body",
    [
        # Redirections stay attached to the command they belong to.
        ("datus hello run > out.txt", f'{_CTX}="${{V}}" datus hello run > out.txt'),
        ("datus hello run >> log 2>&1", f'{_CTX}="${{V}}" datus hello run >> log 2>&1'),
        ("> out.txt datus hello run", f'> out.txt {_CTX}="${{V}}" datus hello run'),
        (">out.txt datus hello run", f'>out.txt {_CTX}="${{V}}" datus hello run'),
        # An fd duplication names its target inside the operator, so it consumes
        # one word — skipping two would make `hello` look like the command word
        # and leave the invocation unbridged.
        ("2>&1 datus hello run", f'2>&1 {_CTX}="${{V}}" datus hello run'),
        ("<&3 datus hello run", f'<&3 {_CTX}="${{V}}" datus hello run'),
        ("3<&4- datus hello run", f'3<&4- {_CTX}="${{V}}" datus hello run'),
        # A here-string is inline: its `<<` must not re-match as a heredoc, whose
        # body skip would swallow the rest of the command.
        ("datus hello run <<< 'x'", f"{_CTX}=\"${{V}}\" datus hello run <<< 'x'"),
        ("echo a <<< 'x'\ndatus hello run", f"echo a <<< 'x'\n{_CTX}=\"${{V}}\" datus hello run"),
        # Braces inside a word are brace expansion, not a command grouping.
        (
            "datus hello run --opt={a,b} --x 1",
            f'{_CTX}="${{V}}" datus hello run --opt={{a,b}} --x 1',
        ),
        # Lists: the assignment goes in front of the plugin command only.
        ("cd /tmp && datus hello run", f'cd /tmp && {_CTX}="${{V}}" datus hello run'),
        ("echo before; datus hello run", f'echo before; {_CTX}="${{V}}" datus hello run'),
        ("datus hello run || echo failed", f'{_CTX}="${{V}}" datus hello run || echo failed'),
        ("datus hello run |& grep ok", f'{_CTX}="${{V}}" datus hello run |& grep ok'),
        ("echo a\ndatus hello run\necho b", f'echo a\n{_CTX}="${{V}}" datus hello run\necho b'),
        # Groupings and compound commands keep their structure; the assignment
        # must land on the command word, never on `(`, `{` or the `do` keyword.
        ("(datus hello run)", f'({_CTX}="${{V}}" datus hello run)'),
        ("{ datus hello run; }", f'{{ {_CTX}="${{V}}" datus hello run; }}'),
        ("for i in 1 2; do datus hello run; done", f'for i in 1 2; do {_CTX}="${{V}}" datus hello run; done'),
        ("if datus hello run; then echo ok; fi", f'if {_CTX}="${{V}}" datus hello run; then echo ok; fi'),
        ("time datus hello run", f'time {_CTX}="${{V}}" datus hello run'),
        # Leading assignments precede the command word too.
        ("FOO=1 datus hello run", f'FOO=1 {_CTX}="${{V}}" datus hello run'),
        # Expansions and substitutions that do not invoke datus are preserved.
        (
            "datus hello run --date $(date +%F)",
            f'{_CTX}="${{V}}" datus hello run --date $(date +%F)',
        ),
        (
            'datus hello run --token "${HELLO_TOKEN}"',
            f'{_CTX}="${{V}}" datus hello run --token "${{HELLO_TOKEN}}"',
        ),
        # A heredoc body is payload, not a command position.
        (
            'datus hello post <<EOF\n{"note": "datus hello run"}\nEOF',
            f'{_CTX}="${{V}}" datus hello post <<EOF\n{{"note": "datus hello run"}}\nEOF',
        ),
    ],
)
def test_prepare_supports_full_shell_command_shapes(monkeypatch, command, expected_body):
    monkeypatch.setattr(runtime_context, "_resolve_plugin_path", lambda config, name: None)

    prepared = runtime_context.prepare_plugin_invocation(command, _Config())

    assert isinstance(prepared, runtime_context.PreparedPluginInvocation)
    assert _normalized_command(prepared) == _expected(expected_body)


@pytest.mark.parametrize(
    "command",
    [
        'datus hello run --sql "select a from t where b > 1 and c < 2"',
        "datus hello run --sql 'select * from t; drop table x'",
        'datus hello run --note "it\'s fine" --re "a|b" | grep -E "x|y"',
        "datus hello run 2> err.log 1>&2",
        "datus hello run &",
        'datus hello --profile prod run --dir "$HOME/x" > "$HOME/o.json"',
        'echo "$(ls)" && datus hello run',
        "datus hello run \\\n  --limit 5 \\\n  --format json",
        "datus hello post <<-'EOF'\n\t{\"a\": 1}\n\tEOF",
        'datus hello run --path "a b/c(d)"',
        "datus hello run --opt={a,b} --nested={x,{y,z}}",
        "while true; do datus hello run; break; done",
        "case x in x) datus hello run;; esac",
    ],
)
def test_prepare_preserves_the_original_command_text(monkeypatch, command):
    """Removing the injected assignment must restore the command byte for byte.

    The rewrite only ever inserts a prologue and one inline assignment, so
    quoting, line continuations, redirections and heredoc bodies written by the
    model reach Bash unchanged.
    """
    monkeypatch.setattr(runtime_context, "_resolve_plugin_path", lambda config, name: None)

    prepared = runtime_context.prepare_plugin_invocation(command, _Config())

    assert isinstance(prepared, runtime_context.PreparedPluginInvocation)
    body = _normalized_command(prepared).replace(_expected(""), "", 1)
    assert body.replace(f'{_CTX}="${{V}}" ', "", 1) == command


@pytest.mark.parametrize(
    "command",
    [
        # Substitutions that do not invoke datus are scanned through, not rejected.
        "echo $(grep 'a b' f) && datus hello run",
        'echo $(grep "a b" f); datus hello run',
        "echo $(echo `date`); datus hello run",
        "echo $(echo $(date +%F)); datus hello run",
        "echo $( (cd /tmp && ls) ); datus hello run",
        # Quoting inside one word: escapes, backticks and expansions.
        'datus hello run --x "`date`"',
        'datus hello run --x "a\\"b"',
        'datus hello run --x "${A:-{brace\\}}"',
        "datus hello run --x \"$(printf '%s' ok)\"",
        # Heredoc delimiter spellings.
        "datus hello post << EOF\npayload\nEOF",
        'datus hello post <<"EOF"\npayload\nEOF',
        "datus hello post <<-EOF\n\tpayload\n\tEOF",
        "datus hello post <<EOF\nunterminated body",
        "datus hello run <&3",
    ],
)
def test_prepare_scans_nested_shell_syntax_without_rejecting(monkeypatch, command):
    """Nested quoting/substitution must be traversed, not treated as a control.

    Each of these carries syntax the scanner has to walk through to find the
    command word; a mis-parse would either reject a valid command or move the
    injected assignment.
    """
    monkeypatch.setattr(runtime_context, "_resolve_plugin_path", lambda config, name: None)

    prepared = runtime_context.prepare_plugin_invocation(command, _Config())

    assert isinstance(prepared, runtime_context.PreparedPluginInvocation)
    body = _normalized_command(prepared).replace(_expected(""), "", 1)
    assert body.replace(f'{_CTX}="${{V}}" ', "", 1) == command


@pytest.mark.parametrize(
    "command,message",
    [
        ('datus hello run --x "unterminated', "unbalanced double quote"),
        ("datus hello run --x `oops", "unbalanced backtick"),
        ("datus hello run --x $(oops", "unterminated command substitution"),
        ("datus hello run --x ${oops", r"unbalanced \$\{\.\.\.\} expansion"),
        ("datus hello post <<'EOF", "unbalanced single quote"),
        ('datus hello post <<"EOF', "unbalanced double quote"),
        ("echo $(grep 'a && datus hello run", "unbalanced single quote"),
        ('echo $(grep "a && datus hello run', "unbalanced double quote"),
        ("echo `datus hello run", "unbalanced backtick"),
        (
            "echo " + "$(" * 20 + "date" + ")" * 20 + "; datus hello run",
            "command substitution nested too deeply",
        ),
    ],
)
def test_prepare_reports_malformed_shell_syntax(monkeypatch, command, message):
    """Unparsable commands fail closed with the reason, never silently unbridged.

    Returning ``None`` here would run the plugin CLI with no runtime context,
    which falls back to local config resolution the managed path forbids.
    """
    monkeypatch.setattr(runtime_context, "_resolve_plugin_path", lambda config, name: None)

    with pytest.raises(runtime_context.PluginRuntimeContextError, match=f"syntax: {message}"):
        runtime_context.prepare_plugin_invocation(command, _Config())


@pytest.mark.parametrize(
    "command,message",
    [
        ("datus hello run | datus hello inspect", "only one plugin CLI"),
        ("datus hello run; datus hello inspect", "only one plugin CLI"),
        ("echo $(datus hello run)", "command substitution"),
        ('echo "$(datus hello run)"', "command substitution"),
        ("echo `datus hello run`", "command substitution"),
        ('datus hello run --token "${x:-$(datus hello token)}"', "command substitution"),
        ("timeout 30 datus hello run", "directly as the command word"),
        ("echo a | xargs datus hello run", "directly as the command word"),
        ("datus hello run 'unbalanced", "syntax"),
    ],
)
def test_prepare_rejects_unsupported_managed_shell_forms(monkeypatch, command, message):
    monkeypatch.setattr(runtime_context, "_resolve_plugin_path", lambda config, name: None)

    with pytest.raises(runtime_context.PluginRuntimeContextError, match=message):
        runtime_context.prepare_plugin_invocation(command, _Config())


@pytest.mark.parametrize(
    "command",
    [
        "ls -la > out.txt && echo ok",
        "echo $(date +%F)",
        "echo hi # datus hello run",
        "datus --help",
        "datus",
    ],
)
def test_prepare_ignores_commands_without_a_plugin_invocation(monkeypatch, command):
    monkeypatch.setattr(runtime_context, "_resolve_plugin_path", lambda config, name: None)

    assert runtime_context.prepare_plugin_invocation(command, _Config()) is None


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


def test_bash_command_list_scopes_context_to_plugin_command(monkeypatch, tmp_path):
    """A `&&` list with a redirection still isolates the context to datus."""
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
    sibling = 'python -c \'import os; print("sibling=" + str(bool(os.environ.get("DATUS_PLUGIN_RUNTIME_CONTEXT"))))\''
    result = tool.bash(f"datus hello run > plugin.txt && {sibling} && cat plugin.txt")

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
        assert isinstance(prepared, runtime_context.PreparedPluginInvocation)
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
            assert isinstance(prepared, runtime_context.PreparedPluginInvocation)
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
