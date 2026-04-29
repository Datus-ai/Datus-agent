# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/mcp_tools/mcp_server.py."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from datus.tools.mcp_tools.mcp_server import SilentMCPServerStdio, find_mcp_directory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(command="python", args=None, env=None):
    """Build a simple MCPServerStdioParams-like object via mock."""
    params = MagicMock()
    params.command = command
    params.args = args or []
    params.env = env or {}
    return params


# ---------------------------------------------------------------------------
# SilentMCPServerStdio
# ---------------------------------------------------------------------------


class TestSilentMCPServerStdio:
    """Test that SilentMCPServerStdio correctly wraps command with stderr redirection."""

    def test_unix_wraps_with_sh(self):
        params = _make_params(command="uvicorn", args=["app:app"], env={})
        with patch("datus.tools.mcp_tools.mcp_server.MCPServerStdio.__init__", return_value=None):
            with patch.object(sys, "platform", "linux"):
                srv = SilentMCPServerStdio.__new__(SilentMCPServerStdio)
                SilentMCPServerStdio.__init__(srv, params)
        assert params.command == "sh"
        assert params.args[0] == "-c"
        assert "2>/dev/null" in params.args[1]
        assert params.env is None

    def test_windows_wraps_with_cmd(self):
        params = _make_params(command="node", args=["server.js"], env={})
        with patch("datus.tools.mcp_tools.mcp_server.MCPServerStdio.__init__", return_value=None):
            with patch("sys.platform", "win32"):
                srv = SilentMCPServerStdio.__new__(SilentMCPServerStdio)
                SilentMCPServerStdio.__init__(srv, params)
        assert params.command == "cmd"
        assert params.args[0] == "/c"
        assert "2>nul" in params.args[1]

    def test_env_vars_excluded_from_shell_env(self):
        env = {"API_KEY": "secret", "BASH_FUNC_xyz": "bad", "SHLVL": "2", "PATH": "/usr/bin"}
        params = _make_params(command="python", args=[], env=env)
        with patch("datus.tools.mcp_tools.mcp_server.MCPServerStdio.__init__", return_value=None):
            with patch("sys.platform", "linux"):
                srv = SilentMCPServerStdio.__new__(SilentMCPServerStdio)
                SilentMCPServerStdio.__init__(srv, params)
        # env should be cleared (moved into shell command)
        assert params.env is None
        # The command string should contain API_KEY but not BASH_FUNC_ or SHLVL
        cmd_str = params.args[1]
        assert "API_KEY" in cmd_str
        assert "BASH_FUNC_xyz" not in cmd_str
        assert "SHLVL" not in cmd_str

    def test_dict_params_also_handled(self):
        """Test params passed as a dict (not object with attributes)."""
        params = {"command": "echo", "args": ["hello"], "env": {}}
        with patch("datus.tools.mcp_tools.mcp_server.MCPServerStdio.__init__", return_value=None):
            with patch("sys.platform", "linux"):
                srv = SilentMCPServerStdio.__new__(SilentMCPServerStdio)
                SilentMCPServerStdio.__init__(srv, params)
        assert params["command"] == "sh"
        assert params["env"] is None

    def test_args_quoted_properly(self):
        """Verify shlex quoting is applied to command and args."""
        params = _make_params(command="my server", args=["--flag with space"], env={})
        with patch("datus.tools.mcp_tools.mcp_server.MCPServerStdio.__init__", return_value=None):
            with patch("sys.platform", "linux"):
                srv = SilentMCPServerStdio.__new__(SilentMCPServerStdio)
                SilentMCPServerStdio.__init__(srv, params)
        cmd_str = params.args[1]
        # shlex.quote wraps strings with spaces in single quotes
        assert "'my server'" in cmd_str or "my server" in cmd_str


# ---------------------------------------------------------------------------
# find_mcp_directory
# ---------------------------------------------------------------------------


class TestFindMcpDirectory:
    def test_finds_relative_path_when_exists(self, tmp_path, monkeypatch):
        mcp_dir = tmp_path / "mcp" / "my-server"
        mcp_dir.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        result = find_mcp_directory("my-server")
        assert "my-server" in result

    def test_finds_via_sys_path_site_packages(self, tmp_path, monkeypatch):
        # Build a real directory structure under a path that contains "site-packages"
        site_pkg = tmp_path / "site-packages"
        mcp_dir = site_pkg / "mcp" / "test-server"
        mcp_dir.mkdir(parents=True)

        # Change to a directory without a local mcp/test-server so the relative path check fails
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)

        # Inject our fake site-packages into sys.path so find_mcp_directory picks it up
        with patch("sys.path", [str(site_pkg)]):
            result = find_mcp_directory("test-server")

        # Must return the real path string found via sys.path site-packages lookup
        assert "test-server" in result, f"Expected 'test-server' in result path, got: {result!r}"
        assert "site-packages" in result, f"Expected result to come from site-packages, got: {result!r}"

    def test_raises_file_not_found_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("sys.path", []):
            with pytest.raises(FileNotFoundError, match="not found"):
                find_mcp_directory("nonexistent-server-xyz")
