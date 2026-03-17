# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/mcp_tools/mcp_config.py"""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from datus.tools.mcp_tools.mcp_config import (
    HTTPServerConfig,
    MCPConfig,
    MCPServerConfig,
    MCPServerType,
    SSEServerConfig,
    STDIOServerConfig,
    ToolFilterConfig,
    expand_config_env_vars,
    expand_env_vars,
)

# ---------------------------------------------------------------------------
# expand_env_vars
# ---------------------------------------------------------------------------


class TestExpandEnvVars:
    def test_plain_string_unchanged(self):
        assert expand_env_vars("hello world") == "hello world"

    def test_expands_simple_var(self):
        with patch.dict(os.environ, {"MY_TOKEN": "secret123"}):
            assert expand_env_vars("${MY_TOKEN}") == "secret123"

    def test_expands_var_with_default_when_set(self):
        with patch.dict(os.environ, {"MY_HOST": "prod.example.com"}):
            assert expand_env_vars("${MY_HOST:-localhost}") == "prod.example.com"

    def test_expands_var_with_default_when_not_set(self):
        env = os.environ.copy()
        env.pop("MISSING_VAR", None)
        with patch.dict(os.environ, env, clear=True):
            assert expand_env_vars("${MISSING_VAR:-fallback}") == "fallback"

    def test_returns_original_when_var_missing_no_default(self):
        env = os.environ.copy()
        env.pop("TOTALLY_MISSING", None)
        with patch.dict(os.environ, env, clear=True):
            result = expand_env_vars("${TOTALLY_MISSING}")
            assert result == "${TOTALLY_MISSING}"

    def test_multiple_vars_in_string(self):
        with patch.dict(os.environ, {"HOST": "example.com", "PORT": "8080"}):
            result = expand_env_vars("${HOST}:${PORT}")
            assert result == "example.com:8080"


# ---------------------------------------------------------------------------
# expand_config_env_vars
# ---------------------------------------------------------------------------


class TestExpandConfigEnvVars:
    def test_expands_string_values(self):
        with patch.dict(os.environ, {"API_KEY": "abc123"}):
            result = expand_config_env_vars({"key": "${API_KEY}"})
            assert result["key"] == "abc123"

    def test_expands_nested_dict(self):
        with patch.dict(os.environ, {"AUTH": "Bearer xyz"}):
            result = expand_config_env_vars({"headers": {"Authorization": "${AUTH}"}})
            assert result["headers"]["Authorization"] == "Bearer xyz"

    def test_expands_list_values(self):
        with patch.dict(os.environ, {"DIR": "/home/user"}):
            result = expand_config_env_vars({"args": ["--dir", "${DIR}"]})
            assert result["args"] == ["--dir", "/home/user"]

    def test_passthrough_non_string_values(self):
        result = expand_config_env_vars({"count": 42, "flag": True})
        assert result["count"] == 42
        assert result["flag"] is True

    def test_nested_dict_non_string_values(self):
        result = expand_config_env_vars({"headers": {"timeout": 30}})
        assert result["headers"]["timeout"] == 30

    def test_list_with_non_string_values(self):
        result = expand_config_env_vars({"ports": [8080, 8443]})
        assert result["ports"] == [8080, 8443]


# ---------------------------------------------------------------------------
# ToolFilterConfig
# ---------------------------------------------------------------------------


class TestToolFilterConfig:
    def test_default_enabled(self):
        tf = ToolFilterConfig()
        assert tf.enabled is True
        assert tf.allowed_tool_names is None
        assert tf.blocked_tool_names is None

    def test_disabled_filter_allows_everything(self):
        tf = ToolFilterConfig(blocked_tool_names=["dangerous"], enabled=False)
        assert tf.is_tool_allowed("dangerous") is True
        assert tf.is_tool_allowed("anything") is True

    def test_allowlist_only(self):
        tf = ToolFilterConfig(allowed_tool_names=["read", "list"])
        assert tf.is_tool_allowed("read") is True
        assert tf.is_tool_allowed("list") is True
        assert tf.is_tool_allowed("delete") is False

    def test_blocklist_only(self):
        tf = ToolFilterConfig(blocked_tool_names=["delete", "exec"])
        assert tf.is_tool_allowed("read") is True
        assert tf.is_tool_allowed("delete") is False
        assert tf.is_tool_allowed("exec") is False

    def test_allowlist_and_blocklist_combined(self):
        tf = ToolFilterConfig(allowed_tool_names=["read", "write", "delete"], blocked_tool_names=["delete"])
        # In allowlist -> pass allowlist check
        # Then blocked -> fail
        assert tf.is_tool_allowed("read") is True
        assert tf.is_tool_allowed("delete") is False
        # Not in allowlist
        assert tf.is_tool_allowed("list") is False


# ---------------------------------------------------------------------------
# MCPServerType
# ---------------------------------------------------------------------------


class TestMCPServerType:
    def test_enum_values(self):
        assert MCPServerType.STDIO == "stdio"
        assert MCPServerType.SSE == "sse"
        assert MCPServerType.HTTP == "http"


# ---------------------------------------------------------------------------
# STDIOServerConfig
# ---------------------------------------------------------------------------


class TestSTDIOServerConfig:
    def test_basic_creation(self):
        cfg = STDIOServerConfig(name="my-server", command="python")
        assert cfg.name == "my-server"
        assert cfg.command == "python"
        assert cfg.type == MCPServerType.STDIO
        assert cfg.args is None
        assert cfg.env is None

    def test_get_connection_info(self):
        cfg = STDIOServerConfig(name="s", command="node", args=["server.js"], env={"FOO": "bar"})
        info = cfg.get_connection_info()
        assert info["type"] == "stdio"
        assert info["command"] == "node"
        assert info["args"] == ["server.js"]
        assert info["env"] == {"FOO": "bar"}

    def test_get_connection_info_defaults(self):
        cfg = STDIOServerConfig(name="s", command="node")
        info = cfg.get_connection_info()
        assert info["args"] == []
        assert info["env"] == {}

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            STDIOServerConfig(name="s", command="node", type="invalid_type")


# ---------------------------------------------------------------------------
# SSEServerConfig
# ---------------------------------------------------------------------------


class TestSSEServerConfig:
    def test_basic_creation(self):
        cfg = SSEServerConfig(name="sse-srv", url="http://localhost:8080/sse")
        assert cfg.name == "sse-srv"
        assert cfg.url == "http://localhost:8080/sse"
        assert cfg.type == MCPServerType.SSE

    def test_get_connection_info(self):
        cfg = SSEServerConfig(name="s", url="http://example.com", headers={"Auth": "tok"}, timeout=15.0)
        info = cfg.get_connection_info()
        assert info["type"] == "sse"
        assert info["url"] == "http://example.com"
        assert info["headers"] == {"Auth": "tok"}
        assert info["timeout"] == 15.0

    def test_negative_timeout_raises(self):
        with pytest.raises(ValidationError):
            SSEServerConfig(name="s", url="http://example.com", timeout=-1.0)

    def test_zero_timeout_raises(self):
        with pytest.raises(ValidationError):
            SSEServerConfig(name="s", url="http://example.com", timeout=0)

    def test_headers_default_empty(self):
        cfg = SSEServerConfig(name="s", url="http://example.com")
        info = cfg.get_connection_info()
        assert info["headers"] == {}


# ---------------------------------------------------------------------------
# HTTPServerConfig
# ---------------------------------------------------------------------------


class TestHTTPServerConfig:
    def test_basic_creation(self):
        cfg = HTTPServerConfig(name="http-srv", url="http://localhost:9090/mcp")
        assert cfg.name == "http-srv"
        assert cfg.type == MCPServerType.HTTP

    def test_get_connection_info(self):
        cfg = HTTPServerConfig(name="s", url="http://api.example.com", headers={"X-Key": "abc"}, timeout=30.0)
        info = cfg.get_connection_info()
        assert info["type"] == "http"
        assert info["url"] == "http://api.example.com"
        assert info["headers"] == {"X-Key": "abc"}
        assert info["timeout"] == 30.0

    def test_negative_timeout_raises(self):
        with pytest.raises(ValidationError):
            HTTPServerConfig(name="s", url="http://example.com", timeout=-5.0)

    def test_headers_default_empty(self):
        cfg = HTTPServerConfig(name="s", url="http://example.com")
        info = cfg.get_connection_info()
        assert info["headers"] == {}


# ---------------------------------------------------------------------------
# MCPServerConfig.from_config_format factory
# ---------------------------------------------------------------------------


class TestMCPServerConfigFactory:
    def test_create_stdio_from_dict(self):
        cfg = MCPServerConfig.from_config_format("srv", {"type": "stdio", "command": "python", "args": ["-m", "app"]})
        assert isinstance(cfg, STDIOServerConfig)
        assert cfg.command == "python"
        assert cfg.args == ["-m", "app"]

    def test_create_sse_from_dict(self):
        cfg = MCPServerConfig.from_config_format("srv", {"type": "sse", "url": "http://example.com/sse"})
        assert isinstance(cfg, SSEServerConfig)
        assert cfg.url == "http://example.com/sse"

    def test_create_http_from_dict(self):
        cfg = MCPServerConfig.from_config_format("srv", {"type": "http", "url": "http://example.com/mcp"})
        assert isinstance(cfg, HTTPServerConfig)
        assert cfg.url == "http://example.com/mcp"

    def test_defaults_to_stdio_when_no_type(self):
        cfg = MCPServerConfig.from_config_format("srv", {"command": "echo"})
        assert isinstance(cfg, STDIOServerConfig)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown server type"):
            MCPServerConfig.from_config_format("srv", {"type": "grpc"})

    def test_env_vars_expanded_in_command(self):
        with patch.dict(os.environ, {"MY_CMD": "uvicorn"}):
            cfg = MCPServerConfig.from_config_format("srv", {"type": "stdio", "command": "${MY_CMD}"})
        assert isinstance(cfg, STDIOServerConfig)
        assert cfg.command == "uvicorn"

    def test_tool_filter_parsed_from_dict(self):
        cfg = MCPServerConfig.from_config_format(
            "srv",
            {
                "type": "stdio",
                "command": "python",
                "tool_filter": {"allowed_tool_names": ["read"], "enabled": True},
            },
        )
        assert cfg.tool_filter is not None
        assert cfg.tool_filter.allowed_tool_names == ["read"]

    def test_tool_filter_as_toolfilterconfig_object(self):
        tf = ToolFilterConfig(allowed_tool_names=["write"])
        cfg = MCPServerConfig.from_config_format(
            "srv",
            {"type": "stdio", "command": "python", "tool_filter": tf},
        )
        assert cfg.tool_filter is tf

    def test_env_vars_expanded_in_headers(self):
        with patch.dict(os.environ, {"AUTH_TOKEN": "mytoken"}):
            cfg = MCPServerConfig.from_config_format(
                "srv", {"type": "sse", "url": "http://example.com", "headers": {"Authorization": "${AUTH_TOKEN}"}}
            )
        assert isinstance(cfg, SSEServerConfig)
        assert cfg.headers["Authorization"] == "mytoken"

    def test_invalid_type_string_raises(self):
        with pytest.raises(ValueError):
            MCPServerConfig.from_config_format("srv", {"type": "invalid_type", "command": "echo"})


# ---------------------------------------------------------------------------
# MCPConfig
# ---------------------------------------------------------------------------


class TestMCPConfig:
    def test_empty_config(self):
        cfg = MCPConfig()
        assert cfg.servers == {}
        assert cfg.version == "1.0"

    def test_add_server(self):
        cfg = MCPConfig()
        srv = STDIOServerConfig(name="test", command="python")
        cfg.add_server(srv)
        assert "test" in cfg.servers

    def test_remove_server_existing(self):
        cfg = MCPConfig()
        srv = STDIOServerConfig(name="test", command="python")
        cfg.add_server(srv)
        removed = cfg.remove_server("test")
        assert removed is True
        assert "test" not in cfg.servers

    def test_remove_server_missing(self):
        cfg = MCPConfig()
        removed = cfg.remove_server("nonexistent")
        assert removed is False

    def test_get_server_existing(self):
        cfg = MCPConfig()
        srv = STDIOServerConfig(name="test", command="python")
        cfg.add_server(srv)
        result = cfg.get_server("test")
        assert result is srv

    def test_get_server_missing(self):
        cfg = MCPConfig()
        assert cfg.get_server("nonexistent") is None

    def test_list_servers_all(self):
        cfg = MCPConfig()
        cfg.add_server(STDIOServerConfig(name="s1", command="python"))
        cfg.add_server(SSEServerConfig(name="s2", url="http://example.com"))
        assert len(cfg.list_servers()) == 2

    def test_list_servers_filtered_by_type(self):
        cfg = MCPConfig()
        cfg.add_server(STDIOServerConfig(name="s1", command="python"))
        cfg.add_server(SSEServerConfig(name="s2", url="http://example.com"))
        cfg.add_server(HTTPServerConfig(name="s3", url="http://example.com/mcp"))
        sse_servers = cfg.list_servers(server_type=MCPServerType.SSE)
        assert len(sse_servers) == 1
        assert sse_servers[0].name == "s2"

    def test_from_config_format(self):
        data = {
            "mcpServers": {
                "stdio-srv": {"type": "stdio", "command": "python"},
                "sse-srv": {"type": "sse", "url": "http://example.com/sse"},
            }
        }
        cfg = MCPConfig.from_config_format(data)
        assert len(cfg.servers) == 2
        assert "stdio-srv" in cfg.servers
        assert "sse-srv" in cfg.servers

    def test_from_config_format_no_mcp_servers_key(self):
        cfg = MCPConfig.from_config_format({})
        assert cfg.servers == {}

    def test_to_config_format_stdio(self):
        cfg = MCPConfig()
        cfg.add_server(STDIOServerConfig(name="s", command="python", args=["-m", "app"], env={"K": "V"}))
        out = cfg.to_config_format()
        assert "mcpServers" in out
        assert "s" in out["mcpServers"]
        s_cfg = out["mcpServers"]["s"]
        assert s_cfg["command"] == "python"
        assert s_cfg["args"] == ["-m", "app"]
        assert s_cfg["env"] == {"K": "V"}

    def test_to_config_format_sse(self):
        cfg = MCPConfig()
        cfg.add_server(SSEServerConfig(name="s", url="http://example.com", headers={"H": "val"}, timeout=10.0))
        out = cfg.to_config_format()
        s_cfg = out["mcpServers"]["s"]
        assert s_cfg["url"] == "http://example.com"
        assert s_cfg["headers"] == {"H": "val"}
        assert s_cfg["timeout"] == 10.0

    def test_to_config_format_http(self):
        cfg = MCPConfig()
        cfg.add_server(HTTPServerConfig(name="s", url="http://api.example.com", timeout=20.0))
        out = cfg.to_config_format()
        s_cfg = out["mcpServers"]["s"]
        assert s_cfg["url"] == "http://api.example.com"
        assert s_cfg["timeout"] == 20.0

    def test_to_config_format_with_tool_filter(self):
        cfg = MCPConfig()
        tf = ToolFilterConfig(allowed_tool_names=["read"], enabled=True)
        cfg.add_server(STDIOServerConfig(name="s", command="python", tool_filter=tf))
        out = cfg.to_config_format()
        s_cfg = out["mcpServers"]["s"]
        assert "tool_filter" in s_cfg
        assert s_cfg["tool_filter"]["allowed_tool_names"] == ["read"]

    def test_roundtrip_config_format(self):
        data = {
            "mcpServers": {
                "my-server": {
                    "type": "stdio",
                    "command": "uvicorn",
                    "args": ["app:app"],
                    "env": {"ENV": "prod"},
                }
            }
        }
        cfg = MCPConfig.from_config_format(data)
        out = cfg.to_config_format()
        assert out["mcpServers"]["my-server"]["command"] == "uvicorn"
        assert out["mcpServers"]["my-server"]["args"] == ["app:app"]

    def test_mcpservertype_validate_type_invalid(self):
        with pytest.raises(ValidationError):
            STDIOServerConfig(name="s", command="echo", type="bad_type")
