# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/models/mcp_utils.py

CI-level: zero external dependencies. MCP servers are fakes whose handshake
either hangs, fails, or succeeds on demand.
"""

import asyncio
import time

import pytest

from datus.models import mcp_utils
from datus.models.mcp_utils import _safe_connect_server, multiple_mcp_servers


class FakeServer:
    """Minimal stand-in for an agents-SDK MCP server."""

    def __init__(self, *, hang: bool = False, fail_times: int = 0):
        self.hang = hang
        self.fail_times = fail_times
        self.enter_calls = 0
        self.exit_calls = 0

    async def __aenter__(self):
        self.enter_calls += 1
        if self.hang:
            await asyncio.sleep(3600)
        if self.enter_calls <= self.fail_times:
            raise ConnectionError("handshake refused")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exit_calls += 1
        return False


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    """Keep the retry backoff out of the test wall clock."""
    monkeypatch.setattr(mcp_utils, "RETRY_BACKOFF_SECONDS", 0.0)


class TestSafeConnectServer:
    @pytest.mark.asyncio
    async def test_hanging_handshake_times_out_per_attempt(self):
        server = FakeServer(hang=True)
        started = time.monotonic()

        with pytest.raises(TimeoutError):
            async with _safe_connect_server("hung", server, max_retries=2, connect_timeout=0.05):
                pytest.fail("should never connect")

        # 2 attempts x 50ms, not 2 x the SDK's own minute-long session timeout.
        assert time.monotonic() - started < 1.0
        assert server.enter_calls == 2
        assert server.exit_calls == 2

    @pytest.mark.asyncio
    async def test_deadline_stops_further_attempts(self):
        server = FakeServer(hang=True)
        deadline = time.monotonic() + 0.05

        with pytest.raises(TimeoutError):
            async with _safe_connect_server("hung", server, max_retries=5, connect_timeout=10.0, deadline=deadline):
                pytest.fail("should never connect")

        # The deadline, not connect_timeout, is what bounds the attempt.
        assert server.enter_calls < 5

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        server = FakeServer(fail_times=1)

        async with _safe_connect_server("flaky", server, max_retries=3, connect_timeout=1.0) as connected:
            assert connected is server

        assert server.enter_calls == 2
        # One teardown for the failed handshake, one for the successful session.
        assert server.exit_calls == 2

    @pytest.mark.asyncio
    async def test_body_is_not_cancelled_by_connect_timeout(self):
        """The timeout covers the handshake only, never the caller's work."""
        server = FakeServer()

        async with _safe_connect_server("ok", server, max_retries=1, connect_timeout=0.05):
            await asyncio.sleep(0.2)

        assert server.exit_calls == 1


class TestMultipleMcpServers:
    @pytest.mark.asyncio
    async def test_unreachable_server_is_skipped_not_fatal(self):
        servers = {"broken": FakeServer(fail_times=99), "good": FakeServer()}

        async with multiple_mcp_servers(servers, connect_budget=5.0) as connected:
            assert "broken" not in connected
            assert connected["good"] is servers["good"]

    @pytest.mark.asyncio
    async def test_budget_bounds_the_whole_connect_phase(self):
        servers = {f"hung{i}": FakeServer(hang=True) for i in range(4)}
        started = time.monotonic()

        async with multiple_mcp_servers(servers, connect_budget=0.2) as connected:
            assert connected == {}

        # Without a shared budget this was servers x retries x per-attempt timeout.
        assert time.monotonic() - started < 2.0

    @pytest.mark.asyncio
    async def test_servers_after_budget_exhaustion_are_not_dialled(self):
        hung = FakeServer(hang=True)
        never = FakeServer()

        async with multiple_mcp_servers({"hung": hung, "never": never}, connect_budget=0.1) as connected:
            assert connected == {}

        assert never.enter_calls == 0

    @pytest.mark.asyncio
    async def test_empty_mapping_yields_empty_dict(self):
        async with multiple_mcp_servers({}) as connected:
            assert connected == {}


class TestEnvOverrides:
    def test_connect_timeout_env_override(self, monkeypatch):
        monkeypatch.setenv("DATUS_MCP_CONNECT_TIMEOUT", "3.5")
        assert mcp_utils.connect_timeout_seconds() == 3.5

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("DATUS_MCP_CONNECT_BUDGET", "not-a-number")
        assert mcp_utils.connect_budget_seconds() == mcp_utils.DEFAULT_CONNECT_BUDGET_SECONDS

    def test_non_positive_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("DATUS_MCP_CONNECT_TIMEOUT", "0")
        assert mcp_utils.connect_timeout_seconds() == mcp_utils.DEFAULT_CONNECT_TIMEOUT_SECONDS
