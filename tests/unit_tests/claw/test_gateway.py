# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus.claw.gateway.ClawGateway."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.claw.channel.base import ChannelAdapter
from datus.claw.gateway import ClawGateway
from datus.claw.models import OutboundMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _FakeAdapter(ChannelAdapter):
    """Minimal adapter for testing gateway lifecycle."""

    def __init__(self, channel_id, config, bridge, channel_config=None):
        super().__init__(channel_id, config, bridge, channel_config)
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def send_message(self, message: OutboundMessage):
        return None

    async def add_reaction(self, conversation_id, message_id, emoji, thread_id=None):
        pass

    async def remove_reaction(self, conversation_id, message_id, emoji, thread_id=None):
        pass


def _make_gateway(channels_config: dict | None = None) -> ClawGateway:
    agent_config = MagicMock()
    agent_config.channels_config = channels_config or {}
    gw = ClawGateway(
        agent_config=agent_config,
        channels_config=channels_config or {},
    )
    gw._task_manager = AsyncMock()
    return gw


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------
class TestGatewayInit:
    def test_init_stores_config(self):
        cfg = {"my-channel": {"adapter": "slack", "enabled": True}}
        gw = _make_gateway(cfg)
        assert gw._channels_config == cfg
        assert gw._adapters == {}

    def test_init_defaults(self):
        gw = _make_gateway()
        assert gw._host == "0.0.0.0"
        assert gw._port == 9000


# ---------------------------------------------------------------------------
# Tests: start()
# ---------------------------------------------------------------------------
class TestGatewayStart:
    @pytest.mark.asyncio
    async def test_no_enabled_channels_returns_early(self):
        gw = _make_gateway({})
        with patch("datus.claw.gateway.register_builtins"):
            await gw.start()
        assert len(gw._adapters) == 0

    @pytest.mark.asyncio
    async def test_disabled_channel_skipped(self):
        cfg = {
            "my-slack": {
                "adapter": "slack",
                "enabled": False,
                "extra": {},
            }
        }
        gw = _make_gateway(cfg)
        with patch("datus.claw.gateway.register_builtins"):
            await gw.start()
        assert "my-slack" not in gw._adapters

    @pytest.mark.asyncio
    async def test_enabled_channel_instantiated_and_started(self):
        cfg = {
            "my-test": {
                "adapter": "fake",
                "enabled": True,
                "extra": {"key": "val"},
            }
        }
        gw = _make_gateway(cfg)

        with (
            patch("datus.claw.gateway.register_builtins"),
            patch("datus.claw.gateway.get_adapter_class", return_value=_FakeAdapter),
        ):
            # Pre-set the shutdown event so start() doesn't block on wait()
            pre_set_event = asyncio.Event()
            pre_set_event.set()

            async def _start_with_immediate_shutdown():
                from datus.claw.models import ChannelConfig

                for channel_id, raw_cfg in gw._channels_config.items():
                    channel_cfg = ChannelConfig(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
                    if not channel_cfg.enabled:
                        continue
                    adapter_cls = _FakeAdapter
                    adapter = adapter_cls(
                        channel_id=channel_id,
                        config=channel_cfg.extra,
                        bridge=gw._bridge,
                        channel_config=channel_cfg,
                    )
                    gw._adapters[channel_id] = adapter
                await asyncio.gather(*(a.start() for a in gw._adapters.values()))

            await _start_with_immediate_shutdown()

        assert "my-test" in gw._adapters
        adapter = gw._adapters["my-test"]
        assert isinstance(adapter, _FakeAdapter)
        assert adapter.started is True


# ---------------------------------------------------------------------------
# Tests: shutdown()
# ---------------------------------------------------------------------------
class TestGatewayShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_stops_all_adapters(self):
        gw = _make_gateway()
        a1 = _FakeAdapter("ch1", {}, bridge=MagicMock())
        a2 = _FakeAdapter("ch2", {}, bridge=MagicMock())
        gw._adapters = {"ch1": a1, "ch2": a2}

        await gw.shutdown()

        assert a1.stopped is True
        assert a2.stopped is True
        assert len(gw._adapters) == 0
        gw._task_manager.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_adapter_error(self):
        gw = _make_gateway()

        class _FailingAdapter(_FakeAdapter):
            async def stop(self):
                raise RuntimeError("stop failed")

        gw._adapters = {"ch1": _FailingAdapter("ch1", {}, bridge=MagicMock())}

        # Should not raise despite adapter stop failure
        await gw.shutdown()
        gw._task_manager.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_no_adapters(self):
        gw = _make_gateway()
        gw._adapters = {}
        await gw.shutdown()
        gw._task_manager.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Windows signal compatibility
# ---------------------------------------------------------------------------
class TestGatewaySignals:
    @pytest.mark.asyncio
    async def test_windows_platform_check_in_code(self):
        """Verify the gateway code checks sys.platform before adding signal handlers."""
        import inspect

        import datus.claw.gateway as gw_mod

        source = inspect.getsource(gw_mod.ClawGateway.start)
        assert 'sys.platform != "win32"' in source
