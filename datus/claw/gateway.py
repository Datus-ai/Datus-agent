# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""ClawGateway — lifecycle orchestrator for IM channel adapters."""

import asyncio
import signal
import sys
from typing import Dict, Optional

from datus.api.services.chat_task_manager import ChatTaskManager
from datus.claw.bridge import ChannelBridge
from datus.claw.channel.base import ChannelAdapter
from datus.claw.channel.registry import get_adapter_class, register_builtins
from datus.claw.models import ChannelConfig
from datus.configuration.agent_config import AgentConfig
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ClawGateway:
    """Manages the lifecycle of all IM channel adapters."""

    def __init__(
        self,
        agent_config: AgentConfig,
        channels_config: Dict[str, dict],
        host: str = "0.0.0.0",
        port: int = 9000,
    ) -> None:
        self._agent_config = agent_config
        self._channels_config = channels_config
        self._host = host
        self._port = port
        self._task_manager = ChatTaskManager(default_interactive=False)
        self._bridge = ChannelBridge(agent_config, self._task_manager)
        self._adapters: Dict[str, ChannelAdapter] = {}
        self._shutdown_event: Optional[asyncio.Event] = None

    async def start(self) -> None:
        """Register built-in adapters, instantiate channels, and run until shutdown signal."""
        register_builtins()

        self._shutdown_event = asyncio.Event()

        # Wire up OS signals (not supported on Windows)
        if sys.platform != "win32":
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._shutdown_event.set)

        # Instantiate and configure adapters
        for channel_id, raw_cfg in self._channels_config.items():
            channel_cfg = ChannelConfig(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
            if not channel_cfg.enabled:
                logger.info("Channel '%s' is disabled, skipping.", channel_id)
                continue

            adapter_cls = get_adapter_class(channel_cfg.adapter)
            adapter = adapter_cls(
                channel_id=channel_id,
                config=channel_cfg.extra,
                bridge=self._bridge,
                channel_config=channel_cfg,
            )
            self._adapters[channel_id] = adapter

        if not self._adapters:
            logger.warning("No enabled channels configured. Gateway has nothing to do.")
            return

        # Start all adapters concurrently
        logger.info("Starting %d channel adapter(s)...", len(self._adapters))
        await asyncio.gather(*(a.start() for a in self._adapters.values()))
        logger.info("All channel adapters started. Waiting for shutdown signal...")

        # Block until signal
        await self._shutdown_event.wait()
        await self.shutdown()

    async def shutdown(self) -> None:
        """Stop all adapters and clean up."""
        logger.info("Shutting down gateway...")
        stop_tasks = [a.stop() for a in self._adapters.values()]
        results = await asyncio.gather(*stop_tasks, return_exceptions=True)
        for channel_id, result in zip(self._adapters.keys(), results):
            if isinstance(result, Exception):
                logger.error("Error stopping adapter '%s': %s", channel_id, result)
        await self._task_manager.shutdown()
        self._adapters.clear()
        logger.info("Gateway shutdown complete.")
