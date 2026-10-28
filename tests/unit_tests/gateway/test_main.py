# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the gateway CLI entry point's config bootstrap."""

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from datus.gateway import main as gateway_main


def _args(**overrides) -> argparse.Namespace:
    values = {"config": "", "datasource": "default", "host": "0.0.0.0", "port": 9000}
    values.update(overrides)
    return argparse.Namespace(**values)


class TestRunGatewayBootstrap:
    """_run_gateway must mark the loaded config for the IM surface."""

    def _run(self, agent_config) -> MagicMock:
        gateway_cls = MagicMock()
        with (
            patch("datus.configuration.agent_config_loader.load_agent_config", return_value=agent_config),
            patch("datus.gateway.runtime.DatusGateway", gateway_cls),
            patch.object(gateway_main.asyncio, "run"),
        ):
            gateway_main._run_gateway(_args())
        return gateway_cls

    def test_bootstrap_flags_are_forced_for_the_im_surface(self):
        agent_config = SimpleNamespace(
            filesystem_strict=False,
            compile_visual_html=True,
            config_mutable=True,
            active_model=MagicMock(),
            channels_config={"slack": {}},
        )

        gateway_cls = self._run(agent_config)

        # No broker to confirm out-of-workspace access.
        assert agent_config.filesystem_strict is True
        # IM channels cannot serve a local HTML artifact.
        assert agent_config.compile_visual_html is False
        # IM users must never be guided to edit the server's config file.
        assert agent_config.config_mutable is False
        assert gateway_cls.call_args.kwargs["agent_config"] is agent_config

    def test_missing_channels_config_exits(self):
        agent_config = SimpleNamespace(
            filesystem_strict=False,
            compile_visual_html=True,
            config_mutable=True,
            active_model=MagicMock(),
            channels_config={},
        )

        with pytest.raises(SystemExit):
            self._run(agent_config)
