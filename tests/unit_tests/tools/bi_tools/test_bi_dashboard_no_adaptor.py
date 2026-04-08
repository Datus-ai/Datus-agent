# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for BI dashboard behavior when no adaptor packages are installed."""

from unittest.mock import MagicMock, patch

import pytest
from datus_bi_core import AuthParam


class TestNoAdaptorInstalled:
    """Verify graceful errors when no BI adaptor plugins are available."""

    @pytest.fixture
    def empty_registry_commands(self):
        """Create BiDashboardCommands with an empty adaptor registry."""
        from datus.cli.bi_dashboard import BiDashboardCommands

        agent_config = MagicMock()
        agent_config.db_type = "postgresql"
        agent_config.namespaces = MagicMock()
        with patch("datus_bi_core.registry.BIAdaptorRegistry.list_adaptors", return_value={}):
            return BiDashboardCommands(agent_config=agent_config, force=True)

    def test_prompt_options_raises_when_no_adaptors(self, empty_registry_commands):
        """_prompt_options should raise ValueError when registry is empty."""
        with pytest.raises(ValueError, match="No BI adaptor implementations found.*pip install datus-agent"):
            empty_registry_commands._prompt_options()

    def test_create_adaptor_raises_for_unknown_platform(self, empty_registry_commands):
        """_create_adaptor should raise ValueError for unregistered platform."""
        from datus.cli.bi_dashboard import DashboardCliOptions

        options = DashboardCliOptions(
            platform="superset",
            dashboard_url="http://localhost:8088/superset/dashboard/1/",
            api_base_url="http://localhost:8088",
            auth_params=AuthParam(username="admin", password="admin"),
        )
        with pytest.raises(ValueError, match="Unsupported platform 'superset'.*pip install datus-bi-superset"):
            empty_registry_commands._create_adaptor(options)
