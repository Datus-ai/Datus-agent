# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for BI dashboard behavior when no adapter packages are installed."""

from unittest.mock import MagicMock, patch

import pytest

# datus-bi-core is a hard dependency (see pyproject.toml [project.dependencies]);
# import directly rather than importorskip so a missing install fails loudly.
from datus_bi_core import AuthParam


class _PaginatedResult:
    def __init__(self, items):
        self.items = items


class TestNoAdapterInstalled:
    """Verify graceful errors when no BI adapter plugins are available."""

    @pytest.fixture
    def empty_registry_commands(self):
        """Create BiDashboardCommands with an empty adapter registry."""
        from datus.cli.bi_dashboard import BiDashboardCommands

        agent_config = MagicMock()
        agent_config.db_type = "postgresql"
        agent_config.datasource_configs = MagicMock()
        with patch("datus_bi_core.registry.BIAdapterRegistry.list_adapters", return_value={}):
            return BiDashboardCommands(agent_config=agent_config, force=True)

    def test_prompt_options_raises_when_no_adapters(self, empty_registry_commands):
        """_prompt_options should raise ValueError when registry is empty."""
        with pytest.raises(ValueError, match="No BI adapter implementations found.*pip install datus-bi-superset"):
            empty_registry_commands._prompt_options()

    def test_create_adapter_raises_for_unknown_platform(self, empty_registry_commands):
        """_create_adapter should raise ValueError for unregistered platform."""
        from datus.cli.bi_dashboard import DashboardCliOptions

        options = DashboardCliOptions(
            platform="superset",
            dashboard_url="http://localhost:8088/superset/dashboard/1/",
            api_base_url="http://localhost:8088",
            auth_params=AuthParam(username="admin", password="admin"),
        )
        with pytest.raises(ValueError, match="Unsupported platform 'superset'.*pip install datus-bi-superset"):
            empty_registry_commands._create_adapter(options)

    def test_items_from_adapter_result_accepts_paginated_result(self, empty_registry_commands):
        """Adapter list methods may return a PaginatedResult envelope."""
        items = [object(), object()]

        assert empty_registry_commands._items_from_adapter_result(_PaginatedResult(items)) == items

    def test_items_from_adapter_result_accepts_plain_sequence(self, empty_registry_commands):
        """Legacy adapters may still return a plain sequence."""
        items = [object(), object()]

        assert empty_registry_commands._items_from_adapter_result(items) == items
